"""Mixed direct-local/reasoning-global native streaming software adapter.

No fitting, answer scoring or adaptive gating. Original native relevance gates
are reused across actual generated prefixes; payload states remain contextual.
The caller owns native identity verification. All numerical work requires Slurm.
"""
from contextlib import ExitStack
import time
from scripts import native_vision_semantic_gate_runtime as semantic
from scripts import native_vision_reasoning_stream as stream
from gnnformer.parallel_local_semantic_gate import ParallelLocalSemanticGate
native=semantic.native
need=native.need
OWN=('scripts/native_vision_reasoning_semantic_stream.py',)


def _controller(model,core,bundle,identity,artifact=None):
    meta=bundle['metadata'];width=meta['prompt_width']
    return ParallelLocalSemanticGate(semantic.native_contract(model,core),model.lm_head,core,
        n_local_rows=meta['n_frames'],mode='native_gate',zero_token_id=15,one_token_id=16,
        origin_identity=semantic.origin_identity(bundle,identity),query_indices=[width-1],
        stream_positions=[width-1],origin_artifact=artifact,capture=True)


def _validate(model,processor,core,bundle,identity):
    semantic.validate_bundle(bundle);meta=bundle['metadata']
    need(meta['local_mode']=='direct_local' and meta['global_system']==stream.SYSTEM
         and meta['global_prompt']==stream.prompt(meta['question']), 'Require the audited mixed Cosmos prompt')
    semantic.origin_identity(bundle,identity);semantic.digit_token_ids(processor.tokenizer)
    norm=native.native_contract(model) if core is None else semantic.native_contract(model,core)
    if core is not None:need(not core.training and not any(p.requires_grad or p.grad is not None for p in core.parameters()),'Freeze the software core')
    return norm


def generate_native(model,processor,core,bundle,*,native_identity_sha256,max_new_tokens=8,
                    capture=True,controller_observer=None):
    """One N+1-row invocation/token, one visual prefill, one gate probe if gated.

    controller_observer is a software-audit callback receiving the active hook
    owner (or None). It does not alter logits. No partial cache escapes this call.
    """
    native.require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
    from gnnformer.runtime import get_rope_index_fn,move_to_device
    norm=_validate(model,processor,core,bundle,native_identity_sha256)
    need(max_new_tokens==8,'This software adapter has a fixed eight-token maximum')
    meta=bundle['metadata'];width,batch=meta['prompt_width'],meta['row_count']
    need(width==meta['original_prompt_width'] and not meta['prefix_ids'],'Start from the original empty prefix')
    modules=[('native',model)]+([] if core is None else [('core',core)])
    versions={(kind,name):p._version for kind,m in modules for name,p in m.named_parameters()}
    original_rope=getattr(model.model,'rope_deltas',None)
    layout=native.audit_layout(get_rope_index_fn(model),bundle)
    config,policy=stream.generation_policy(model,processor.tokenizer,max_new_tokens)
    recorder=stream.GlobalLogitRecorder(max_steps=8,full_vectors=True)
    broadcast=GlobalBroadcastLogitsProcessor(n_local_rows=meta['n_frames'],prompt_length=width)
    counts=dict(model=0,visual=0,language=0,norm=0,head=0);step_inputs=[];positions=[];captures=[]
    fusion=None
    def before(module,args,kwargs):
        step=counts['model'];counts['model']+=1
        if fusion is not None:fusion.configure_queries([width-1 if step==0 else 0],[width+step-1])
        ids,mask=kwargs['input_ids'],kwargs['attention_mask']
        need(ids.shape==(batch,width if step==0 else 1) and mask.shape==(batch,width+step)
             and torch.equal(mask[:,:width].cpu(),bundle['inputs']['attention_mask'])
             and bool((mask[:,width:]==1).all()),'Native query/history mask differs')
        if step==0:need(torch.equal(ids.cpu(),bundle['inputs']['input_ids']) and kwargs.get('pixel_values') is not None,'Prefill inputs/images differ')
        else:need(kwargs.get('pixel_values') is None and kwargs.get('past_key_values') is not None
                  and kwargs['past_key_values'].get_seq_length()==width+step-1
                  and torch.equal(ids,ids[-1:].expand_as(ids)),'Native cache/shared prefix differs')
        step_inputs.append(ids.detach().cpu().clone())
    def count(key):
        def hook(*_):counts[key]+=1
        return hook
    def language(module,args,kwargs):
        step=counts['language'];counts['language']+=1;actual=kwargs['position_ids'].detach().cpu();mask=kwargs['attention_mask'].detach().cpu()
        text=mask.long().cumsum(-1)-1
        expected=layout['position_ids'] if step==0 else (layout['rope_deltas'].view(1,batch,1)+width+step-1).expand(3,-1,-1)
        need(actual.shape==(4,batch,width if step==0 else 1) and torch.equal(actual[1:],expected),'Native mRoPE differs')
        need(torch.equal(actual[0][mask.bool()],text[mask.bool()]) if step==0 else torch.equal(actual[0],text[:,-1:]),'Native text positions differ')
        positions.append(native.tensor_info(actual))
    def after(module,args,output):
        if fusion is not None:
            need(fusion.calls==counts['model'] and fusion.probe_norm_calls==fusion.probe_head_calls==fusion.probability_calls==1,'Gate probe/call counts differ')
            if capture:captures.append(fusion.export_last_capture(cpu=True))
    start=time.perf_counter()
    with ExitStack() as stack:
        stack.callback(setattr,model.model,'rope_deltas',original_rope)
        if core is not None:fusion=stack.enter_context(_controller(model,core,bundle,native_identity_sha256))
        if controller_observer is not None:
            controller_observer(fusion);stack.callback(controller_observer,None)
        for handle in (model.register_forward_pre_hook(before,with_kwargs=True),model.register_forward_hook(recorder),
            model.register_forward_hook(after),model.model.visual.register_forward_pre_hook(count('visual')),
            model.model.language_model.register_forward_pre_hook(language,with_kwargs=True),
            norm.register_forward_pre_hook(count('norm')),model.lm_head.register_forward_pre_hook(count('head'))):stack.callback(handle.remove)
        with torch.inference_mode():
            output=model.generate(**move_to_device(bundle['inputs'],model.device),generation_config=config,
                logits_processor=LogitsProcessorList([broadcast]),logits_to_keep=1)
        torch.cuda.synchronize();suffix=output.sequences[:,width:];ids=suffix[-1].detach().cpu().tolist();steps=len(ids)
        need(getattr(output,'logits',None) is None and getattr(output,'scores',None) is None,'HF retained all-row logits/scores')
        need(1<=steps<=8 and torch.equal(suffix,suffix[-1:].expand_as(suffix))
             and counts==dict(model=steps,visual=1,language=steps,norm=steps,head=steps)
             and broadcast.calls==len(recorder.records)==steps,'Native generation inventory differs')
        need([r['top1_token_id'] for r in recorder.records]==ids
             and all(bool((step_inputs[i]==ids[i-1]).all()) for i in range(1,steps)),'Raw argmax/shared history differs')
        eos=policy['native_eos_token_ids'];completed=ids[-1] in eos
        need(not any(x in eos for x in ids[:-1]) and (completed or steps==8),'Unexpected native stopping')
        artifact=None if fusion is None else fusion.export_origin_artifact(cpu=True)
        status=None if fusion is None else fusion.assert_complete()
    need((fusion is None or not fusion.active) and versions=={(kind,name):p._version for kind,m in modules for name,p in m.named_parameters()},'Parameter mutation or leaked fusion hook')
    return dict(generated_ids=ids,raw_logits=torch.stack(recorder.vectors),logit_records=recorder.records,
        raw_text=processor.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
        completed=completed,truncated=not completed,metadata=dict(meta,generation=policy,layout=layout['metadata'],generation_position_ids=positions),
        counters=dict(counts,broadcast=steps,fusion=0 if fusion is None else steps,
            probe_norm=0 if fusion is None else 1,probe_head=0 if fusion is None else 1,probability=0 if fusion is None else 1),
        captures=captures,origin_artifact=artifact,fusion_audit=status,model_seconds=time.perf_counter()-start)


def forward_native(model,processor,core,bundle,*,native_identity_sha256,origin_artifact,controller_observer=None):
    """Actual-prefix uncached replay; reuse the origin even for empty-prefix t0."""
    native.require_slurm(gpu=True)
    import torch
    from gnnformer.runtime import get_rope_index_fn,move_to_device
    _validate(model,processor,core,bundle,native_identity_sha256)
    need(origin_artifact is not None,'Every replay must reuse the trajectory origin')
    layout=native.audit_layout(get_rope_index_fn(model),bundle);original_rope=getattr(model.model,'rope_deltas',None)
    with ExitStack() as stack:
        stack.callback(setattr,model.model,'rope_deltas',original_rope)
        fusion=stack.enter_context(_controller(model,core,bundle,native_identity_sha256,origin_artifact))
        if controller_observer is not None:controller_observer(fusion);stack.callback(controller_observer,None)
        with torch.inference_mode():output=model(**move_to_device(bundle['inputs'],model.device),use_cache=False,logits_to_keep=1)
        need(output.past_key_values is None and output.logits.dtype==torch.float16,'Uncached native output contract differs')
        result=dict(global_logits=output.logits[-1,-1].detach().float().cpu().clone(),fusion=fusion.export_last_capture(cpu=True),
                    origin_artifact_sha256=origin_artifact['artifact_sha256'],fusion_audit=fusion.assert_complete(),layout=layout['metadata'])
        need(fusion.probe_head_calls==fusion.probe_norm_calls==fusion.probability_calls==0,'Replay reclassified the current prefix')
    need(not fusion.active,'Replay hook survived cleanup');return result

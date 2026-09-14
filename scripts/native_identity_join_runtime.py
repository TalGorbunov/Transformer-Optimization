"""Held binary-origin native identity-join integration; no fit or efficacy.

Original inclusion measurement is frozen across native answer prefixes. The
vector/scalar mode is explicit and immutable; only the global final-norm query
receives a residual. Invalid native measurements return retained failure data.
"""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import time
from scripts import native_vision_v7_runtime as native
from scripts import native_vision_semantic_gate_runtime as shared
from scripts.native_vision_reasoning_stream import GlobalLogitRecorder
from scripts.probe_native_identity_join import local_prompt
from gnnformer.parallel_local_binary_aggregation import ParallelLocalBinaryAggregation
from gnnformer.parallel_local_binary_gate import ParallelLocalBinaryGate,InvalidBinaryMeasurement,BINARY_GATE_RULE
need=native.need
OWN=('scripts/native_identity_join_runtime.py','gnnformer/parallel_local_binary_gate.py',
     'gnnformer/parallel_local_binary_aggregation.py','tests/test_parallel_local_binary_gate.py',
     'gnnformer/parallel_local_semantic_gate.py','gnnformer/parallel_local_semantic_aggregation.py',
     'gnnformer/parallel_local_aggregation.py','gnnformer/parallel_local_native.py','gnnformer/runtime.py','gnnformer/constants.py',
     'scripts/native_vision_v7_runtime.py','scripts/native_vision_semantic_gate_runtime.py',
     'scripts/native_vision_reasoning_stream.py','scripts/probe_native_identity_join.py')


def sources():
    root=Path(__file__).resolve().parents[1]
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in OWN}


validate_bundle=shared.validate_bundle
origin_identity=shared.origin_identity
append_observed_prefix=shared.append_observed_prefix


def prepare_scene(processor,sample,*,prefix_ids=(),verify_processor_parity=True):
    """Consume only SID, actual N, the complete question and original images."""
    native.require_slurm()
    need(set(sample)=={'sid','n_frames','question','image_files'},'Model input view may not contain labels/roles/answers')
    need(type(sample['n_frames']) is int and sample['n_frames']>0 and len(sample['image_files'])==sample['n_frames'],'Actual image count differs')
    import torch
    from PIL import Image
    images=[];old=processor.tokenizer.padding_side
    try:
        for image in sample['image_files']:
            path=Path(image['path']);need(hashlib.sha256(path.read_bytes()).hexdigest()==image['sha256'],'Original image bytes changed')
            with Image.open(path) as original:
                rgb=original.convert('RGB')
                try:images.append(rgb.resize((392,392)))
                finally:rgb.close()
        question=sample['question'];text=local_prompt(question)
        conversations=[[dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=text)])] for image in images]
        conversations.append([dict(role='user',content=[dict(type='text',text=question)])])
        rows=[dict(processor.apply_chat_template(c,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')) for c in conversations]
        packed=native.pack_rows(rows,processor.tokenizer.pad_token_id)
        if verify_processor_parity:
            processor.tokenizer.padding_side='left'
            ordinary=dict(processor.apply_chat_template(conversations,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',padding=True))
            need(set(ordinary)==set(packed) and all(torch.equal(ordinary[k],packed[k]) for k in packed),'Ordinary mixed HF processor parity failed')
        n=sample['n_frames'];width=packed['input_ids'].shape[1]
        meta=dict(arm='parallel',sid=sample['sid'],n_frames=n,question=question,question_sha256=native.object_sha(question),
            row_count=n+1,global_row=n,local_elements=n,row_kinds=['local']*n+['global'],original_prompt_width=width,prompt_width=width,
            prefix_ids=[],row_prompt_tokens=[r['input_ids'].shape[1] for r in rows],local_prompt=text,global_prompt=question,resize=392,
            image_paths=[x['path'] for x in sample['image_files']],image_sha256=[x['sha256'] for x in sample['image_files']],
            input_identity={k:native.tensor_info(v) for k,v in packed.items()},processor_parity_checked=verify_processor_parity)
        return append_observed_prefix(dict(inputs=packed,row_inputs=rows,metadata=meta),prefix_ids)
    finally:
        processor.tokenizer.padding_side=old
        for image in images:image.close()


def native_contract(model,core,branch_mode):
    norm=native.native_contract(model,core)
    if core is None:need(branch_mode is None,'Bare profile has no branch mode')
    else:need(type(core) is ParallelLocalBinaryAggregation and branch_mode in ('vector','scalar') and core.mode==branch_mode
        and not core.training and not any(p.requires_grad or p.grad is not None for p in core.parameters()),'Frozen core and explicit checkpoint branch_mode must agree')
    return norm


def _controller(model,core,bundle,identity,artifact):
    width=bundle['metadata']['prompt_width']
    return ParallelLocalBinaryGate(model.model.language_model.norm,model.lm_head,core,n_local_rows=bundle['metadata']['n_frames'],
        zero_token_id=15,one_token_id=16,origin_identity=origin_identity(bundle,identity),query_indices=[width-1],stream_positions=[width-1],
        origin_artifact=artifact,capture=True)


def _failure(exc,metadata,branch_mode,counters):
    return dict(measurement_passed=False,measurement_performed=True,measurement_failure=True,failure_type='invalid_binary_measurement',
        invalid_rows=list(exc.artifact['invalid_rows']),origin_artifact=exc.artifact,generated_ids=[],raw_logits=None,
        completed=False,truncated=False,finish_reason='invalid_binary_measurement',branch_mode=branch_mode,
        metadata=metadata,counters=counters,gate_rule=BINARY_GATE_RULE)


def generate_native(model,processor,core,bundle,*,branch_mode=None,native_identity_sha256,max_new_tokens=4,
                    capture=True,controller_observer=None):
    native.require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.runtime import move_to_device,get_rope_index_fn
    from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
    norm=native_contract(model,core,branch_mode);validate_bundle(bundle);shared.digit_token_ids(processor.tokenizer)
    meta=bundle['metadata'];width,batch=meta['prompt_width'],meta['row_count']
    need(meta['local_prompt']==local_prompt(meta['question']) and meta['global_prompt']==meta['question']
         and not meta['prefix_ids'] and width==meta['original_prompt_width'],'Require exact generic prompts and empty initial prefix')
    origin_identity(bundle,native_identity_sha256)
    need(max_new_tokens==4,'Identity-join answer budget is fixed at four native tokens')
    config,policy=native.generation_policy(model,processor.tokenizer,max_new_tokens=4)
    config.output_logits=False;config.output_scores=False;policy=dict(policy,output_logits=False,output_scores=False,global_row_streaming=True)
    recorder=GlobalLogitRecorder(max_steps=4,full_vectors=True);broadcast=GlobalBroadcastLogitsProcessor(n_local_rows=meta['n_frames'],prompt_length=width)
    layout=native.audit_layout(get_rope_index_fn(model),bundle);rope=getattr(model.model,'rope_deltas',None)
    modules=[('native',model)]+([] if core is None else [('core',core)])
    versions={(k,n):p._version for k,m in modules for n,p in m.named_parameters()}
    counts=dict(model=0,visual=0,language=0,norm=0,head=0);inputs_seen=[];position_rows=[];captures=[];fusion=None;failed=None
    def before(module,args,kwargs):
        step=counts['model'];counts['model']+=1
        if fusion is not None:fusion.configure_queries([width-1 if step==0 else 0],[width+step-1])
        ids,mask=kwargs['input_ids'],kwargs['attention_mask']
        need(ids.shape==(batch,width if step==0 else 1) and mask.shape==(batch,width+step)
             and torch.equal(mask[:,:width].cpu(),bundle['inputs']['attention_mask']) and bool((mask[:,width:]==1).all()),'Native query/history mask differs')
        if step==0:need(torch.equal(ids.cpu(),bundle['inputs']['input_ids']) and kwargs.get('pixel_values') is not None,'Original prefill inputs differ')
        else:need(kwargs.get('pixel_values') is None and kwargs.get('past_key_values') is not None
            and kwargs['past_key_values'].get_seq_length()==width+step-1 and torch.equal(ids,ids[-1:].expand_as(ids)),'Cached global broadcast or visual ownership differs')
        inputs_seen.append(ids.detach().cpu().clone())
    def count(key):
        def hook(*_):counts[key]+=1
        return hook
    def language(module,args,kwargs):
        step=counts['language'];counts['language']+=1;pos=kwargs['position_ids'].detach().cpu();mask=kwargs['attention_mask'].detach().cpu();text=mask.long().cumsum(-1)-1
        expected=layout['position_ids'] if step==0 else (layout['rope_deltas'].view(1,batch,1)+width+step-1).expand(3,-1,-1)
        need(pos.shape==(4,batch,width if step==0 else 1) and torch.equal(pos[1:],expected),'Native mRoPE differs')
        need(torch.equal(pos[0][mask.bool()],text[mask.bool()]) if step==0 else torch.equal(pos[0],text[:,-1:]),'Native text positions differ')
        position_rows.append(native.tensor_info(pos))
    def after(module,args,output):
        if fusion is not None:
            need(fusion.calls==counts['model'] and fusion.argmax_calls==fusion.probability_calls==fusion.probe_head_calls==fusion.probe_norm_calls==1,'Binary origin was not measured exactly once')
            if capture:captures.append(fusion.export_last_capture(cpu=True))
    start=time.perf_counter()
    with ExitStack() as stack:
        stack.callback(setattr,model.model,'rope_deltas',rope)
        if core is not None:fusion=stack.enter_context(_controller(model,core,bundle,native_identity_sha256,None))
        if controller_observer is not None:controller_observer(fusion);stack.callback(controller_observer,None)
        for handle in (model.register_forward_pre_hook(before,with_kwargs=True),model.register_forward_hook(recorder),model.register_forward_hook(after),
            model.model.visual.register_forward_pre_hook(count('visual')),model.model.language_model.register_forward_pre_hook(language,with_kwargs=True),
            norm.register_forward_pre_hook(count('norm'),prepend=True),model.lm_head.register_forward_pre_hook(count('head'))):stack.callback(handle.remove)
        try:
            with torch.inference_mode():result=model.generate(**move_to_device(bundle['inputs'],model.device),generation_config=config,
                logits_processor=LogitsProcessorList([broadcast]),logits_to_keep=1)
        except InvalidBinaryMeasurement as exc:failed=exc
        if failed is None:
            torch.cuda.synchronize();suffix=result.sequences[:,width:];ids=suffix[-1].detach().cpu().tolist();steps=len(ids)
            need(getattr(result,'logits',None) is None and getattr(result,'scores',None) is None,'HF retained per-row output histories')
            need(1<=steps<=4 and torch.equal(suffix,suffix[-1:].expand_as(suffix)) and counts==dict(model=steps,visual=1,language=steps,norm=steps,head=steps)
                 and len(recorder.records)==broadcast.calls==steps and all(r['native_dtype']=='torch.float16' for r in recorder.records),'Native generation/call inventory differs')
            need([r['top1_token_id'] for r in recorder.records]==ids and all(bool((inputs_seen[t]==ids[t-1]).all()) for t in range(1,steps)),'Unmasked global argmax/history differs')
            eos=policy['native_eos_token_ids'];completed=ids[-1] in eos
            need(not any(x in eos for x in ids[:-1]) and (completed or steps==4),'Unexpected native answer stopping')
        artifact=None if fusion is None else fusion.export_origin_artifact(cpu=True)
        status=None if fusion is None or failed is not None else fusion.assert_complete()
    need((fusion is None or not fusion.active) and (core is None or core.mode==branch_mode)
         and versions=={(k,n):p._version for k,m in modules for n,p in m.named_parameters()},'Native/core mutation or leaked hook')
    counters=dict(counts,model_completed=len(recorder.records),broadcast=broadcast.calls,fusion=0 if fusion is None else fusion.calls,
        probe_norm=0 if fusion is None else fusion.probe_norm_calls,probe_head=0 if fusion is None else fusion.probe_head_calls,
        probability=0 if fusion is None else fusion.probability_calls,argmax=0 if fusion is None else fusion.argmax_calls,
        artifact_validation_argmax=0 if fusion is None else fusion.artifact_validation_argmax_calls)
    metadata=dict(meta,branch_mode=branch_mode,generation=policy,layout=layout['metadata'],generation_position_ids=position_rows,
        native_identity_sha256=native_identity_sha256,gate_rule=BINARY_GATE_RULE)
    if failed is not None:return dict(_failure(failed,metadata,branch_mode,counters),model_seconds=time.perf_counter()-start)
    return dict(measurement_passed=None if core is None else True,measurement_performed=core is not None,measurement_failure=False,branch_mode=branch_mode,generated_ids=ids,raw_logits=torch.stack(recorder.vectors),
        text=processor.tokenizer.decode(ids,skip_special_tokens=True),raw_text=processor.tokenizer.decode(ids,skip_special_tokens=False),
        completed=completed,truncated=not completed,finish_reason='eos' if completed else 'length',metadata=metadata,
        counters=counters,logit_records=recorder.records,captures=captures,origin_artifact=artifact,fusion_audit=status,model_seconds=time.perf_counter()-start)


def forward_native(model,processor,core,bundle,*,branch_mode=None,native_identity_sha256,origin_artifact=None,controller_observer=None):
    """Uncached actual-prefix replay; a nonempty prefix requires bound origin."""
    native.require_slurm(gpu=True)
    import torch
    from gnnformer.runtime import get_rope_index_fn,move_to_device
    norm=native_contract(model,core,branch_mode);validate_bundle(bundle);shared.digit_token_ids(processor.tokenizer)
    need(core is not None or origin_artifact is None,'Bare replay cannot consume a gate artifact')
    need(core is None or not bundle['metadata']['prefix_ids'] or origin_artifact is not None,'Observed prefix requires original measurement')
    need(bundle['metadata']['local_prompt']==local_prompt(bundle['metadata']['question']) and bundle['metadata']['global_prompt']==bundle['metadata']['question'],'Generic prompts differ')
    origin_identity(bundle,native_identity_sha256);layout=native.audit_layout(get_rope_index_fn(model),bundle);rope=getattr(model.model,'rope_deltas',None)
    modules=[('native',model)]+([] if core is None else [('core',core)]);versions={(k,n):p._version for k,m in modules for n,p in m.named_parameters()}
    counts=dict(model=0,visual=0,language=0,norm=0,head=0);observed={};failed=None;fusion=None
    def count(key):
        def hook(*_):counts[key]+=1
        return hook
    def language(module,args,kwargs):counts['language']+=1;observed.update(position_ids=kwargs['position_ids'].detach().cpu().clone(),attention_mask=kwargs['attention_mask'].detach().cpu().clone())
    with ExitStack() as stack:
        stack.callback(setattr,model.model,'rope_deltas',rope)
        if core is not None:fusion=stack.enter_context(_controller(model,core,bundle,native_identity_sha256,origin_artifact))
        if controller_observer is not None:controller_observer(fusion);stack.callback(controller_observer,None)
        for handle in (model.register_forward_pre_hook(count('model')),model.model.visual.register_forward_pre_hook(count('visual')),
            model.model.language_model.register_forward_pre_hook(language,with_kwargs=True),norm.register_forward_pre_hook(count('norm'),prepend=True),
            model.lm_head.register_forward_pre_hook(count('head'))):stack.callback(handle.remove)
        try:
            with torch.inference_mode():output=model(**move_to_device(bundle['inputs'],model.device),use_cache=False,logits_to_keep=1)
        except InvalidBinaryMeasurement as exc:failed=exc
        need(torch.equal(observed['position_ids'],layout['position_ids']) and torch.equal(observed['attention_mask'],bundle['inputs']['attention_mask']),'Actual full-prefix positions/mask differ')
        if failed is None:
            need(output.past_key_values is None and output.logits.dtype==torch.float16 and output.logits.shape[:2]==(bundle['metadata']['row_count'],1)
                 and bool(torch.isfinite(output.logits).all()) and counts==dict(model=1,visual=1,language=1,norm=1,head=1),'Uncached native call/cache contract differs')
            values=native.normal_copies(dict(native_logits=output.logits,global_logits=output.logits[-1,-1]),cpu=True)
            capture=None if fusion is None else fusion.export_last_capture(cpu=True);status=None if fusion is None else fusion.assert_complete()
        artifact=None if fusion is None else fusion.export_origin_artifact(cpu=True)
    need((fusion is None or not fusion.active) and (core is None or core.mode==branch_mode)
         and versions=={(k,n):p._version for k,m in modules for n,p in m.named_parameters()},'Replay mutated native/core or leaked hook')
    counts.update(fusion=0 if fusion is None else fusion.calls,probe_head=0 if fusion is None else fusion.probe_head_calls,
        probe_norm=0 if fusion is None else fusion.probe_norm_calls,probability=0 if fusion is None else fusion.probability_calls,argmax=0 if fusion is None else fusion.argmax_calls,
        artifact_validation_argmax=0 if fusion is None else fusion.artifact_validation_argmax_calls)
    meta=dict(bundle['metadata'],branch_mode=branch_mode,layout=layout['metadata'],native_identity_sha256=native_identity_sha256)
    if failed is not None:return _failure(failed,meta,branch_mode,counts)
    return dict(values,measurement_passed=None if core is None else True,measurement_performed=core is not None,measurement_failure=False,branch_mode=branch_mode,origin_artifact=artifact,
                capture=capture,fusion_audit=status,metadata=meta,counters=counts)

"""Held native N+1 semantic-gate runtime, without an external answer tally.

The original empty-prefix native 0/1 readout gates every coordinate of each
local payload. Its bound artifact is reused at later generated positions. Both
arms execute one extra native norm/head and full-vocabulary reduction at prefill;
all-open replaces the applied gates by ones after the same work. Canonical image
pixels and prompts are unchanged. No fitting, selection or efficacy release is
implemented here. All tensor/image/model work requires Slurm.
"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_vision_v7_runtime as native
from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation
from gnnformer.parallel_local_semantic_gate import ParallelLocalSemanticGate,GATE_RULE
need=native.need
OWN=('gnnformer/parallel_local_semantic_aggregation.py','gnnformer/parallel_local_semantic_gate.py',
     'scripts/native_vision_semantic_gate_runtime.py','tests/test_parallel_local_semantic_aggregation.py',
     'tests/test_parallel_local_semantic_gate.py')


def validate_bundle(bundle):
    native.require_slurm()
    meta=bundle['metadata'];inputs=bundle['inputs'];rows=bundle['row_inputs']
    n,p=meta['n_frames'],meta['original_prompt_width'];prefix=meta.get('prefix_ids',[])
    need(type(n) is int and n>0 and meta['arm']=='parallel' and meta['row_count']==n+1
         and meta['global_row']==n and meta['local_elements']==n
         and meta['row_kinds']==['local']*n+['global'] and len(rows)==n+1,
         'Require exactly N actual image streams and one final text-only global stream')
    need(meta['prompt_width']==p+len(prefix) and inputs['input_ids'].shape==(n+1,p+len(prefix))
         and inputs['attention_mask'].shape==inputs['input_ids'].shape
         and inputs['image_grid_thw'].shape[0]==n
         and len(meta['row_prompt_tokens'])==n+1
         and len(meta['image_sha256'])==n,
         'Original/observed prefix width or image occurrence layout differs')
    need('pixel_values' not in rows[-1] and all('pixel_values' in row for row in rows[:-1]),
         'Only actual image rows may carry pixels')
    need(all(row['input_ids'].shape[1]==length+len(prefix) for row,length in zip(rows,meta['row_prompt_tokens'])),
         'Complete row lengths do not match the original prompts plus observed prefix')
    return dict(passed=True,n_frames=n,row_count=n+1,original_prompt_width=p,prefix_tokens=len(prefix))


def append_observed_prefix(bundle,prefix_ids):
    """Keep every supplied observed ID, including specials; never inspect gold."""
    native.require_slurm();validate_bundle(bundle)
    prefix=list(prefix_ids);meta=bundle['metadata']
    need(meta['prompt_width']==meta['original_prompt_width'] and not meta.get('prefix_ids'),
         'Append to the original complete prompt exactly once')
    inputs=native.append_prefix(bundle['inputs'],prefix)
    rows=[native.append_prefix(row,prefix) for row in bundle['row_inputs']]
    updated=dict(meta,prefix_ids=prefix,prompt_width=int(inputs['input_ids'].shape[1]),
                 input_identity={k:native.tensor_info(v) for k,v in inputs.items()},observed_prefix_unfiltered=True)
    result=dict(inputs=inputs,row_inputs=rows,metadata=updated);validate_bundle(result);return result


def prepare_scene(processor,sample,*,prefix_ids=(),verify_processor_parity=False):
    bare=native.prepare_scene(processor,sample,'parallel',verify_processor_parity=verify_processor_parity)
    return append_observed_prefix(bare,prefix_ids)


def origin_identity(bundle,native_identity_sha256):
    """Bind original inputs even when bundle contains an observed continuation.

    The caller supplies a frozen digest binding model, processor, actual native
    norm/head weights and installed runtime. The runtime validates its shape;
    the experiment must verify that binding against the live model once before
    execution. Input hashes below are independently recomputed from this bundle.
    No gold labels, reference data or generated answer enter the gate identity.
    """
    native.require_slurm();validate_bundle(bundle)
    need(isinstance(native_identity_sha256,str) and len(native_identity_sha256)==64
         and all(c in '0123456789abcdef' for c in native_identity_sha256),
         'Require the frozen native model/runtime/processor identity SHA256')
    meta=bundle['metadata'];p=meta['original_prompt_width']
    original={k:(v[:,:p] if k in ('input_ids','attention_mask') else v) for k,v in bundle['inputs'].items()}
    binding=dict(inputs={k:native.tensor_info(v) for k,v in original.items()},
        metadata={k:meta[k] for k in ('arm','n_frames','row_kinds','row_count','global_row','local_elements',
            'original_prompt_width','row_prompt_tokens','local_prompt','global_prompt','resize')})
    return dict(sid=meta['sid'],question_sha256=meta['question_sha256'],image_sha256=list(meta['image_sha256']),
        prompt_width=p,input_identity_sha256=native.object_sha(binding),native_identity_sha256=native_identity_sha256)


def digit_token_ids(tokenizer):
    native.require_slurm();values=[]
    for digit in ('0','1'):
        ids=list(tokenizer(digit,add_special_tokens=False)['input_ids'])
        need(len(ids)==1 and ids[0] not in tokenizer.all_special_ids
             and tokenizer.decode(ids,skip_special_tokens=False)==digit,'ASCII digit gate tokenization differs')
        values.append(ids[0])
    need(values==[15,16],'Frozen Qwen native ASCII0/1 IDs differ')
    return tuple(values)


def native_contract(model,core):
    norm=native.native_contract(model,core)
    need(type(core) is ParallelLocalSemanticAggregation,'Require the explicit gated payload core')
    return norm


def forward_native(model,processor,core,bundle,*,mode,native_identity_sha256,
                   origin_artifact=None,capture=True,cpu=True):
    """One full-prefix native forward, using an explicitly bound original gate.

    A nonempty prefix requires the artifact from that scene's empty-prefix
    execution. Every observed prefix token is retained. This helper creates no
    cache, performs no answer parsing and never selects a prefix from gold.
    """
    native.require_slurm(gpu=True)
    import torch
    from gnnformer.runtime import get_rope_index_fn,move_to_device
    norm=native_contract(model,core);validate_bundle(bundle)
    need(not core.training and not any(p.requires_grad for p in core.parameters()),
         'Native forward requires a frozen/eval semantic core')
    meta=bundle['metadata'];width=meta['prompt_width']
    origin=origin_identity(bundle,native_identity_sha256);zero,one=digit_token_ids(processor.tokenizer)
    need(width==meta['original_prompt_width'] or origin_artifact is not None,
         'Observed continuation requires the bound empty-prefix gate artifact')
    layout=native.audit_layout(get_rope_index_fn(model),bundle)
    inputs=move_to_device(bundle['inputs'],model.device)
    original_rope=getattr(model.model,'rope_deltas',None)
    versions={(kind,name):p._version for kind,m in [('native',model),('core',core)] for name,p in m.named_parameters()}
    counts=dict(model=0,visual=0,language=0,norm=0,head=0);observed={}
    def model_hook(*_):counts['model']+=1
    def visual_hook(*_):counts['visual']+=1
    def head_hook(*_):counts['head']+=1
    def norm_hook(module,args):
        counts['norm']+=1
        need(args[0].dtype==torch.float16,'Native hidden dtype changed')
        if capture:observed.update(native.normal_copies(dict(local_states=args[0][:-1,-1:,:],
            global_states=args[0][-1,-1:,:]),cpu=cpu))
    def language_hook(module,args,kwargs):
        counts['language']+=1
        observed['positions']=kwargs['position_ids'].detach().cpu().clone()
        observed['mask']=kwargs['attention_mask'].detach().cpu().clone()
    begin=time.perf_counter()
    with ExitStack() as stack:
        stack.callback(setattr,model.model,'rope_deltas',original_rope)
        for register in (lambda:model.register_forward_pre_hook(model_hook),
                         lambda:model.model.visual.register_forward_pre_hook(visual_hook),
                         lambda:norm.register_forward_pre_hook(norm_hook),
                         lambda:model.lm_head.register_forward_pre_hook(head_hook),
                         lambda:model.model.language_model.register_forward_pre_hook(language_hook,with_kwargs=True)):
            handle=register();stack.callback(handle.remove)
        fusion=stack.enter_context(ParallelLocalSemanticGate(norm,model.lm_head,core,
            n_local_rows=meta['n_frames'],mode=mode,zero_token_id=zero,one_token_id=one,
            origin_identity=origin,query_indices=[width-1],stream_positions=[width-1],
            origin_artifact=origin_artifact,capture=capture))
        with torch.inference_mode():output=model(**inputs,use_cache=False,logits_to_keep=1)
        torch.cuda.synchronize()
        need(counts==dict(model=1,visual=1,language=1,norm=1,head=1) and fusion.calls==1,
             'Require one ordinary VLM/vision/norm/head/fusion per full-prefix forward')
        need(torch.equal(observed['positions'],layout['position_ids'])
             and torch.equal(observed['mask'],bundle['inputs']['attention_mask'])
             and torch.equal(model.model.rope_deltas.detach().cpu(),layout['rope_deltas']),
             'Native full-prefix positions, padding or rope deltas differ')
        need(output.past_key_values is None and output.logits.shape[:2]==(meta['row_count'],1)
             and output.logits.dtype==torch.float16 and bool(torch.isfinite(output.logits).all()),
             'Full-prefix native output shape/dtype/cache differs')
        values=native.normal_copies(dict(native_logits=output.logits,global_logits=output.logits[-1,-1]),cpu=cpu)
        if capture:
            values.update({k:observed[k] for k in ('local_states','global_states')})
            values['fusion']=fusion.export_last_capture(cpu=cpu)
        artifact=fusion.export_origin_artifact(cpu=cpu);audit=fusion.assert_complete()
    need(not fusion.active and versions=={(kind,name):p._version for kind,m in [('native',model),('core',core)]
         for name,p in m.named_parameters()},'Native/core parameters changed or semantic hook remained active')
    return dict(values,origin_artifact=artifact,fusion_audit=audit,
        metadata=dict(meta,layout=layout['metadata'],native_dtype='torch.float16',branch_dtype='torch.float32',
            gate_mode=mode,gate_rule=GATE_RULE,origin_identity=origin,native_identity_sha256=native_identity_sha256),
        counters=dict(counts,fusion=fusion.calls,probe_norm=fusion.probe_norm_calls,
            probe_head=fusion.probe_head_calls,probability=fusion.probability_calls),
        model_seconds=time.perf_counter()-begin)


def generate_native(model, processor, core, bundle, *, mode, native_identity_sha256,
                    max_new_tokens=4, capture=False, cpu=True):
    """Native cached generation; parallel choices use the global row only.

    Return global generated_ids and [generated_tokens,vocabulary] raw_logits;
    raw means native output before the broadcast processor, without normalization
    or numeric masking. EOS completion/truncation are separate from any parser.
    Optional captures retain local/global states and gated vector contributions.
    The full original probe artifact is returned once for bound replay.
    No comparison with uncached decoding or reasoning efficacy is implied.
    """
    native.require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = native_contract(model, core)
    need(not core.training and not any(p.requires_grad for p in core.parameters()),
         'Native generation requires a frozen/eval semantic core')
    versions={(kind,name):p._version for kind,m in [('native',model),('core',core)] for name,p in m.named_parameters()}
    original_rope=getattr(model.model,'rope_deltas',None)
    meta, inputs = bundle['metadata'], move_to_device(bundle['inputs'], model.device)
    layout = native.audit_layout(get_rope_index_fn(model), bundle)
    config, policy = native.generation_policy(model, processor.tokenizer, max_new_tokens=max_new_tokens)
    width, batch = meta['prompt_width'], meta['row_count']
    need(meta['arm'] == 'parallel' and width == meta['original_prompt_width'],
         'Natural generation starts from the unchanged parallel prompt without supplied answer tokens')
    validate_bundle(bundle)
    need(mode in ('native_gate','all_open'), 'Unknown semantic gate mode')
    origin = origin_identity(bundle, native_identity_sha256)
    zero, one = digit_token_ids(processor.tokenizer)
    processors = LogitsProcessorList()
    broadcast = None
    if meta['arm'] == 'parallel':
        broadcast = GlobalBroadcastLogitsProcessor(n_local_rows=meta['n_frames'], prompt_length=width)
        processors.append(broadcast)
    counts = dict(model=0, visual=0, language=0, norm=0, head=0)
    step_inputs, captures, position_rows = [], [], []
    def model_hook(module, args, kwargs):
        index = counts['model']; counts['model'] += 1
        fusion.configure_queries([width-1 if index==0 else 0],[width+index-1])
        ids, mask = kwargs['input_ids'], kwargs['attention_mask']
        need(ids.shape == (batch, width if index == 0 else 1), 'Unexpected prefill/cache token shape')
        expected_mask = native.append_prefix(bundle['inputs'], [0] * index)['attention_mask']
        need(torch.equal(mask.detach().cpu(), expected_mask), 'Native cached padding/key mask differs')
        if index == 0:
            need(torch.equal(ids.detach().cpu(), bundle['inputs']['input_ids'])
                 and kwargs.get('pixel_values') is not None, 'Native prefill changed inputs or omitted images')
        else:
            need(kwargs.get('pixel_values') is None and kwargs.get('past_key_values') is not None
                 and kwargs['past_key_values'].get_seq_length() == width + index - 1,
                 'Decode must use the native cache and no repeated visual forward')
            need(torch.equal(ids, ids[-1:].expand_as(ids)), 'Parallel generated prefixes diverged')
        step_inputs.append(ids.detach().cpu().clone())
    def visual_hook(*_): counts['visual'] += 1
    def norm_hook(*_): counts['norm'] += 1
    def head_hook(*_): counts['head'] += 1
    def language_hook(module, args, kwargs):
        index = counts['language']; counts['language'] += 1
        positions = kwargs['position_ids'].detach().cpu()
        need(positions.shape == (4, batch, width if index == 0 else 1),
             'Installed native generation must supply text plus three mRoPE axes')
        mask = kwargs['attention_mask'].detach().cpu()
        text = mask.long().cumsum(-1) - 1
        if index == 0:
            expected = layout['position_ids']
            valid = mask.bool()
            need(torch.equal(positions[0][valid], text[valid]), 'Native text positions differ')
        else:
            expected = (layout['rope_deltas'].view(1, batch, 1) + width + index - 1).expand(3, -1, -1)
            need(torch.equal(positions[0], text[:, -1:]), 'Native cached text positions differ')
        need(torch.equal(positions[1:], expected), 'Native generation logical mRoPE differs')
        position_rows.append(native.tensor_info(positions))
    begin = time.perf_counter()
    with ExitStack() as stack:
        # Every call owns a fresh native cache. Restore the caller's mRoPE state
        # after normal completion or an exception; no partial cache is returned.
        stack.callback(setattr,model.model,'rope_deltas',original_rope)
        fusion = stack.enter_context(ParallelLocalSemanticGate(norm,model.lm_head,core,
            n_local_rows=meta['n_frames'],mode=mode,zero_token_id=zero,one_token_id=one,
            origin_identity=origin,query_indices=[width-1],stream_positions=[width-1],capture=capture))
        def output_hook(module, args, output):
            need(fusion.calls==counts['model'] and fusion.probe_norm_calls==fusion.probe_head_calls==fusion.probability_calls==1,
                 'Semantic fusion/original probe counts differ')
            if capture: captures.append(fusion.export_last_capture(cpu=cpu))
        for register in (lambda:model.register_forward_pre_hook(model_hook,with_kwargs=True),
                         lambda:model.model.visual.register_forward_pre_hook(visual_hook),
                         lambda:norm.register_forward_pre_hook(norm_hook),
                         lambda:model.lm_head.register_forward_pre_hook(head_hook),
                         lambda:model.model.language_model.register_forward_pre_hook(language_hook,with_kwargs=True),
                         lambda:model.register_forward_hook(output_hook)):
            handle=register();stack.callback(handle.remove)
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=config,
                                    logits_processor=processors, logits_to_keep=1)
        torch.cuda.synchronize()
        suffix = output.sequences[:, width:]
        steps = suffix.shape[1]
        need(1 <= steps <= 4 and len(output.logits) == steps and counts == dict(model=steps, visual=1, language=steps, norm=steps, head=steps)
             and fusion.calls == steps and (broadcast is None or broadcast.calls == steps),
             'Require exactly one native model/fusion invocation per generated token and one visual prefill')
        need(torch.equal(suffix, suffix[-1:].expand_as(suffix)), 'Final broadcast sequences differ')
        generated = suffix[-1].detach().cpu().tolist()
        for index in range(1, steps):
            need(bool((step_inputs[index] == generated[index-1]).all()), 'Executed cached token differs from output history')
        raw = torch.stack([value[-1] for value in output.logits])
        need(raw.ndim == 2 and raw.shape[0] == steps and bool(torch.isfinite(raw).all()), 'Raw global logits are malformed')
        # Greedy choices must be the GLOBAL raw argmax, not a hidden mask/penalty.
        need(raw.argmax(-1).detach().cpu().tolist() == generated, 'Generation altered raw global greedy choices')
        eos = policy['native_eos_token_ids']
        completed = generated[-1] in eos
        need(not any(token in eos for token in generated[:-1]) and (completed or steps == 4),
             'Unexpected early stop or an interior EOS')
        values = native.normal_copies(dict(raw_logits=raw), cpu=cpu)
        fusion_audit = fusion.assert_complete()
        origin_artifact = fusion.export_origin_artifact(cpu=cpu)
    need(not fusion.active and versions=={(kind,name):p._version for kind,m in [('native',model),('core',core)]
         for name,p in m.named_parameters()},'Native/core parameters changed or semantic hook remained active')
    return dict(values, generated_ids=generated, raw_text=processor.tokenizer.decode(generated, skip_special_tokens=False),
        text=processor.tokenizer.decode(generated, skip_special_tokens=True), completed=completed,
        truncated=not completed, finish_reason='eos' if completed else 'length',
        metadata=dict(meta, layout=layout['metadata'], generation=policy, native_dtype='torch.float16',
                      gate_mode=mode,gate_rule=GATE_RULE,origin_identity=origin,
                      native_identity_sha256=native_identity_sha256,read_boundary='before_native_final_norm',
                      branch_dtype='torch.float32', generation_position_ids=position_rows),
        counters=dict(counts, fusion=steps, broadcast=0 if broadcast is None else broadcast.calls,
                      probe_norm=fusion.probe_norm_calls,probe_head=fusion.probe_head_calls,
                      probability=fusion.probability_calls),
        captures=captures,origin_artifact=origin_artifact,fusion_audit=fusion_audit,
        model_seconds=time.perf_counter()-begin)

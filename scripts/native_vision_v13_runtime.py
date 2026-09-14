"""V13 native N+1-stream generation with a separate conditional null predictor.

Only offset and centered deployment are supported. The actual generated prefix
is shared by ordinary native rows; no reference images, lookup table or labels
enter inference. Original core and predictor stay separate from the backbone.
"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_vision_v7_runtime as native
from gnnformer.parallel_local_learned_null import ParallelLocalLearnedNull
need=native.need


def prepare_scene(processor,sample,*,prefix_ids=(),verify_processor_parity=False):
    bundle=native.prepare_scene(processor,sample,'parallel',prefix_ids=prefix_ids,
        verify_processor_parity=verify_processor_parity)
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    meta=bundle['metadata'];n=meta['n_frames']
    need(type(n) is int and n>0 and meta['arm']=='parallel' and meta['row_count']==n+1
        and meta['global_row']==n and meta['local_elements']==n
        and len(meta['image_sha256'])==n and len(bundle['row_inputs'])==n+1,
        'Require exactly N actual image rows and one native global row')
    return dict(passed=True,n_frames=n,rows=n+1,no_reference_rows=True)


def generate_native(model, processor, branch, predictor, bundle, *, mode, max_new_tokens=4, capture=False, cpu=True):
    """Native cached generation; parallel choices use the global row only.

    Return global generated_ids and [generated_tokens,vocabulary] raw_logits;
    raw means native output before the broadcast processor, without normalization
    or numeric masking. EOS completion/truncation are separate from any parser.
    Optional captures contain local/global states and the learned-null readout write.
    No comparison with uncached decoding or reasoning efficacy is implied.
    """
    native.require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = native.native_contract(model, branch)
    need(branch is not None and max_new_tokens==4, 'V13 requires the explicit original core and registered max4')
    meta, inputs = bundle['metadata'], move_to_device(bundle['inputs'], model.device)
    layout = native.audit_layout(get_rope_index_fn(model), bundle)
    config, policy = native.generation_policy(model, processor.tokenizer, max_new_tokens=max_new_tokens)
    width, batch = meta['prompt_width'], meta['row_count']
    need(meta['arm'] == 'parallel' and width == meta['original_prompt_width'],
         'Natural generation starts from the unchanged parallel prompt without supplied answer tokens')
    validate_bundle(bundle)
    need(mode in ('offset','centered'), 'Unregistered learned-null mode')
    processors = LogitsProcessorList()
    broadcast = None
    if meta['arm'] == 'parallel':
        broadcast = GlobalBroadcastLogitsProcessor(n_local_rows=meta['n_frames'], prompt_length=width)
        processors.append(broadcast)
    counts = dict(model=0, visual=0, language=0)
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
        fusion = stack.enter_context(ParallelLocalLearnedNull(norm,branch,predictor,n_local_rows=meta['n_frames'],mode=mode,
            query_indices=[width-1],stream_positions=[width-1],capture=capture))
        def output_hook(module, args, output):
            need(fusion.calls==counts['model'],'Learned-null query read/write counts differ')
            if capture: captures.append(fusion.export_last_capture(cpu=cpu))
        for handle in (model.register_forward_pre_hook(model_hook, with_kwargs=True),
                       model.model.visual.register_forward_pre_hook(visual_hook),
                       model.model.language_model.register_forward_pre_hook(language_hook, with_kwargs=True),
                       model.register_forward_hook(output_hook)):
            stack.callback(handle.remove)
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=config,
                                    logits_processor=processors, logits_to_keep=1)
        torch.cuda.synchronize()
        suffix = output.sequences[:, width:]
        steps = suffix.shape[1]
        need(1 <= steps <= 4 and len(output.logits) == steps and counts == dict(model=steps, visual=1, language=steps)
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
    return dict(values, generated_ids=generated, raw_text=processor.tokenizer.decode(generated, skip_special_tokens=False),
        text=processor.tokenizer.decode(generated, skip_special_tokens=True), completed=completed,
        truncated=not completed, finish_reason='eos' if completed else 'length',
        metadata=dict(meta, layout=layout['metadata'], generation=policy, native_dtype='torch.float16',
                      null_mode=mode,read_boundary='before_native_final_norm',
                      null_predictor_parameters=sum(p.numel() for p in predictor.parameters()),
                      branch_dtype='torch.float32', generation_position_ids=position_rows),
        counters=dict(counts, fusion=steps, broadcast=0 if broadcast is None else broadcast.calls),
        captures=captures, model_seconds=time.perf_counter()-begin)



def forward_native(model,bundle,core,predictor,*,mode,capture=True,cpu=False):
    """One ordinary uncached prefill and one vision call; no replay inside.

    Captures preserve the full native batch at only its current last query, so
    a caller can replay the actual norm/head batch shape without extra images.
    Strict-prefix construction remains the caller's explicit responsibility.
    """
    native.require_slurm(gpu=True)
    import torch
    from gnnformer.runtime import get_rope_index_fn,move_to_device
    validate_bundle(bundle);norm=native.native_contract(model,core)
    meta=bundle['metadata'];layout=native.audit_layout(get_rope_index_fn(model),bundle)
    inputs=move_to_device(bundle['inputs'],model.device);counts=dict(model=0,visual=0,norm=0);observed={}
    def model_hook(*_):counts['model']+=1
    def visual_hook(*_):counts['visual']+=1
    def norm_before(module,args):
        counts['norm']+=1;need(args[0].dtype==torch.float16,'Native carry must stay FP16')
        if capture:observed.update(native.normal_copies(dict(native_query_hidden=args[0][:,-1:,:]),cpu=cpu))
    def norm_after(module,args):
        if capture:observed.update(native.normal_copies(dict(fused_query_hidden=args[0][:,-1:,:]),cpu=cpu))
    def language_hook(module,args,kwargs):
        observed['positions']=kwargs['position_ids'].detach().cpu().clone()
        observed['mask']=kwargs['attention_mask'].detach().cpu().clone()
    started=time.perf_counter()
    with ExitStack() as stack:
        for h in (model.register_forward_pre_hook(model_hook),model.model.visual.register_forward_pre_hook(visual_hook),
                  norm.register_forward_pre_hook(norm_before),
                  model.model.language_model.register_forward_pre_hook(language_hook,with_kwargs=True)):
            stack.callback(h.remove)
        query=[inputs['input_ids'].shape[1]-1]
        fusion=stack.enter_context(ParallelLocalLearnedNull(norm,core,predictor,n_local_rows=meta['n_frames'],mode=mode,
            query_indices=query,stream_positions=query,capture=capture))
        stack.callback(norm.register_forward_pre_hook(norm_after).remove)
        with torch.inference_mode():output=model(**inputs,use_cache=False,logits_to_keep=1)
        torch.cuda.synchronize()
        need(counts==dict(model=1,visual=1,norm=1) and fusion.calls==1,'One native prefill/vision/fusion required')
        need(torch.equal(observed['positions'],layout['position_ids']) and torch.equal(observed['mask'],bundle['inputs']['attention_mask'])
            and torch.equal(model.model.rope_deltas.detach().cpu(),layout['rope_deltas']),'Native positions/mask changed')
        need(output.past_key_values is None and output.logits.shape[:2]==(meta['row_count'],1)
            and bool(torch.isfinite(output.logits).all()),'Ordinary uncached native output differs')
        values=native.normal_copies(dict(global_logits=output.logits[-1,-1]),cpu=cpu)
        if capture:
            values.update({k:observed[k] for k in ('native_query_hidden','fused_query_hidden')})
            values.update(local_states=observed['native_query_hidden'][:-1],global_states=observed['native_query_hidden'][-1],
                          fusion=fusion.export_last_capture(cpu=cpu))
    return dict(values,metadata=dict(meta,layout=layout['metadata'],native_dtype='torch.float16',branch_dtype='torch.float32',
                                    null_mode=mode,null_predictor_parameters=sum(p.numel() for p in predictor.parameters())),
                counters=counts,model_seconds=time.perf_counter()-started)

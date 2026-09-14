"""Unreleased diagnostic source: matched learned/bank null means, N+25 rows.

Reuses immutable V12 image preparation and V7 ordinary native generation. Both
conditions see identical real/reference/global rows. There is no fitting,
checkpoint selection, output parser or efficacy/resource release here. Actual
GPU integration still requires a separately registered software profile.
"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_vision_v12_runtime as reference
from scripts import native_vision_v7_runtime as native
from gnnformer.parallel_local_null_comparison import ParallelLocalNullComparison
from gnnformer.conditional_null_mean import ConditionalNullMean
need=native.need
reference_bank=reference.reference_bank
REFERENCE_MANIFEST=reference.REFERENCE_MANIFEST
OWN=('gnnformer/parallel_local_null_comparison.py','scripts/native_vision_null_comparison_runtime.py',
     'tests/test_parallel_local_null_comparison.py')


def append_observed_prefix(bundle,prefix_ids):
    """Append exact observed IDs, including specials, without interpreting gold.

    For full-prefix diagnostics only. Native layout/replay checks remain the
    caller's responsibility; an invalid native structural token is never removed
    or replaced to make a replay pass. Natural generation starts with no prefix.
    """
    native.require_slurm()
    prefix=list(prefix_ids);meta=bundle['metadata']
    need(meta['prompt_width']==meta['original_prompt_width'] and not meta.get('prefix_ids'),
         'Append to the original complete prompt exactly once')
    inputs=native.append_prefix(bundle['inputs'],prefix)
    rows=[native.append_prefix(row,prefix) for row in bundle['row_inputs']]
    updated=dict(meta,prefix_ids=prefix,prompt_width=int(inputs['input_ids'].shape[1]),
                 input_identity={k:native.tensor_info(v) for k,v in inputs.items()},observed_prefix_unfiltered=True)
    result=dict(inputs=inputs,row_inputs=rows,metadata=updated)
    validate_bundle(result)
    return result


def prepare_scene(processor,sample,bank,*,prefix_ids=(),verify_processor_parity=False):
    # V12/V7's explicit-prefix constructor rejects specials. Append actual
    # observed tokens after preparing the unchanged empty prompt instead.
    bare=reference.prepare_scene(processor,sample,bank,verify_processor_parity=verify_processor_parity)
    return append_observed_prefix(bare,prefix_ids)


def validate_bundle(bundle):
    result=reference.validate_bundle(bundle);meta=bundle['metadata']
    need(meta['prompt_width']==meta['original_prompt_width']+len(meta.get('prefix_ids',[])),
         'Observed prefix width differs')
    return result


def generate_native(model, processor, core, predictor, bundle, *, condition, mean_source,
                    core_key, max_new_tokens=4, capture=False, cpu=True):
    """Native cached generation; parallel choices use the global row only.

    Return global generated_ids and [generated_tokens,vocabulary] raw_logits;
    raw means native output before the broadcast processor, without normalization
    or numeric masking. EOS completion/truncation are separate from any parser.
    Optional captures contain actual/reference/global states before final norm and the readout write.
    No comparison with uncached decoding or reasoning efficacy is implied.
    """
    native.require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = native.native_contract(model, core)
    need(isinstance(predictor,ConditionalNullMean) and predictor.rank==core.rank
         and not core.training and not predictor.training
         and not any(p.requires_grad for m in (core,predictor) for p in m.parameters())
         and all(p.dtype==torch.float32 and p.device==norm.weight.device for p in predictor.parameters()),
         'Comparison requires frozen/eval original core and separate FP32 predictor')
    versions={(kind,name):p._version for kind,m in [('native',model),('core',core),('predictor',predictor)] for name,p in m.named_parameters()}
    original_rope=getattr(model.model,'rope_deltas',None)
    meta, inputs = bundle['metadata'], move_to_device(bundle['inputs'], model.device)
    layout = native.audit_layout(get_rope_index_fn(model), bundle)
    config, policy = native.generation_policy(model, processor.tokenizer, max_new_tokens=max_new_tokens)
    width, batch = meta['prompt_width'], meta['row_count']
    need(meta['arm'] == 'parallel' and width == meta['original_prompt_width'],
         'Natural generation starts from the unchanged parallel prompt without supplied answer tokens')
    validate_bundle(bundle)
    need(condition in ('offset','centered') and mean_source in ('learned','bank') and isinstance(core_key,str) and core_key,
         'Unknown coefficient/mean source or missing core identity')
    processors = LogitsProcessorList()
    broadcast = None
    if meta['arm'] == 'parallel':
        broadcast = GlobalBroadcastLogitsProcessor(n_local_rows=meta['processed_image_rows'], prompt_length=width)
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
        # Every call owns a fresh native cache. Restore the caller's mRoPE state
        # after normal completion or an exception; no partial cache is returned.
        stack.callback(setattr,model.model,'rope_deltas',original_rope)
        fusion = stack.enter_context(ParallelLocalNullComparison(norm,core,predictor,n_actual_rows=meta['actual_n_frames'],
            condition=condition,mean_source=mean_source,
            query_indices=[width-1],stream_positions=[width-1],capture=capture))
        def output_hook(module, args, output):
            need(fusion.calls==fusion.actual_reads==fusion.reference_reads==fusion.predictor_reads==counts['model'],'Reference query read/write counts differ')
            if capture: captures.append(fusion.export_last_capture(cpu=cpu))
        for register in (lambda:model.register_forward_pre_hook(model_hook,with_kwargs=True),
                         lambda:model.model.visual.register_forward_pre_hook(visual_hook),
                         lambda:model.model.language_model.register_forward_pre_hook(language_hook,with_kwargs=True),
                         lambda:model.register_forward_hook(output_hook)):
            handle=register();stack.callback(handle.remove)
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
    need(not fusion.active and versions=={(kind,name):p._version for kind,m in [('native',model),('core',core),('predictor',predictor)]
         for name,p in m.named_parameters()},'Native/learned parameters changed or comparison hook remained active')
    return dict(values, generated_ids=generated, raw_text=processor.tokenizer.decode(generated, skip_special_tokens=False),
        text=processor.tokenizer.decode(generated, skip_special_tokens=True), completed=completed,
        truncated=not completed, finish_reason='eos' if completed else 'length',
        metadata=dict(meta, layout=layout['metadata'], generation=policy, native_dtype='torch.float16',
                      null_condition=condition,mean_source=mean_source,coefficient=meta['actual_n_frames'] if condition=='centered' else 1,
                      core_key=core_key,reference_streams_diagnostic_only=True,read_boundary='before_native_final_norm',
                      branch_dtype='torch.float32', generation_position_ids=position_rows),
        counters=dict(counts, fusion=steps, broadcast=0 if broadcast is None else broadcast.calls),
        captures=captures, model_seconds=time.perf_counter()-begin)


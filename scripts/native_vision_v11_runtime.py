"""Proposed V11 native cached generation with persistent or readout writes.

Keeps the immutable V7 ordinary generation, broadcast, mask and mRoPE contract.
Only the common penultimate read and residual write location change. No training
or efficacy release is implied by this source file. Native integration must be
profiled before use; no inference feature cache substitutes for image inputs.
"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts import native_vision_v7_runtime as native
from gnnformer.parallel_local_memory import ParallelLocalMemory
need=native.need

def generate_native(model, processor, branch, bundle, *, write_location, max_new_tokens=4, capture=False, cpu=True):
    """Native cached generation; parallel choices use the global row only.

    Return global generated_ids and [generated_tokens,vocabulary] raw_logits;
    raw means native output before the broadcast processor, without normalization
    or numeric masking. EOS completion/truncation are separate from any parser.
    Optional captures contain each token's common penultimate reads and write states.
    No comparison with uncached decoding or reasoning efficacy is implied.
    """
    native.require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = native.native_contract(model, branch)
    need(branch is not None, 'Native V11 generation requires the explicit shared branch')
    meta, inputs = bundle['metadata'], move_to_device(bundle['inputs'], model.device)
    layout = native.audit_layout(get_rope_index_fn(model), bundle)
    config, policy = native.generation_policy(model, processor.tokenizer, max_new_tokens=max_new_tokens)
    width, batch = meta['prompt_width'], meta['row_count']
    need(meta['arm'] == 'parallel' and width == meta['original_prompt_width'],
         'Natural generation starts from the unchanged parallel prompt without supplied answer tokens')
    need(write_location in ('pre_last', 'post_last') and len(model.model.language_model.layers) == 28,
         'Require the registered 28-block model and one write location')
    processors = LogitsProcessorList()
    broadcast = None
    if meta['arm'] == 'parallel':
        broadcast = GlobalBroadcastLogitsProcessor(n_local_rows=meta['n_frames'], prompt_length=width)
        processors.append(broadcast)
    counts = dict(model=0, visual=0, language=0)
    step_inputs, captures, position_rows = [], [], []
    def model_hook(module, args, kwargs):
        index = counts['model']; counts['model'] += 1
        fusion.configure_queries([width-1 if index == 0 else 0], [width+index-1])
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
        fusion = stack.enter_context(ParallelLocalMemory(
            model.model.language_model.layers[-2], norm, branch,
            n_local_rows=meta['n_frames'], write_location=write_location,
            query_indices=[width-1], stream_positions=[width-1], capture=capture))
        def output_hook(module, args, output):
            fusion.assert_complete()
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
                      write_location=write_location, read_block_index=26, final_block_index=27,
                      branch_dtype='torch.float32', generation_position_ids=position_rows),
        counters=dict(counts, fusion=steps, broadcast=0 if broadcast is None else broadcast.calls),
        captures=captures, model_seconds=time.perf_counter()-begin)


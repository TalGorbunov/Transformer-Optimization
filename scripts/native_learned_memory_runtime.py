"""Held native learned aggregation with common lower reads and two write placements.

The pre_last write enters the existing final-block KV; post_last is the matched
final-norm control. Both read immediately after block26. Full-prefix execution
recomputes every historical write; generation writes the current query once.
No cache, fit, reasoning prompt or efficacy release is implied by this module.
"""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import time
from scripts import native_learned_selection_runtime as shared
from scripts import native_vision_v7_runtime as native
from scripts import native_vision_reasoning_stream as stream
from scripts.native_vision_reasoning_stream import GlobalLogitRecorder
from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
from gnnformer.parallel_local_learned_memory import ParallelLocalLearnedMemory

need = native.need
prepare_scene = shared.prepare_scene
append_observed_prefix = shared.append_observed_prefix
validate_bundle = shared.validate_bundle
input_identity = shared.input_identity
_versions = shared._versions
OWN = ('scripts/native_learned_memory_runtime.py',
       'gnnformer/parallel_local_learned_memory.py', 'gnnformer/parallel_local_memory.py')


def sources():
    root = Path(__file__).resolve().parents[1]
    return {**shared.sources(), **{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in OWN}}


def memory_contract(model, core, selection_mode, write_location):
    norm = shared.native_contract(model, core, selection_mode)
    need(len(model.model.language_model.layers) == 28, 'The common read/write boundary requires28layers')
    need(write_location is None if core is None else write_location in ('pre_last', 'post_last'),
         'Bare execution has no placement; a fused run requires explicit placement')
    return norm


def forward_native(model, bundle, core=None, *, selection_mode=None, native_identity_sha256,
                   write_location=None, capture=True, cpu=True, controller_observer=None):
    """Full-prefix native forward with every historical write live at its own query."""
    native.require_slurm(gpu=True)
    import torch
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = memory_contract(model, core, selection_mode, write_location); validate_bundle(bundle)
    binding = input_identity(bundle, native_identity_sha256)
    layout = native.audit_layout(get_rope_index_fn(model), bundle)
    rope = getattr(model.model, 'rope_deltas', None); versions = _versions(model, core)
    counts = dict(model=0, visual=0, language=0, norm=0, head=0); observed = {}; fusion = None
    def count(key):
        def hook(*_): counts[key] += 1
        return hook
    def language(module, args, kwargs):
        counts['language'] += 1
        observed.update(position_ids=kwargs['position_ids'].detach().cpu().clone(),
                        attention_mask=kwargs['attention_mask'].detach().cpu().clone())
    def before_write(module, args, kwargs, output):
        if capture:
            hidden = output[0] if isinstance(output, tuple) else output
            observed.update(native.normal_copies(dict(local_states=hidden[:-1, -1:, :],
                global_states=hidden[-1, -1:, :]), cpu=cpu))
    def before_norm(module, args):
        counts['norm'] += 1
        if capture: observed.update(native.normal_copies(dict(native_query_hidden=args[0][:, -1:, :]), cpu=cpu))
    start = time.perf_counter()
    with ExitStack() as stack:
        stack.callback(setattr, model.model, 'rope_deltas', rope)
        for handle in (model.register_forward_pre_hook(count('model')), model.model.visual.register_forward_pre_hook(count('visual')),
            model.model.language_model.register_forward_pre_hook(language, with_kwargs=True),
            model.model.language_model.layers[-2].register_forward_hook(before_write, with_kwargs=True, prepend=True),
            norm.register_forward_pre_hook(before_norm), model.lm_head.register_forward_pre_hook(count('head'))): stack.callback(handle.remove)
        if core is not None:
            positions = list(range(bundle['metadata']['original_prompt_width']-1, bundle['metadata']['prompt_width']))
            fusion = stack.enter_context(ParallelLocalLearnedMemory(model.model.language_model.layers[-2],
                norm, core, n_local_rows=bundle['metadata']['n_frames'], write_location=write_location,
                query_indices=positions, stream_positions=positions, selection_mode=selection_mode, capture=capture))
        if controller_observer is not None: stack.callback(controller_observer, None); controller_observer(fusion)
        with torch.inference_mode(): output = model(**move_to_device(bundle['inputs'], model.device), use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        need(counts == dict(model=1, visual=1, language=1, norm=1, head=1) and (fusion is None or fusion.calls == 1),
             'One native forward/vision/norm/head and one optional branch call required')
        need(torch.equal(observed['position_ids'], layout['position_ids'])
             and torch.equal(observed['attention_mask'], bundle['inputs']['attention_mask'])
             and torch.equal(model.model.rope_deltas.detach().cpu(), layout['rope_deltas']), 'Actual full-prefix native layout differs')
        need(output.past_key_values is None and output.logits.dtype == torch.float16
             and output.logits.shape[:2] == (bundle['metadata']['row_count'], 1)
             and bool(torch.isfinite(output.logits).all()), 'Native uncached output contract differs')
        values = native.normal_copies(dict(native_logits=output.logits, global_logits=output.logits[-1, -1]), cpu=cpu)
        if fusion is not None: fusion.assert_complete()
        cap = None if fusion is None else fusion.export_last_capture(cpu=cpu)
    need(versions == _versions(model, core) and (fusion is None or not fusion.active)
         and (core is None or core.mode == selection_mode), 'Native/core mutation or leaked controller')
    return dict(values, **{k: v for k, v in observed.items() if k not in ('position_ids', 'attention_mask')},
        capture=cap, counters=dict(counts, selection=0 if fusion is None else fusion.calls, probe_head=0),
        metadata=dict(bundle['metadata'], selection_mode=selection_mode, write_location=write_location,
                      read_block_index=26, final_block_index=27, all_historical_writes=True, layout=layout['metadata'],
                      native_identity_sha256=native_identity_sha256, scene_input_identity=binding),
        model_seconds=time.perf_counter()-start)


def generate_native(model, processor, core, bundle, *, selection_mode=None, native_identity_sha256,
                    write_location=None, max_new_tokens=4, capture=False, full_vectors=False, controller_observer=None):
    """Bounded native generation with one current write/call and ordinary persistent KV."""
    native.require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = memory_contract(model, core, selection_mode, write_location); validate_bundle(bundle)
    meta = bundle['metadata']; width, batch = meta['prompt_width'], meta['row_count']
    need(not meta['prefix_ids'] and width == meta['original_prompt_width'], 'Generation starts from the original prompt')
    need(type(max_new_tokens) is int and 1 <= max_new_tokens <= 4096, 'Explicit bounded native token budget required')
    binding = input_identity(bundle, native_identity_sha256)
    config, policy = stream.generation_policy(model, processor.tokenizer, max_new_tokens)
    config.output_logits = False; config.output_scores = False
    policy = dict(policy, output_logits=False, output_scores=False, global_row_streaming=True)
    recorder = GlobalLogitRecorder(max_steps=max_new_tokens, full_vectors=full_vectors)
    broadcast = GlobalBroadcastLogitsProcessor(n_local_rows=meta['n_frames'], prompt_length=width)
    layout = native.audit_layout(get_rope_index_fn(model), bundle)
    rope = getattr(model.model, 'rope_deltas', None); versions = _versions(model, core)
    counts = dict(model=0, visual=0, language=0, norm=0, head=0)
    inputs_seen = []; position_rows = []; captures = []; fusion = None
    def before(module, args, kwargs):
        step = counts['model']; counts['model'] += 1
        if fusion is not None: fusion.configure_queries([width-1 if step == 0 else 0], [width+step-1])
        ids, mask = kwargs['input_ids'], kwargs['attention_mask']
        need(ids.shape == (batch, width if step == 0 else 1) and mask.shape == (batch, width+step)
             and torch.equal(mask[:, :width].cpu(), bundle['inputs']['attention_mask'])
             and bool((mask[:, width:] == 1).all()), 'Native query/history mask differs')
        if step == 0:
            need(torch.equal(ids.cpu(), bundle['inputs']['input_ids']) and kwargs.get('pixel_values') is not None,
                 'Original prefill inputs differ')
        else:
            need(kwargs.get('pixel_values') is None and kwargs.get('past_key_values') is not None
                 and kwargs['past_key_values'].get_seq_length() == width+step-1
                 and torch.equal(ids, ids[-1:].expand_as(ids)), 'Cached global history or visual ownership differs')
        inputs_seen.append(ids.detach().cpu().clone())
    def count(key):
        def hook(*_): counts[key] += 1
        return hook
    def language(module, args, kwargs):
        step = counts['language']; counts['language'] += 1
        pos = kwargs['position_ids'].detach().cpu(); mask = kwargs['attention_mask'].detach().cpu()
        text = mask.long().cumsum(-1)-1
        expected = layout['position_ids'] if step == 0 else (layout['rope_deltas'].view(1, batch, 1)+width+step-1).expand(3, -1, -1)
        need(pos.shape == (4, batch, width if step == 0 else 1) and torch.equal(pos[1:], expected), 'Native mRoPE differs')
        need(torch.equal(pos[0][mask.bool()], text[mask.bool()]) if step == 0 else torch.equal(pos[0], text[:, -1:]),
             'Native text positions differ')
        position_rows.append(native.tensor_info(pos))
    def after(module, args, output):
        if fusion is not None:
            fusion.assert_complete()
            need(fusion.calls == counts['model'], 'Selection must be computed once at every current query')
            if capture: captures.append(fusion.export_last_capture(cpu=True))
    start = time.perf_counter()
    with ExitStack() as stack:
        stack.callback(setattr, model.model, 'rope_deltas', rope)
        if core is not None:
            fusion = stack.enter_context(ParallelLocalLearnedMemory(model.model.language_model.layers[-2],
                norm, core, n_local_rows=meta['n_frames'], write_location=write_location,
                query_indices=[width-1], stream_positions=[width-1], selection_mode=selection_mode, capture=capture))
        if controller_observer is not None: stack.callback(controller_observer, None); controller_observer(fusion)
        for handle in (model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(recorder),
            model.register_forward_hook(after), model.model.visual.register_forward_pre_hook(count('visual')),
            model.model.language_model.register_forward_pre_hook(language, with_kwargs=True),
            norm.register_forward_pre_hook(count('norm'), prepend=True), model.lm_head.register_forward_pre_hook(count('head'))): stack.callback(handle.remove)
        with torch.inference_mode():
            result = model.generate(**move_to_device(bundle['inputs'], model.device), generation_config=config,
                                   logits_processor=LogitsProcessorList([broadcast]), logits_to_keep=1)
        torch.cuda.synchronize()
        suffix = result.sequences[:, width:]; ids = suffix[-1].detach().cpu().tolist(); steps = len(ids)
        need(getattr(result, 'logits', None) is None and getattr(result, 'scores', None) is None, 'HF retained per-row output histories')
        need(1 <= steps <= max_new_tokens and torch.equal(suffix, suffix[-1:].expand_as(suffix))
             and counts == dict(model=steps, visual=1, language=steps, norm=steps, head=steps)
             and len(recorder.records) == broadcast.calls == steps
             and all(r['native_dtype'] == 'torch.float16' for r in recorder.records), 'Native generation/call inventory differs')
        need([r['top1_token_id'] for r in recorder.records] == ids
             and all(bool((inputs_seen[t] == ids[t-1]).all()) for t in range(1, steps)), 'Unmasked global argmax/history differs')
        eos = policy['native_eos_token_ids']; completed = ids[-1] in eos
        need(not any(x in eos for x in ids[:-1]) and (completed or steps == max_new_tokens), 'Unexpected native answer stopping')
    need(versions == _versions(model, core) and (fusion is None or not fusion.active)
         and (core is None or core.mode == selection_mode), 'Native/core mutation or leaked controller')
    return dict(selection_mode=selection_mode, generated_ids=ids, raw_logits=torch.stack(recorder.vectors) if full_vectors else None,
        text=processor.tokenizer.decode(ids, skip_special_tokens=True), raw_text=processor.tokenizer.decode(ids, skip_special_tokens=False),
        completed=completed, truncated=not completed, finish_reason='eos' if completed else 'length',
        metadata=dict(meta, selection_mode=selection_mode, write_location=write_location, read_block_index=26,
            final_block_index=27, generation=policy, layout=layout['metadata'],
            generation_position_ids=position_rows, native_identity_sha256=native_identity_sha256, scene_input_identity=binding),
        counters=dict(counts, broadcast=broadcast.calls, selection=0 if fusion is None else fusion.calls, probe_head=0),
        logit_records=recorder.records, captures=captures, model_seconds=time.perf_counter()-start)

"""Held learned-selection native N+1 execution, with the same question per row.

No local binary instruction, vocabulary probe, reference, origin artifact or
answer parser. Selection is computed from each current causal query. The sole
write is the global last query before final norm, so previous native KV is not
rewritten. The caller owns model/source/input freezing and raw publication.
"""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import time
import weakref

from scripts import native_vision_v7_runtime as native
from scripts import native_vision_semantic_gate_runtime as shared
from scripts.native_vision_reasoning_stream import GlobalLogitRecorder
from gnnformer.parallel_local_native import ParallelLocalNative, GlobalBroadcastLogitsProcessor
from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection, MODES

need = native.need
OWN = ('scripts/native_learned_selection_runtime.py',
       'gnnformer/parallel_local_learned_selection.py', 'tests/test_parallel_local_learned_selection.py',
       'gnnformer/parallel_local_aggregation.py', 'gnnformer/parallel_local_native.py',
       'scripts/native_vision_v7_runtime.py', 'scripts/native_vision_reasoning_stream.py',
       'scripts/native_vision_semantic_gate_runtime.py', 'gnnformer/parallel_local_semantic_gate.py',
       'gnnformer/parallel_local_semantic_aggregation.py', 'gnnformer/runtime.py', 'gnnformer/constants.py')


def sources():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in OWN}


append_observed_prefix = shared.append_observed_prefix
input_identity = shared.origin_identity  # Pure input/native binding; no gate artifact is created.


def validate_bundle(bundle):
    result = shared.validate_bundle(bundle)
    meta = bundle['metadata']
    need(meta['local_prompt'] == meta['global_prompt'] == meta['question']
         and meta['question_sha256'] == native.object_sha(meta['question']),
         'Every local and global prompt must be the unaltered complete question')
    return result


def prepare_scene(processor, sample, *, prefix_ids=(), verify_processor_parity=True):
    native.require_slurm()
    need(set(sample) == {'sid', 'n_frames', 'question', 'image_files'},
         'Model input view may not contain labels/roles/answers')
    need(type(sample['n_frames']) is int and sample['n_frames'] > 0
         and len(sample['image_files']) == sample['n_frames'], 'Actual image count differs')
    question = sample['question']
    need(isinstance(question, str) and bool(question.strip()), 'A complete question is required')
    import torch
    from PIL import Image
    images = []; old = processor.tokenizer.padding_side
    try:
        for item in sample['image_files']:
            path = Path(item['path'])
            need(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Original image bytes changed')
            with Image.open(path) as original:
                rgb = original.convert('RGB')
                try: images.append(rgb.resize((392, 392)))
                finally: rgb.close()
        conversations = [[dict(role='user', content=[dict(type='image', image=image),
                          dict(type='text', text=question)])] for image in images]
        conversations.append([dict(role='user', content=[dict(type='text', text=question)])])
        rows = [dict(processor.apply_chat_template(c, add_generation_prompt=True, tokenize=True,
                     return_dict=True, return_tensors='pt')) for c in conversations]
        packed = native.pack_rows(rows, processor.tokenizer.pad_token_id)
        if verify_processor_parity:
            processor.tokenizer.padding_side = 'left'
            ordinary = dict(processor.apply_chat_template(conversations, add_generation_prompt=True,
                            tokenize=True, return_dict=True, return_tensors='pt', padding=True))
            need(set(ordinary) == set(packed) and all(torch.equal(ordinary[k], packed[k]) for k in packed),
                 'Ordinary mixed HF processor parity failed')
        n = sample['n_frames']; width = packed['input_ids'].shape[1]
        meta = dict(arm='parallel', sid=sample['sid'], n_frames=n, question=question,
            question_sha256=native.object_sha(question), row_count=n+1, global_row=n, local_elements=n,
            row_kinds=['local']*n+['global'], original_prompt_width=width, prompt_width=width,
            prefix_ids=[], row_prompt_tokens=[r['input_ids'].shape[1] for r in rows],
            local_prompt=question, global_prompt=question, resize=392,
            image_paths=[x['path'] for x in sample['image_files']], image_sha256=[x['sha256'] for x in sample['image_files']],
            input_identity={k: native.tensor_info(v) for k, v in packed.items()},
            processor_parity_checked=verify_processor_parity, selection_from_current_prefix=True)
        bundle = append_observed_prefix(dict(inputs=packed, row_inputs=rows, metadata=meta), prefix_ids)
        validate_bundle(bundle)
        return bundle
    finally:
        processor.tokenizer.padding_side = old
        for image in images: image.close()


class LearnedSelectionNative(ParallelLocalNative):
    """One branch call per native norm call; no second pass for capture.

    Reuses the immutable controller lifecycle/export contract. This small hook
    override adds live selection capture to the same native cast-before-add.
    The controller stores detached captures only and is never a model child.
    """
    _owners = weakref.WeakKeyDictionary()

    def __init__(self, norm, branch, *, n_local_rows, selection_mode, capture=False):
        need(type(branch) is ParallelLocalLearnedSelection and selection_mode in MODES
             and branch.mode == selection_mode, 'Explicit selection mode must match the core')
        super().__init__(norm, branch, n_local_rows=n_local_rows, capture=capture)
        self.selection_mode = selection_mode
        self._initial_mode = selection_mode

    def _mode_guard(self):
        need(self.selection_mode == self._initial_mode and self.branch.mode == self._initial_mode,
             'Selection mode changed during native execution')

    def __enter__(self):
        self._mode_guard()
        need(self.norm not in self._owners, 'Nested learned-selection controllers are forbidden')
        result = super().__enter__()
        self._owners[self.norm] = self
        return result

    def close(self):
        super().close()
        if self._owners.get(self.norm) is self: del self._owners[self.norm]

    def _before_norm(self, module, args, kwargs):
        import torch
        self._mode_guard()
        need(module is self.norm and self.active and not any(p.requires_grad for p in module.parameters()),
             'Unexpected or unfrozen native norm')
        positional = bool(args)
        need(not (positional and 'hidden_states' in kwargs), 'Ambiguous native norm input')
        need(positional or 'hidden_states' in kwargs, 'Missing native hidden input')
        hidden = args[0] if positional else kwargs['hidden_states']
        need(isinstance(hidden, torch.Tensor) and hidden.dtype == torch.float16 and hidden.ndim == 3
             and hidden.shape[0] == self.n_local_rows+1 and hidden.shape[1] > 0
             and hidden.shape[2] == self.branch.hidden_size and bool(torch.isfinite(hidden).all()), 'Expected finite native FP16 [N+1,L,H]')
        local, global_states = hidden[:-1, -1:, :], hidden[-1, -1:, :]
        result = self.branch(local, global_states, output_dtype=hidden.dtype, capture=self.capture)
        delta, values = result if self.capture else (result, {})
        need(delta.shape == global_states.shape and delta.dtype == hidden.dtype and bool(torch.isfinite(delta).all()),
             'Native residual shape/dtype or finite cast differs')
        fused_global = global_states + delta
        need(bool(torch.isfinite(fused_global).all()), 'Native residual addition overflowed')
        fused = hidden.clone(); fused[-1, -1:, :] = fused_global
        self.calls += 1; self.last_query_position = hidden.shape[1]-1; self.last_capture = None
        if self.capture:
            values = dict(values, local_states=local, global_states=global_states,
                native_query_hidden=hidden[:, -1:, :], native_delta=delta,
                fused_global=fused[-1, -1:, :], fused_query_hidden=fused[:, -1:, :])
            self.last_capture = {k: v.detach().clone() for k, v in values.items()}
        if positional: return (fused,) + args[1:], kwargs
        updated = dict(kwargs); updated['hidden_states'] = fused
        return args, updated


def native_contract(model, core, selection_mode):
    norm = native.native_contract(model, None)  # Native-only ancestor; its old core count does not apply.
    if core is None:
        need(selection_mode is None, 'Bare execution has no selection mode')
    else:
        import torch
        need(type(core) is ParallelLocalLearnedSelection and core.mode == selection_mode
             and selection_mode in MODES and core.hidden_size == 3584 and core.rank == 96
             and core.merge == 'sum' and core.post_activation == 'silu', 'Learned selection core differs')
        need(sum(p.numel() for p in core.parameters()) == 1041697 and not core.training
             and all(p.dtype == torch.float32 and p.device == norm.weight.device
                     and not p.requires_grad and p.grad is None for p in core.parameters()),
             'Require the frozen/eval 1041697-parameter FP32 core on the native device')
    return norm


def _versions(model, core):
    return {(kind, name): p._version for kind, m in [('native', model)]+([] if core is None else [('core', core)])
            for name, p in m.named_parameters()}


def forward_native(model, bundle, core=None, *, selection_mode=None, native_identity_sha256,
                   capture=True, cpu=True, controller_observer=None):
    """One bare/fused uncached last-query forward at any supplied strict prefix."""
    native.require_slurm(gpu=True)
    import torch
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = native_contract(model, core, selection_mode); validate_bundle(bundle)
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
    def before_norm(module, args):
        counts['norm'] += 1
        if capture: observed.update(native.normal_copies(dict(native_query_hidden=args[0][:, -1:, :],
            local_states=args[0][:-1, -1:, :], global_states=args[0][-1, -1:, :]), cpu=cpu))
    start = time.perf_counter()
    with ExitStack() as stack:
        stack.callback(setattr, model.model, 'rope_deltas', rope)
        for handle in (model.register_forward_pre_hook(count('model')), model.model.visual.register_forward_pre_hook(count('visual')),
            model.model.language_model.register_forward_pre_hook(language, with_kwargs=True),
            norm.register_forward_pre_hook(before_norm), model.lm_head.register_forward_pre_hook(count('head'))): stack.callback(handle.remove)
        if core is not None:
            fusion = stack.enter_context(LearnedSelectionNative(norm, core, n_local_rows=bundle['metadata']['n_frames'],
                                         selection_mode=selection_mode, capture=capture))
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
        cap = None if fusion is None else fusion.export_last_capture(cpu=cpu)
    need(versions == _versions(model, core) and (fusion is None or not fusion.active)
         and (core is None or core.mode == selection_mode), 'Native/core mutation or leaked controller')
    return dict(values, **{k: v for k, v in observed.items() if k not in ('position_ids', 'attention_mask')},
        capture=cap, counters=dict(counts, selection=0 if fusion is None else fusion.calls, probe_head=0),
        metadata=dict(bundle['metadata'], selection_mode=selection_mode, layout=layout['metadata'],
                      native_identity_sha256=native_identity_sha256, scene_input_identity=binding),
        model_seconds=time.perf_counter()-start)


def generate_native(model, processor, core, bundle, *, selection_mode=None, native_identity_sha256,
                    max_new_tokens=4, capture=True, controller_observer=None):
    """Four-token unmasked native generation, one initial vision pass, no origin probe."""
    native.require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = native_contract(model, core, selection_mode); validate_bundle(bundle)
    meta = bundle['metadata']; width, batch = meta['prompt_width'], meta['row_count']
    need(not meta['prefix_ids'] and width == meta['original_prompt_width'], 'Generation starts from the original prompt')
    need(type(max_new_tokens) is int and max_new_tokens == 4, 'The native answer budget is four tokens')
    binding = input_identity(bundle, native_identity_sha256)
    config, policy = native.generation_policy(model, processor.tokenizer, max_new_tokens=4)
    config.output_logits = False; config.output_scores = False
    policy = dict(policy, output_logits=False, output_scores=False, global_row_streaming=True)
    recorder = GlobalLogitRecorder(max_steps=4, full_vectors=True)
    broadcast = GlobalBroadcastLogitsProcessor(n_local_rows=meta['n_frames'], prompt_length=width)
    layout = native.audit_layout(get_rope_index_fn(model), bundle)
    rope = getattr(model.model, 'rope_deltas', None); versions = _versions(model, core)
    counts = dict(model=0, visual=0, language=0, norm=0, head=0)
    inputs_seen = []; position_rows = []; captures = []; fusion = None
    def before(module, args, kwargs):
        step = counts['model']; counts['model'] += 1
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
            need(fusion.calls == counts['model'], 'Selection must be computed once at every current query')
            if capture: captures.append(fusion.export_last_capture(cpu=True))
    start = time.perf_counter()
    with ExitStack() as stack:
        stack.callback(setattr, model.model, 'rope_deltas', rope)
        if core is not None:
            fusion = stack.enter_context(LearnedSelectionNative(norm, core, n_local_rows=meta['n_frames'],
                                         selection_mode=selection_mode, capture=capture))
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
        need(1 <= steps <= 4 and torch.equal(suffix, suffix[-1:].expand_as(suffix))
             and counts == dict(model=steps, visual=1, language=steps, norm=steps, head=steps)
             and len(recorder.records) == broadcast.calls == steps
             and all(r['native_dtype'] == 'torch.float16' for r in recorder.records), 'Native generation/call inventory differs')
        need([r['top1_token_id'] for r in recorder.records] == ids
             and all(bool((inputs_seen[t] == ids[t-1]).all()) for t in range(1, steps)), 'Unmasked global argmax/history differs')
        eos = policy['native_eos_token_ids']; completed = ids[-1] in eos
        need(not any(x in eos for x in ids[:-1]) and (completed or steps == 4), 'Unexpected native answer stopping')
    need(versions == _versions(model, core) and (fusion is None or not fusion.active)
         and (core is None or core.mode == selection_mode), 'Native/core mutation or leaked controller')
    return dict(selection_mode=selection_mode, generated_ids=ids, raw_logits=torch.stack(recorder.vectors),
        text=processor.tokenizer.decode(ids, skip_special_tokens=True), raw_text=processor.tokenizer.decode(ids, skip_special_tokens=False),
        completed=completed, truncated=not completed, finish_reason='eos' if completed else 'length',
        metadata=dict(meta, selection_mode=selection_mode, generation=policy, layout=layout['metadata'],
            generation_position_ids=position_rows, native_identity_sha256=native_identity_sha256, scene_input_identity=binding),
        counters=dict(counts, broadcast=broadcast.calls, selection=0 if fusion is None else fusion.calls, probe_head=0),
        logit_records=recorder.records, captures=captures, model_seconds=time.perf_counter()-start)

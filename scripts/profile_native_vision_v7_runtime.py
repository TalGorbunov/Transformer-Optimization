"""Fixed V7 native runtime integration, without training or count scoring.

Two old software scenes, both paired arms, native/zero/active generations and
all candidate prefixes. Cache/full TV<=.02/top1 is descriptive, with every
failure retained. Zero-U identity and exact input/mask/cache/call mechanics are
mandatory. Active U is a fixed .001 normal draw; there is no fit or gold prefix.
All work runs through CPU/GPU Slurm. Original mixed failure remains unchanged.
"""
from __future__ import annotations
import argparse
import copy
import json
import importlib.metadata
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from scripts import native_vision_v7_runtime as runtime
from scripts import probe_native_vision_parallel_local_mixed as mixed
from scripts import probe_native_vision_parallel_local as local
from scripts.probe_native_vision_mixed_cache_localization import mask_check
from scripts.stage_native_vision_v6_teacher import MODEL, need, read, save, sha, model_metadata
OUT = REPO / 'outputs/native_aggregation_vlm/v7/runtime'
DATA = Path('/mnt/data/gabriele/gnn_transformer/v7_runtime')
CKPT = Path('/mnt/ckpts/gabriele/gnn_transformer/v7_runtime')
SEED = 20260927
OWN = ('scripts/profile_native_vision_v7_runtime.py', 'scripts/native_vision_v7_runtime.py',
       'slurm/native_vision_v7_runtime_selftest.sbatch', 'slurm/native_vision_v7_runtime_check.sbatch',
       'slurm/native_vision_v7_runtime_profile.sbatch', 'gnnformer/parallel_local_aggregation.py',
       'gnnformer/parallel_local_native.py', 'gnnformer/parallel_local_prompts.py', 'gnnformer/runtime.py',
       'gnnformer/data.py', 'scripts/probe_native_vision_parallel_local_mixed.py',
       'scripts/probe_native_vision_parallel_local.py', 'scripts/probe_native_vision_mixed_cache_localization.py',
       'scripts/stage_native_vision_v6_teacher.py', 'scripts/cache_native_vision_v6_teacher.py',
       'scripts/probe_native_vision_v2_prefix.py', 'scripts/probe_native_vision_v5_local_readability.py')


def sources(): return {name: sha(REPO / name) for name in OWN}


def snapshot(out):
    (out / 'source').mkdir()
    for name in OWN: (out / 'source' / name.replace('/', '_')).write_bytes((REPO / name).read_bytes())
    save(out / 'source_hashes.json', sources())


def initial_states(torch):
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    with torch.random.fork_rng():
        torch.manual_seed(SEED)
        core = ParallelLocalAggregation()
        zero = {name: value.detach().clone() for name, value in core.state_dict().items()}
        with torch.no_grad(): core.up.weight.normal_(0, .001)
        active = {name: value.detach().clone() for name, value in core.state_dict().items()}
    need(sum(x.numel() for x in zero.values()) == 1041600 and not bool(zero['up.weight'].any())
         and all(torch.equal(zero[k], active[k]) for k in zero if k != 'up.weight'), 'Fixed initialization differs')
    return dict(zero=zero, active=active)


def state_identity(states):
    return {condition: {name: runtime.tensor_info(value) for name, value in state.items()}
            for condition, state in states.items()}


def check():
    import torch, transformers
    from transformers import AutoProcessor, GenerationConfig
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4); begin = time.perf_counter(); frozen = sources()
    tests = runtime.self_test()
    teacher, _, _, _, binding = local.teacher_inputs()
    scenes, manifest = mixed.selected_sources()
    prior_path = mixed.OUT/'run_441729'/'summary.json'
    need(read(prior_path)['numerical_gate_passed'] is False, 'Original mixed failure must remain false')
    original_failure = dict(path=str(prior_path), sha256=sha(prior_path), numerical_gate_passed=False)
    processor = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True, use_fast=False)
    need(fingerprint(processor, str(transformers.__version__)) == teacher['processor'], 'Native processor ancestor differs')
    owner, fn, native = local.native_api(processor)
    def rope(**kwargs): return fn(owner, **kwargs)
    policy_model = SimpleNamespace(generation_config=GenerationConfig.from_pretrained(str(MODEL)))
    config, policy = runtime.generation_policy(policy_model, processor.tokenizer)
    states = initial_states(torch)
    bundles, cases, shared_pixels = {}, [], {}
    for arm in ('parallel', 'joint'):
        for scene in scenes:
            case_id = f"{arm}_N{scene['n_frames']}"
            # Scene labels are used only by the immutable selection audit, never by this runtime.
            sample = {key: scene[key] for key in ('sid', 'n_frames', 'question', 'image_files')}
            bundle = runtime.prepare_scene(processor, sample, arm, verify_processor_parity=True)
            layout = runtime.audit_layout(rope, bundle)
            meta = bundle['metadata']; n = meta['n_frames']
            if arm == 'parallel':
                need(meta['row_prompt_tokens'][-1] < meta['prompt_width'], 'Profile must exercise genuine global left padding')
            # Deduplicate identical processed pixels in the serialization, without changing any tensor bytes.
            if n in shared_pixels:
                need(torch.equal(shared_pixels[n], bundle['inputs']['pixel_values']), 'Arm-specific pixels differ')
                bundle['inputs']['pixel_values'] = shared_pixels[n]
            else:
                shared_pixels[n] = bundle['inputs']['pixel_values']
            offset = 0
            for row in bundle['row_inputs']:
                if 'pixel_values' in row:
                    size = row['pixel_values'].shape[0]
                    row['pixel_values'] = shared_pixels[n][offset:offset+size]
                    offset += size
            need(offset == shared_pixels[n].shape[0], 'Serialized image-row order differs')
            bundles[case_id] = bundle
            cases.append(dict(case_id=case_id, metadata=meta, layout=layout['metadata']))
    job = os.environ['SLURM_JOB_ID']; out = OUT / f'check_{job}'; out.mkdir(parents=True, exist_ok=False); snapshot(out)
    data = DATA / f'check_{job}'; data.mkdir(parents=True, exist_ok=False)
    weights = CKPT / f'check_{job}'; weights.mkdir(parents=True, exist_ok=False)
    prepared = data / 'prepared.pt'; initial = weights / 'initial.pt'
    torch.save(dict(schema_version=1, bundles=bundles), prepared)
    torch.save(dict(schema_version=1, states=states), initial)
    plan = dict(schema_version=1, protocol='v7_native_runtime_integration', seed=SEED, source_sha256=frozen,
        ancestor=binding, model=model_metadata(), runtime=teacher['runtime'], processor=teacher['processor'],
        native_api=native, source_manifest=manifest, original_mixed_failure=original_failure, cases=cases, generation_policy=policy,
        generation_config=config.to_dict(), initial_state_identity=state_identity(states),
        initial_file=str(initial), initial_sha256=sha(initial), prepared_file=str(prepared), prepared_sha256=sha(prepared),
        tests=tests, maximum_calls=dict(model=80, visual=44),
        zero_identity='Exact global raw logits and generated IDs versus native no-fusion baseline',
        numerical_policy='Every candidate prefix TV<=.02 and top1 exact reported descriptively; every failure retained',
        active_initialization='CPU seed20260927 ordinary core init; then up.weight normal_(0,.001), same tensors all arms/scenes',
        scope='Software only; no count scoring/training/gold-prefix selection; original mixed441729 gate remains failed',
        native_dtype='torch.float16', branch_dtype='torch.float32', slurm_job_id=job)
    need(sources() == frozen, 'Source changed during CPU freeze')
    path = out / 'plan.json'; save(path, plan); path.with_suffix('.sha256').write_text(sha(path)+'\n')
    verify(path)
    save(out / 'summary.json', dict(passed=True, plan_file=str(path), plan_sha256=sha(path),
        source_sha256=frozen, seconds=time.perf_counter()-begin, cases=len(cases)))
    local.index(out, 'V7 native runtime CPU freeze', [('Plan', 'plan.json'), ('Summary', 'summary.json'), ('Sources', 'source/')])
    local.index(data, 'V7 prepared software inputs', [('Prepared tensors', 'prepared.pt')])
    local.index(weights, 'V7 untrained fixed software branches', [('Initial tensors', 'initial.pt')])
    print(json.dumps(dict(passed=True, plan=str(path), plan_sha256=sha(path))), flush=True)


def verify(path):
    path = Path(path)
    need(path.is_relative_to(OUT) and sha(path) == path.with_suffix('.sha256').read_text().strip(), 'Plan sidecar differs')
    plan = read(path)
    need(plan['schema_version'] == 1 and plan['protocol'] == 'v7_native_runtime_integration'
         and plan['seed'] == SEED and plan['source_sha256'] == sources() and plan['model'] == model_metadata(),
         'Frozen native software source/model differs')
    for item in (plan['source_manifest'], plan['original_mixed_failure']): need(sha(item['path']) == item['sha256'], 'Source manifest changed')
    need(sha(plan['prepared_file']) == plan['prepared_sha256'] and sha(plan['initial_file']) == plan['initial_sha256'],
         'Prepared tensors/initial branches changed')
    for name, digest in plan['source_sha256'].items():
        need(sha(path.parent / 'source' / name.replace('/', '_')) == digest, 'CPU source snapshot changed')
    need([(x['metadata']['arm'], x['metadata']['n_frames']) for x in plan['cases']] ==
         [('parallel', 16), ('parallel', 64), ('joint', 16), ('joint', 64)], 'Fixed case inventory differs')
    return plan


def metric(torch, a, b):
    a, b = a.double(), b.double()
    need(a.ndim == b.ndim == 1 and a.shape == b.shape and bool(torch.isfinite(a).all())
         and bool(torch.isfinite(b).all()), 'Nonfinite or malformed native logits')
    difference = a-b; centered = difference-difference.mean()
    tv = float(.5*(torch.softmax(a, -1)-torch.softmax(b, -1)).abs().sum())
    same = int(a.argmax()) == int(b.argmax())
    return dict(full_vocabulary_tv=tv, top1_equal=same, raw_maximum_absolute=float(difference.abs().max()),
        centered_maximum_absolute=float(centered.abs().max()), centered_rms=float(centered.square().mean().sqrt()),
        numerical_rule_passed=tv <= .02 and same)


def prefixed_bundle(bundle, ids):
    value = dict(inputs=runtime.append_prefix(bundle['inputs'], ids),
                 row_inputs=[runtime.append_prefix(row, ids) for row in bundle['row_inputs']],
                 metadata=copy.deepcopy(bundle['metadata']))
    value['metadata']['prefix_ids'] = list(ids)
    value['metadata']['prompt_width'] += len(ids)
    value['metadata']['input_identity'] = {key: runtime.tensor_info(x) for key, x in value['inputs'].items()}
    return value


class NativeAudit:
    """Observe actual model inputs, valid-query masks and retained native KV prefixes."""
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.records, self.handles = [], []
        self.counts = dict(model=0, visual=0, language=0, norm=0, attention=0)
        self.context, self.capture, self.previous = {}, {}, None

    def __enter__(self):
        from gnnformer.runtime import get_layers
        self.layers = get_layers(self.model)
        self.handles = [self.model.register_forward_pre_hook(self.before, with_kwargs=True),
            self.model.register_forward_hook(self.after),
            self.model.model.visual.register_forward_pre_hook(self.visual),
            self.model.model.language_model.register_forward_pre_hook(self.language, with_kwargs=True),
            self.model.model.language_model.norm.register_forward_pre_hook(self.norm),
            self.layers[0].self_attn.register_forward_pre_hook(self.attention, with_kwargs=True)]
        return self

    def __exit__(self, *_):
        for handle in self.handles: handle.remove()
        self.previous = None
        return False

    def before(self, module, args, kwargs):
        import torch
        self.capture = {}; self.counts['model'] += 1
        self.capture['input_ids'] = kwargs['input_ids'].detach().cpu().clone()
        self.capture['attention_mask'] = kwargs['attention_mask'].detach().cpu().clone()
        self.capture['visual_expected'] = kwargs.get('pixel_values') is not None
        self.capture['use_cache'] = kwargs.get('use_cache', False)
        self.capture['before_counts'] = dict(self.counts)
        self.capture['expected_visual_identity'] = None if kwargs.get('pixel_values') is None else dict(
            pixel_values=runtime.tensor_info(kwargs['pixel_values']), image_grid_thw=runtime.tensor_info(kwargs['image_grid_thw']))
        cache = kwargs.get('past_key_values'); length = 0 if cache is None else cache.get_seq_length()
        self.capture['old_length'] = length; self.previous = None
        if length:
            self.previous, _ = mixed.cache_snapshot(cache, kwargs['input_ids'].shape[0], length, len(self.layers), clone=True)
        torch.cuda.synchronize(); self.capture['started'] = time.perf_counter()

    def visual(self, *_): self.counts['visual'] += 1

    def language(self, module, args, kwargs):
        self.counts['language'] += 1
        self.capture['position_ids'] = kwargs['position_ids'].detach().cpu().clone()
        need(runtime.tensor_info(kwargs['attention_mask']) == runtime.tensor_info(self.capture['attention_mask']),
             'Language model changed the actual key padding mask')

    def norm(self, module, args):
        self.counts['norm'] += 1
        self.capture['pre_final_rms_hidden'] = args[0][:, -1, :].detach().cpu().clone()

    def attention(self, module, args, kwargs):
        self.counts['attention'] += 1
        mask = kwargs.get('attention_mask')
        self.capture['causal_mask'] = None if mask is None else mask.detach().cpu().clone()
        self.capture['cache_position'] = kwargs['cache_position'].detach().cpu().clone()

    def after(self, module, args, output):
        import torch
        torch.cuda.synchronize(); elapsed = time.perf_counter()-self.capture['started']
        value = self.capture; current = self.counts
        need(all(current[key]-value['before_counts'][key] == 1 for key in ('language', 'norm', 'attention'))
             and current['visual']-value['before_counts']['visual'] == int(value['visual_expected']),
             'Actual native submodule counts differ')
        masks = mask_check(torch, value['causal_mask'], value['attention_mask'], value['cache_position'])
        logits = output.logits[:, -1, :].detach().cpu().clone()
        hidden = value['pre_final_rms_hidden']
        need(hidden.dtype == torch.float16 and logits.dtype == torch.float16
             and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(hidden).all()), 'Native outputs changed dtype/finite contract')
        cache = output.past_key_values; retained = None; cache_meta = None
        if value['use_cache']:
            length = value['attention_mask'].shape[1]
            _, cache_meta = mixed.cache_snapshot(cache, hidden.shape[0], length, len(self.layers))
            if self.previous is not None:
                retained = mixed.prefix_preserved(torch, cache, self.previous, value['old_length'])
        else:
            need(cache is None and self.previous is None, 'Uncached reference returned/used KV state')
        self.previous = None
        state = dict(schema_version=1, context=dict(self.context), native_logits=logits,
                     **{key: value[key] for key in ('input_ids', 'attention_mask', 'position_ids', 'causal_mask',
                                                   'cache_position', 'pre_final_rms_hidden')})
        path = self.data / f"forward_{len(self.records):03d}.pt"; torch.save(state, path)
        record = dict(context=dict(self.context), forward_index=len(self.records), path=str(path), sha256=sha(path),
            batch_size=hidden.shape[0], query_tokens=value['input_ids'].shape[1], key_tokens=value['attention_mask'].shape[1],
            visual=int(value['visual_expected']), mask_audit=masks, native_forward_with_observer_seconds=elapsed,
            native_logits=runtime.tensor_info(logits), native_hidden=runtime.tensor_info(hidden),
            cache_layers=cache_meta, previous_prefix_exact_by_layer=retained,
            expected_visual_identity=value['expected_visual_identity'])
        self.records.append(record)


def native_baseline(model, processor, bundle):
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
    from gnnformer.runtime import move_to_device
    config, policy = runtime.generation_policy(model, processor.tokenizer)
    processors = LogitsProcessorList()
    meta = bundle['metadata']; width = meta['prompt_width']
    if meta['arm'] == 'parallel':
        processors.append(GlobalBroadcastLogitsProcessor(n_local_rows=meta['n_frames'], prompt_length=width))
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**move_to_device(bundle['inputs'], model.device), generation_config=config,
                                logits_processor=processors, logits_to_keep=1)
    torch.cuda.synchronize()
    suffix = output.sequences[:, width:]
    need(torch.equal(suffix, suffix[-1:].expand_as(suffix)), 'Native baseline broadcast history differs')
    ids = suffix[-1].cpu().tolist(); raw = torch.stack([x[-1] for x in output.logits]).cpu()
    need(1 <= len(ids) <= 4 and raw.argmax(-1).tolist() == ids and bool(torch.isfinite(raw).all()),
         'Native baseline is not unrestricted greedy output')
    eos = policy['native_eos_token_ids']; completed = ids[-1] in eos
    need(not any(x in eos for x in ids[:-1]) and (completed or len(ids) == 4), 'Native baseline stopped unexpectedly')
    return dict(generated_ids=ids, raw_logits=raw, model_seconds=time.perf_counter()-start,
                completed=completed, truncated=not completed, native_without_fusion=True)


def audit_sequence(torch, model, bundle, records, generated_ids):
    """Check all executed native positions/masks/tokens, including no-fusion baseline."""
    from gnnformer.runtime import get_rope_index_fn
    meta = bundle['metadata']; width = meta['prompt_width']; batch = meta['row_count']
    need(len(records) == len(generated_ids), 'Exactly one model call per generated token required')
    for step, record in enumerate(records):
        state = torch.load(record['path'], map_location='cpu', weights_only=True)
        full = runtime.append_prefix(bundle['inputs'], generated_ids[:step])
        pos, delta = get_rope_index_fn(model)(input_ids=full['input_ids'], image_grid_thw=full['image_grid_thw'],
                                           attention_mask=full['attention_mask'])
        actual = state['position_ids']; mask = full['attention_mask']; text = mask.cumsum(-1)-1
        need(actual.shape == (4, batch, width if step == 0 else 1)
             and torch.equal(state['attention_mask'], mask), 'Generation position/key-mask shape differs')
        if step == 0:
            need(torch.equal(state['input_ids'], full['input_ids']) and torch.equal(actual[1:], pos)
                 and torch.equal(actual[0][mask.bool()], text[mask.bool()]) and record['visual'] == 1,
                 'Native prefill tokens/positions/vision differ')
            need(record['expected_visual_identity'] == {key: meta['input_identity'][key] for key in
                 ('pixel_values', 'image_grid_thw')}, 'Native prefill pixels/grid differ')
        else:
            need(state['input_ids'].shape == (batch, 1) and bool((state['input_ids'] == generated_ids[step-1]).all())
                 and torch.equal(actual[1:], pos[:, :, -1:]) and torch.equal(actual[0], text[:, -1:])
                 and record['visual'] == 0 and all(record['previous_prefix_exact_by_layer']),
                 'Native cached history/logical positions/prefix retention differ')
        need(state['native_logits'][-1].argmax().item() == generated_ids[step], 'Saved native global argmax differs')


def run(args):
    import torch, transformers
    from gnnformer.runtime import load_runtime, get_rope_index_fn
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4); begin = time.perf_counter(); plan = verify(args.plan)
    cpu_summary = read(Path(args.plan).parent/'summary.json')
    need(cpu_summary['passed'] and cpu_summary['plan_sha256'] == sha(args.plan), 'Completed CPU freeze required')
    teacher, _, _, _, binding = local.teacher_inputs()
    actual_runtime = dict(torch_version=str(torch.__version__), transformers_version=str(transformers.__version__),
                          bitsandbytes_version=importlib.metadata.version('bitsandbytes'))
    need(actual_runtime == plan['runtime'], 'Actual installed runtime version differs')
    need(binding == plan['ancestor'] and teacher['runtime'] == plan['runtime'], 'Frozen ancestor/runtime differs')
    blob = torch.load(plan['prepared_file'], map_location='cpu', weights_only=True)
    weights = torch.load(plan['initial_file'], map_location='cpu', weights_only=True)
    need(blob['schema_version'] == weights['schema_version'] == 1
         and state_identity(weights['states']) == plan['initial_state_identity'], 'Input/initial-state blob differs')
    job = os.environ['SLURM_JOB_ID']; out = OUT/f'profile_{job}'; out.mkdir(parents=True, exist_ok=False); snapshot(out)
    data = DATA/f'profile_{job}'; data.mkdir(parents=True, exist_ok=False)
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    local.index(out, 'V7 native runtime integration', [('Frozen plan', 'plan.json'), ('Summary after completion', 'summary.json'),
        ('Native forward evidence', 'forwards.json'), ('Cache/full differences', 'comparisons.json')])
    tick = time.perf_counter(); loaded = load_runtime(str(MODEL), use_4bit=True, attn_implementation='sdpa', device_map='cuda')
    model = loaded.model; model.eval(); model.requires_grad_(False)
    runtime.native_contract(model)
    need(fingerprint(loaded.processor, str(transformers.__version__)) == plan['processor'], 'Loaded processor differs')
    _, _, native = local.native_api(loaded.processor)
    need(native == plan['native_api'], 'Installed native implementation differs')
    config, policy = runtime.generation_policy(model, loaded.tokenizer)
    need(policy == plan['generation_policy'] and config.to_dict() == plan['generation_config'], 'Generation policy differs')
    torch.cuda.synchronize(); load_seconds = time.perf_counter()-tick
    branch = ParallelLocalAggregation().to(device=loaded.device)
    observations, comparisons, row_comparisons, artifacts = [], [], [], []
    with NativeAudit(model, data) as audit:
        for case in plan['cases']:
            case_id = case['case_id']; bundle = blob['bundles'][case_id]
            need(bundle['metadata'] == case['metadata'] and runtime.audit_layout(get_rope_index_fn(model), bundle)['metadata'] == case['layout'],
                 'Prepared case/layout changed')
            baseline = None
            for condition in ('native', 'zero', 'active'):
                audit.context = dict(case_id=case_id, condition=condition, phase='generation')
                offset = len(audit.records); torch.cuda.reset_peak_memory_stats(); phase_start = time.perf_counter()
                if condition == 'native':
                    result = native_baseline(model, loaded.processor, bundle); baseline = result
                else:
                    branch.load_state_dict(weights['states'][condition], strict=True)
                    result = runtime.generate_native(model, loaded.processor, branch, bundle, capture=True)
                records = audit.records[offset:]
                audit_sequence(torch, model, bundle, records, result['generated_ids'])
                zero_exact = None
                if condition == 'zero':
                    zero_exact = result['generated_ids'] == baseline['generated_ids'] and torch.equal(result['raw_logits'], baseline['raw_logits'])
                    need(zero_exact, 'Zero-U fusion is not identical to native no-fusion generation')
                    need(all(not bool(x['delta'].any()) for x in result['captures']), 'Zero-U produced a nonzero residual')
                path = data/f'{case_id}__{condition}__generation.pt'; torch.save(result, path)
                artifacts.append(dict(case_id=case_id, condition=condition, phase='generation', path=str(path), sha256=sha(path)))
                row = dict(case_id=case_id, arm=case['metadata']['arm'], n_frames=case['metadata']['n_frames'],
                    condition=condition, generated_ids=result['generated_ids'], completed=result['completed'],
                    model_seconds=result['model_seconds'], total_instrumented_seconds=time.perf_counter()-phase_start,
                    peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(), zero_identity_exact=zero_exact,
                    model_calls=len(records), visual_calls=sum(x['visual'] for x in records),
                    artifact_path=str(path), artifact_sha256=sha(path))
                observations.append(row)
                if condition == 'native': continue
                prefix_start = time.perf_counter(); full_offset = len(audit.records)
                for step in range(len(result['generated_ids'])):
                    audit.context = dict(case_id=case_id, condition=condition, phase='full_prefix', step=step)
                    prefixed = prefixed_bundle(bundle, result['generated_ids'][:step])
                    reference = runtime.forward_native(model, prefixed, branch, capture=True)
                    item = metric(torch, result['raw_logits'][step], reference['global_logits'])
                    item.update(case_id=case_id, arm=case['metadata']['arm'], n_frames=case['metadata']['n_frames'],
                                condition=condition, step=step, generated_prefix_ids=result['generated_ids'][:step], descriptive_only=True)
                    comparisons.append(item)
                    cached_state = torch.load(records[step]['path'], map_location='cpu', weights_only=True)
                    full_state = torch.load(audit.records[-1]['path'], map_location='cpu', weights_only=True)
                    need(cached_state['native_logits'].shape == full_state['native_logits'].shape,
                         'All-row cache/full vocabulary shape differs')
                    for row_index, (cached_logits, full_logits) in enumerate(zip(cached_state['native_logits'], full_state['native_logits'])):
                        row_item = metric(torch, cached_logits, full_logits)
                        ch = cached_state['pre_final_rms_hidden'][row_index].double()
                        fh = full_state['pre_final_rms_hidden'][row_index].double()
                        difference = ch-fh; denominator = float(fh.norm())
                        row_item.update(case_id=case_id, condition=condition, step=step, row_index=row_index,
                            row_kind=bundle['metadata']['row_kinds'][row_index], descriptive_only=True,
                            hidden_rms=float(difference.square().mean().sqrt()),
                            hidden_relative_l2=float(difference.norm())/denominator if denominator else None,
                            hidden_cosine=float(torch.dot(ch,fh)/(ch.norm()*fh.norm())) if float(ch.norm()*fh.norm()) else None,
                            cached_state_file=records[step]['path'], full_state_file=audit.records[-1]['path'])
                        row_comparisons.append(row_item)
                    refpath = data/f'{case_id}__{condition}__full_{step}.pt'; torch.save(reference, refpath)
                    artifacts.append(dict(case_id=case_id, condition=condition, phase='full_prefix', step=step,
                                          path=str(refpath), sha256=sha(refpath)))
                row['full_prefix_replay_seconds'] = time.perf_counter()-prefix_start
                row['full_prefix_calls'] = len(audit.records)-full_offset
                need(row['full_prefix_calls'] == len(result['generated_ids']), 'Incomplete prefix coverage')
                print(json.dumps(dict(case_id=case_id, condition=condition, generated_tokens=len(result['generated_ids']),
                                      generation_seconds=row['model_seconds'], replay_seconds=row['full_prefix_replay_seconds'])), flush=True)
        counts = dict(audit.counts); records = list(audit.records)
    expected_generation = sum(row['model_calls'] for row in observations)
    expected_replay = len(comparisons)
    need(len(observations) == 12 and counts == dict(model=expected_generation+expected_replay, visual=12+expected_replay,
         language=expected_generation+expected_replay, norm=expected_generation+expected_replay, attention=expected_generation+expected_replay)
         and counts['model'] <= 80 and counts['visual'] <= 44, 'Registered software invocation coverage differs')
    need(verify(args.plan) == plan, 'Source/input state changed during software execution')
    for path, value in (('forwards.json', records), ('observations.json', observations), ('comparisons.json', comparisons),
                        ('artifacts.json', artifacts), ('all_row_comparisons.json', row_comparisons)): save(out/path, value)
    failures = [x for x in comparisons if not x['numerical_rule_passed']]
    all_failures = [x for x in row_comparisons if not x['numerical_rule_passed']]
    summary = dict(schema_version=1, completed=True, computational_integrity_passed=True, zero_identity_passed=True,
        strict_numerical_gates_passed=not all_failures, global_numerical_failures=failures, numerical_failures=all_failures,
        numerical_policy=plan['numerical_policy'], original_mixed_numerical_gate_passed=False, original_mixed_failure=plan['original_mixed_failure'],
        plan_file=str(Path(args.plan).resolve()), plan_sha256=sha(args.plan), source_sha256=plan['source_sha256'],
        model=plan['model'], runtime=plan['runtime'], processor=plan['processor'], generation_policy=policy,
        seed=SEED, native_dtype='torch.float16', branch_dtype='torch.float32', model_load_seconds=load_seconds,
        total_seconds=time.perf_counter()-begin, calls=counts, generated_tokens=expected_generation,
        prefix_comparisons=len(comparisons), all_row_comparisons=len(row_comparisons), observations=observations, gpu=torch.cuda.get_device_name(0), slurm_job_id=job,
        files={name: dict(path=str(out/name), sha256=sha(out/name)) for name in
               ('forwards.json', 'observations.json', 'comparisons.json', 'all_row_comparisons.json', 'artifacts.json')},
        limitations=['Fixed random branches and old software scenes; no fitting, count scoring or efficacy conclusion.',
            'Native EOS list is explicit and differs from historical V4-V6 tokenizer-EOS-only decoding.',
            'Cache/full TV and top1 failures remain descriptive and are never relabeled passed.',
            'KV-prefix retention is an actual GPU assertion; full KV values are not saved for independent replay.',
            'Timing includes audit hooks, CPU transfers, hashes and tensor serialization; prefix loops are separate.',
            'An untrained active branch can produce any token; no prompt/seed/scale is chosen from its outputs.'])
    save(out/'summary.json', summary)
    lines = ['# V7 native runtime software profile', '',
        f"Computational integrity and zero-U identity passed. Global numerical failures: {len(failures)}/{len(comparisons)}; all-row failures: {len(all_failures)}/{len(row_comparisons)}.", '',
        '| Arm | N | Condition | Tokens | Generation seconds | Prefix replay seconds |',
        '|---|---:|---|---:|---:|---:|']
    for row in observations:
        lines.append(f"| {row['arm']} | {row['n_frames']} | {row['condition']} | {len(row['generated_ids'])} | {row['model_seconds']:.3f} | {row.get('full_prefix_replay_seconds',0):.3f} |")
    lines += ['', 'No aggregation accuracy was measured. All numerical failures and the original mixed/cache failure remain recorded.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    local.index(data, 'Native V7 software evidence', [('Profile summary', str(out/'summary.json')),
                                                   ('Full native forward inventory', str(out/'forwards.json'))])
    print(json.dumps(dict(directory=str(out), computational_integrity_passed=True, zero_identity_passed=True,
                         strict_numerical_gates_passed=not all_failures, numerical_failures=len(all_failures), calls=counts)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true'); mode.add_argument('--run', action='store_true')
    parser.add_argument('--plan', type=Path); args = parser.parse_args()
    runtime.require_slurm(gpu=args.run)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION') == 'cpu' and not os.environ.get('SLURM_JOB_GPUS'), 'CPU freeze only')
        check()
    else:
        need(args.plan is not None, 'Exact frozen CPU plan required')
        run(args)


if __name__ == '__main__': main()

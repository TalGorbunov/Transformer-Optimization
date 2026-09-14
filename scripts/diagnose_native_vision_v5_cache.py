"""V5 cache discrepancy localization. Diagnostic only; no new acceptance gate.

CPU --check binds failed-profile checkpoints/data/source hashes. GPU --plan runs
four fixed cases, never training, and saves native-precision traces on DATA_BASE.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = Path('/mnt/data/gabriele/gnn_transformer')
OUT = REPO / 'outputs/native_aggregation_vlm/v5/cache_diagnostic'
RUNS = {
    'sum': REPO / 'outputs/native_aggregation_vlm/v5/profile/sum/sum_seed4_20260910_193917_441417_1444693',
    'mean': REPO / 'outputs/native_aggregation_vlm/v5/profile/mean/mean_seed4_20260910_193917_441416_2900405',
}
OWN = ('scripts/diagnose_native_vision_v5_cache.py',
       'slurm/native_aggregation_vision_v5_cache_check.sbatch',
       'slurm/native_aggregation_vision_v5_cache_diagnostic.sbatch')


def ensure(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sources():
    return {name: sha(REPO / name) for name in OWN}


def selected_epoch(history):
    def score(row):
        count = sum(x['n'] for x in row['dev'])
        return (sum(x['correct'] for x in row['dev']) / count,
                -sum(x['gold_first_token_nll'] * x['n'] for x in row['dev']) / count)
    return max(history, key=score)['epoch']


def make_plan(torch):
    from scripts.native_aggregation_vlm_v4 import sample_metadata
    frozen, runs, rows = {}, {}, None
    manifest_path = DATA / 'v4_diversity/profile_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = [manifest['splits'][f'test_N{n}']['samples'][0] for n in (16, 64)]
    for record in records:
        parsed = sample_metadata(Path(record['path']), record['n_frames'])
        for key in ('sid', 'gold', 'qa_sha256'):
            ensure(parsed[key] == record[key], f'QA metadata mismatch: {key}')
        record['question'] = parsed['question']
        ensure(sha(Path(record['path']) / 'qa.txt') == record['qa_sha256'], 'QA hash changed')
        for picture in record['image_files']:
            ensure(sha(picture['path']) == picture['sha256'], 'Image hash changed')
    for arm, directory in RUNS.items():
        config = json.loads((directory / 'config.json').read_text())
        history = json.loads((directory / 'training.json').read_text())
        ensure(config['profile'] and config['condition'] == arm and config['seed'] == 4, 'Wrong profile')
        ensure(len(history) == 2 and config['manifest_sha256'] == sha(manifest_path), 'Wrong profile data/history')
        for name, digest in config['code_sha256'].items():
            ensure(sha(REPO / name) == digest, f'Frozen profile source changed: {name}')
            frozen[name] = digest
        checkpoint = Path(config['checkpoint_root']) / 'native_aggregation_vlm_v5' / config['run_id'] / 'best.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        epoch = selected_epoch(history)
        ensure(saved['architecture'] == config['architecture'] and saved['epoch'] == epoch, 'Wrong selected checkpoint')
        ensure(saved['dev'] == history[epoch - 1]['dev'] and saved['step'] == epoch, 'Selected checkpoint dev/step differs')
        ensure(float(saved['branch']['up.weight'].float().norm()) > 0, 'Selected checkpoint has zero U')
        staged = json.loads((directory / 'data_manifest.json').read_text())
        for record in records:
            expected = staged[f'length_N{record["n_frames"]}']['samples'][0]
            ensure(all(expected[k] == record[k] for k in ('sid', 'gold', 'qa_sha256', 'path')), 'Profile helper case differs')
        runs[arm] = dict(directory=str(directory), config=config, checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint), selected_epoch=epoch,
            config_sha256=sha(directory / 'config.json'), training_sha256=sha(directory / 'training.json'))
        del saved
    return dict(schema_version=1, source_sha256=sources(), frozen_profile_source_sha256=frozen,
        manifest=str(manifest_path), manifest_sha256=sha(manifest_path), records=records, runs=runs,
        scope='Software discrepancy localization only; no threshold changes, efficacy scoring, training, or new model weights',
        comparisons='Cached second raw logits versus full prompt+first token; ON and forced-prefix OFF; branch-only replay',
        caveat='ON/OFF changes native activations. Numerical traces localize disagreement but do not alone prove a causal kernel or general correctness.')


def tensor_difference(torch, a, b):
    ensure(a.shape == b.shape, 'Compared tensor shapes differ')
    x, y = a.detach().float(), b.detach().float()
    ensure(bool(torch.isfinite(x).all() and torch.isfinite(y).all()), 'Nonfinite trace')
    d = x - y
    return dict(exact=torch.equal(a, b), max_abs=float(d.abs().max()),
                rms=float(d.square().mean().sqrt()), mean=float(d.mean()),
                a_norm=float(x.norm()), b_norm=float(y.norm()))


def logit_comparison(torch, cached, full):
    result = tensor_difference(torch, cached, full)
    d = cached.float() - full.float()
    centered = d - d.mean()
    lc, lf = cached.float().log_softmax(-1), full.float().log_softmax(-1)
    pc, pf = lc.exp(), lf.exp()
    result.update(centered_max_abs=float(centered.abs().max()),
        centered_rms=float(centered.square().mean().sqrt()),
        common_shift_squared_fraction=float(d.mean().square() / d.square().mean()) if bool(d.square().mean() > 0) else 0.,
        log_softmax=tensor_difference(torch, lc, lf),
        kl_cached_to_full=float((pc * (lc - lf)).sum()),
        kl_full_to_cached=float((pf * (lf - lc)).sum()),
        total_variation=float((pc - pf).abs().sum() / 2),
        cached_top1=int(cached.argmax()), full_top1=int(full.argmax()),
        top1_equal=bool(cached.argmax() == full.argmax()),
        cached_top5=cached.topk(5).indices.cpu().tolist(), full_top5=full.topk(5).indices.cpu().tolist())
    return result


def cpu_self_checks(torch):
    x = torch.tensor([1., -2., 4., 0., 2.])
    r = logit_comparison(torch, x + 8, x)
    ensure(r['rms'] == 8 and r['centered_rms'] == 0 and r['top1_equal'], 'Common-shift diagnostic failed')
    ensure(r['log_softmax']['max_abs'] == 0 and r['total_variation'] == 0, 'Shift invariance failed')
    history = [dict(epoch=1, dev=[dict(n=2, correct=1, gold_first_token_nll=2.)]),
               dict(epoch=2, dev=[dict(n=2, correct=1, gold_first_token_nll=2.)])]
    ensure(selected_epoch(history) == 1, 'Checkpoint ties must preserve earliest epoch')


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(torch, v) for v in value]
    return value


class Trace:
    def __init__(self, torch, model, branch, attn):
        self.torch, self.branch, self.traces, self.current = torch, branch, [], None
        self.last_read = None
        self.handles = [
            model.register_forward_pre_hook(self.begin, with_kwargs=True),
            attn.register_forward_pre_hook(self.before_attention, with_kwargs=True),
            attn.o_proj.register_forward_hook(self.native_output),
            attn.register_forward_hook(self.patched_output),
            branch.query.register_forward_hook(self.query),
            branch.memory.register_forward_hook(self.memory),
            branch.read.register_forward_hook(self.read),
            branch.register_forward_hook(self.branch_output),
        ]

    def begin(self, module, args, kwargs):
        self.current = dict(mode=self.branch.mode, reads=[])
        self.traces.append(self.current)
        self.last_read = None
        for key in ('input_ids', 'attention_mask', 'cache_position', 'position_ids', 'image_grid_thw'):
            value = kwargs.get(key)
            if value is not None:
                self.current[key] = value.detach().clone()

    def before_attention(self, module, args, kwargs):
        h = kwargs.get('hidden_states', args[0] if args else None)
        controller = self.branch._controller
        self.current.update(hidden=h[0, -1].detach().clone(),
            raw_memory=tuple(x.detach() for x in controller.image_memory),
            query_positions=controller.positions.detach().clone(),
            language_mask=controller.language_mask.detach().clone(),
            image_ends=controller.image_ends.detach().clone())
        embeddings = kwargs.get('position_embeddings')
        if embeddings is not None:
            self.current['rotary_cos_sin_last'] = tuple(x[..., -1, :].detach().clone() for x in embeddings)

    def native_output(self, module, args, output):
        self.current['native_attention_output'] = output[0, -1].detach().clone()

    def patched_output(self, module, args, output):
        self.current['patched_attention_output'] = output[0][0, -1].detach().clone()

    def query(self, module, args, output):
        self.current['query'] = output[-1].detach().clone()

    def flush_read(self):
        if self.last_read is not None:
            self.current['reads'].append(self.last_read)
            self.last_read = None

    def memory(self, module, args, output):
        self.flush_read()

    def read(self, module, args, output):
        self.last_read = args[0][-1].detach().clone()

    def branch_output(self, module, args, output):
        self.flush_read()
        self.current['delta'] = output[0, -1].detach().clone()
        self.current['messages'] = module.export_last_query_diagnostics(cpu=False)

    def last(self):
        ensure(self.traces, 'No captured native forward')
        result = self.traces[-1]
        self.traces = []
        self.current = None
        return result

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.traces = []
        self.current = None


def compare_traces(torch, cached, full):
    ensure(int(cached['query_positions'][-1]) == int(full['query_positions'][-1]), 'Final absolute positions differ')
    ensure(bool(cached['language_mask'][-1] and full['language_mask'][-1]), 'Compared query is not language')
    ensure(torch.equal(cached['image_ends'], full['image_ends']), 'Image-end metadata mismatch')
    ensure(cached['input_ids'].shape[1] == 1 and int(cached['input_ids'][0, -1]) == int(full['input_ids'][0, -1]), 'Compared cached token differs')
    result = dict(final_position=int(cached['query_positions'][-1]), image_ends_exact=True,
                  complete_visible_images=int((cached['query_positions'][-1] > cached['image_ends']).sum()))
    for key in ('hidden', 'native_attention_output', 'patched_attention_output', 'query', 'delta'):
        if key in cached:
            result[key] = tensor_difference(torch, cached[key], full[key])
    ensure(len(cached['raw_memory']) == len(full['raw_memory']), 'Memory image counts differ')
    result['memory_per_image'] = [tensor_difference(torch, x, y) for x, y in zip(cached['raw_memory'], full['raw_memory'])]
    result['all_raw_memory_exact'] = all(x['exact'] for x in result['memory_per_image'])
    if 'query' in cached:
        ensure(len(cached['reads']) == len(full['reads']) == len(cached['raw_memory']), 'Final per-image reads missing')
        result['reads'] = tensor_difference(torch, torch.stack(cached['reads']), torch.stack(full['reads']))
        result['frame_messages'] = tensor_difference(torch, cached['messages']['frame_messages'], full['messages']['frame_messages'])
    if 'rotary_cos_sin_last' in cached:
        result['rotary_cos_sin_last'] = [tensor_difference(torch, x, y) for x, y in zip(cached['rotary_cos_sin_last'], full['rotary_cos_sin_last'])]
    return result


def execute(torch, plan, plan_path):
    from transformers import LogitsProcessor, LogitsProcessorList, __version__ as transformers_version
    from gnnformer.runtime import load_runtime, move_to_device, get_layers
    from gnnformer.data import load_mmred_sample, build_prompt_inputs, build_count_prompt
    from gnnformer.carriers import attach_lora
    from gnnformer.independent_vision_aggregation import attach_independent_vision_aggregation
    ensure(torch.cuda.is_available(), 'CUDA required')
    started = time.monotonic()
    job = os.environ['SLURM_JOB_ID']
    output = OUT / f'run_{job}'
    tensors_root = DATA / 'v5_cache_diagnostics' / f'run_{job}'
    output.mkdir(parents=True, exist_ok=False)
    tensors_root.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_bytes(plan_path.read_bytes())
    runtime = load_runtime('Qwen/Qwen2.5-VL-7B-Instruct', use_4bit=True,
                           attn_implementation='sdpa', device_map='cuda')
    model, processor = runtime.model, runtime.processor
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    layers = get_layers(model)
    if runtime.tokenizer.pad_token_id is None:
        runtime.tokenizer.pad_token_id = runtime.tokenizer.eos_token_id
    image_settings = {k: getattr(processor.image_processor, k, None) for k in
        ('size', 'min_pixels', 'max_pixels', 'patch_size', 'temporal_patch_size', 'merge_size')}
    observations = []

    class ForceFirst(LogitsProcessor):
        def __init__(self, length, token):
            self.length, self.token = length, token
        def __call__(self, ids, scores):
            if ids.shape[1] == self.length:
                forced = torch.full_like(scores, -torch.inf)
                forced[:, self.token] = 0
                return forced
            return scores

    def generate(inputs, processor_list=None):
        return model.generate(**inputs, do_sample=False, use_cache=True,
            min_new_tokens=2, max_new_tokens=2, repetition_penalty=1.,
            logits_processor=processor_list, return_dict_in_generate=True, output_logits=True)

    with torch.no_grad():
        for arm, source in plan['runs'].items():
            config = source['config']
            ensure(str(torch.__version__) == config['torch_version'] and str(transformers_version) == config['transformers_version'], 'Runtime versions differ')
            ensure(image_settings == config['image_processor_settings'], 'Processor settings differ')
            branch = attach_independent_vision_aggregation(model, layer_index=14, rank=96, merge=arm)
            branch.capture_last_query_messages = True
            lora = attach_lora(layers, len(layers) - 4, rank=8, alpha=16., device=runtime.device)
            saved = torch.load(source['checkpoint'], map_location='cpu', weights_only=True)
            ensure(saved['architecture'] == config['architecture'] and saved['epoch'] == source['selected_epoch'], 'Checkpoint metadata changed')
            branch.load_state_dict(saved['branch'], strict=True)
            ensure(set(saved['lora']) == {f'{i}.{n}' for i, n in lora.params}, 'LoRA keys differ')
            for (index, name), (a, b) in lora.params.items():
                a.copy_(saved['lora'][f'{index}.{name}'][0].to(a))
                b.copy_(saved['lora'][f'{index}.{name}'][1].to(b))
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()
            del saved
            try:
                for record in plan['records']:
                    sid, frames, question, _states, answer = load_mmred_sample(Path(record['path']))
                    ensure(sid == record['sid'] and question == record['question'] and int(answer) == record['gold'], 'Sample changed')
                    resized = []
                    try:
                        resized = [frame.resize((392, 392)) for frame in frames]
                        inputs = move_to_device(build_prompt_inputs(processor, resized, build_count_prompt(question, len(resized))), runtime.device)
                    finally:
                        for frame in frames + resized:
                            frame.close()
                    branch.mode = 'all'
                    trace = Trace(torch, model, branch, layers[14].self_attn)
                    try:
                        generated = generate(inputs)
                        ensure(len(generated.logits) == 2, 'Expected two native outputs')
                        ids = generated.sequences[0, inputs['input_ids'].shape[1]:]
                        ensure(len(ids) == 2, 'Expected two generated tokens')
                        cached_logits = generated.logits[1][0].detach().clone()
                        on_cached = trace.last()
                        full = dict(inputs)
                        full['input_ids'] = torch.cat((inputs['input_ids'], ids[:1][None]), dim=1)
                        for key, value in (('attention_mask', 1), ('token_type_ids', 0)):
                            if key in full:
                                full[key] = torch.cat((full[key], torch.full_like(ids[:1][None], value)), dim=1)
                        full_logits = model(**full, use_cache=False, logits_to_keep=1).logits[0, -1].detach().float().clone()
                        on_full = trace.last()
                        branch.mode = 'off'
                        native = generate(inputs, LogitsProcessorList([ForceFirst(inputs['input_ids'].shape[1], int(ids[0]))]))
                        ensure(len(native.logits) == 2 and int(native.sequences[0, inputs['input_ids'].shape[1]]) == int(ids[0]), 'OFF prefix differs')
                        off_cached_logits = native.logits[1][0].detach().clone()
                        off_cached = trace.last()
                        off_full_logits = model(**full, use_cache=False, logits_to_keep=1).logits[0, -1].detach().float().clone()
                        off_full = trace.last()
                    finally:
                        trace.remove()
                    branch.mode = 'all'
                    replays = {}
                    for name, point in (('cached', on_cached), ('full', on_full)):
                        delta = branch(point['hidden'][None, None], point['raw_memory'], point['image_ends'],
                                       point['query_positions'][-1:], point['language_mask'][-1:])[0, -1].detach().clone()
                        replays[name] = dict(delta=delta, messages=branch.export_last_query_diagnostics(cpu=False))
                    # Same cached hidden, swapped full memory: isolates any visual-memory change.
                    hybrid = branch(on_cached['hidden'][None, None], on_full['raw_memory'], on_full['image_ends'],
                                    on_cached['query_positions'][-1:], on_cached['language_mask'][-1:])[0, -1].detach().clone()
                    result = dict(arm=arm, sid=sid, n_frames=record['n_frames'], selected_epoch=source['selected_epoch'],
                        first_generated_token=int(ids[0]), generated_ids=ids.cpu().tolist(),
                        on_logits=logit_comparison(torch, cached_logits, full_logits),
                        off_logits=logit_comparison(torch, off_cached_logits, off_full_logits),
                        on_trace=compare_traces(torch, on_cached, on_full),
                        off_trace=compare_traces(torch, off_cached, off_full),
                        branch_replay_cached=tensor_difference(torch, on_cached['delta'], replays['cached']['delta']),
                        branch_replay_full=tensor_difference(torch, on_full['delta'], replays['full']['delta']),
                        fixed_cached_query_memory_swap=tensor_difference(torch, replays['cached']['delta'], hybrid))
                    tensors = dict(on_cached=on_cached, on_full=on_full, off_cached=off_cached, off_full=off_full,
                        on_cached_logits=cached_logits, on_full_logits=full_logits,
                        off_cached_logits=off_cached_logits, off_full_logits=off_full_logits,
                        raw_on_logit_difference=cached_logits.float() - full_logits.float(),
                        centered_on_logit_difference=(cached_logits.float() - full_logits.float()) - (cached_logits.float() - full_logits.float()).mean(),
                        branch_replays=replays, hybrid_delta=hybrid)
                    tensor_path = tensors_root / f'{arm}_{sid}.pt'
                    torch.save(cpu_tree(torch, tensors), tensor_path)
                    result.update(tensors_path=str(tensor_path), tensors_sha256=sha(tensor_path))
                    observations.append(result)
                    save_json(output / 'observations.json', observations)
                    print(json.dumps({k: result[k] for k in ('arm', 'sid', 'n_frames', 'on_logits', 'off_logits')}), flush=True)
                    del tensors, replays, hybrid, on_cached, on_full, off_cached, off_full, generated, native, inputs, full
                    branch.reset_memory()
                    gc.collect()
            finally:
                lora.remove()
                branch.remove()
                del branch, lora
                gc.collect()
                torch.cuda.empty_cache()
    torch.cuda.synchronize()
    summary = dict(schema_version=1, plan_sha256=sha(plan_path), source_sha256=sources(),
        slurm_job_id=job, gpu=torch.cuda.get_device_name(0), elapsed_seconds=time.monotonic() - started,
        scope=plan['scope'], caveat=plan['caveat'], observations=observations)
    save_json(output / 'summary.json', summary)
    (output / 'INDEX.md').write_text('# V5 cache discrepancy diagnosis\n\n[Summary](summary.json) · [Exact plan](plan.json).\n\n'
        'Four fixed software cases; no efficacy scores or tolerance decisions. Native-precision tensors are linked by path and SHA in the summary.\n')
    print(json.dumps(dict(output=str(output), elapsed_seconds=summary['elapsed_seconds'])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--plan', type=Path)
    a = p.parse_args()
    ensure(os.environ.get('SLURM_JOB_ID'), 'All diagnostic work requires Slurm')
    partition = os.environ.get('SLURM_JOB_PARTITION', '')
    ensure(partition == ('cpu' if a.check else 'gpu'), 'Wrong Slurm partition')
    if a.check:
        ensure(not os.environ.get('SLURM_JOB_GPUS', ''), 'CPU check must not allocate GPUs')
    else:
        ensure(os.environ.get('SLURM_JOB_GPUS', ''), 'GPU allocation required')
    import torch
    torch.set_num_threads(max(1, min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))))
    if a.check:
        cpu_self_checks(torch)
        plan = make_plan(torch)
        destination = OUT / f'check_{os.environ["SLURM_JOB_ID"]}'
        destination.mkdir(parents=True, exist_ok=False)
        for name in OWN:
            target = destination / 'code' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / name).read_bytes())
        save_json(destination / 'plan.json', plan)
        (destination / 'plan.sha256').write_text(sha(destination / 'plan.json') + '\n')
        print(json.dumps(dict(passed=True, plan=str(destination / 'plan.json'), plan_sha256=sha(destination / 'plan.json'))), flush=True)
    else:
        ensure(sha(a.plan) == a.plan.with_suffix('.sha256').read_text().strip(), 'Plan sidecar differs')
        plan = json.loads(a.plan.read_text())
        ensure(plan['source_sha256'] == sources(), 'Diagnostic sources changed')
        for name, digest in plan['frozen_profile_source_sha256'].items():
            ensure(sha(REPO / name) == digest, 'Frozen profile source changed')
        ensure(sha(plan['manifest']) == plan['manifest_sha256'], 'Manifest changed')
        for arm, run in plan['runs'].items():
            ensure(sha(run['checkpoint']) == run['checkpoint_sha256'], 'Checkpoint changed')
            ensure(sha(Path(run['directory']) / 'config.json') == run['config_sha256'], 'Config changed')
        for row in plan['records']:
            ensure(sha(Path(row['path']) / 'qa.txt') == row['qa_sha256'], 'QA changed')
            for image in row['image_files']:
                ensure(sha(image['path']) == image['sha256'], 'Image changed')
        execute(torch, plan, a.plan)


if __name__ == '__main__':
    main()

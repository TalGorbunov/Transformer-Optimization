"""Timing-only forced-cap calibration; never an efficacy evaluation.

Uses the unchanged frozen baseline plan and its two fixed software scenes:
K3/N64 and K6/N16. min_new_tokens=max_new_tokens suppresses native EOS for
32 direct or512 reasoning tokens, solely to measure sustained cached decoding.
No output is parsed or scored. The original conservative gate stays failed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.probe_native_vision_reasoning import (
    CAPS, MODEL_PATHS, OUTPUT_ROOT, SEED, fingerprint, prepare, require,
    save_json, sha_file, verify_images, verify_plan,
)

BASELINE_PLAN = OUTPUT_ROOT / 'check_441210/plan.json'
BASELINE_SHA256 = '6de7e0c1610c4ab928fec6b513fc5ef70afb2d90477c7a1b0a88c37812c3373f'
ROOT = OUTPUT_ROOT / 'throughput'
SOURCES = (
    'scripts/profile_native_vision_reasoning_throughput.py',
    'slurm/native_vision_reasoning_throughput_check.sbatch',
    'slurm/native_vision_reasoning_throughput.sbatch',
)
LIMITATIONS = [
    'Software timing only: EOS suppression forces full caps. These outputs are not natural reasoning traces and have no accuracy interpretation.',
    'The original software-profile cost gates remain failed and are not replaced or silently reclassified.',
    'Prompts, data, models, main-policy budgets and parsing remain frozen. Only this separate timing procedure suppresses EOS.',
    'Two fixed software scenes measure sustained native cache cost; the projected bound is conservative but not a guarantee.',
    'The512-token main policy remains far below the official Cosmos recommendation; the timing calibration does not expand its budget.',
]


def sources():
    return {name: sha_file(REPO / name) for name in SOURCES}


def snapshot(output):
    for name in SOURCES:
        target = output / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())


def baseline(path):
    path = Path(path)
    digest = sha_file(path)
    require(digest == BASELINE_SHA256 == path.with_suffix('.sha256').read_text().strip(),
            'Expected exact unchanged check441210 baseline plan')
    plan = json.loads(path.read_text())
    verify_plan(plan)
    return plan


def select_records(plan):
    records = []
    for gold, n in ((3, 64), (6, 16)):
        matches = [r for r in plan['profile_records'] if r['gold'] == gold and r['n_frames'] == n]
        require(len(matches) == 1, 'Fixed software cell does not contain exactly one scene')
        records.append(matches[0])
    require(len({r['sid'] for r in records}) == 2, 'Selected software scenes collide')
    return records


def projection(rows, model_load_seconds):
    require(len(rows) == 4 and {r['condition'] for r in rows} == set(CAPS), 'Expected four timing trajectories')
    for condition, cap in CAPS.items():
        group = [r for r in rows if r['condition'] == condition]
        require(len(group) == 2 and all(r['generated_tokens'] == cap for r in group), 'Full policy cap was not measured')
    prefill = max(r['prefill_gpu_seconds'] for r in rows)
    preparation = max(r['preparation_seconds_shared_between_conditions'] for r in rows)
    decode = {condition: max(r['decode_wall_seconds_per_token'] for r in rows if r['condition'] == condition)
              for condition in CAPS}
    estimated = model_load_seconds + 1.25 * 54 * (preparation + 2 * prefill +
                                                31 * decode['direct'] + 511 * decode['reason']) + 30
    return dict(estimated_main_worst_caps_seconds=estimated, within_26_minute_cap=estimated <= 1560,
                maximum_preparation_seconds=preparation, maximum_prefill_gpu_seconds=prefill,
                maximum_decode_wall_seconds_per_token=decode, scenes_per_policy=54,
                direct_decode_steps_per_scene=31, reason_decode_steps_per_scene=511,
                safety_multiplier=1.25, reserve_seconds=30,
                formula='load +1.25*54*(maxprep +2*maxprefill +31*max_direct_decode +511*max_reason_decode)+30')


def self_test():
    rows = [dict(condition=c, generated_tokens=CAPS[c], prefill_gpu_seconds=0.2,
                 preparation_seconds_shared_between_conditions=0.1,
                 decode_wall_seconds_per_token=0.01 if c == 'direct' else 0.02)
            for _ in range(2) for c in CAPS]
    result = projection(rows, 10)
    require(abs(result['estimated_main_worst_caps_seconds'] - 784.525) < 1e-9,
            'Policy-specific cap arithmetic differs')
    rows[0]['generated_tokens'] = 2
    try:
        projection(rows, 10)
    except ValueError:
        pass
    else:
        raise ValueError('Short direct traces incorrectly accepted as full-cap calibration')
    return dict(passed=True, tests=2)


def verify_timing_plan(path):
    path = Path(path)
    require(sha_file(path) == path.with_suffix('.sha256').read_text().strip(), 'Timing plan sidecar mismatch')
    plan = json.loads(path.read_text())
    require(plan['schema_version'] == 1 and plan['protocol'] == 'forced_cap_timing_only', 'Unexpected timing protocol')
    require(plan['source_sha256'] == sources() and plan['baseline_plan_sha256'] == BASELINE_SHA256,
            'Frozen timing source/baseline identity changed')
    require(plan['caps'] == CAPS and plan['decoding_override'] == 'min_new_tokens=max_new_tokens; native EOS suppression solely for timing',
            'Frozen timing policy changed')
    base = baseline(plan['baseline_plan'])
    require(plan['records'] == select_records(base), 'Fixed timing scenes changed')
    return plan, base


def check(args):
    base = baseline(args.baseline_plan)
    records = select_records(base)
    verify_images(records)
    output = Path(args.output or ROOT / f'check_{os.environ["SLURM_JOB_ID"]}')
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(schema_version=1, protocol='forced_cap_timing_only',
                baseline_plan=str(Path(args.baseline_plan).resolve()), baseline_plan_sha256=BASELINE_SHA256,
                records=records, caps=CAPS, source_sha256=sources(), seed=SEED,
                decoding_override='min_new_tokens=max_new_tokens; native EOS suppression solely for timing',
                tests=self_test(), limitations=LIMITATIONS, slurm_job_id=os.environ['SLURM_JOB_ID'])
    save_json(output / 'plan.json', plan)
    (output / 'plan.sha256').write_text(sha_file(output / 'plan.json') + '\n')
    verify_timing_plan(output / 'plan.json')
    snapshot(output)
    (output / 'REPORT.md').write_text('Timing CPU plan passed. Exactly K3/N64 and K6/N16 software scenes, unchanged baseline plan, forced32/512 caps, no efficacy scoring.\n')
    print(json.dumps(dict(plan=str(output / 'plan.json'), sha256=sha_file(output / 'plan.json'))), flush=True)


def run(args):
    require(args.model_key in MODEL_PATHS and args.timing_plan, 'Timing run requires model key and CPU plan')
    plan, base = verify_timing_plan(args.timing_plan)
    record_model = base['models'][args.model_key]
    records = plan['records']
    verify_images(records)
    output = Path(args.output or ROOT / f'{args.model_key}_{os.environ["SLURM_JOB_ID"]}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_bytes(Path(args.timing_plan).read_bytes())
    (output / 'plan.sha256').write_text(sha_file(output / 'plan.json') + '\n')
    snapshot(output)
    save_json(output / 'config.json', dict(protocol=plan['protocol'], model_key=args.model_key,
              timing_plan_sha256=sha_file(args.timing_plan), baseline_plan_sha256=BASELINE_SHA256,
              source_sha256=plan['source_sha256'], selected_model=record_model,
              slurm_job_id=os.environ['SLURM_JOB_ID'], limitations=LIMITATIONS))
    import torch
    import transformers
    from gnnformer.runtime import load_runtime, move_to_device
    require(torch.cuda.is_available(), 'Timing mode requires a GPU')
    require(torch.__version__ == base['runtime']['torch_version'] and
            transformers.__version__ == base['runtime']['transformers_version'], 'Frozen runtime version changed')
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = time.perf_counter()
    rt = load_runtime(record_model['path'], use_4bit=True, attn_implementation='sdpa', device_map='cuda')
    rt.model.requires_grad_(False)
    rt.model.eval()
    require(fingerprint(rt.processor, transformers.__version__) == record_model['processor'], 'Processor differs from baseline')
    require(rt.model.generation_config.eos_token_id == record_model['eos_token_ids'], 'Native EOS config differs')
    require(not any(p.requires_grad for p in rt.model.parameters()), 'Model is not frozen')
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    rows = []
    with (output / 'trajectories.jsonl').open('x') as stream:
        for record in records:
            prep_start = time.perf_counter()
            inputs, layouts = prepare(rt.processor, record)
            require(layouts == record_model['layouts'][record['sid']], 'Actual prompts differ from frozen baseline')
            preparation_seconds = time.perf_counter() - prep_start
            for condition, cap in CAPS.items():
                item = move_to_device(inputs[condition], rt.device)
                events = []
                def before_forward(module, positional):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    events.append((begin, end))
                def after_forward(module, positional, result):
                    events[-1][1].record()
                hooks = [rt.model.register_forward_pre_hook(before_forward),
                         rt.model.register_forward_hook(after_forward)]
                torch.cuda.synchronize()
                generation_start = time.perf_counter()
                try:
                    with torch.inference_mode():
                        generated = rt.model.generate(
                            **item, do_sample=False, use_cache=True, min_new_tokens=cap, max_new_tokens=cap,
                            repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                            eos_token_id=record_model['eos_token_ids'], pad_token_id=record_model['pad_token_id'],
                            return_dict_in_generate=True, output_scores=False,
                        )
                    torch.cuda.synchronize()
                finally:
                    for hook in hooks:
                        hook.remove()
                elapsed = time.perf_counter() - generation_start
                prompt_length = item['input_ids'].shape[1]
                require(generated.sequences.shape[0] == 1 and
                        torch.equal(generated.sequences[0, :prompt_length], item['input_ids'][0]), 'Generation prefix differs')
                ids = generated.sequences[0, prompt_length:].tolist()
                require(len(ids) == cap and not any(token in record_model['eos_token_ids'] for token in ids),
                        'Timing generation did not suppress EOS and reach exact cap')
                forward_seconds = [begin.elapsed_time(end) / 1000 for begin, end in events]
                require(len(forward_seconds) == cap, 'Forward/token count differs under forced cap')
                row = dict(model_key=args.model_key, sid=record['sid'], pair_id=record['pair_id'],
                           n_frames=record['n_frames'], condition=condition, timing_only=True,
                           min_new_tokens=cap, max_new_tokens=cap, generated_tokens=len(ids), generated_ids=ids,
                           raw_text=rt.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                           prompt_tokens=prompt_length, input_ids_sha256=layouts[condition]['input_ids_sha256'],
                           qa_sha256=record['qa_sha256'], native_eos_ids=record_model['eos_token_ids'],
                           native_eos_suppressed=True, finish_reason='forced_timing_cap',
                           preparation_seconds_shared_between_conditions=preparation_seconds,
                           generation_seconds=elapsed, prefill_gpu_seconds=forward_seconds[0],
                           decode_gpu_seconds=sum(forward_seconds[1:]), forward_gpu_seconds=forward_seconds,
                           decode_wall_seconds_per_token=(elapsed - forward_seconds[0]) / (cap - 1),
                           peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                print(json.dumps({k: row[k] for k in ('model_key', 'sid', 'condition', 'generated_tokens',
                                                       'generation_seconds', 'decode_wall_seconds_per_token')}), flush=True)
                del generated, item, events
            del inputs
    verify_timing_plan(args.timing_plan)
    result = dict(schema_version=1, protocol=plan['protocol'], model_key=args.model_key, completed=True,
                  timing_plan_sha256=sha_file(args.timing_plan), baseline_plan_sha256=BASELINE_SHA256,
                  source_sha256=plan['source_sha256'], scene_count=2, generation_count=4,
                  total_generated_tokens=sum(r['generated_tokens'] for r in rows),
                  total_generation_seconds=sum(r['generation_seconds'] for r in rows),
                  model_load_seconds=model_load_seconds, total_seconds=time.perf_counter() - started,
                  peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
                  trajectories_sha256=sha_file(output / 'trajectories.jsonl'),
                  cost_gate=projection(rows, model_load_seconds), limitations=LIMITATIONS)
    save_json(output / 'summary.json', result)
    gate = result['cost_gate']
    (output / 'REPORT.md').write_text(
        '# Forced-cap timing calibration\n\n'
        f'Model {args.model_key};4 timing-only trajectories,1088generated tokens. No output is scored.\n\n'
        f'Projected main worst-cap time: {gate["estimated_main_worst_caps_seconds"]:.2f}s; '
        f'within26minutes: {gate["within_26_minute_cap"]}.\n\n'
        f'Maximum wall decode seconds/token: {gate["maximum_decode_wall_seconds_per_token"]}.\n\n'
        + '\n'.join('- ' + item for item in LIMITATIONS) + '\n')
    print(json.dumps(dict(output=str(output), cost_gate=gate)), flush=True)


def main():
    require(bool(os.environ.get('SLURM_JOB_ID')), 'This profiler requires Slurm for CPU and GPU work')
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--baseline-plan', default=str(BASELINE_PLAN))
    parser.add_argument('--timing-plan')
    parser.add_argument('--model-key', choices=tuple(MODEL_PATHS))
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.check_plan:
        check(args)
    else:
        run(args)


if __name__ == '__main__':
    main()

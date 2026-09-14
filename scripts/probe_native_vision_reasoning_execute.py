"""Execution and independently rescored report for the frozen reasoning assay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
from statistics import mean
import time

from scripts.probe_native_vision_reasoning import (
    CAPS, DATA_ROOT, LIMITS, MODEL_PATHS, OUTPUT_ROOT, SEED, check_plan, fingerprint,
    messages, metrics, prepare, require, save_json, score_tokens, self_test,
    sha_file, snapshot_sources, source_hashes, user_text, verify_images, verify_plan,
)


def run(args):
    require(args.model_key in MODEL_PATHS and args.plan is not None, 'Run requires model key and frozen plan')
    plan_path = Path(args.plan)
    require(sha_file(plan_path) == plan_path.with_suffix('.sha256').read_text().strip(), 'CPU plan sidecar hash differs')
    plan = json.loads(plan_path.read_text())
    verify_plan(plan)
    records = plan['profile_records'] if args.profile else plan['main_records']
    if args.profile:
        records = [r for r in records if r['n_frames'] in (16, 64)]
    require(len(records) == (4 if args.profile else 54), 'Unexpected run size')
    verify_images(records)
    mode = 'profile' if args.profile else 'main'
    output = Path(args.output or OUTPUT_ROOT / mode / f'{args.model_key}_{os.environ["SLURM_JOB_ID"]}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_bytes(plan_path.read_bytes())
    (output / 'plan.sha256').write_text(sha_file(plan_path) + '\n')
    snapshot_sources(output)
    manifest_copies = output / 'manifests'
    manifest_copies.mkdir()
    for key, item in plan['manifests'].items():
        (manifest_copies / f'{key}.json').write_bytes(Path(item['path']).read_bytes())
    config = dict(protocol=plan['protocol'], model_key=args.model_key, profile=args.profile,
                  plan_path=str(plan_path.resolve()), plan_sha256=sha_file(plan_path),
                  sources=plan['sources'], model=plan['models'][args.model_key],
                  slurm_job_id=os.environ['SLURM_JOB_ID'], scene_count=len(records),
                  generation_count=2 * len(records), conditions_order=['direct', 'reason'])
    save_json(output / 'config.json', config)
    import torch
    import transformers
    from gnnformer.runtime import load_runtime, move_to_device
    require(torch.cuda.is_available(), 'GPU mode requires CUDA')
    require(torch.__version__ == plan['runtime']['torch_version'] and
            transformers.__version__ == plan['runtime']['transformers_version'], 'CPU/GPU package versions differ')
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    start = time.perf_counter()
    rt = load_runtime(config['model']['path'], use_4bit=True, attn_implementation='sdpa', device_map='cuda')
    rt.model.requires_grad_(False)
    rt.model.eval()
    require(fingerprint(rt.processor, transformers.__version__) == config['model']['processor'], 'Processor changed')
    require(rt.model.generation_config.eos_token_id == config['model']['eos_token_ids'], 'Loaded native EOS changed')
    require(not any(p.requires_grad for p in rt.model.parameters()), 'Model is not frozen')
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    rows = []
    with (output / 'predictions.jsonl').open('x') as stream:
        for record in records:
            preparation_start = time.perf_counter()
            inputs, layouts = prepare(rt.processor, record)
            require(layouts == config['model']['layouts'][record['sid']], 'Actual processor layout differs from CPU plan')
            prep_seconds = time.perf_counter() - preparation_start
            for condition in CAPS:
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
                            **item, do_sample=False, use_cache=True, max_new_tokens=CAPS[condition],
                            repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                            eos_token_id=config['model']['eos_token_ids'],
                            pad_token_id=config['model']['pad_token_id'],
                            return_dict_in_generate=True, output_scores=False,
                        )
                    torch.cuda.synchronize()
                finally:
                    for hook in hooks:
                        hook.remove()
                generation_seconds = time.perf_counter() - generation_start
                prompt_tokens = item['input_ids'].shape[1]
                require(generated.sequences.shape[0] == 1, 'Expected one unrestricted greedy trajectory')
                require(torch.equal(generated.sequences[0, :prompt_tokens], item['input_ids'][0]), 'Prompt prefix changed')
                ids = generated.sequences[0, prompt_tokens:].tolist()
                require(0 < len(ids) <= CAPS[condition], 'Generated length outside cap')
                eos_positions = [i for i, token in enumerate(ids) if token in config['model']['eos_token_ids']]
                require(not eos_positions or eos_positions == [len(ids) - 1], 'EOS is not the unique final generated token')
                require(bool(eos_positions) or len(ids) == CAPS[condition], 'Generation stopped early without native EOS')
                endpoints = {str(CAPS[condition]): score_tokens(
                    ids, rt.tokenizer, config['model']['eos_token_ids'], CAPS[condition], condition, record['gold'])}
                if condition == 'reason':
                    endpoints['128'] = score_tokens(ids, rt.tokenizer, config['model']['eos_token_ids'],
                                                     128, condition, record['gold'])
                forward_seconds = [begin.elapsed_time(end) / 1000 for begin, end in events]
                require(len(forward_seconds) == len(ids), 'Native cached forward/token count differs')
                row = dict(model_key=args.model_key, condition=condition, sid=record['sid'],
                           pair_id=record['pair_id'], n_frames=record['n_frames'], gold=record['gold'],
                           qa_sha256=record['qa_sha256'], input_ids_sha256=layouts[condition]['input_ids_sha256'],
                           user_text=user_text(record), prompt_tokens=prompt_tokens,
                           generation_budget=CAPS[condition], generated_tokens=len(ids),
                           generated_ids=ids, raw_text=rt.tokenizer.decode(ids, skip_special_tokens=False,
                                                                         clean_up_tokenization_spaces=False),
                           finish_reason='native_eos' if eos_positions else 'max_new_tokens',
                           native_eos_ids=config['model']['eos_token_ids'], endpoints=endpoints,
                           preparation_seconds_shared_between_conditions=prep_seconds,
                           generation_seconds=generation_seconds, forward_gpu_seconds=forward_seconds,
                           prefill_gpu_seconds=forward_seconds[0],
                           decode_gpu_seconds=sum(forward_seconds[1:]),
                           decode_wall_seconds_per_token=(generation_seconds - forward_seconds[0]) /
                                                        max(1, len(ids) - 1),
                           peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                print(json.dumps({key: row[key] for key in ('model_key', 'condition', 'sid', 'n_frames',
                                                           'generated_tokens', 'finish_reason', 'generation_seconds')}), flush=True)
                del generated, item, events
            del inputs
    verify_plan(plan)
    elapsed = time.perf_counter() - start
    summary = summarize(rows, profile=args.profile)
    summary.update(model_key=args.model_key, profile=args.profile, model_load_seconds=load_seconds,
                   total_seconds=elapsed, total_generated_tokens=sum(r['generated_tokens'] for r in rows),
                   total_generation_seconds=sum(r['generation_seconds'] for r in rows),
                   peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
                   plan_sha256=config['plan_sha256'], predictions_sha256=sha_file(output / 'predictions.jsonl'),
                   source_sha256=config['sources'], completed=True, limitations=LIMITS)
    summary['cost_by_policy'] = policy_costs(rows)
    if args.profile:
        # No correctness enters this cost gate. Project all54 scenes at BOTH caps.
        prefill = max(r['prefill_gpu_seconds'] for r in rows)
        decode = max(r['decode_wall_seconds_per_token'] for r in rows)
        prep = max(r['preparation_seconds_shared_between_conditions'] for r in rows)
        projected = load_seconds + 1.25 * 54 * (prep + 2 * prefill + (31 + 511) * decode) + 30
        summary['cost_gate'] = dict(estimated_main_worst_caps_seconds=projected,
                                   within_26_minute_cap=projected <= 26 * 60,
                                   maximum_measured_prefill_gpu_seconds=prefill,
                                   maximum_measured_decode_wall_seconds_per_token=decode,
                                   maximum_preparation_seconds=prep, safety_multiplier=1.25, reserve_seconds=30,
                                   caveat='Four software scenes estimate throughput, not a guarantee. Failed/timed-out profiles do not authorize dropping scenes or conditions.')
    save_json(output / 'summary.json', summary)
    (output / 'REPORT.md').write_text(render_report({args.model_key: summary}, profile=args.profile))
    print(json.dumps({'output': str(output), 'completed': True, 'seconds': elapsed}), flush=True)


def policy_costs(rows):
    return {condition: dict(
        generations=len([r for r in rows if r['condition'] == condition]),
        generation_seconds=sum(r['generation_seconds'] for r in rows if r['condition'] == condition),
        generated_tokens=sum(r['generated_tokens'] for r in rows if r['condition'] == condition),
        mean_generated_tokens=mean(r['generated_tokens'] for r in rows if r['condition'] == condition),
        mean_prompt_tokens=mean(r['prompt_tokens'] for r in rows if r['condition'] == condition),
        max_prompt_tokens=max(r['prompt_tokens'] for r in rows if r['condition'] == condition),
    ) for condition in CAPS}


def bootstrap(values, seed=SEED):
    require(len(values) > 1, 'Too few paired anchors')
    rng = random.Random(seed)
    draws = sorted(mean(rng.choice(values) for _ in values) for _ in range(10000))
    return dict(estimate=mean(values), lower=draws[249], upper=draws[9749], replicates=10000,
                seed=seed, clusters=len(values), unit='whole_paired_anchor', quantiles='ordered_draws[249,9749]')


def summarize(rows, profile=False):
    indexed = defaultdict(dict)
    for row in rows:
        key = (row['pair_id'], row['n_frames'])
        require(row['condition'] not in indexed[key], 'Duplicate condition for scene')
        indexed[key][row['condition']] = row
    expected_n = (16, 64) if profile else (16, 32, 64)
    expected_anchors = 2 if profile else 18
    anchors = sorted({pair for pair, _ in indexed})
    require(len(anchors) == expected_anchors and len(indexed) == len(expected_n) * expected_anchors,
            'Missing scenes or anchors')
    require(all(set(indexed[pair, n]) == {'direct', 'reason'} for pair in anchors for n in expected_n),
            'Incomplete policy pairing')
    endpoints = {'direct32': ('direct', '32'), 'reason128': ('reason', '128'), 'reason512': ('reason', '512')}
    cells = {}
    for endpoint, (condition, budget) in endpoints.items():
        cells[endpoint] = {}
        for label, lengths in [('all', expected_n), ('familiar_ood', tuple(n for n in expected_n if n > 16))] + [
                (f'N{n}', (n,)) for n in expected_n]:
            selected = [indexed[pair, n][condition]['endpoints'][budget] for pair in anchors for n in lengths]
            cells[endpoint][label] = metrics(selected)
        cells[endpoint]['per_k'] = {
            f'N{n}_K{k}': metrics([indexed[pair, n][condition]['endpoints'][budget] for pair in anchors
                                  if indexed[pair, n][condition]['gold'] == k])
            for n in expected_n for k in sorted({r['gold'] for r in rows})}
    contrasts = {}
    for budget in ('128', '512'):
        contrasts[budget] = {}
        for label, lengths in [('familiar_ood', tuple(n for n in expected_n if n > 16)), ('N16', (16,))]:
            values = [mean(int(indexed[pair, n]['reason']['endpoints'][budget]['correct']) -
                                int(indexed[pair, n]['direct']['endpoints']['32']['correct']) for n in lengths)
                      for pair in anchors]
            contrasts[budget][label] = bootstrap(values)
    extension = {}
    for endpoint, (condition, budget) in endpoints.items():
        extension[endpoint] = {}
        for n in expected_n:
            if n == 16:
                continue
            pairs = [(indexed[pair, 16][condition]['endpoints'][budget],
                      indexed[pair, n][condition]['endpoints'][budget]) for pair in anchors]
            extension[endpoint][f'N16_N{n}'] = dict(
                pairs=len(pairs), both_correct=sum(a['correct'] and b['correct'] for a, b in pairs),
                short_correct_long_wrong=sum(a['correct'] and not b['correct'] for a, b in pairs),
                both_parsed=sum(a['parsed'] and b['parsed'] for a, b in pairs),
                invariant_among_parsed=sum(a['parsed'] and b['parsed'] and a['prediction'] == b['prediction']
                                          for a, b in pairs))
    return dict(scene_count=len(indexed), generation_count=len(rows), anchors=len(anchors), cells=cells,
                contrasts_reason_minus_direct=contrasts, extension_stability=extension,
                primary_screen_pass=contrasts['512']['familiar_ood']['estimate'] >= 0.05,
                primary_screen='reason512 minus direct32 familiar OOD exact >=5 percentage points per fixed model; descriptive, not a significance criterion')


def render_report(summaries, profile=False):
    title = 'Frozen reasoning baseline software profile' if profile else 'Frozen reasoning baseline on fresh MMReD Vision scenes'
    lines = [f'# {title}', '',
             'The primary contrast is within each fixed model: reasoning512 minus direct32 on paired N32/N64 scenes. All malformed/truncated outputs remain wrong.']
    for model, summary in summaries.items():
        lines += ['', f'## {model}', '',
                  '| Model | Endpoint | N16 exact | N32/N64 exact | OOD parsed | OOD truncated | Parsed OOD MAE |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for endpoint, cells in summary['cells'].items():
            short, long = cells['N16'], cells['familiar_ood']
            mae = '—' if long['mae'] is None else f'{long["mae"]:.3f} ({long["mae_denominator"]})'
            lines.append(f'| {model} | {endpoint} | {short["correct"]}/{short["n"]} | '
                         f'{long["correct"]}/{long["n"]} | {long["parsed"]}/{long["n"]} | '
                         f'{long["truncated"]}/{long["n"]} | {mae} |')
        delta = summary['contrasts_reason_minus_direct']['512']['familiar_ood']
        loss = summary['contrasts_reason_minus_direct']['512']['N16']['estimate']
        lines += ['', f'{model}: primary difference {100 * delta["estimate"]:+.2f} pp '
                  f'(paired-anchor interval [{100 * delta["lower"]:+.2f}, {100 * delta["upper"]:+.2f}]); '
                  f'5 pp descriptive screen: {summary["primary_screen_pass"]}. N16 difference {100 * loss:+.2f} pp.',
                  f'Measured total {summary.get("total_seconds", 0):.1f}s; generation '
                  f'{summary.get("total_generation_seconds", 0):.1f}s; '
                  f'{summary.get("total_generated_tokens", 0)} generated tokens.']
        for policy, cost in summary.get('cost_by_policy', {}).items():
            lines.append(f'{policy}: {cost["generation_seconds"]:.1f}s generation, '
                         f'{cost["generated_tokens"]} tokens ({cost["mean_generated_tokens"]:.1f}/scene), '
                         f'mean prompt {cost["mean_prompt_tokens"]:.1f} tokens.')
        if 'cost_gate' in summary:
            gate = summary['cost_gate']
            lines.append(f'Profile cost projection for all54 main scenes at both caps: '
                         f'{gate["estimated_main_worst_caps_seconds"]:.1f}s; within26minutes: {gate["within_26_minute_cap"]}.')
    lines += ['', '## Interpretation limits', ''] + [f'- {text}' for text in LIMITS]
    return '\n'.join(lines) + '\n'


def report(args):
    require(args.runs and len(args.runs) == 2, 'Report requires the two completed model run directories')
    output = Path(args.output or OUTPUT_ROOT / f'report_{os.environ["SLURM_JOB_ID"]}')
    output.mkdir(parents=True, exist_ok=False)
    summaries, provenance = {}, {}
    plan_hashes = set()
    for name in args.runs:
        directory = Path(name)
        config = json.loads((directory / 'config.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        plan = json.loads((directory / 'plan.json').read_text())
        require(sha_file(directory / 'plan.json') == (directory / 'plan.sha256').read_text().strip(), 'Run plan sidecar differs')
        verify_plan(plan)
        require(not config['profile'] and summary['completed'], 'Expected completed main run')
        key = config['model_key']
        require(key in MODEL_PATHS and key not in summaries, 'Unexpected or duplicate model')
        require(sha_file(directory / 'plan.json') == config['plan_sha256'] == summary['plan_sha256'], 'Plan hash differs')
        require(sha_file(directory / 'predictions.jsonl') == summary['predictions_sha256'], 'Prediction hash differs')
        require(config['sources'] == summary['source_sha256'] == plan['sources'], 'Run source ledger differs')
        require(config['model'] == plan['models'][key], 'Selected frozen model differs from CPU plan')
        require(all(sha_file(directory / 'code' / source) == digest for source, digest in plan['sources'].items()),
                'Run source snapshot differs')
        rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
        require(len(rows) == 108, 'Expected108 trajectories/model')
        records = {r['sid']: r for r in plan['main_records']}
        import transformers
        from transformers import AutoProcessor
        require(transformers.__version__ == plan['runtime']['transformers_version'], 'Report transformers version differs')
        processor = AutoProcessor.from_pretrained(config['model']['path'], trust_remote_code=True,
                                                 use_fast=False, local_files_only=True)
        require(fingerprint(processor, transformers.__version__) == config['model']['processor'], 'Report processor differs')
        for row in rows:
            record = records[row['sid']]
            require(row['pair_id'] == record['pair_id'] and row['gold'] == record['gold'] and
                    row['n_frames'] == record['n_frames'] and row['qa_sha256'] == record['qa_sha256'], 'Prediction identity mismatch')
            condition = row['condition']
            require(row['model_key'] == key and row['generation_budget'] == CAPS[condition], 'Prediction policy differs')
            require(row['input_ids_sha256'] == config['model']['layouts'][row['sid']][condition]['input_ids_sha256'],
                    'Prediction processor layout differs')
            ids = row['generated_ids']
            require(0 < len(ids) == row['generated_tokens'] <= CAPS[condition], 'Raw token count differs')
            eos_ids = config['model']['eos_token_ids']
            eos_positions = [i for i, token in enumerate(ids) if token in eos_ids]
            require(not eos_positions or eos_positions == [len(ids) - 1], 'Report found nonterminal/multiple EOS')
            require(bool(eos_positions) or len(ids) == CAPS[condition], 'Report found short stop without EOS')
            require(row['native_eos_ids'] == eos_ids and row['finish_reason'] ==
                    ('native_eos' if eos_positions else 'max_new_tokens'), 'Stored finish metadata differs')
            require(row['raw_text'] == processor.tokenizer.decode(ids, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False), 'Raw text differs from stored token IDs')
            require(row['prompt_tokens'] == config['model']['layouts'][row['sid']][condition]['prompt_tokens']
                    and row['user_text'] == user_text(record), 'Stored prompt metadata differs')
            require(len(row['forward_gpu_seconds']) == len(ids), 'Forward/token accounting differs')
            expected = {str(CAPS[condition]): score_tokens(ids, processor.tokenizer, config['model']['eos_token_ids'],
                                                         CAPS[condition], condition, record['gold'])}
            if condition == 'reason':
                expected['128'] = score_tokens(ids, processor.tokenizer, config['model']['eos_token_ids'],
                                               128, condition, record['gold'])
            require(expected == row['endpoints'], 'Stored parsing differs from strict raw-token rescoring')
        computed = summarize(rows)
        require(all(summary[field] == value for field, value in computed.items()), 'Summary differs from predictions')
        require(summary['cost_by_policy'] == policy_costs(rows), 'Stored policy costs differ from trajectories')
        require(summary['total_generated_tokens'] == sum(r['generated_tokens'] for r in rows) and
                summary['total_generation_seconds'] == sum(r['generation_seconds'] for r in rows),
                'Stored aggregate token/time totals differ')
        summaries[key] = summary
        plan_hashes.add(config['plan_sha256'])
        provenance[key] = dict(path=str(directory.resolve()), summary_sha256=sha_file(directory / 'summary.json'),
                               predictions_sha256=summary['predictions_sha256'], plan_sha256=config['plan_sha256'])
    require(set(summaries) == set(MODEL_PATHS) and len(plan_hashes) == 1, 'Models do not share one frozen CPU plan')
    analysis = dict(schema_version=1, protocol='frozen_native_vision_reasoning_baseline',
                    results=summaries, provenance=provenance, source_sha256=source_hashes(), limitations=LIMITS)
    save_json(output / 'analysis.json', analysis)
    snapshot_sources(output)
    (output / 'REPORT.md').write_text(render_report(summaries))
    plot_report(summaries, output)
    (output / 'INDEX.md').write_text('# Reasoning baseline report\n\n- [Report](REPORT.md)\n- [Audited analysis](analysis.json)\n- [Figure](comparison.png)\n- [Figure PDF](comparison.pdf)\n')
    print(json.dumps({'output': str(output), 'verified_models': list(summaries)}), flush=True)


def plot_report(summaries, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
    for ax, (model, summary) in zip(axes, summaries.items()):
        for endpoint, marker in [('direct32', 'o'), ('reason128', 's'), ('reason512', '^')]:
            ax.plot([16, 32, 64], [100 * summary['cells'][endpoint][f'N{n}']['exact'] for n in (16, 32, 64)],
                    marker=marker, label=endpoint)
        ax.set(title=model, xlabel='Frames', xticks=[16, 32, 64], ylim=(-2, 102))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel('Strict completed exact (%)')
    fig.savefig(output / 'comparison.png', dpi=180)
    fig.savefig(output / 'comparison.pdf')
    plt.close(fig)


def cli():
    require(bool(os.environ.get('SLURM_JOB_ID')), 'This script must run inside a Slurm allocation')
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--self-test', action='store_true')
    modes.add_argument('--check-plan', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--report', action='store_true')
    parser.add_argument('--manifest', default=str(DATA_ROOT / 'main_manifest.json'))
    parser.add_argument('--profile-manifest', default=str(DATA_ROOT / 'profile_manifest.json'))
    parser.add_argument('--plan')
    parser.add_argument('--model-key', choices=tuple(MODEL_PATHS))
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--runs', nargs='+')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    elif args.check_plan:
        check_plan(args)
    elif args.run:
        run(args)
    else:
        report(args)

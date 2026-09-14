"""Verify and report V3 nonlinear-value placement on CPU Slurm only.

This is an exploratory comparison on the reused V2 test data. Anchor bootstrap
conditions on the two trained seeds; it does not estimate seed-population error.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.report_native_vision_v2 import (CELLS, compare, extension, summarize,
                                            verify_anchors, verify_manifests)

ARMS = ('lift_pre', 'lift_post')
SEEDS = (0, 1)
V3_CHECKPOINT_ROOT = Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v3')
CONTRASTS = {'familiar_ood': ('length_N32', 'length_N64'), 'N16': ('length_N16',),
             'unseen_count': ('unseen_count_N32', 'unseen_count_N64')}
REPORT_SOURCES = ('scripts/report_native_vision_v3.py', 'scripts/report_native_vision_v2.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def compare_pooled(candidate, control, cells, np, rng):
    """Resample anchor families, preserving both lengths AND both fixed seeds."""
    if set(candidate) != set(control) or not candidate:
        raise ValueError('Pooled comparison has mismatched seeds')
    anchor_effects, anchor_keys = defaultdict(list), None
    candidate_only = control_only = total = 0
    for seed in sorted(candidate):
        aa = {(r['cell'], r['sid']): r for r in candidate[seed] if r['cell'] in cells}
        bb = {(r['cell'], r['sid']): r for r in control[seed] if r['cell'] in cells}
        expected_count = sum(r['cell'] in cells for r in candidate[seed])
        if not aa or len(aa) != expected_count or aa.keys() != bb.keys():
            raise ValueError('Duplicate or mismatched pooled examples')
        by_anchor = defaultdict(dict)
        for key, row in aa.items():
            other = bb[key]
            if (row['path'], row['gold'], row['pair_id']) != (other['path'], other['gold'], other['pair_id']):
                raise ValueError('Pooled pair identity mismatch')
            if row['cell'] in by_anchor[row['pair_id']]:
                raise ValueError('Duplicate pooled anchor/cell')
            by_anchor[row['pair_id']][row['cell']] = int(row['exact']) - int(other['exact'])
            candidate_only += int(row['exact'] and not other['exact'])
            control_only += int(other['exact'] and not row['exact'])
        if any(set(group) != set(cells) for group in by_anchor.values()):
            raise ValueError('Incomplete paired lengths in pooled comparison')
        keys = {(row['pair_id'], row['cell'], row['sid'], row['path'], row['gold']) for row in aa.values()}
        if anchor_keys is not None and keys != anchor_keys:
            raise ValueError('Seeds do not share identical anchor observations')
        anchor_keys = keys
        for anchor, group in by_anchor.items():
            anchor_effects[anchor].append(mean(group.values()))
        total += len(aa)
    values = np.asarray([mean(anchor_effects[k]) for k in sorted(anchor_effects)])
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return dict(n=total, unique_examples=total // len(candidate), anchors=len(values),
                seeds=sorted(candidate), exact_gain=float(values.mean()),
                bootstrap95=np.quantile(samples, [.025, .975]).tolist(),
                candidate_only=candidate_only, control_only=control_only,
                resampling='One draw per shared anchor; all lengths and both fixed seeds retained together')


def verify_run(run, arm, seed, expected_hashes, torch, *, historical=False):
    config = json.loads((run / 'config.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    rows = json.loads((run / 'predictions.json').read_text())
    manifest = json.loads((run / 'data_manifest.json').read_text())
    if config['code_sha256'] != expected_hashes:
        raise ValueError('Frozen launch hashes differ from run configuration')
    for name, sha in expected_hashes.items():
        if digest(run / 'code' / name.replace('/', '_')) != sha:
            raise ValueError(f'Training snapshot changed: {name}')
    required = dict(arm=arm, seed=seed, epochs=9, accumulation=4, layer_index=14,
                    lora_layers=4, lora_rank=8, lora_alpha=16.0, resize=392,
                    lr=0.001, lr_lora=0.0001, max_grad_norm=1.0, max_steps=0,
                    train_ns=[8, 16], dev_ns=[8, 16], eval_ns=[16, 32, 64],
                    gold_max=16, eval_modes='all')
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f'Registered {key} differs for {arm}/seed{seed}')
    if config['generation_policy'] != dict(do_sample=False, repetition_penalty=1.0,
                                            max_new_tokens=4, output_logits=True):
        raise ValueError('Generation policy differs')
    architecture = config['architecture']
    if historical:
        if architecture['protocol'] != 'vision_v2_clean' or arm != 'hidden' or seed != 0:
            raise ValueError('Historical reference must be V2 hidden seed0')
        if architecture.get('hidden_rank') != 96:
            raise ValueError('Historical hidden width differs')
        checkpoint_root = Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v2')
        expected_branch, expected_total = 688224, 1409120
    else:
        expected_architecture = dict(protocol='vision_v3_value_lifting', model='Qwen/Qwen2.5-VL-7B-Instruct',
            arm=arm, layer_index=14, rank=64, lift_rank=8,
            placement='before' if arm == 'lift_pre' else 'after', activation='silu',
            lora_layers=4, lora_rank=8, lora_alpha=16.0, resize=392, quantization='nf4')
        if architecture != expected_architecture:
            raise ValueError('V3 architecture differs from registered nonlinear-placement comparison')
        checkpoint_root = V3_CHECKPOINT_ROOT
        expected_branch, expected_total = 696896, 1417792
    for key, expected in [('branch', expected_branch), ('lora', 720896), ('total_trainable', expected_total)]:
        if summary['parameters'][key] != expected or config['parameters'][key] != expected:
            raise ValueError(f'Unexpected active parameter count: {key}')
    if summary['run_id'] != config['run_id'] or summary['arm'] != arm or run.name != config['run_id']:
        raise ValueError('Run identity mismatch')
    verify_manifests(run, config, manifest)
    expected_rows = {(cell, row['sid']): (row['path'], row['gold'], row['pair_id'])
                     for cell in CELLS for row in manifest[cell]['samples']}
    actual_rows = {(row['cell'], row['sid']): (row['path'], row['gold'], row['pair_id']) for row in rows}
    if len(rows) != 452 or len(actual_rows) != 452 or actual_rows != expected_rows:
        raise ValueError('Incomplete or mismatched held-out predictions')
    for row in rows:
        if row['mode'] != 'all' or row['tag'] != 'test' or row['exact'] != (row['prediction'] == row['gold']):
            raise ValueError('Invalid exact-answer bookkeeping')
    verify_anchors(rows, 'length', CELLS[:3], 108)
    verify_anchors(rows, 'unseen_count', CELLS[3:], 64)
    history = summary['training']
    if [entry['epoch'] for entry in history] != list(range(1, 10)):
        raise ValueError('Nine complete training epochs required')
    if any(entry['n'] != 180 or entry['step'] != 45 * entry['epoch']
           or {item['cell']: item['n'] for item in entry['dev']} != {'dev_N8': 36, 'dev_N16': 36}
           or len(entry['dev']) != 2 or not math.isfinite(entry['mean_answer_token_ce']) for entry in history):
        raise ValueError('Registered training/development coverage differs')
    def dev_score(entry):
        return (sum(item['correct'] for item in entry['dev']) / 72,
                -sum(item['gold_first_token_nll'] * item['n'] for item in entry['dev']) / 72)
    best = max(history, key=dev_score)
    checkpoint = Path(summary['selected_checkpoint']).resolve()
    if not checkpoint.is_relative_to(checkpoint_root) or checkpoint.name != 'best.pt' or checkpoint.parent.name != run.name:
        raise ValueError('Selected checkpoint is outside the registered run/root')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (saved['epoch'] != best['epoch'] or saved['step'] != best['step'] or saved['dev'] != best['dev']
            or saved['architecture'] != architecture or saved['config']['run_id'] != config['run_id']
            or saved['config']['code_sha256'] != expected_hashes
            or saved['config']['manifest_sha256'] != config['manifest_sha256']
            or saved['config']['count_manifest_sha256'] != config['count_manifest_sha256']):
        raise ValueError('Checkpoint metadata does not match development selection')
    if sum(value.numel() for value in saved['branch'].values()) != expected_branch:
        raise ValueError('Checkpoint branch tensor count differs')
    if sum(value.numel() for pair in saved['lora'].values() for value in pair) != 720896:
        raise ValueError('Checkpoint LoRA tensor count differs')
    if not all(torch.isfinite(value).all() for value in saved['branch'].values()):
        raise ValueError('Nonfinite saved branch parameters')
    if not all(torch.isfinite(value).all() for pair in saved['lora'].values() for value in pair):
        raise ValueError('Nonfinite saved LoRA parameters')
    checkpoint_hash = digest(checkpoint)
    del saved
    info = describe_rows(rows)
    info.update(parameters=summary['parameters'], selected_epoch=best['epoch'], selected_checkpoint=str(checkpoint),
                selected_checkpoint_sha256=checkpoint_hash, training=history,
                manifest_sha256=config['manifest_sha256'], count_manifest_sha256=config['count_manifest_sha256'],
                memory=dict(training_peak_allocated_bytes=max(entry['peak_cuda_allocated_bytes'] for entry in history),
                            training_peak_reserved_bytes=max(entry['peak_cuda_reserved_bytes'] for entry in history),
                            evaluation_peak_allocated_bytes=max(item['peak_cuda_allocated_bytes'] for item in summary['results'])))
    if len(summary['results']) != len(CELLS) or {item['cell'] for item in summary['results']} != set(CELLS):
        raise ValueError('Published cell coverage differs')
    for item in summary['results']:
        computed = info['cells'][item['cell']]
        if computed['correct'] != item['correct'] or computed['n'] != item['n']:
            raise ValueError('Summary differs from saved predictions')
    return rows, info


def describe_rows(rows):
    return dict(cells={cell: summarize([row for row in rows if row['cell'] == cell]) for cell in CELLS},
                familiar_ood=summarize([row for row in rows if row['cell'] in CONTRASTS['familiar_ood']]),
                unseen_count=summarize([row for row in rows if row['cell'] in CONTRASTS['unseen_count']]),
                per_k={f'{cell}_K{k}': summarize([row for row in rows if row['cell'] == cell and row['gold'] == k])
                       for cell in CELLS for k in sorted({row['gold'] for row in rows if row['cell'] == cell})},
                count_answer_support={name: summarize([row for row in rows if row['cell'] in CONTRASTS['unseen_count'] and predicate(row)])
                    for name, predicate in [('K9', lambda row: row['gold'] == 9), ('K10_16', lambda row: row['gold'] >= 10)]})


def decision_criteria(contrasts, historical):
    per_seed = {seed: bool(contrasts[seed]['familiar_ood']['exact_gain'] >= .05 - 1e-12
                           and contrasts[seed]['N16']['exact_gain'] >= -.05 - 1e-12)
                for seed in SEEDS}
    secondary = bool(historical[0]['familiar_ood']['exact_gain'] >= .05 - 1e-12
                     and historical[0]['N16']['exact_gain'] >= -.05 - 1e-12)
    return dict(V3_1=all(per_seed.values()), V3_2=secondary,
                primary_each_seed=per_seed, primary_both_seeds=all(per_seed.values()),
                secondary_seed0_vs_historical_hidden=secondary,
                secondary_reference='PRE seed0 versus historical V2 hidden seed0; seed1 descriptive only')


def self_test(np):
    candidate, control = {}, {}
    for seed in SEEDS:
        candidate[seed], control[seed] = [], []
        for index in range(4):
            for cell in CONTRASTS['familiar_ood']:
                base = dict(cell=cell, sid=f'{index}_{cell}', path=f'/sample/{index}/{cell}', gold=index, pair_id=str(index))
                candidate[seed].append(dict(base, exact=index % 2 == 0))
                control[seed].append(dict(base, exact=False))
    rng = lambda: np.random.default_rng(20260910)
    one = compare(candidate[0], control[0], CONTRASTS['familiar_ood'], np, rng())
    pooled = compare_pooled(candidate, control, CONTRASTS['familiar_ood'], np, rng())
    if pooled['anchors'] != 4 or pooled['n'] != 16 or pooled['unique_examples'] != 8:
        raise AssertionError('Pooled denominators wrong')
    np.testing.assert_allclose(pooled['bootstrap95'], one['bootstrap95'])
    bad = {seed: [dict(row) for row in rows] for seed, rows in candidate.items()}
    bad[1][0]['path'] = '/wrong'
    try:
        compare_pooled(bad, control, CONTRASTS['familiar_ood'], np, rng())
    except ValueError:
        pass
    else:
        raise AssertionError('Cross-seed identity mismatch accepted')
    good = {seed: {'familiar_ood': {'exact_gain': .06}, 'N16': {'exact_gain': 0.0}} for seed in SEEDS}
    history = {0: {'familiar_ood': {'exact_gain': .06}, 'N16': {'exact_gain': -.06}},
               1: {'familiar_ood': {'exact_gain': .50}, 'N16': {'exact_gain': .50}}}
    if decision_criteria(good, history)['V3_2']:
        raise AssertionError('Secondary criterion ignored seed0 N16 loss')
    history[0]['N16']['exact_gain'] = -.05
    history[1]['familiar_ood']['exact_gain'] = -.50
    if not decision_criteria(good, history)['V3_2']:
        raise AssertionError('Descriptive seed1 altered seed0 secondary criterion')
    good[1]['familiar_ood']['exact_gain'] = 0.0
    if decision_criteria(good, history)['V3_1']:
        raise AssertionError('Primary criterion did not require both seeds')
    print('PASS: pooled anchor dependence, identity checks, both-seed primary and seed0-only historical criterion with N16 guard')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/native_aggregation_vlm/v3'))
    p.add_argument('--historical-root', type=Path, default=Path('outputs/native_aggregation_vlm/v2'))
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run report/checks on CPU Slurm')
    import numpy as np
    if a.self_test:
        self_test(np)
        return
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    report_hashes = json.loads((a.root / 'analysis_source_hashes.json').read_text())
    for name in REPORT_SOURCES:
        if report_hashes.get(name) != digest(REPO / name):
            raise ValueError(f'Analysis source differs from frozen launch version: {name}')
    expected_hashes = json.loads((a.root / 'main/source_hashes.json').read_text())
    results, predictions, runs = {}, {}, {}
    for arm in ARMS:
        results[arm], predictions[arm], runs[arm] = {}, {}, {}
        for seed in SEEDS:
            candidates = list((a.root / 'main' / arm / f'seed{seed}').glob('*/summary.json'))
            if len(candidates) != 1:
                raise ValueError(f'Need exactly one completed {arm}/seed{seed}: {candidates}')
            run = candidates[0].parent
            rows, info = verify_run(run, arm, seed, expected_hashes, torch)
            info['extensions'] = {f'{family}_{lo}_to_{hi}': extension(rows, family, lo, hi)
                for family, lo, hi in [('length', 16, 32), ('length', 32, 64), ('length', 16, 64), ('unseen_count', 32, 64)]}
            predictions[arm][seed], results[arm][seed], runs[arm][seed] = rows, info, str(run)
    data_hashes = {(info['manifest_sha256'], info['count_manifest_sha256']) for seeds in results.values() for info in seeds.values()}
    if len(data_hashes) != 1:
        raise ValueError('V3 runs used different data')
    historical_candidates = list((a.historical_root / 'main/hidden').glob('*/summary.json'))
    if len(historical_candidates) != 1:
        raise ValueError('Need exactly one completed historical V2 hidden run')
    historical_run = historical_candidates[0].parent
    historical_hashes = json.loads((a.historical_root / 'main/source_hashes.json').read_text())
    historical_rows, historical_info = verify_run(historical_run, 'hidden', 0, historical_hashes, torch, historical=True)
    if (historical_info['manifest_sha256'], historical_info['count_manifest_sha256']) not in data_hashes:
        raise ValueError('Historical reference used different manifests')
    rng = lambda: np.random.default_rng(20260910)
    contrasts = {seed: {name: compare(predictions['lift_pre'][seed], predictions['lift_post'][seed], cells, np, rng())
                       for name, cells in CONTRASTS.items()} for seed in SEEDS}
    pooled = {name: compare_pooled(predictions['lift_pre'], predictions['lift_post'], cells, np, rng())
              for name, cells in CONTRASTS.items()}
    historical = {seed: {name: compare(predictions['lift_pre'][seed], historical_rows, cells, np, rng())
                         for name, cells in CONTRASTS.items()} for seed in SEEDS}
    criteria = decision_criteria(contrasts, historical)
    per_seed_primary = criteria['primary_each_seed']
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', local_files_only=True)
    notes = [
        'Exploratory comparison on reused V2 test data; no new sealed test set.',
        'Primary criterion must hold separately in each seed: PRE minus POST familiar N32/N64 >=5pp, N16 loss <=5pp.',
        'Anchor bootstrap conditions on fixed trained models. Both lengths and both seeds stay together in the pooled resample.',
        'Two seeds are inadequate for population inference over training seeds; no seed bootstrap or independent-seed significance claim.',
        'Historical V2 hidden seed0 is a reused reference, not a contemporaneous third arm. V3.2 applies only to PRE seed0: familiar OOD gain >=5pp and N16 loss <=5pp; PRE seed1 versus this reference is descriptive only.',
        'GQA uses 4 KV heads and 28 query heads: PRE applies value SiLU on KV heads, POST applies SiLU on expanded query-head reads. PRE recomputes its lifted value activation over cached V at each decode step; activation work and cache costs are not matched.',
        'Lift and outer maps use fp32 while SDPA inputs use bf16. PRE and POST activation placement also straddles bf16 SDPA rounding; this is not a pure exact-arithmetic ordering contrast.',
        'Both arms initialize outer-up and lift-up to zero. This identically delays initial lift_down learning until those paths have nonzero weights; equal parameters do not imply every parameter updates on the first step.',
        'K9 is a single token; K10..16 are multi-digit. First-token NLL is not whole-answer likelihood.',
        'All selected examples remain in exact-accuracy denominators. MAE/bias are conditional on parsing and shown with parse rate.',
        'Timing and memory include concurrent cluster conditions; measured implementation cost is not an optimized algorithmic lower bound.',
        'Ordinary short native answers are evaluated; neither successful placement nor two-seed replication establishes reasoning composition or a general counting algorithm.',
    ]
    analysis = dict(criteria=criteria, arms=results, runs=runs, pre_minus_post=contrasts,
                    pooled_pre_minus_post=pooled,
                    pooled_arms={arm: describe_rows([row for seed in SEEDS for row in predictions[arm][seed]]) for arm in ARMS},
                    pre_minus_historical_hidden=historical,
                    historical_hidden=dict(run=str(historical_run), results=historical_info),
                    answer_tokenization={str(k): tokenizer(str(k), add_special_tokens=False).input_ids for k in range(17)},
                    training_source_sha256=expected_hashes, analysis_source_sha256={name: digest(REPO / name) for name in REPORT_SOURCES},
                    analysis_job=os.environ['SLURM_JOB_ID'], notes=notes)
    write_json(a.root / 'analysis.json', analysis)
    code = a.root / 'analysis_code'
    code.mkdir(exist_ok=True)
    for name in REPORT_SOURCES:
        (code / name.replace('/', '_')).write_bytes((REPO / name).read_bytes())
    lines = ['# Native nonlinear-value placement: V3', '',
             'Exploratory reused-test comparison; development-selected checkpoints and two training seeds.', '',
             '| Arm | Seed | N16 K0–8 | N32 K0–8 | N64 K0–8 | Familiar OOD | K9–16 | Epoch |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for seed in SEEDS:
            info = results[arm][seed]
            cells = [info['cells'][cell] for cell in CELLS[:3]] + [info['familiar_ood'], info['unseen_count']]
            values = [f"{cell['correct']}/{cell['n']} ({100*cell['exact']:.1f}%)" for cell in cells]
            lines.append('| ' + ' | '.join([arm, str(seed), *values, str(info['selected_epoch'])]) + ' |')
    lines += ['', f"Primary passes in both seeds: **{criteria['primary_both_seeds']}**.",
              'Each familiar-length cell has108 examples; pooled N32/N64 has216 observations from108 paired anchors. The count test has128 observations from64 anchors.', '',
              '| PRE minus POST | Familiar OOD gain | Paired-anchor 95% interval | N16 gain | Primary passes |',
              '|---|---:|---:|---:|---:|']
    for label, group, passed in [(f'seed{seed}', contrasts[seed], per_seed_primary[seed]) for seed in SEEDS] + [('pooled fixed seeds', pooled, 'not a decision criterion')]:
        value = group['familiar_ood']
        low, high = value['bootstrap95']
        lines.append(f"| {label} | {100*value['exact_gain']:+.1f}pp | [{100*low:+.1f}, {100*high:+.1f}]pp | {100*group['N16']['exact_gain']:+.1f}pp | {passed} |")
    lines += ['', 'The pooled contrast preserves both seeds and both lengths within each resampled anchor. It cannot replace a failure in either seed.', '',
              '## Historical reference and unseen counts', '']
    for seed in SEEDS:
        value = historical[seed]['familiar_ood']; low, high = value['bootstrap95']
        status = f"V3.2={criteria['V3_2']}" if seed == 0 else 'descriptive only; not a registered criterion'
        n16 = historical[seed]['N16']['exact_gain']
        lines.append(f"- PRE seed{seed} minus historical V2 hidden seed0: familiar OOD {100*value['exact_gain']:+.1f}pp, interval [{100*low:+.1f}, {100*high:+.1f}]pp; N16 {100*n16:+.1f}pp; {status}.")
    lines += ['', '| Arm/seed | K9 exact /16 | K10–16 exact /112 | Count parse rate | Count MAE |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for seed in SEEDS:
            info = results[arm][seed]; support = info['count_answer_support']; count = info['unseen_count']
            mae = 'NA' if count['mae_parsed'] is None else f"{count['mae_parsed']:.3f}"
            lines.append(f"| {arm}/seed{seed} | {support['K9']['correct']}/16 | {support['K10_16']['correct']}/112 | {100*count['parse_rate']:.1f}% | {mae} |")
    lines += ['', '## Measured implementation cost', '',
              '| Arm/seed | Familiar OOD model sec/example | Total sec/example | Peak training allocated GiB |', '|---|---:|---:|---:|']
    for arm in ARMS:
        for seed in SEEDS:
            info = results[arm][seed]; cell = info['familiar_ood']
            lines.append(f"| {arm}/seed{seed} | {cell['model_seconds']:.3f} | {cell['total_seconds']:.3f} | {info['memory']['training_peak_allocated_bytes']/2**30:.2f} |")
    lines += ['', 'Per-K outcomes, parse rates, MAE/bias, likelihoods, memory, paired extension transitions, checkpoint hashes and all contrasts are in [analysis.json](analysis.json).', '',
              '## Interpretation limits', ''] + ['- ' + note for note in notes] + ['']
    (a.root / 'REPORT.md').write_text('\n'.join(lines))
    colors = {'lift_pre': '#0072B2', 'lift_post': '#D55E00'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for arm in ARMS:
        for seed in SEEDS:
            info = results[arm][seed]; style = '-' if seed == 0 else '--'
            axes[0, 0].plot([16, 32, 64], [info['cells'][f'length_N{n}']['exact'] for n in [16, 32, 64]], marker='o', linestyle=style, color=colors[arm], label=f'{arm} seed{seed}')
            axes[0, 1].plot([32, 64], [info['cells'][f'unseen_count_N{n}']['exact'] for n in [32, 64]], marker='o', linestyle=style, color=colors[arm])
            axes[1, 0].plot(range(1, 10), [sum(item['correct'] for item in epoch['dev']) / 72 for epoch in info['training']], linestyle=style, color=colors[arm])
    groups = [contrasts[0]['familiar_ood'], contrasts[1]['familiar_ood'], pooled['familiar_ood']]
    means = np.array([group['exact_gain'] for group in groups]) * 100
    bounds = np.array([group['bootstrap95'] for group in groups]) * 100
    axes[1, 1].errorbar(means, [0, 1, 2], xerr=np.array([means-bounds[:, 0], bounds[:, 1]-means]), fmt='o', capsize=4, color='#333333')
    axes[1, 1].axvline(0, color='grey', linewidth=1)
    axes[1, 1].axvline(5, color='#009E73', linestyle=':', label='Registered +5pp screen')
    axes[1, 1].set(yticks=[0, 1, 2], yticklabels=['seed0', 'seed1', 'pooled fixed seeds'], xlabel='PRE minus POST (percentage points)', title='Familiar OOD · anchor intervals')
    axes[0, 0].set(title='Familiar counts 0–8', xlabel='Frames', ylabel='Exact answer', xticks=[16, 32, 64])
    axes[0, 1].set(title='Unseen counts 9–16', xlabel='Frames', ylabel='Exact answer', xticks=[32, 64])
    axes[1, 0].set(title='In-range development', xlabel='Epoch', ylabel='Exact answer')
    for axis in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        axis.set_ylim(-.02, 1.02)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False)
    fig.suptitle('Native value lifting · exploratory reused V2 test', y=.945)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(a.root / 'comparison.png', dpi=170)
    fig.savefig(a.root / 'comparison.pdf')
    print('\n'.join(lines[:22]), flush=True)


if __name__ == '__main__':
    main()

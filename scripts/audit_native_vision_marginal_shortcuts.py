"""Audit oracle marginal shortcuts in the V2 generator, on CPU Slurm only.

Theory, conditional on fixed N,K under the nominal generator before content
exclusions: every positive contributes one to both C (target-character frames)
and R (target-room frames). Each of the N-K negatives contributes (1,0), (0,1)
or (0,0) with equal probability. Thus E[C]=E[R]=N/3+2K/3. For
B=C+R-2K ~ Binomial(N-K,2/3), Khat=.75(C+R)-.5N is unbiased and
Var(Khat)=(9/16)*(N-K)*(2/3)*(1/3)=(N-K)/8. The individual references
1.5C-.5N and 1.5R-.5N are also unbiased, with variance (N-K)/2 each.

These are fixed oracle references using known generator probabilities. They do
not read images, model predictions or hidden states, and are not VLM methods.
Predictive marginal information does not identify K exactly: at N=4, counts
(match,same-char, same-room,neither)=(2,0,0,2) and (1,1,1,1) both give C=R=2,
but K differs. High model accuracy can therefore exploit a statistical shortcut;
this audit alone cannot show that a model uses it or solves conjunction binding.
No clipping, training, fitted calibration, or tuning is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from statistics import mean, pvariance
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DATA_BASE = Path('/mnt/data/gabriele/gnn_transformer')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.audit_native_vision_data import inspect

NAMES = ('joint_marginal_moment', 'character_only_moment', 'room_only_moment')
COUNTS = ('matches', 'same_character_wrong_room', 'queried_room_other_character', 'neither')
EXPECTED = {
    'main': {'train_N8': (90, range(9), 10), 'train_N16': (90, range(9), 10),
             'dev_N8': (36, range(9), 4), 'dev_N16': (36, range(9), 4),
             'test_N16': (108, range(9), 12), 'test_N32': (108, range(9), 12),
             'test_N64': (108, range(9), 12)},
    'count': {'test_N32': (64, range(9, 17), 8), 'test_N64': (64, range(9, 17), 8)},
}


def estimates(n, c, r):
    if not 0 <= c <= n or not 0 <= r <= n:
        raise ValueError('Invalid oracle marginal counts')
    return dict(zip(NAMES, (.75 * (c + r) - .5 * n, 1.5 * c - .5 * n, 1.5 * r - .5 * n)))


def variances(n, k):
    return dict(zip(NAMES, ((n - k) / 8, (n - k) / 2, (n - k) / 2)))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def synthetic_checks():
    """Exact finite enumeration, deterministic and independent of real labels."""
    cases = 0
    for k in (0, 2, 6):
        for negatives in range(7):
            n = k + negatives
            values = {name: [] for name in NAMES}
            for types in itertools.product(range(3), repeat=negatives):
                c = k + types.count(0)
                r = k + types.count(1)
                for name, estimate in estimates(n, c, r).items():
                    values[name].append(estimate)
                cases += 1
            for name in NAMES:
                if not math.isclose(mean(values[name]), k, abs_tol=1e-12):
                    raise AssertionError('Nominal estimator mean differs from K')
                if not math.isclose(pvariance(values[name]), variances(n, k)[name], abs_tol=1e-12):
                    raise AssertionError('Nominal estimator variance formula failed')
    # Same marginals can arise from different conjunction counts.
    first, second = (2, 0, 0, 2), (1, 1, 1, 1)
    def from_categories(counts):
        match, character, room, neither = counts
        return estimates(sum(counts), match + character, match + room)
    if first[0] == second[0] or from_categories(first) != from_categories(second):
        raise AssertionError('Equal-marginal, different-K sanity check failed')
    if estimates(4, 0, 0)['joint_marginal_moment'] != -2.0:
        raise AssertionError('Unrequested clipping introduced')
    if math.floor(.5 + .5) != 1 or math.floor(-.5 + .5) != 0:
        raise AssertionError('Rounding convention changed')
    return dict(passed=True, enumerated_configurations=cases,
                checks=['Exact enumeration of nominal means/variances',
                        'Equal marginals do not determine conjunction count',
                        'Negative estimates retained without clipping', 'floor(x+0.5) rounding'])


def summarize(rows):
    out = dict(n=len(rows), gold_histogram=dict(sorted(Counter(row['gold'] for row in rows).items())),
               mean_N=mean(row['n_frames'] for row in rows), mean_K=mean(row['gold'] for row in rows),
               mean_C=mean(row['C'] for row in rows), mean_R=mean(row['R'] for row in rows),
               estimators={})
    for name in NAMES:
        values = [row['estimates'][name] for row in rows]
        errors = [value - row['gold'] for value, row in zip(values, rows)]
        correct = sum(math.floor(value + .5) == row['gold'] for value, row in zip(values, rows))
        out['estimators'][name] = dict(mae=mean(abs(error) for error in errors), bias=mean(errors),
            mse=mean(error * error for error in errors), empirical_error_variance=pvariance(errors),
            nominal_mean_conditional_variance=mean(row['nominal_variance'][name] for row in rows),
            rounded_correct=correct, rounded_exact=correct / len(rows),
            minimum_estimate=min(values), maximum_estimate=max(values))
    return out


def collect(main_path, count_path):
    rows, metadata, paths, content_hashes = [], {}, set(), set()
    shared_root = None
    for purpose, path in [('main', main_path), ('count', count_path)]:
        payload = path.read_bytes()
        manifest = json.loads(payload)
        root = Path(manifest['dataset_root']).resolve()
        if manifest.get('schema_version') != 1 or not root.is_relative_to(DATA_BASE):
            raise ValueError('Invalid manifest schema or data root')
        if shared_root is not None and root != shared_root:
            raise ValueError('Main and count manifests use different roots')
        shared_root = root
        if set(manifest['splits']) != set(EXPECTED[purpose]):
            raise ValueError('Unexpected manifest cells')
        metadata[purpose] = dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest())
        for cell, (size, golds, per_gold) in EXPECTED[purpose].items():
            selected = manifest['splits'][cell]['samples']
            if len(selected) != size or Counter(record['gold'] for record in selected) != Counter({k: per_gold for k in golds}):
                raise ValueError(f'Unexpected registered sample count/gold balance: {purpose}/{cell}')
            split, n = cell.split('_N'); n = int(n)
            for record in selected:
                directory = Path(record['path']).resolve()
                if directory.parent != root / 'mmred_vfiltered' / f'seq_len_{n}' / split:
                    raise ValueError('Sample outside declared data split')
                if Path(record['source_path']).resolve() != directory:
                    raise ValueError('V2 generated source must equal staged sample path')
                if str(directory) in paths or record['content_sha256'] in content_hashes:
                    raise ValueError('Duplicate selected content/path')
                paths.add(str(directory)); content_hashes.add(record['content_sha256'])
                checked = inspect(record, purpose, cell)
                for field in COUNTS:
                    if record['semantic_counts'].get(field) != checked[field]:
                        raise ValueError(f'Manifest semantic count differs from independent QA recount: {field}')
                if checked['target_character'] != record['target_character'] or checked['target_room'] != record['target_room']:
                    raise ValueError('Question target differs from manifest metadata')
                c = checked['target_character_frames']
                r = checked['matches'] + checked['queried_room_other_character']
                family = 'unseen_count' if purpose == 'count' else ('familiar_count' if split == 'test' else split)
                rows.append(dict(row_index=len(rows), path=str(directory), sid=record['sid'],
                    cell=('count_N' + str(n)) if purpose == 'count' else cell,
                    split=split, family=family, n_frames=n, gold=checked['gold'], C=c, R=r,
                    pair_id=record.get('pair_id'), qa_sha256=record['qa_sha256'],
                    semantic_counts={field: checked[field] for field in COUNTS},
                    estimates=estimates(n, c, r), nominal_variance=variances(n, checked['gold'])))
    if len(rows) != 704:
        raise ValueError('Expected exactly 704 registered main/count samples')
    return rows, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DATA_BASE / 'v2_clean/main_manifest.json')
    p.add_argument('--count-manifest', type=Path, default=DATA_BASE / 'v2_clean/count_manifest.json')
    p.add_argument('--output', type=Path, default=REPO / 'outputs/native_aggregation_vlm/v2/marginal_audit')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run the metadata-only marginal diagnostic inside Slurm CPU')
    started = time.monotonic()
    checks = synthetic_checks()
    if a.self_test:
        print(json.dumps(checks), flush=True)
        return
    for path in (a.manifest, a.count_manifest):
        if not path.resolve().is_relative_to(DATA_BASE):
            raise SystemExit(f'Manifest paths must remain under {DATA_BASE}')
    rows, manifests = collect(a.manifest, a.count_manifest)
    groups = defaultdict(list)
    for row in rows:
        groups[row['cell']].append(row)
        groups[f"{row['cell']}_K{row['gold']}"] .append(row)
        groups['all_' + row['family']].append(row)
        if row['family'] == 'familiar_count' and row['n_frames'] in (32, 64):
            groups['familiar_count_N32_N64'].append(row)
        if row['family'] == 'unseen_count':
            suffix = 'K9' if row['gold'] == 9 else 'K10_16'
            groups['unseen_count_' + suffix].append(row)
            groups[f"{row['cell']}_{suffix}"] .append(row)
    summaries = {name: summarize(group) for name, group in sorted(groups.items())}
    output = a.output / f'audit_{os.environ["SLURM_JOB_ID"]}'
    output.mkdir(parents=True, exist_ok=False)
    code = output / 'code'; code.mkdir()
    hashes = {}
    for name in ('scripts/audit_native_vision_marginal_shortcuts.py', 'scripts/audit_native_vision_data.py'):
        payload = (REPO / name).read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        (code / name.replace('/', '_')).write_bytes(payload)
    notes = [
        'Oracle marginals and known generator probabilities; no images, model features/predictions, model fitting, or proposed inference method.',
        'No clipping or tuning: continuous estimates are rounded only by floor(x+0.5) for secondary exact accuracy.',
        'Under the nominal conditional law E[C]=E[R]=N/3+2K/3, so marginal frequencies predict K without observing which frames realize the conjunction.',
        'Marginals do not determine K exactly; equal-marginal scenes can have different conjunction counts. The audit quantifies a statistical shortcut, not a complete substitute for binding.',
        'Unbiasedness and variance formulas hold under the nominal IID-negative law before content-exclusion/rejection conditioning. Empirical bias/variance need not equal nominal moments in this finite selected sample.',
        'Paired lengths share anchors and these descriptive rows are not independent inferential replicates. No confidence or causal claim is made.',
        'Oracle-reference accuracies must remain separate from competitive model tables. This audit cannot show that any model uses the shortcut.',
    ]
    result = dict(schema_version=1, slurm_job_id=os.environ['SLURM_JOB_ID'], samples=len(rows),
                  manifests=manifests, code_sha256=hashes, synthetic_checks=checks,
                  all_qa_hashes_verified=True, all_gold_recounts_verified=True,
                  all_manifest_semantic_counts_verified=True, image_files_read=0,
                  definitions=dict(C='Number of frames containing the target character',
                    R='Number of frames containing any character in the target room',
                    joint_marginal_moment='0.75*(C+R)-0.5*N',
                    character_only_moment='1.5*C-0.5*N', room_only_moment='1.5*R-0.5*N'),
                  cells=summaries, notes=notes, elapsed_seconds=time.monotonic() - started)
    write_json(output / 'analysis.json', result)
    write_json(output / 'rows.json', rows)
    lines = ['# V2 oracle-marginal sampling-law audit', '',
             'These are fixed oracle references using known generator probabilities, not VLM results or a proposed method.', '',
             '| Data cell | n | Joint-marginal MAE | Joint bias | Joint rounded exact | Character-only MAE | Room-only MAE |',
             '|---|---:|---:|---:|---:|---:|---:|']
    display = ['train_N8', 'train_N16', 'dev_N8', 'dev_N16', 'test_N16', 'test_N32', 'test_N64',
               'familiar_count_N32_N64', 'count_N32', 'count_N64', 'unseen_count_K9', 'unseen_count_K10_16']
    for cell in display:
        info = summaries[cell]; est = info['estimators']; joint = est[NAMES[0]]
        lines.append(f"| {cell} | {info['n']} | {joint['mae']:.3f} | {joint['bias']:+.3f} | {joint['rounded_correct']}/{info['n']} ({100*joint['rounded_exact']:.1f}%) | {est[NAMES[1]]['mae']:.3f} | {est[NAMES[2]]['mae']:.3f} |")
    lines += ['', 'For each negative frame, the character/room marginal indicator is (1,0), (0,1) or (0,0), each with probability 1/3. Consequently C+R=2K+B with B~Binomial(N-K,2/3). The joint estimator is unbiased with variance (N-K)/8 under the nominal law; each individual estimator has variance (N-K)/2.', '',
              'At N=4, category counts (match, character-only, room-only, neither)=(2,0,0,2) and (1,1,1,1) both yield C=R=2 but have different K. This illustrates both the predictive shortcut and its inability to identify individual conjunction counts exactly.', '',
              'All 704 QA hashes, gold counts and manifest semantic counts were independently checked. Per-K metrics, all three estimators\' bias/exact scores, empirical error variances and nominal variances are in [analysis.json](analysis.json); sample-level estimates are in [rows.json](rows.json).', '',
              '## Interpretation limits', ''] + ['- ' + note for note in notes] + ['']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    index = a.output / 'INDEX.md'
    if not index.exists():
        index.write_text('# V2 marginal-shortcut audit\n\nOracle references only; separate from model efficacy results.\n\n')
    with index.open('a') as stream:
        stream.write(f"- Slurm {os.environ['SLURM_JOB_ID']}: [report]({output.name}/REPORT.md), [verified analysis]({output.name}/analysis.json), [sample estimates]({output.name}/rows.json). All704 QA/gold/semantic checks passed; no images read.\n")
    print('\n'.join(lines[:19]), flush=True)
    print(f'Report: {output}', flush=True)


if __name__ == '__main__':
    main()

"""CPU Slurm ridge diagnostic on frozen native vision h/r captures.

Final test outcomes never select regularization or a representation. Fitted
weights are not persisted; reports contain continuous predictions and metrics.
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
from scripts.probe_native_vision_recoverability import DATA_BASE, LastRowCapture, require_slurm, sha256, write_json

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def ridge_grid(x_train, y_train, x_eval, alphas=ALPHAS):
    """Training-only standardization; centered-target dual ridge, no saved model."""
    import numpy as np
    x_train = np.asarray(x_train, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    a, b = (x_train - mean) / scale, (x_eval - mean) / scale
    target_mean = float(y_train.mean())
    eig, vectors = np.linalg.eigh(a @ a.T)
    eig = np.maximum(eig, 0.0)
    rhs = vectors.T @ (y_train - target_mean)
    cross = (b @ a.T) @ vectors
    predictions = {float(alpha): cross @ (rhs / (eig + alpha)) + target_mean for alpha in alphas}
    return predictions


def metrics(y, prediction):
    import numpy as np
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.size == 0:
        return None
    error = prediction - y
    denominator = float(((y - y.mean()) ** 2).sum())
    return dict(n=int(y.size), mae=float(np.abs(error).mean()),
                r2=None if denominator <= 1e-12 else float(1.0 - (error ** 2).sum() / denominator),
                rounded_exact=float((np.floor(prediction + 0.5) == y).mean()),
                bias=float(error.mean()), mean_gold=float(y.mean()),
                mean_prediction=float(prediction.mean()))


def select_alpha(predictions, gold, dev_indices):
    import numpy as np
    scores = {alpha: float(np.abs(value[dev_indices] - gold[dev_indices]).mean())
              for alpha, value in predictions.items()}
    chosen = min(scores, key=lambda alpha: (scores[alpha], -alpha))
    return chosen, scores


def report_groups(records):
    """Each set is descriptive: paired lengths are correlated observations."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[row['cell']].append(i)
        groups['all_' + row['split']].append(i)
        groups[f"{row['cell']}_K{row['gold']}"] .append(i)
        if row['split'] == 'test':
            groups['family_' + row['family']].append(i)
            groups[f"family_{row['family']}_K{row['gold']}"] .append(i)
            if row['family'] == 'familiar_count' and row['n_frames'] in (32, 64):
                groups['familiar_count_long_N32_N64'].append(i)
            if row['family'] == 'unseen_count':
                subset = 'K9' if row['gold'] == 9 else 'K10_16'
                groups['unseen_count_' + subset].append(i)
                groups[row['cell'] + '_' + subset].append(i)
    return dict(groups)


def self_test():
    """Tiny CPU checks of the real hook sites and ridge solution, not model runs."""
    import numpy as np
    import torch
    from torch import nn

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = nn.Linear(4, 4, bias=False)

        def forward(self, hidden_states):
            return self.o_proj(hidden_states * 2.0)

    attention = Attention()
    values = torch.arange(24.0).view(1, 6, 4)
    expected_output = attention(values).detach()
    capture = LastRowCapture(attention)
    actual_output = attention(hidden_states=values)
    found = capture.take()
    torch.testing.assert_close(found['h'], values[0, -1])
    torch.testing.assert_close(found['r'], values[0, -1] * 2.0)
    torch.testing.assert_close(actual_output, expected_output)
    attention(values)
    capture.take()
    capture.close()
    if attention._forward_pre_hooks or attention.o_proj._forward_pre_hooks:
        raise AssertionError('Hooks not removed')

    rng = np.random.default_rng(7)
    train, held = rng.normal(size=(18, 5)), rng.normal(size=(7, 5)) + 15
    train[:, -1], held[:, -1] = 3.0, 3.0
    gold = 4.0 + train[:, 0] - 2 * train[:, 1]
    predictions = ridge_grid(train, gold, held)
    mean, scale = train.mean(0), train.std(0)
    scale[scale <= 1e-12] = 1.0
    x, z = (train - mean) / scale, (held - mean) / scale
    for alpha in ALPHAS:
        primal = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (gold - gold.mean()))
        np.testing.assert_allclose(predictions[alpha], z @ primal + gold.mean(), atol=1e-9)
    chosen, _ = select_alpha({0.1: np.array([2.0, 100.0]), 100.0: np.array([2.0, -100.0])},
                             np.array([2.0, 7.0]), np.array([0]))
    if chosen != 100.0:
        raise AssertionError('Tie selection or dev-only selection failed')
    if metrics([9, 9], [9, 9])['r2'] is not None or metrics([1], [0.5])['rounded_exact'] != 1.0:
        raise AssertionError('Metric conventions changed')
    print('PASS: observational kwargs/positional hooks, hook removal, dual/primal ridge parity, training-only scaling, dev-only tie rule, constant-target R2, rounding', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-input', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    require_slurm()
    if a.self_test:
        self_test()
        return
    if a.data_input is None or a.output is None:
        raise SystemExit('--data-input and --output are required unless --self-test')
    if not a.data_input.resolve().is_relative_to(DATA_BASE):
        raise SystemExit(f'Feature data must reside under {DATA_BASE}')
    import numpy as np
    started = time.time()
    source = json.loads((a.data_input / 'harvest.json').read_text())
    feature_path, rows_path = a.data_input / 'features.npz', a.data_input / 'records.json'
    if not source.get('complete') or source['feature_sha256'] != sha256(feature_path) or source['records_sha256'] != sha256(rows_path):
        raise ValueError('Incomplete or altered feature harvest')
    records = json.loads(rows_path.read_text())
    with np.load(feature_path, allow_pickle=False) as data:
        h, r = data['h'].astype(np.float64), data['r'].astype(np.float64)
    if h.shape != (704, 3584) or r.shape != h.shape or len(records) != 704:
        raise ValueError('Expected exactly two 704x3584 feature matrices')
    if not np.isfinite(h).all() or not np.isfinite(r).all():
        raise ValueError('Nonfinite features')
    for i, row in enumerate(records):
        if row['row_index'] != i:
            raise ValueError('Feature row order has changed')
    train = np.array([i for i, row in enumerate(records) if row['split'] == 'train'])
    dev = np.array([i for i, row in enumerate(records) if row['split'] == 'dev'])
    if (len(train), len(dev)) != (180, 72):
        raise ValueError('Unexpected training/development row counts')
    y = np.array([row['gold'] for row in records], dtype=np.float64)
    n = np.array([row['n_frames'] for row in records], dtype=np.float64)[:, None]
    groups = report_groups(records)
    matrices = {'h': h, 'r': r, 'h+r': np.concatenate([h, r], axis=1)}
    matrices.update({key + '+N': np.concatenate([value, n], axis=1) for key, value in list(matrices.items())})
    matrices['N_only'] = n
    results, predictions = {}, {}
    for name, matrix in matrices.items():
        grid = ridge_grid(matrix[train], y[train], matrix)
        alpha, dev_mae = select_alpha(grid, y, dev)
        chosen = grid[alpha]
        predictions[name] = chosen
        results[name] = dict(alpha=alpha, dev_mae_by_alpha=dev_mae,
                             n_features=matrix.shape[1],
                             metrics={key: metrics(y[indices], chosen[indices]) for key, indices in groups.items()})
        print(f'{name}: alpha={alpha:g}, dev MAE={dev_mae[alpha]:.6f}', flush=True)
    constant = np.full(len(y), y[train].mean())
    predictions['train_mean'] = constant
    results['train_mean'] = dict(alpha=None, n_features=0,
                                metrics={key: metrics(y[indices], constant[indices]) for key, indices in groups.items()})
    out = a.output / f'fit_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True, exist_ok=False)
    config = dict(schema_version=1, slurm_job_id=os.environ['SLURM_JOB_ID'],
                  data_input=str(a.data_input), feature_sha256=source['feature_sha256'],
                  records_sha256=source['records_sha256'], manifests=source['manifests'],
                  numpy_version=np.__version__, alphas=ALPHAS, n_train=len(train), n_dev=len(dev),
                  normalization='Training-only mean and population standard deviation; constant columns scale=1',
                  intercept='Training target mean; unpenalized',
                  regularization_selection='Pooled dev MAE, ties toward larger alpha; independent per representation',
                  h_plus_r='Concatenation, not elementwise sum',
                  rounding='floor(prediction+0.5), no clipping to training range',
                  interpretation='Linear diagnostic only; correlated paired test rows are descriptive, not independent inference',
                  code_sha256=sha256(Path(__file__)), elapsed_seconds=time.time() - started)
    write_json(out / 'config.json', config)
    write_json(out / 'results.json', results)
    with (out / 'predictions.jsonl').open('w') as stream:
        for i, row in enumerate(records):
            item = {key: row.get(key) for key in ('row_index', 'sid', 'path', 'split', 'cell', 'family',
                                                 'n_frames', 'gold', 'pair_id', 'test_family')}
            item['predictions'] = {key: float(value[i]) for key, value in predictions.items()}
            stream.write(json.dumps(item, allow_nan=False) + '\n')
    lines = ['# Frozen V2 native-read linear recoverability', '',
             'Alpha selected by development MAE only. Predictions are continuous; h+r concatenates the two channels.', '',
             '| Feature | alpha | Dev MAE | Familiar N32/64 MAE | K9 MAE | K10–16 MAE |',
             '|---|---:|---:|---:|---:|---:|']
    for key, value in results.items():
        ms = value['metrics']
        fields = [f'{ms[group]["mae"]:.4f}' for group in ('all_dev', 'familiar_count_long_N32_N64', 'unseen_count_K9', 'unseen_count_K10_16')]
        lines.append('| ' + ' | '.join([key, str(value['alpha'])] + fields) + ' |')
    lines += ['', 'R-squared and rounded-integer exact are in results.json, including every original split/N and per-K cell.',
              'No fitted weights are saved. Low probe accuracy does not establish that information is absent.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    (out / 'fit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(output=str(out), elapsed_seconds=time.time() - started)), flush=True)


if __name__ == '__main__':
    main()

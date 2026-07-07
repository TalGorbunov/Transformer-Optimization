#!/usr/bin/env python3
"""Confirm the 'nonlinearity-before-sum' claim and the readout ceiling, CPU on the L19 cache.

- per-frame sigmoid-then-sum: fit a per-frame is-evidence logistic (the 0.96 classifier), then for each
  example SUM the per-frame probabilities (soft) and hard decisions -> count. If this lands high while
  linear-on-S_all is 0.45, the missing ingredient is the per-frame THRESHOLD, not capacity.
- MLP vs linear on S_all and on the real last-token rep: is the residual count nonlinearly recoverable
  from the single summed/aggregated vector? (Garcia/readout-ceiling style.)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.glstm.probe_message_sum_decodability import load_cache, agg_sum
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, default=8)
    args = ap.parse_args()
    for spec in args.caches:
        label, _, path = spec.partition(":")
        exs = load_cache(Path(path), args.seq_len)
        N = exs[0].reps.shape[0]
        print(f"\n===== {label}: {len(exs)} examples =====")
        # example-level split
        idx = np.arange(len(exs))
        tr, te = train_test_split(idx, test_size=0.35, random_state=0)
        # per-frame logistic on TRAIN frames
        Xtr = np.stack([exs[i].reps[f] for i in tr for f in range(N)])
        ytr = np.asarray([int(exs[i].labels[f]) for i in tr for f in range(N)])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(Xtr), ytr)
        # per-example soft/hard count on TEST
        gold_te = np.asarray([exs[i].gold for i in te])
        soft, hard = [], []
        for i in te:
            p = clf.predict_proba(sc.transform(exs[i].reps))[:, 1]
            soft.append(p.sum()); hard.append(int((p > 0.5).sum()))
        soft = np.rint(soft).astype(int); hard = np.asarray(hard)
        print(f"  per-frame sigmoid-then-sum:  SOFT acc={accuracy_score(gold_te, soft):.3f} "
              f"mae={mean_absolute_error(gold_te, soft):.2f}   "
              f"HARD acc={accuracy_score(gold_te, hard):.3f} mae={mean_absolute_error(gold_te, hard):.2f}")
        # linear vs MLP on S_all and last-token rep
        for name, feat in [("S_all", np.stack([agg_sum(e.reps) for e in exs])),
                           ("last_tok", np.stack([e.query for e in exs]))]:
            gold = np.asarray([e.gold for e in exs])
            scf = StandardScaler().fit(feat[tr])
            lin = Ridge(alpha=1.0).fit(scf.transform(feat[tr]), gold[tr])
            mlp = MLPRegressor(hidden_layer_sizes=(256,), max_iter=400, random_state=0,
                               early_stopping=True).fit(scf.transform(feat[tr]), gold[tr])
            la = accuracy_score(gold[te], np.rint(lin.predict(scf.transform(feat[te]))).astype(int))
            ma = accuracy_score(gold[te], np.rint(mlp.predict(scf.transform(feat[te]))).astype(int))
            print(f"  {name:9s}: linear acc={la:.3f}   MLP acc={ma:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

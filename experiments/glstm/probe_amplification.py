#!/usr/bin/env python3
"""Amplification experiment: at FIXED N, (1) is the softmax MEAN (divide by N) really worse than the
SUM for the count? (2) does amplifying the per-frame evidence signal help — and does it need to be
NONLINEAR? On the frozen L19 caches.

  sum_all   : decode count from Sigma reps        (= S_all, the model's effective extensive read)
  mean_all  : decode count from (1/N) Sigma reps  (= the softmax-mean read; should EQUAL sum at fixed N)
  lin-amp   : scale each rep's evidence-direction component by lambda, decode from mean (LINEAR amp)
  gate(kappa): c = Sigma sigmoid(kappa * per-frame score); sum vs mean readout, kappa = gate sharpness
If mean==sum -> dividing by N does NOT suppress the count at fixed N. If lin-amp==raw but gate>raw ->
amplification must be NONLINEAR (the gate), not a linear rescale.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.glstm.probe_message_sum_decodability import load_cache, fit_ridge, agg_sum, agg_mean
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    for spec in args.caches:
        label, _, path = spec.partition(":")
        exs = load_cache(Path(path), args.seq_len)
        N = exs[0].reps.shape[0]
        gold = np.asarray([e.gold for e in exs])
        print(f"\n===== {label}: {len(exs)} ex, N={N} =====")

        # (1) mean vs sum (is dividing by N benign at fixed N?)
        S_all = np.stack([agg_sum(e.reps) for e in exs])
        M_all = np.stack([agg_mean(e.reps) for e in exs])
        a_sum = fit_ridge(S_all, gold, seeds); a_mean = fit_ridge(M_all, gold, seeds)
        print(f"  sum_all  -> count: acc={a_sum['acc']:.3f} r2={a_sum['r2']:.3f}")
        print(f"  mean_all -> count: acc={a_mean['acc']:.3f} r2={a_mean['r2']:.3f}   "
              f"(==sum => dividing by N is benign at fixed N)")

        # evidence direction (per-frame logistic) for amplification
        XF = np.stack([e.reps[f] for e in exs for f in range(N)])
        yF = np.asarray([int(e.labels[f]) for e in exs for f in range(N)])
        sc = StandardScaler().fit(XF)
        clf = LogisticRegression(max_iter=300).fit(sc.transform(XF), yF)
        w = clf.coef_[0] / sc.scale_; dhat = w / (np.linalg.norm(w) + 1e-9)

        # (2a) LINEAR amplification along dhat -> decode from mean
        print("  -- LINEAR amplification along evidence dir (decode from MEAN) --")
        for lam in (0.0, 4.0, 20.0):
            R = np.stack([e.reps + lam * (e.reps @ dhat)[:, None] * dhat[None, :] for e in exs])
            Mlam = R.mean(1)
            a = fit_ridge(Mlam, gold, seeds)
            print(f"    lambda={lam:>5}: mean->count acc={a['acc']:.3f}   (flat => linear amp is a no-op)")

        # (2b) NONLINEAR gate sharpness sweep -> sum vs mean readout
        print("  -- NONLINEAR gate: c = Sigma sigmoid(kappa*score); sum vs mean readout --")
        idx = np.arange(len(exs)); tr, te = train_test_split(idx, test_size=0.35, random_state=0)
        # per-frame standardized score
        def scores(e):
            return clf.decision_function(sc.transform(e.reps))
        smean = np.mean([scores(exs[i]).mean() for i in tr]); sstd = np.std([scores(exs[i]) for i in tr]) + 1e-9
        for kap in (0.0, 0.5, 1.0, 2.0, 5.0, 20.0):
            sum_pred, mean_pred = [], []
            for i in te:
                z = (scores(exs[i]) - smean) / sstd
                g = 1.0 / (1.0 + np.exp(-kap * z)) if kap > 0 else 0.5 * np.ones_like(z)
                sum_pred.append(int(round(g.sum())))
                mean_pred.append(int(round(N * g.mean())))
            gt = gold[te]
            asum = accuracy_score(gt, sum_pred); amean = accuracy_score(gt, mean_pred)
            print(f"    kappa={kap:>5}: SUM acc={asum:.3f}  MEAN(xN) acc={amean:.3f}   "
                  f"({'sharper gate => higher' if kap>0 else 'no gate'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

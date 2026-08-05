#!/usr/bin/env python3
"""Deconfound the hop: is the count still in the parent state, hidden behind gain?

Probe families per cell (all fit on standardized PCA-512, 3 sample-splits):
  lr          multiclass logistic (the probe used so far)
  ridge_round ridge regression -> round to nearest int  (residual-shape check:
              if rr >> lr, deep-cell 'exactness loss' was partly probe artifact)
  lr_dirnorm  logistic on L2-NORMALIZED states (direction only — cancels any
              GLOBAL multiplicative gain; improvement => gain-noise evidence)
  rr_dirnorm  ridge-round on normalized states
  lr_ratio    logistic on h / <h, pc1>  (pc1 = dominant shared 'arrival' component
              as a per-sample gain estimate; improvement => per-read gain noise)

Cells: natural b2 lvl2@L20 (frozen sweep), patch P0/P2 lvl2@24 and lvl3@27.
Usage: python scripts/superquery/deconfound.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/superquery"))

from probe_tree import leaf_sets, tree_levels  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402

levels = tree_levels(8, 2)
lsets = leaf_sets(levels)


def counts(Y, lv):
    return np.stack([[int(Y[s, sorted(ls)].sum()) for ls in lsets[lv]]
                     for s in range(len(Y))])


def evaluate(X, y, mode):
    n = X.shape[0]
    accs = []
    for seed in range(3):
        idx = np.random.default_rng(seed).permutation(n)
        tr, ev = idx[: n // 2], idx[n // 2:]
        Xtr = X[tr].reshape(-1, X.shape[-1]).astype(np.float32)
        Xev = X[ev].reshape(-1, X.shape[-1]).astype(np.float32)
        ytr, yev = y[tr].reshape(-1), y[ev].reshape(-1)
        if "dirnorm" in mode:
            Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-6)
            Xev = Xev / (np.linalg.norm(Xev, axis=1, keepdims=True) + 1e-6)
        if "ratio" in mode:
            pc1 = PCA(n_components=1).fit(Xtr).components_[0]
            gtr = Xtr @ pc1
            gev = Xev @ pc1
            Xtr = Xtr / (np.abs(gtr[:, None]) + 1e-6)
            Xev = Xev / (np.abs(gev[:, None]) + 1e-6)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xev = (Xtr - mu) / sd, (Xev - mu) / sd
        k = min(512, Xtr.shape[0] - 1, Xtr.shape[1])
        p = PCA(n_components=k, random_state=0).fit(Xtr)
        Xtr, Xev = p.transform(Xtr), p.transform(Xev)
        if mode.startswith("lr"):
            clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            pr = clf.predict(Xev)
        else:
            rg = Ridge(alpha=10.0).fit(Xtr, ytr)
            pr = np.clip(np.round(rg.predict(Xev)), y.min(), y.max())
        accs.append(float((pr == yev).mean()))
    return float(np.mean(accs))


def main() -> int:
    d_nat = np.load(_REPO / "outputs/superquery/capture_16_64_128790/feats_N8.npz")
    d_pat = np.load(_REPO / "outputs/superquery/patch_n8/patch_feats.npz")
    cells = [
        ("natural lvl2 @L20", d_nat["b2|1|20|mean"], counts(d_nat["Y"], 1)),
        ("P0 lvl2 @L24", d_pat["P0|1|24|mean"], counts(d_pat["Y"], 1)),
        ("P2 lvl2 @L24", d_pat["P2|1|24|mean"], counts(d_pat["Y"], 1)),
        ("P0 lvl3 @L27", d_pat["P0|2|27|last"], counts(d_pat["Y"], 2)),
        ("P2 lvl3 @L27", d_pat["P2|2|27|last"], counts(d_pat["Y"], 2)),
    ]
    modes = ("lr", "ridge_round", "lr_dirnorm", "rr_dirnorm", "lr_ratio")
    print(f"{'cell':22s}" + "".join(f"{m:>12s}" for m in modes))
    for name, X, y in cells:
        row = [evaluate(X.astype(np.float32), y, m) for m in modes]
        print(f"{name:22s}" + "".join(f"{v:12.3f}" for v in row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

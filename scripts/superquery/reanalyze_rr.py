#!/usr/bin/env python3
"""Ridge-round reanalysis of the frozen capacity table (LR understated exactness).

For every (N, arm, level) cell @L20 (mean feat): LR acc vs ridge-round acc, plus
ridge-round composed tally (sum of per-node rounded predictions == gold).
Usage: python scripts/superquery/reanalyze_rr.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/superquery"))

from probe_tree import build_arms, leaf_sets  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402

D = _REPO / "outputs/superquery/capture_16_64_128790"


def main() -> int:
    rows = []
    for N in (8, 16, 32, 64):
        d = np.load(D / f"feats_N{N}.npz")
        Y, G = d["Y"], d["G"]
        n = len(G)
        arms = build_arms(N)
        for arm, levels in arms.items():
            lsets = leaf_sets(levels)
            for lv in range(min(len(levels), 2)):        # lvl1 + first hop
                key = f"{arm}|{lv}|20|mean"
                if key not in d.files:
                    continue
                X = d[key].astype(np.float32)
                _, nn, H = X.shape
                cnt = np.stack([[int(Y[s, sorted(ls)].sum()) for ls in lsets[lv]]
                                for s in range(n)])
                fan = float(np.mean([len(g) for g in levels[lv]]))
                lr_a, rr_a, rr_t = [], [], []
                for seed in range(3):
                    idx = np.random.default_rng(seed).permutation(n)
                    tr, ev = idx[: n // 2], idx[n // 2:]
                    Xtr = X[tr].reshape(-1, H)
                    Xev = X[ev].reshape(-1, H)
                    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                    Xtr, Xev = (Xtr - mu) / sd, (Xev - mu) / sd
                    k = min(512, Xtr.shape[0] - 1, H)
                    p = PCA(n_components=k, random_state=0).fit(Xtr)
                    Xtr, Xev = p.transform(Xtr), p.transform(Xev)
                    ytr = cnt[tr].reshape(-1)
                    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
                    lr_a.append(float((clf.predict(Xev).reshape(len(ev), nn)
                                       == cnt[ev]).mean()))
                    rg = Ridge(alpha=10.0).fit(Xtr, ytr)
                    pr = np.clip(np.round(rg.predict(Xev)), cnt.min(), cnt.max())
                    pr = pr.reshape(len(ev), nn)
                    rr_a.append(float((pr == cnt[ev]).mean()))
                    rr_t.append(float((pr.sum(1) == G[ev]).mean()))
                rows.append([N, arm, lv + 1, fan, np.mean(lr_a), np.mean(rr_a),
                             np.mean(rr_t)])
                print(f"[N={N} {arm} lvl{lv+1} fan{fan:.1f}] lr {np.mean(lr_a):.3f} "
                      f"rr {np.mean(rr_a):.3f} rr-tally {np.mean(rr_t):.3f}", flush=True)
    out = _REPO / "outputs/superquery/fits_all/reanalysis_rr.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([["N", "arm", "level", "fan_in", "lr_acc", "rr_acc",
                                  "rr_tally"], *rows])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

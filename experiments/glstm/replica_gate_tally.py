#!/usr/bin/env python3
"""Gate->tally on a replica_carrier_probe messages_cache.pt (CPU, 2026-07-18 E1).

Loads the probe cache (keys: rep {L: [n,NF,H]}, labels [n,NF], gold [n]), fits a per-frame
logistic gate on a train split (5 seeds), reports held-out per-frame err, tally exact +- std,
MAE, and the per-count row. Majority-class and gold-histogram context printed for honesty.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="replica probe messages_cache.pt")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    c = torch.load(args.cache, map_location="cpu", weights_only=False)
    X = np.asarray(c["rep"][args.layer], dtype=np.float32)
    Y = np.asarray(c["labels"], dtype=int)
    G = np.asarray(c["gold"], dtype=int)
    n, NF, H = X.shape
    hist = {int(g): int((G == g).sum()) for g in np.unique(G)}
    maj = max(hist.values()) / n

    accs, ferrs, maes = [], [], []
    per_tot = {}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        ntr = int(args.train_frac * n)
        tr, ev = idx[:ntr], idx[ntr:]
        clf = LogisticRegression(max_iter=2000).fit(X[tr].reshape(-1, H), Y[tr].reshape(-1))
        pr = clf.predict(X[ev].reshape(-1, H)).reshape(len(ev), NF)
        ferrs.append(float((pr != Y[ev]).mean()))
        tally = pr.sum(1)
        accs.append(float((tally == G[ev]).mean()))
        maes.append(float(np.abs(tally - G[ev]).mean()))
        for g in np.unique(G[ev]):
            m = G[ev] == g
            hit, tot = per_tot.setdefault(int(g), [0, 0])
            per_tot[int(g)] = [hit + int((tally[m] == g).sum()), tot + int(m.sum())]

    pc = " ".join(f"g{g}:{h}/{t}" for g, (h, t) in sorted(per_tot.items()))
    lines = [f"=== REPLICA GATE->TALLY (cache={args.cache}, L{args.layer}, n={n}, NF={NF}, "
             f"{args.seeds} seeds, train_frac={args.train_frac}) ===",
             "[gold-hist] " + " ".join(f"g{g}:{cnt}" for g, cnt in sorted(hist.items()))
             + f"  (majority baseline {maj:.3f})",
             f"per-frame err {np.mean(ferrs):.4f}±{np.std(ferrs):.4f}",
             f"tally exact {np.mean(accs):.3f}±{np.std(accs):.3f}  MAE {np.mean(maes):.2f}",
             f"per-count (summed over seeds): {pc}"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

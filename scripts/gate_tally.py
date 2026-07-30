#!/usr/bin/env python3
"""Gate->tally scaffold rung (CPU): fit a per-frame logistic gate on cached carrier
messages, tally = sum of verdicts. The external-readout ceiling for every supply cell.

Anchors (RESULTS.md): E1 blockfence cache -> exact 0.998 +/- 0.001 ([2026-07-18]);
P3a natural 0.980 +/- 0.012 ([2026-07-24]).

Works on any messages/carrier-states cache with keys {rep {L: [n,NF,H]}, labels, gold}
(probe_supply and eval_carrier --dump-carrier-states both write this shape).

Usage:
  python scripts/gate_tally.py --cache <run>/messages_cache.pt --layer 16 --output <run>/gate_tally
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--zscore", action="store_true", help="z-score features before the fit")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    c = torch.load(args.cache, map_location="cpu", weights_only=False)
    X = np.asarray(c["rep"][args.layer], dtype=np.float32)
    Y = np.asarray(c["labels"], dtype=int)
    G = np.asarray(c["gold"], dtype=int)
    n, NF, H = X.shape
    if args.zscore:
        mu, sd = X.reshape(-1, H).mean(0), X.reshape(-1, H).std(0) + 1e-8
        X = (X - mu) / sd
    hist = {int(g): int((G == g).sum()) for g in np.unique(G)}
    maj = max(hist.values()) / n

    accs, ferrs, maes = [], [], []
    per_tot: dict = {}
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
    lines = [f"=== GATE->TALLY (cache={args.cache}, L{args.layer}, n={n}, NF={NF}, "
             f"{args.seeds} seeds, train_frac={args.train_frac}, zscore={args.zscore}) ===",
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

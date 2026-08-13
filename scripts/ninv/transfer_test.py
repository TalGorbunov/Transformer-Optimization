#!/usr/bin/env python3
"""Cross-N head-transfer test — THE campaign gate instrument (CPU, ~30s).

Fits a ridge head on pair-node states from one capture (train N), tests it on
another (test N). Measures whether the tree-node count code is N-invariant.

ANCHORS (must reproduce before trusting any new number — 2026-08-09 baseline,
node positions NOT yet reset):
  in-N heldout N=8:                       0.979
  cross-capture N=8 (128773 -> 128914):   1.000
  N=8 -> N=16 / 32 / 64 (capture 128790): 0.643 / 0.445 / 0.320   <- the leak

GATE after the node-posreset fix: N=8 -> N=64 must be >= 0.95, AND the in-N
heldout must stay >= 0.95 (the fix must not break in-N decodability).

Usage:
  python scripts/ninv/transfer_test.py \
    --train-npz outputs/superquery/20260805_142644_tree_128773/feats_N8.npz --train-n 8 \
    --test-npz  outputs/superquery/capture_16_64_128790/feats_N64.npz      --test-n 64 \
    [--key-template "b2|0|20|mean"]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/superquery"))

from probe_tree import build_arms, leaf_sets  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_skew import class_report, cluster_bootstrap_ci, format_report  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402


def pair_counts(Y: np.ndarray, N: int) -> np.ndarray:
    lsets = leaf_sets(build_arms(N)["b2"])
    return np.stack([[int(Y[s, sorted(ls)].sum()) for ls in lsets[0]]
                     for s in range(len(Y))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--train-n", type=int, required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--test-n", type=int, required=True)
    ap.add_argument("--key-template", default="b2|0|20|mean",
                    help="feature key: arm|level|layer|feat")
    ap.add_argument("--gate", type=float, default=0.95)
    args = ap.parse_args()

    dtr = np.load(args.train_npz)
    Xtr3 = dtr[args.key_template].astype(np.float32)
    ytr2 = pair_counts(dtr["Y"], args.train_n)
    n, k, H = Xtr3.shape
    Xtr, ytr = Xtr3.reshape(-1, H), ytr2.reshape(-1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    p = PCA(n_components=min(512, Xtr.shape[0] - 1, H), random_state=0).fit((Xtr - mu) / sd)
    rg = Ridge(alpha=10.0).fit(p.transform((Xtr - mu) / sd), ytr)

    # in-N heldout reference (50/50 split, seed 0)
    idx = np.random.default_rng(0).permutation(n)
    tr, ev = idx[: n // 2], idx[n // 2:]
    Xt, yt = Xtr3[tr].reshape(-1, H), ytr2[tr].reshape(-1)
    mu2, sd2 = Xt.mean(0), Xt.std(0) + 1e-6
    p2 = PCA(n_components=min(512, Xt.shape[0] - 1, H), random_state=0).fit((Xt - mu2) / sd2)
    rg2 = Ridge(alpha=10.0).fit(p2.transform((Xt - mu2) / sd2), yt)
    pr = np.clip(np.round(rg2.predict(p2.transform(
        (Xtr3[ev].reshape(-1, H) - mu2) / sd2))), 0, 2).reshape(len(ev), k)
    in_n = float((pr == ytr2[ev]).mean())

    dte = np.load(args.test_npz)
    Xte3 = dte[args.key_template].astype(np.float32)
    yte2 = pair_counts(dte["Y"], args.test_n)
    nte, kte, _ = Xte3.shape
    pr = np.clip(np.round(rg.predict(p.transform(
        (Xte3.reshape(-1, H) - mu) / sd))), 0, 2).reshape(nte, kte)
    cross = float((pr == yte2).mean())

    print(f"in-N heldout  (N={args.train_n}):           {in_n:.3f}")
    print(f"cross-N transfer (N={args.train_n} -> N={args.test_n}): {cross:.3f}")
    ok = cross >= args.gate and in_n >= args.gate
    print(f"GATE (both >= {args.gate}): {'PASS' if ok else 'FAIL'}")

    # DIRECTIVE 2026-08-09 item 4: raw accuracy is flattering on MMReD-HF, whose
    # pair-count prior is ~0.78-0.81 zeros (majority alone scores ~0.80). Always
    # print the skew-aware companions. The gate decision above is UNCHANGED — these
    # are additional reporting, so the docstring anchors still hold.
    print("\nskew-aware breakdown (reporting only; the gate above is on raw):")
    print(format_report(class_report(pr, yte2), prefix=f"  cross-N  N={args.test_n}: "))
    lo, hi, _ = cluster_bootstrap_ci(pr == yte2)
    print(f"  cross-N cluster-bootstrap 95% CI over samples: [{lo:.3f}, {hi:.3f}]"
          f"   (n={len(yte2)} samples x {kte} nodes)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

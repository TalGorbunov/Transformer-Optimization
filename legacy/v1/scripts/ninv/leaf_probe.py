#!/usr/bin/env python3
"""Leaf (per-frame verdict) probe — is the HF c2 deficit PERCEPTION or MERGE LOSS?

Phase 0 on MMReD-HF passed on N-invariance, but c2 recall (pair nodes where BOTH
leaves carry evidence) sat at 0.811 @N=8 and 0.880 @N=16, with every error being
true-2 -> predicted-1. Two incompatible stories:

  PERCEPTION   a leaf never encoded "character is in room" for one of its two
               frames, so the merge had nothing to add. Then pair success is an
               independent product of two leaf detections and
                   c2_recall ~= (leaf recall on evidence frames)^2
               sqrt(0.811) = 0.900 and sqrt(0.880) = 0.938 — so this story predicts
               a leaf evidence-recall near 0.90/0.94.
  MERGE LOSS   both leaves encoded it and fan-2 aggregation dropped one. Then leaf
               evidence-recall is ~0.99 and the square-law prediction fails badly.

This fits a probe on the level-0 replica-span states (key "leaf|0|<L>|mean", added to
probe_tree_ninv.py for exactly this) and reports:
  1. leaf verdict accuracy / balanced / per-class recall / majority baseline
  2. the square-law check: leaf_evidence_recall^2 vs the OBSERVED c2 recall
  3. coincidence: do the node's c2 failures land on pairs that contain a leaf miss?
     Under PERCEPTION they should; under MERGE LOSS they should not.
Both probes are fit on the SAME train half and read on the SAME eval half, so the
coincidence is measured cell-by-cell rather than compared in aggregate.

Usage:
  python scripts/ninv/leaf_probe.py --npz outputs/ninv/<run>/feats_N8.npz --n 8
  python scripts/ninv/leaf_probe.py --npz <392run>/feats_N8.npz --n 8 \
      --compare <512run>/feats_N8.npz --compare-label "resize 512"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/superquery"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_skew import class_report  # noqa: E402
from transfer_matrix import fit_head, pair_counts  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


def fit_leaf(X3: np.ndarray, y2: np.ndarray):
    """Binary per-frame verdict probe (logistic on standardized PCA-512).

    Logistic, not the ridge-round used for counts: the leaf target is a single bit,
    and a classifier's decision boundary is the honest read of "is the verdict
    linearly present", without a rounding threshold in the way.
    """
    H = X3.shape[-1]
    X, y = X3.reshape(-1, H), y2.reshape(-1)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    p = PCA(n_components=min(512, X.shape[0] - 1, H), random_state=0).fit((X - mu) / sd)
    clf = LogisticRegression(max_iter=1000).fit(p.transform((X - mu) / sd), y)
    return lambda Z3: clf.predict(
        p.transform((Z3.reshape(-1, Z3.shape[-1]) - mu) / sd)).reshape(Z3.shape[:2])


def analyse(path: str, n: int, layer: int, label: str, seeds: int = 5):
    d = np.load(path)
    lk, nk = f"leaf|0|{layer}|mean", f"b2|0|{layer}|mean"
    if lk not in d.files:
        raise SystemExit(f"{path} has no '{lk}' — recapture with the leaf-dump build "
                         f"of probe_tree_ninv.py (keys: {[k for k in d.files][:6]}...)")
    XL, Y = d[lk].astype(np.float32), d["Y"].astype(np.int64)   # (n, N, H), (n, N)
    XN, C = d[nk].astype(np.float32), pair_counts(d["Y"], n)     # (n, N/2, H), (n, N/2)
    m = len(Y)
    # Pool several 50/50 splits: one eval half holds only ~50 c2 cells, so a
    # single-seed c2 recall moves by ~0.04 per flipped cell and cannot separate
    # 0.88 from 0.95. Pooling keeps every cell heldout while shrinking that noise.
    LP, NP, LT, CT = [], [], [], []
    for s in range(seeds):
        idx = np.random.default_rng(s).permutation(m)
        tr, ev = idx[: m // 2], idx[m // 2 :]
        LP.append(fit_leaf(XL[tr], Y[tr])(XL[ev]))
        NP.append(fit_head(XN[tr], C[tr])(XN[ev]))
        LT.append(Y[ev])
        CT.append(C[ev])
    lp, np_, lt, ct = (np.concatenate(v) for v in (LP, NP, LT, CT))

    print(f"\n{'='*74}\n{label}   ({path})\n{'='*74}")
    print(f"  {m} samples x {Y.shape[1]} frames   layer {layer}")

    lr = class_report(lp, lt, n_classes=2)
    print(f"\n1. LEAF verdict probe (per-frame 'character is in room'):")
    print(f"     raw {lr['raw']:.3f}   majority {lr['majority']:.3f}   "
          f"balanced {lr['balanced']:.3f}")
    print(f"     recall  no-evidence {lr['recall'][0]:.3f}   "
          f"EVIDENCE {lr['recall'][1]:.3f}   support {lr['support']}")

    nr = class_report(np_, ct)
    leaf_ev = lr["recall"][1]
    pred_c2 = leaf_ev ** 2
    obs_c2 = nr["recall"][2]
    print(f"\n2. SQUARE-LAW CHECK (does pair success factor as two leaf detections?):")
    print(f"     leaf evidence-recall p          = {leaf_ev:.3f}")
    print(f"     p^2 (PERCEPTION prediction)     = {pred_c2:.3f}")
    print(f"     OBSERVED c2 recall              = {obs_c2:.3f}   "
          f"(support {nr['support'][2]})")
    print(f"     residual (observed - p^2)       = {obs_c2 - pred_c2:+.3f}")
    verdict = ("PERCEPTION-consistent (within 0.05 of the square law)"
               if abs(obs_c2 - pred_c2) <= 0.05 else
               "MERGE LOSS: pairs do WORSE than two independent leaves"
               if obs_c2 < pred_c2 else
               "pairs do BETTER than two independent leaves -> the merge is "
               "recovering information the linear leaf probe misses")
    print(f"     -> {verdict}")

    print(f"\n3. COINCIDENCE (cell-by-cell, eval half only):")
    lmiss = ((lt == 1) & (lp == 0))
    pair_lmiss = lmiss[:, ::2] | lmiss[:, 1::2]     # >=1 missed evidence leaf in pair
    is2 = ct == 2
    perr = (np_ != ct) & is2
    pok = (np_ == ct) & is2
    def pct(a, b):
        return f"{(a.sum() / max(b.sum(), 1)):.3f} ({int(a.sum())}/{int(b.sum())})"
    print(f"     among c2 pairs the node got WRONG: fraction containing a leaf miss "
          f"= {pct(perr & pair_lmiss, perr)}")
    print(f"     among c2 pairs the node got RIGHT: fraction containing a leaf miss "
          f"= {pct(pok & pair_lmiss, pok)}")
    print(f"     (PERCEPTION -> first >> second. MERGE LOSS -> both similar/low.)")
    both = ((lt == 1) & (lp == 1))
    pair_both = both[:, ::2] & both[:, 1::2]
    sub = is2 & pair_both
    if sub.sum():
        print(f"     node c2 recall RESTRICTED to pairs where BOTH leaves were "
              f"correctly read: {float((np_[sub] == 2).mean()):.3f} ({int(sub.sum())} "
              f"cells)   <- if this is ~1.0 the merge itself is intact")
    return {"leaf_ev": leaf_ev, "leaf_raw": lr["raw"], "leaf_bal": lr["balanced"],
            "obs_c2": obs_c2, "pair_raw": nr["raw"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--compare", default=None, help="second npz (e.g. a 512px arm)")
    ap.add_argument("--compare-label", default="comparison")
    args = ap.parse_args()

    a = analyse(args.npz, args.n, args.layer, args.label)
    if args.compare:
        b = analyse(args.compare, args.n, args.layer, args.compare_label)
        print(f"\n{'='*74}\nDELTA  {args.compare_label} - {args.label}\n{'='*74}")
        for k, nm in (("leaf_ev", "leaf evidence-recall"), ("leaf_raw", "leaf raw acc"),
                      ("leaf_bal", "leaf balanced"), ("obs_c2", "pair c2 recall"),
                      ("pair_raw", "pair raw acc")):
            print(f"  {nm:<24} {a[k]:.3f} -> {b[k]:.3f}   {b[k]-a[k]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

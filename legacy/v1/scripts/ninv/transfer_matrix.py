#!/usr/bin/env python3
"""Full cross-N transfer MATRIX + error anatomy (CPU, ~1 min) — Phase 0 diagnosis.

transfer_test.py answers "does an N=8 head work at N=64?" with one number. When that
number lands just under the gate you need to know WHY, and the single number cannot
separate the two candidate causes:

  (a) residual N-dependence  -> accuracy degrades monotonically with test N
  (b) domain shift           -> accuracy drops the moment you cross data roots and
                                then stays FLAT in N (our N=8 root is
                                data/mmred_images_park, N=16/32/64 are
                                data/mmred_longN_park — different generators)

So this fits a ridge head on EVERY capture and evaluates it on EVERY capture. Read the
rows: a row that is flat across test-N columns but low is (b); a row that decays with
test N is (a). The same-root pair (train N=16 -> test N=64, both longN) is the decisive
cell — it holds the root fixed and varies only N.

Also prints, for each cell, the error anatomy that distinguishes a positional leak from
plain noise:
  bias   mean signed error (pred - true). A negative bias at large N is the known
         small-n deflation / under-counting signature.
  byidx  accuracy split by node index within the level (first third / last third).
         A positional leak that survives posreset shows up as late nodes being worse.
  conf   confusion over pair counts {0,1,2}.

Usage:
  python scripts/ninv/transfer_matrix.py \
    --npz 8=outputs/ninv/<cap8>/feats_N8.npz 16=outputs/ninv/<cap16>/feats_N16.npz \
          32=outputs/ninv/<cap32>/feats_N32.npz 64=outputs/ninv/<cap64>/feats_N64.npz \
    [--key b2|0|20|mean] [--anatomy 8:64]
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
from metrics_skew import class_report  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402


def pair_counts(Y: np.ndarray, N: int) -> np.ndarray:
    lsets = leaf_sets(build_arms(N)["b2"])
    return np.stack([[int(Y[s, sorted(ls)].sum()) for ls in lsets[0]]
                     for s in range(len(Y))])


def fit_head(X3: np.ndarray, y2: np.ndarray):
    """Ridge-on-PCA head, identical recipe to transfer_test.py (so numbers compare)."""
    H = X3.shape[-1]
    X, y = X3.reshape(-1, H), y2.reshape(-1)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    p = PCA(n_components=min(512, X.shape[0] - 1, H), random_state=0).fit((X - mu) / sd)
    rg = Ridge(alpha=10.0).fit(p.transform((X - mu) / sd), y)
    return lambda Z3: np.clip(np.round(rg.predict(p.transform(
        (Z3.reshape(-1, Z3.shape[-1]) - mu) / sd))), 0, 2).reshape(Z3.shape[:2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", nargs="+", required=True, metavar="N=PATH")
    ap.add_argument("--key", default="b2|0|20|mean")
    ap.add_argument("--anatomy", default=None, metavar="TRAIN:TEST",
                    help="also print the error anatomy for this cell, e.g. 8:64")
    args = ap.parse_args()

    caps = {}
    for spec in args.npz:
        n_s, path = spec.split("=", 1)
        d = np.load(path)
        n = int(n_s)
        caps[n] = {"X": d[args.key].astype(np.float32), "y": pair_counts(d["Y"], n),
                   "path": path, "root": Path(path).parent.name}
    ns = sorted(caps)

    heads = {n: fit_head(caps[n]["X"], caps[n]["y"]) for n in ns}

    print(f"key={args.key}   cell = pair-count accuracy (level-1 b2 nodes, 0/1/2)")
    print("rows = head trained on N, cols = evaluated at N\n")
    acc, rep = {}, {}
    for tn in ns:
        for en in ns:
            pr = heads[tn](caps[en]["X"])
            rep[(tn, en)] = class_report(pr, caps[en]["y"])
            acc[(tn, en)] = rep[(tn, en)]["raw"]

    # DIRECTIVE 2026-08-09 item 4: three grids, never raw alone. On MMReD-HF the
    # zero prior is ~0.78-0.81, so a raw 0.97 can hide a c2 recall of 0.88 — and c2
    # (both leaves carry evidence) is the only class that exercises fan-2 at all.
    for title, pick in (("RAW accuracy", lambda r: r["raw"]),
                        ("BALANCED (mean per-class recall)", lambda r: r["balanced"]),
                        ("c2 RECALL (both leaves have evidence)",
                         lambda r: r["recall"][2])):
        print(f"--- {title} ---")
        print("train\\test " + "".join(f"{n:>10}" for n in ns) + "   samples")
        for tn in ns:
            print(f"N={tn:<8}" + "".join(f"{pick(rep[(tn, en)]):>10.3f}" for en in ns)
                  + f"   n={len(caps[tn]['y'])}")
        print()
    print("majority-class baseline per test capture (what 'always predict 0' scores):")
    print("           " + "".join(f"{rep[(ns[0], en)]['majority']:>10.3f}" for en in ns))
    print("class support per test capture (c0/c1/c2):")
    print("           " + "".join(
        f"{'/'.join(str(s) for s in rep[(ns[0], en)]['support']):>10}" for en in ns))
    print()

    print("\n(diagonal cells are IN-SAMPLE — the head saw those rows; they are an upper "
          "bound, not a heldout number. Off-diagonal cells are the real transfer.)")

    # Length dependence must be read on the EXTRAPOLATION cells (test N > train N)
    # ONLY. A naive spread over all off-diagonal cells conflates "decays as N grows"
    # with "one particular column is odd", and here it mislabels the N=16 head as
    # decaying when its weak cell is at test N=8 — the SMALLEST N.
    print("\nlength dependence — extrapolation cells only (test N > train N):")
    for tn in ns:
        up = [(en, acc[(tn, en)]) for en in ns if en > tn]
        if not up:
            continue
        vals = [v for _, v in up]
        trend = vals[-1] - vals[0]
        print(f"  head N={tn:<3} " + " ".join(f"->{en}:{v:.3f}" for en, v in up)
              + f"   drift(last-first) {trend:+.3f}"
              + ("   <- FLAT in N" if abs(trend) < 0.02 else
                 "   <- DEGRADES with N" if trend < 0 else "   <- IMPROVES with N"))

    # Column means expose an odd-one-out capture (e.g. one from a different data
    # root), which is a domain effect masquerading as a length effect.
    print("\nodd-one-out check — mean off-diagonal accuracy by capture:")
    for en in ns:
        col = [acc[(tn, en)] for tn in ns if tn != en]
        row = [acc[(en, tn)] for tn in ns if tn != en]
        print(f"  N={en:<3} as TEST {np.mean(col):.3f}   as TRAIN {np.mean(row):.3f}"
              f"   root={caps[en]['root']}")
    print("  (a capture that is weak BOTH as train and as test is the outlier — "
          "suspect its data root, not the sequence length)")

    if args.anatomy:
        tn, en = (int(x) for x in args.anatomy.split(":"))
        pr, gt = heads[tn](caps[en]["X"]), caps[en]["y"]
        err = pr - gt
        k = gt.shape[1]
        third = max(1, k // 3)
        print(f"\n=== error anatomy: head N={tn} -> test N={en} "
              f"({caps[en]['root']}) ===")
        print(f"  acc {float((pr == gt).mean()):.3f}   "
              f"bias(pred-true) {float(err.mean()):+.3f}   "
              f"|err|<=1 {float((np.abs(err) <= 1).mean()):.3f}")
        print(f"  by node index: first {third} nodes "
              f"{float((pr[:, :third] == gt[:, :third]).mean()):.3f}   "
              f"last {third} nodes "
              f"{float((pr[:, -third:] == gt[:, -third:]).mean()):.3f}"
              "   (a surviving positional leak makes LATE nodes worse)")
        print("  confusion (rows true 0/1/2, cols pred 0/1/2):")
        for t in range(3):
            sel = gt == t
            counts = [int(((pr == p_) & sel).sum()) for p_ in range(3)]
            tot = max(int(sel.sum()), 1)
            print(f"    true {t}: " + " ".join(f"{c:>6}" for c in counts)
                  + f"   (n={tot}, acc {counts[t]/tot:.3f})")
        print("  per-sample: "
              f"{float((pr == gt).all(1).mean()):.3f} of samples have EVERY node right")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

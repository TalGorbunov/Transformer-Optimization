#!/usr/bin/env python3
"""Merge anatomy — WHY does an HF fan-2 node under-report c2 with perfect leaves?

Established (2026-08-10): HF leaf evidence-recall is 0.995 @392 / 1.000 @512, yet
pair c2 recall is 0.822 / 0.919, versus park 0.983-0.998. The merge fails on pairs
whose leaves are demonstrably intact. Two candidate mechanisms:

  OR-not-SUM   the node encodes "at least one child fired" but not "how many". Then
               a c>=1 probe is near-perfect while a 2-vs-1 probe (restricted to
               cells that HAVE evidence) is weak. That is a REPRESENTATION claim and
               is what section 1 tests.
  AMPLITUDE    the node encodes a scalar sum whose two-child value is not separable
               from the one-child value when the children are too alike, i.e. the
               code is an amplitude rather than a direction. Then errors concentrate
               where the two children's leaf states are MOST similar and where the
               weaker child's verdict margin is LOWEST. Section 2 tests that, and
               park (which merges correctly) should show measurably LESS similar
               children on the same statistic.

The two are not exclusive — an OR-like read is what an amplitude code degenerates to
once the second child adds no new direction.

Section 1 needs only the node key; section 2 additionally needs "leaf|0|<L>|mean"
(present in captures made after the leaf-dump change).

Usage:
  python scripts/ninv/merge_anatomy.py --npz outputs/ninv/<run>/feats_N8.npz --n 8 \
      --label "HF @392"
  python scripts/ninv/merge_anatomy.py --npz A --n 8 --compare B --compare-label park
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


def _prep(Xtr, Xev):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    p = PCA(n_components=min(512, Xtr.shape[0] - 1, Xtr.shape[1]),
            random_state=0).fit((Xtr - mu) / sd)
    return p.transform((Xtr - mu) / sd), p.transform((Xev - mu) / sd)


def bin_probe(Xtr, ytr, Xev, yev):
    """Binary logistic probe -> (class_report, decision margins on eval)."""
    if len(np.unique(ytr)) < 2 or len(np.unique(yev)) < 2:
        return None, None
    A, B = _prep(Xtr, Xev)
    clf = LogisticRegression(max_iter=1000).fit(A, ytr)
    return class_report(clf.predict(B), yev, n_classes=2), clf.decision_function(B)


def analyse(path: str, n: int, layer: int, label: str, seeds: int = 5):
    d = np.load(path)
    nk, lk = f"b2|0|{layer}|mean", f"leaf|0|{layer}|mean"
    XN, C = d[nk].astype(np.float32), pair_counts(d["Y"], n)
    Y = d["Y"].astype(np.int64)
    XL = d[lk].astype(np.float32) if lk in d.files else None
    m = len(C)
    H = XN.shape[-1]
    print(f"\n{'='*76}\n{label}   ({path})\n{'='*76}")
    print(f"  {m} samples x {C.shape[1]} pair-nodes   layer {layer}"
          f"   leaf states: {'yes' if XL is not None else 'NO (section 2 skipped)'}")

    or_r, two_r, sum_r, cnt_c2 = [], [], [], []
    sim_ok, sim_bad, marg_ok, marg_bad = [], [], [], []
    for s in range(seeds):
        idx = np.random.default_rng(s).permutation(m)
        tr, ev = idx[: m // 2], idx[m // 2 :]
        ctr, cev = C[tr].reshape(-1), C[ev].reshape(-1)
        Atr, Aev = XN[tr].reshape(-1, H), XN[ev].reshape(-1, H)

        r, _ = bin_probe(Atr, (ctr >= 1).astype(int), Aev, (cev >= 1).astype(int))
        or_r.append(r)
        r, _ = bin_probe(Atr, (ctr == 2).astype(int), Aev, (cev == 2).astype(int))
        sum_r.append(r)
        # 2-vs-1 RESTRICTED to cells that have evidence: isolates "how many" from
        # "any at all". This is the cell the OR-not-SUM story must fail.
        mtr, mev = ctr >= 1, cev >= 1
        r, _ = bin_probe(Atr[mtr], (ctr[mtr] == 2).astype(int),
                         Aev[mev], (cev[mev] == 2).astype(int))
        two_r.append(r)

        pr = fit_head(XN[tr], C[tr])(XN[ev])
        cnt_c2.append(class_report(pr, C[ev])["recall"][2])

        if XL is not None:                      # section 2 statistics
            L = XL[ev]
            a, b = L[:, ::2], L[:, 1::2]        # the two children of each pair
            cos = ((a * b).sum(-1)
                   / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9))
            _, mg = bin_probe(XL[tr].reshape(-1, H), Y[tr].reshape(-1),
                              L.reshape(-1, H), Y[ev].reshape(-1))
            mg = mg.reshape(L.shape[:2])
            minmarg = np.minimum(mg[:, ::2], mg[:, 1::2])
            is2 = C[ev] == 2
            good = is2 & (pr == 2)
            bad = is2 & (pr != 2)
            sim_ok += cos[good].tolist()
            sim_bad += cos[bad].tolist()
            marg_ok += minmarg[good].tolist()
            marg_bad += minmarg[bad].tolist()

    def avg(rs, key, cls=None):
        v = [r[key][cls] if cls is not None else r[key] for r in rs if r]
        return float(np.mean(v)) if v else float("nan")

    print("\n1. OR vs SUM — what does the node state actually encode?")
    print(f"     'at least one' (c>=1)      raw {avg(or_r,'raw'):.3f}  "
          f"balanced {avg(or_r,'balanced'):.3f}  recall+ {avg(or_r,'recall',1):.3f}")
    print(f"     'exactly two'  (c==2)      raw {avg(sum_r,'raw'):.3f}  "
          f"balanced {avg(sum_r,'balanced'):.3f}  recall+ {avg(sum_r,'recall',1):.3f}")
    print(f"     2-vs-1 GIVEN evidence      raw {avg(two_r,'raw'):.3f}  "
          f"balanced {avg(two_r,'balanced'):.3f}  recall+ {avg(two_r,'recall',1):.3f}"
          "   <- the counting question")
    print(f"     3-class c2 recall (ref)    {float(np.mean(cnt_c2)):.3f}")
    gap = avg(or_r, "balanced") - avg(two_r, "balanced")
    print(f"     OR-minus-COUNT balanced gap {gap:+.3f}   "
          + ("<- OR-NOT-SUM: detection intact, counting is what fails"
             if gap > 0.05 else "<- no OR/SUM split; both behave alike"))

    if XL is None:
        return {"or": avg(or_r, "balanced"), "two": avg(two_r, "balanced"),
                "c2": float(np.mean(cnt_c2)), "cos_all": float("nan"),
                "d_cos": float("nan"), "d_marg": float("nan")}

    so, sb = np.array(sim_ok), np.array(sim_bad)
    mo, mb = np.array(marg_ok), np.array(marg_bad)
    print("\n2. AMPLITUDE test — where do the c2 failures live? (pooled over seeds)")
    print(f"     child-child cosine similarity   correct pairs {so.mean():.4f} "
          f"(n={len(so)})   FAILED pairs {sb.mean():.4f} (n={len(sb)})"
          f"   delta {sb.mean()-so.mean():+.4f}")
    print(f"     min child verdict margin        correct pairs {mo.mean():+.3f}   "
          f"FAILED pairs {mb.mean():+.3f}   delta {mb.mean()-mo.mean():+.3f}")
    print("     amplitude hypothesis predicts: failures at HIGHER similarity "
          "(delta > 0) and LOWER min-margin (delta < 0)")
    hits = []
    if len(sb) and sb.mean() > so.mean():
        hits.append("similarity")
    if len(mb) and mb.mean() < mo.mean():
        hits.append("margin")
    print(f"     -> matches on: {hits if hits else 'NEITHER — amplitude story unsupported'}")
    return {"or": avg(or_r, "balanced"), "two": avg(two_r, "balanced"),
            "c2": float(np.mean(cnt_c2)), "cos_all": float(np.concatenate([so, sb]).mean()),
            "d_cos": float(sb.mean() - so.mean()) if len(sb) else float("nan"),
            "d_marg": float(mb.mean() - mo.mean()) if len(mb) else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--label", default="A")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--compare-n", type=int, default=None)
    ap.add_argument("--compare-label", default="B")
    args = ap.parse_args()
    a = analyse(args.npz, args.n, args.layer, args.label)
    if args.compare:
        b = analyse(args.compare, args.compare_n or args.n, args.layer,
                    args.compare_label)
        print(f"\n{'='*76}\nCOMPARISON  {args.label}  vs  {args.compare_label}\n{'='*76}")
        for k, nm in (("or", "OR balanced (c>=1)"), ("two", "COUNT balanced (2-vs-1)"),
                      ("c2", "c2 recall"),
                      ("cos_all", "mean child-child cosine"),
                      ("d_cos", "cosine delta (fail-correct)"),
                      ("d_marg", "margin delta (fail-correct)")):
            print(f"  {nm:<30} {a[k]:>8.4f}   {b[k]:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

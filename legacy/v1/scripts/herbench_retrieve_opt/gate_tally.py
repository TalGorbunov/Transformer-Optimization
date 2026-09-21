#!/usr/bin/env python3
"""Gate->tally count exact-match from saved probe messages (CPU, no GPU).

The carrier method's readout: a logistic GATE over per-unit carrier messages predicts
occurrence(1)/not(0); TALLY = sum of gate decisions over a question's units -> predicted
count. Measures count-EM/MAE directly, pooled and per-verb, with group(question)-disjoint
train/test splits so a question's own units never straddle the fold.

Construction caveat: each question here = K positive clips + ~K matched negatives (pos:neg
~1:1), i.e. N≈2K units. This is DENSER in positives than armB (K evidence / 16 frames), so
the measured EM is OPTIMISTIC vs the sparse HERBench setting. We also print the codebase
adequacy-law EM at the measured per-unit d' for N=16 (the sparse reference).

Reads outputs/herbench_retrieve_opt/probe/<arm>/<ts>/messages.npz. Usage:
  python scripts/herbench_retrieve_opt/gate_tally.py --arm C_d1_r672 [--arm B_d1_r448]
"""
from __future__ import annotations
import argparse, glob, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


def law_em(dp, N):
    return max(2 * norm.cdf(dp / (2 * np.sqrt(N))) - 1.0, 0.0)


def dprime_auc(scores, y):
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(y, scores)
    return np.sqrt(2) * norm.ppf(min(max(a, 1e-4), 1 - 1e-4)), a


def run_arm(arm: str, base: Path, layers=(16, 18, 20), offsets=(0, 1, 2, 3),
            seeds=(0, 1, 2, 3, 4)):
    p = sorted(glob.glob(str(base / arm / "2*" / "messages.npz")))[-1]
    d = np.load(p, allow_pickle=True)
    y = d["y"].astype(int); groups = d["groups"]; verbs = d["verbs"]
    uniq = np.unique(groups)
    qverb = {g: verbs[groups == g][0] for g in uniq}
    qgold = {g: int((y[groups == g] == 1).sum()) for g in uniq}   # true count = #pos units

    # pick the (layer,offset) feature with best pooled held-out gate AUROC (quick 1-split)
    best = None
    rng0 = np.random.RandomState(0); perm = rng0.permutation(len(uniq))
    trG = set(uniq[perm[: int(0.6 * len(uniq))]].tolist())
    tr = np.array([i for i in range(len(y)) if groups[i] in trG])
    te = np.array([i for i in range(len(y)) if groups[i] not in trG])
    for L in layers:
        for o in offsets:
            X = d[f"L{L}_o{o}"]
            clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
            _, auc = dprime_auc(clf.predict_proba(X[te])[:, 1], y[te])
            if best is None or auc > best[0]:
                best = (auc, L, o)
    _, L, o = best
    X = d[f"L{L}_o{o}"]

    # group-split gate -> tally, averaged over seeds
    ems, maes, dps = [], [], []
    per_verb_em = defaultdict(list)
    biases = []
    for s in seeds:
        rng = np.random.RandomState(s); perm = rng.permutation(len(uniq))
        trG = set(uniq[perm[: int(0.6 * len(uniq))]].tolist())
        tr = np.array([i for i in range(len(y)) if groups[i] in trG])
        te = np.array([i for i in range(len(y)) if groups[i] not in trG])
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        pu = clf.predict(X[te])                     # per-unit gate decision on test units
        dp, _ = dprime_auc(clf.predict_proba(X[te])[:, 1], y[te]); dps.append(dp)
        teG = [g for g in uniq if g not in trG]
        q_em, q_ae, q_bias = [], [], []
        by_verb = defaultdict(list)
        idx_of = {}
        for gi, g in enumerate(teG):
            mask = groups[te] == g
            pred = int(pu[mask].sum()); gold = qgold[g]
            ok = int(pred == gold)
            q_em.append(ok); q_ae.append(abs(pred - gold)); q_bias.append(pred - gold)
            by_verb[qverb[g]].append(ok)
        ems.append(np.mean(q_em)); maes.append(np.mean(q_ae)); biases.append(np.mean(q_bias))
        for v, oks in by_verb.items():
            per_verb_em[v].append(np.mean(oks))

    dbar = float(np.mean(dps))
    counts = list(qgold.values())
    res = {"arm": arm, "feature": f"L{L}_o{o}", "n_questions": len(uniq),
           "per_unit_dprime": round(dbar, 3),
           "count_EM": round(float(np.mean(ems)), 3), "EM_std": round(float(np.std(ems)), 3),
           "MAE": round(float(np.mean(maes)), 3), "bias": round(float(np.mean(biases)), 3),
           "law_EM_N16": round(law_em(dbar, 16), 3),
           "law_EM_N8": round(law_em(dbar, 8), 3),
           "median_count": int(np.median(counts)),
           "per_verb_EM": {v: round(float(np.mean(e)), 3) for v, e in sorted(per_verb_em.items())
                           if len([g for g in uniq if qverb[g] == v]) >= 8}}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=None,
                    help="repeatable; default C_d1_r672 + B_d1_r448")
    ap.add_argument("--probe-base", default="outputs/herbench_retrieve_opt/probe")
    ap.add_argument("--out", default="outputs/herbench_retrieve_opt/phase3_gate_tally")
    args = ap.parse_args()
    arms = args.arm or ["C_d1_r672", "B_d1_r448", "C_d2_r672"]
    base = Path(args.probe_base)
    results = [run_arm(a, base) for a in arms]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "gate_tally.json").write_text(json.dumps(results, indent=2))
    print("=== GATE->TALLY count exact-match (frozen carrier gate; frozen native EM=0.049) ===")
    print(f"{'arm':14s} {'feat':8s} {'d′/unit':>8} {'count-EM':>9} {'MAE':>6} {'bias':>6} "
          f"{'lawN16':>7} {'lawN8':>7}")
    for r in results:
        print(f"{r['arm']:14s} {r['feature']:8s} {r['per_unit_dprime']:>8.2f} "
              f"{r['count_EM']:>9.3f} {r['MAE']:>6.2f} {r['bias']:>6.2f} "
              f"{r['law_EM_N16']:>7.3f} {r['law_EM_N8']:>7.3f}")
        print(f"    per-verb count-EM: {r['per_verb_EM']}  (median true count {r['median_count']})")
    print(f"\nwrote {out/'gate_tally.json'}")
    print("NOTE: pos:neg ~1:1 here (N≈2·count) → EM optimistic vs sparse armB (K/16); "
          "law_EM_N16 is the sparse reference at the measured per-unit d′.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

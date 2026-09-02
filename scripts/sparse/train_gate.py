#!/usr/bin/env python3
"""SPARSE S2 — the model gate: logistic regression on captured L20 replica-slot
states (capture_verdicts.py output) -> per-frame evidence yes/no.

Train on the S1 training-split captures; evaluate on per-N exam captures (gate
accuracy, per-class recall/precision, d' per N — H-S2 band: >= 0.99 per frame at
every N). Saves gate.npz (w, b, layer) for the two-forward gated decode.

Probe-family lesson (2026-08-10): readout must match label structure — labels here
are per-frame binary (the natural unit), and we report per-class + per-N, never a
pooled headline alone.

Usage:
  python scripts/sparse/train_gate.py --train-npz <cap_train/verdicts.npz> \
      --eval-npz N8=<capN8/verdicts.npz> --eval-npz N16=... --output outputs/sparse/s2/gate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


def dprime(y, score):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    s = np.sqrt(0.5 * (pos.var(ddof=1) + neg.var(ddof=1)))
    return float((pos.mean() - neg.mean()) / max(s, 1e-9))


def hit_fa_dprime(y, pred):
    hit = float(((pred == 1) & (y == 1)).sum()) / max(1, (y == 1).sum())
    fa = float(((pred == 1) & (y == 0)).sum()) / max(1, (y == 0).sum())
    eps = 0.5 / max(1, len(y))
    return float(norm.ppf(np.clip(hit, eps, 1 - eps)) - norm.ppf(np.clip(fa, eps, 1 - eps)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-npz", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="held-out SAMPLE fraction of the training captures")
    ap.add_argument("--eval-npz", action="append", default=[],
                    help="NAME=path/verdicts.npz (per-N exam captures)")
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    d = np.load(args.train_npz)
    X, y, sidx = d["X"].astype(np.float32), d["y"].astype(int), d["sample_idx"]
    rng = np.random.default_rng(args.seed)
    samples = np.unique(sidx)
    rng.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val_s = set(samples[:n_val].tolist())
    va = np.array([s in val_s for s in sidx])
    tr = ~va
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=args.c).fit(Xn[tr], y[tr])
    lines = [f"train frames {tr.sum()} (pos {y[tr].mean():.4f}) "
             f"val frames {va.sum()} (pos {y[va].mean():.4f})"]
    for name, m in (("train", tr), ("val", va)):
        pred = clf.predict(Xn[m])
        sc = clf.decision_function(Xn[m])
        acc = float((pred == y[m]).mean())
        rec = float((pred[y[m] == 1] == 1).mean()) if (y[m] == 1).any() else float("nan")
        spec = float((pred[y[m] == 0] == 0).mean()) if (y[m] == 0).any() else float("nan")
        lines.append(f"{name}: acc {acc:.4f} recall(evid) {rec:.4f} "
                     f"specificity {spec:.4f} d'(score) {dprime(y[m], sc):.2f} "
                     f"d'(rate) {hit_fa_dprime(y[m], pred):.2f}")
    rows = ["cell,frames,pos_frac,acc,recall,specificity,dprime_score,dprime_rate"]
    for spec_ in args.eval_npz:
        name, path = spec_.split("=", 1)
        de = np.load(path)
        Xe = (de["X"].astype(np.float32) - mu) / sd
        ye = de["y"].astype(int)
        pred = clf.predict(Xe)
        sc = clf.decision_function(Xe)
        acc = float((pred == ye).mean())
        rec = float((pred[ye == 1] == 1).mean()) if (ye == 1).any() else float("nan")
        spc = float((pred[ye == 0] == 0).mean()) if (ye == 0).any() else float("nan")
        lines.append(f"EVAL {name}: frames {len(ye)} pos {ye.mean():.4f} acc {acc:.4f} "
                     f"recall {rec:.4f} specificity {spc:.4f} "
                     f"d'(score) {dprime(ye, sc):.2f} d'(rate) {hit_fa_dprime(ye, pred):.2f}")
        rows.append(f"{name},{len(ye)},{ye.mean():.4f},{acc:.4f},{rec:.4f},{spc:.4f},"
                    f"{dprime(ye, sc):.2f},{hit_fa_dprime(ye, pred):.2f}")
    np.savez(out / "gate.npz", w=clf.coef_[0].astype(np.float32),
             b=np.float32(clf.intercept_[0]), mu=mu.astype(np.float32),
             sd=sd.astype(np.float32), layer=d["layer"])
    (out / "eval.csv").write_text("\n".join(rows) + "\n")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""D4 (CPU): is the per-frame fact linearly readable at the <|vision_end|> slot? Fits a logistic
regression on gate_capture.npz files (z-scored features, held out BY SAMPLE) and reports per-qtype,
per-N held-out accuracy / AUC / recall per (arm, layer). Twin: legacy/v1/scripts/sparse/train_gate.py
(LR on L20 replica-slot states, >=0.999/frame on park). Also cross-N transfer (--train-runs /
--test-runs) — the gate is trained at seq_len <= 16 and must hold at 32..128.
Usage:
  python experiments/gate_fit.py --runs outputs/diag/gate/N8/*/2*/ outputs/diag/gate/N16/*/2*/ --out outputs/diag/fits/gate_d4.json
  python experiments/gate_fit.py --train-runs outputs/diag/gate/N{8,16}/fenced_qfirst/2*/ --test-runs outputs/diag/gate/N{32,64,128}/fenced_qfirst/2*/ --out ...
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load(paths: List[str]):
    caps = []
    for pat in paths:
        for d in sorted(glob.glob(pat)):
            f = Path(d) / "capture.npz"
            m = Path(d) / "meta.json"
            if f.exists() and m.exists():
                z = np.load(f, allow_pickle=False)
                caps.append((json.loads(m.read_text()), z))
    if not caps:
        raise SystemExit(f"no capture.npz under {paths}")
    return caps


def _stack(caps, arm):
    xs, meta = [], defaultdict(list)
    for m, z in caps:
        if m["arm"] != arm:
            continue
        xs.append(z["X"].astype(np.float32))
        for k in ("qid", "qtype", "N", "frame", "is_evid"):
            meta[k].append(z[k])
        layers = z["layers"].tolist()
    if not xs:
        return None, None, None
    return np.concatenate(xs), {k: np.concatenate(v) for k, v in meta.items()}, layers


def _fit_eval(Xtr, ytr, Xte, C=1.0):
    from sklearn.linear_model import LogisticRegression

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=3000, C=C)
    clf.fit((Xtr - mu) / sd, ytr)
    return clf.decision_function((Xte - mu) / sd), clf


def _scores(y, s):
    from sklearn.metrics import roc_auc_score

    pred = (s > 0).astype(int)
    acc = float((pred == y).mean()) if len(y) else float("nan")
    rec = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    spec = float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    try:
        auc = float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "recall": rec, "specificity": spec, "auc": auc, "n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else float("nan")}


def _table(y, s, meta, idx):
    out = {"all": _scores(y[idx], s[idx])}
    for qt in sorted(set(meta["qtype"][idx].tolist())):
        sel = idx[meta["qtype"][idx] == qt]
        out[qt] = _scores(y[sel], s[sel])
        for N in sorted(set(meta["N"][sel].tolist())):
            sel2 = sel[meta["N"][sel] == N]
            out[f"{qt}@N{N}"] = _scores(y[sel2], s[sel2])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="run dirs (globs) for the held-out-by-sample fit")
    ap.add_argument("--train-runs", nargs="*", default=None)
    ap.add_argument("--test-runs", nargs="*", default=None)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--per-qtype", action="store_true", help="also fit one classifier per qtype")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    results: Dict[str, Any] = {"mode": "heldout" if args.runs else "transfer", "cells": []}
    if args.runs:
        caps = _load(args.runs)
        arms = sorted({m["arm"] for m, _ in caps})
        for arm in arms:
            X, meta, layers = _stack(caps, arm)
            y = meta["is_evid"].astype(int)
            samples = np.unique(meta["qid"])
            rng.shuffle(samples)
            n_val = max(1, int(round(args.val_frac * len(samples))))
            val = set(samples[:n_val].tolist())
            te = np.array([i for i in range(len(y)) if meta["qid"][i] in val])
            tr = np.array([i for i in range(len(y)) if meta["qid"][i] not in val])
            for li, L in enumerate(layers):
                s = np.zeros(len(y))
                s[te], _ = _fit_eval(X[tr, li], y[tr], X[te, li], args.C)
                cell = {"arm": arm, "layer": int(L), "fit": "pooled", "train_frames": int(len(tr)), "test": _table(y, s, meta, te)}
                results["cells"].append(cell)
                print(f"{arm:14s} L{L:<3d} pooled  held-out acc {cell['test']['all']['acc']:.4f} auc {cell['test']['all']['auc']:.4f} n={cell['test']['all']['n']}")
                if args.per_qtype:
                    per = {}
                    for qt in sorted(set(meta["qtype"].tolist())):
                        trq = tr[meta["qtype"][tr] == qt]; teq = te[meta["qtype"][te] == qt]
                        if len(set(y[trq].tolist())) < 2 or len(teq) == 0:
                            continue
                        sq, _ = _fit_eval(X[trq, li], y[trq], X[teq, li], args.C)
                        per[qt] = _scores(y[teq], sq)
                    results["cells"].append({"arm": arm, "layer": int(L), "fit": "per-qtype", "test": per})
    else:
        caps_tr, caps_te = _load(args.train_runs), _load(args.test_runs)
        for arm in sorted({m["arm"] for m, _ in caps_tr}):
            Xtr, mtr, layers = _stack(caps_tr, arm)
            Xte, mte, _ = _stack(caps_te, arm)
            if Xte is None:
                continue
            for li, L in enumerate(layers):
                s, _ = _fit_eval(Xtr[:, li], mtr["is_evid"].astype(int), Xte[:, li], args.C)
                idx = np.arange(len(s))
                cell = {"arm": arm, "layer": int(L), "fit": "transfer", "train_frames": int(len(Xtr)),
                        "test": _table(mte["is_evid"].astype(int), s, mte, idx)}
                results["cells"].append(cell)
                print(f"{arm:14s} L{L:<3d} transfer acc {cell['test']['all']['acc']:.4f} auc {cell['test']['all']['auc']:.4f} n={cell['test']['all']['n']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    with open(args.out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "layer", "fit", "cell", "acc", "recall", "specificity", "auc", "n", "pos_rate"])
        for c in results["cells"]:
            for name, sc in c["test"].items():
                w.writerow([c["arm"], c["layer"], c["fit"], name] + [f"{sc[k]:.4f}" if isinstance(sc[k], float) else sc[k]
                                                                    for k in ("acc", "recall", "specificity", "auc", "n", "pos_rate")])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

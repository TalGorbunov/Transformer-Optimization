#!/usr/bin/env python3
"""D4 / SCALEUP S2 (CPU): is "this block holds an evidence frame" linearly readable at the slot?
Fits a logistic regression on gate_capture.npz files (z-scored features, held out BY GROUP: the
`group` array = video_id on real video, qid on MMReD — questions of one video share frames, so a
qid split would leak frame identity) and reports per-qtype, per-N held-out accuracy, balanced
accuracy, AUC, recall and specificity per (arm, layer). Twin: legacy/v1/scripts/sparse/train_gate.py
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
import warnings

warnings.filterwarnings("ignore", "Mean of empty slice")


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


def _stack(caps, arm, features="slot"):
    """features: slot = X (the block's end token), pool = P (mean over the block's image tokens),
    both = [X | P] concatenated per layer. The judge logits J (if captured) ride along in meta."""
    xs, meta = [], defaultdict(list)
    for m, z in caps:
        if m["arm"] != arm:
            continue
        if features == "slot":
            F = z["X"]
        elif features == "pool":
            if "P" not in z.files:
                raise SystemExit(f"--features pool needs a capture made with --pool ({m.get('split_name')})")
            F = z["P"]
        else:
            F = np.concatenate([z["X"], z["P"]], axis=2)
        xs.append(F)                                       # float16 on disk; cast per layer slice in _fit_eval
        for k in ("qid", "qtype", "N", "frame", "is_evid"):
            meta[k].append(z[k])
        meta["group"].append(z["group"] if "group" in z.files else z["qid"])   # pre-seam captures: group = qid
        if "J" in z.files:
            meta["J"].append(z["J"]); meta["judge_words"] = m.get("judge_words")
        layers = z["layers"].tolist()
    if not xs:
        return None, None, None
    out = {k: np.concatenate(v) for k, v in meta.items() if k != "judge_words"}
    if "judge_words" in meta:
        out["judge_words"] = meta["judge_words"]
    return np.concatenate(xs), out, layers


def _judge_score(meta):
    """Training-free per-block score: log P(yes) - log P(no) over the judge's first answer token
    (logsumexp over the spelling variants), from the J array of a --judge capture."""
    J = meta["J"].astype(np.float64)
    words = meta["judge_words"]
    yes = [i for i, w in enumerate(words) if w.strip().lower() == "yes"]
    no = [i for i, w in enumerate(words) if w.strip().lower() == "no"]
    lse = lambda a: np.log(np.exp(a - a.max(1, keepdims=True)).sum(1)) + a.max(1)
    return lse(J[:, yes]) - lse(J[:, no])


def _fit_eval(Xtr, ytr, Xte, C=1.0):
    from sklearn.linear_model import LogisticRegression

    # Captures are stored in float16; a dimension beyond its range (Gemma 3 has one at depth > 0.8)
    # is saved as inf. Read it as the format's largest value: a saturated feature, not a missing frame.
    Xtr = np.nan_to_num(Xtr.astype(np.float32), nan=0.0, posinf=65504.0, neginf=-65504.0)
    Xte = np.nan_to_num(Xte.astype(np.float32), nan=0.0, posinf=65504.0, neginf=-65504.0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=3000, C=C)
    clf.fit((Xtr - mu) / sd, ytr)
    return clf.decision_function((Xte - mu) / sd), clf


def _position_features(meta):
    """What position alone knows: the frame's relative index (cubic), per task. The baseline every
    slot-state AUC is read against — planted evidence is not uniform in time, and an unfenced
    frame's state carries its position."""
    rel = meta["frame"].astype(np.float32) / np.maximum(meta["N"].astype(np.float32) - 1.0, 1.0)
    poly = np.stack([np.ones_like(rel), rel, rel ** 2, rel ** 3], axis=1)
    tasks = sorted(set(meta["qtype"].tolist()))
    onehot = np.stack([(meta["qtype"] == t).astype(np.float32) for t in tasks], axis=1)
    return (onehot[:, :, None] * poly[:, None, :]).reshape(len(rel), -1)


METRICS = ("acc", "bal_acc", "recall", "precision", "specificity", "auc", "recall@spec95", "precision@spec95",
           "recall@spec99", "precision@spec99", "n", "pos_rate")


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
    bal = float(np.nanmean([rec, spec])) if not (np.isnan(rec) and np.isnan(spec)) else float("nan")
    prec = float(y[pred == 1].mean()) if (pred == 1).any() else float("nan")
    out = {"acc": acc, "bal_acc": bal, "recall": rec, "precision": prec, "specificity": spec, "auc": auc, "n": int(len(y)),
           "pos_rate": float(y.mean()) if len(y) else float("nan")}
    # Operating points a gate would run at when positives are rare: the threshold that lets through
    # 5 % / 1 % of the negatives, and what it then keeps of the positives (recall) and how clean the
    # kept set is (precision). AUC alone hides this.
    for tag, q in (("spec95", 0.95), ("spec99", 0.99)):
        if (y == 0).sum() >= 20 and (y == 1).any():
            thr = float(np.quantile(s[y == 0], q))
            keep = s > thr
            out[f"recall@{tag}"] = float(keep[y == 1].mean())
            out[f"precision@{tag}"] = float(y[keep].mean()) if keep.any() else float("nan")
        else:
            out[f"recall@{tag}"] = out[f"precision@{tag}"] = float("nan")
    return out


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
    ap.add_argument("--group-by", choices=["group", "qid"], default="group", help="held-out unit (group = video on real video)")
    ap.add_argument("--folds", type=int, default=1, help="> 1: group K-fold over --group-by (every video held out once)")
    ap.add_argument("--train-n", nargs="*", type=int, default=None, help="with --folds: train only on frames from these N; test on every N")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--per-qtype", action="store_true", help="also fit one classifier per qtype")
    ap.add_argument("--features", choices=["slot", "pool", "both"], default="slot", help="readout: block end token / mean of image tokens / both")
    ap.add_argument("--per-task-fold", action="store_true", help="with --folds: fit one classifier per qtype (its own frames only) instead of one pooled classifier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    results: Dict[str, Any] = {"mode": "heldout" if args.runs else "transfer", "cells": []}
    if args.runs and args.folds > 1:
        # Group K-fold: every group (video) is held out once; the classifier is trained on the other
        # groups' frames at N in --train-n only and tested on the held-out groups at EVERY N. On real
        # video this is the only leak-free form of "train short, test long": planted evidence frames
        # are the same image files at every N, so the split must be by video, not by N.
        results["mode"] = f"group-{args.folds}fold"
        results["train_n"] = args.train_n
        results["features"] = args.features
        caps = _load(args.runs)
        for arm in sorted({m["arm"] for m, _ in caps}):
            X, meta, layers = _stack(caps, arm, args.features)
            y = meta["is_evid"].astype(int)
            key = meta["group"] if args.group_by == "group" else meta["qid"]
            groups = np.unique(key)
            rng.shuffle(groups)
            fold_of = {g: i % args.folds for i, g in enumerate(groups.tolist())}
            fold = np.array([fold_of[g] for g in key.tolist()])
            in_train_n = np.isin(meta["N"], args.train_n) if args.train_n else np.ones(len(y), bool)
            P = _position_features(meta)[:, None, :]             # pseudo-layer -1 = position only
            # pseudo-layer -2 = the per-block judge (no training; scored on the same held-out folds)
            pseudo = [(-1, -1)] + ([(-2, -2)] if "J" in meta else [])
            for li, L in pseudo + list(enumerate(layers)):
                F = P if li == -1 else X
                per_fold = []
                for k in range(args.folds):
                    tr = np.where((fold != k) & in_train_n)[0]
                    te = np.where(fold == k)[0]
                    if len(set(y[tr].tolist())) < 2 or len(te) == 0:
                        continue
                    s = np.zeros(len(y))
                    if li == -2:
                        s[te] = _judge_score(meta)[te]
                    elif args.per_task_fold:
                        for qt in sorted(set(meta["qtype"][te].tolist())):
                            trq = tr[meta["qtype"][tr] == qt]; teq = te[meta["qtype"][te] == qt]
                            if len(set(y[trq].tolist())) < 2 or len(teq) == 0:
                                s[teq] = np.nan
                                continue
                            s[teq], _ = _fit_eval(F[trq, max(li, 0)], y[trq], F[teq, max(li, 0)], args.C)
                        te = te[~np.isnan(s[te])]
                        if len(te) == 0:
                            continue
                    else:
                        s[te], _ = _fit_eval(F[tr, max(li, 0)], y[tr], F[te, max(li, 0)], args.C)
                    per_fold.append(_table(y, s, meta, te))
                names = sorted({n for t in per_fold for n in t})
                agg = {}
                for n in names:
                    vals = [t[n] for t in per_fold if n in t]
                    agg[n] = {m: float(np.nanmean([v[m] for v in vals])) for m in METRICS if m != "n"}
                    agg[n]["auc_sd"] = float(np.nanstd([v["auc"] for v in vals]))
                    agg[n]["n"] = int(sum(v["n"] for v in vals))
                    agg[n]["folds"] = len(vals)
                tag = {-1: "POS", -2: "JUDGE"}.get(L, f"L{L}")
                cell = {"arm": arm, "layer": int(L), "fit": f"group-{args.folds}fold" + ("-pertask" if args.per_task_fold and li >= 0 else ""),
                        "features": "judge" if li == -2 else ("position" if li == -1 else args.features), "train_n": args.train_n,
                        "groups": int(len(groups)), "test": agg}
                results["cells"].append(cell)
                a = agg.get("all", {})
                print(f"{arm:14s} {tag:<5s} {cell['features']:8s} {args.folds}-fold by {args.group_by} (train N={args.train_n or 'all'}): "
                      f"acc {a.get('acc', float('nan')):.4f} bal {a.get('bal_acc', float('nan')):.4f} "
                      f"auc {a.get('auc', float('nan')):.4f} +- {a.get('auc_sd', float('nan')):.4f} "
                      f"recall@spec95 {a.get('recall@spec95', float('nan')):.3f} prec@spec95 {a.get('precision@spec95', float('nan')):.3f} n={a.get('n')}", flush=True)
    elif args.runs:
        caps = _load(args.runs)
        arms = sorted({m["arm"] for m, _ in caps})
        for arm in arms:
            X, meta, layers = _stack(caps, arm)
            y = meta["is_evid"].astype(int)
            key = meta["group"] if args.group_by == "group" else meta["qid"]
            samples = np.unique(key)
            rng.shuffle(samples)
            n_val = max(1, int(round(args.val_frac * len(samples))))
            val = set(samples[:n_val].tolist())
            te = np.array([i for i in range(len(y)) if key[i] in val])
            tr = np.array([i for i in range(len(y)) if key[i] not in val])
            for li, L in enumerate(layers):
                s = np.zeros(len(y))
                s[te], _ = _fit_eval(X[tr, li], y[tr], X[te, li], args.C)
                cell = {"arm": arm, "layer": int(L), "fit": "pooled", "train_frames": int(len(tr)), "test": _table(y, s, meta, te)}
                results["cells"].append(cell)
                print(f"{arm:14s} L{L:<3d} pooled  held-out acc {cell['test']['all']['acc']:.4f} bal {cell['test']['all']['bal_acc']:.4f} auc {cell['test']['all']['auc']:.4f} n={cell['test']['all']['n']} (by {args.group_by})")
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
        w.writerow(["arm", "layer", "fit", "features", "cell"] + list(METRICS))
        for c in results["cells"]:
            for name, sc in c["test"].items():
                w.writerow([c["arm"], c["layer"], c["fit"], c.get("features", "slot"), name]
                           + [(f"{sc[k]:.4f}" if isinstance(sc.get(k), float) else sc.get(k, "")) for k in METRICS])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

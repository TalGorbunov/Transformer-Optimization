#!/usr/bin/env python3
"""B2: per-frame gate calibration vs N (dilution test) — CPU on the B1 caches.

The deployed tally solution is a per-frame gate (is this frame evidence?) summed over frames.
B1 caches carry per-frame carrier messages msg_f AND the per-frame attention mass
m_f = sum_{j in f} A[c,j]. Under joint attention the softmax renormalizes over ~N*196 visual
tokens, so raw msg_f magnitudes dilute ~1/N — a gate THRESHOLD fit at N=8 should drift
(FN inflation) even if the direction stays valid. Mass-normalized messages msg_f / m_f
(attention-weighted mean) remove the dilution; fenced messages should too.

Registered predictions (plan 2026-07-08 B): raw FN inflates with N; mass-norm + fenced ~flat;
tally bias(g) ~= N*FP - g*(FN+FP).

Inputs: --train-cache (N=8 joint), --eval-caches (comma N=path list), optional fenced pair.
Gate = logistic regression (fit at N=8, frozen: direction AND threshold). Reports per
(arm, layer, N): FN, FP, AUC (threshold-free), tally exact/MAE/bias vs predicted bias.
Outputs: results.csv, report.txt, fig_b2.png under --output.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_cache(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def feats(c, L, o, kind):
    X = np.asarray(c["msgs"][L][o], dtype=np.float32)          # [n, NF, H]
    if kind == "massnorm":
        m = np.asarray(c["mass"][L][o], dtype=np.float32)      # [n, NF]
        X = X / np.clip(m, 1e-6, None)[:, :, None]
    return X


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True, help="N=8 joint messages_cache.pt")
    ap.add_argument("--eval-caches", required=True,
                    help="comma list N=path (joint caches incl. N=8 holdout)")
    ap.add_argument("--train-cache-fenced", default="")
    ap.add_argument("--eval-caches-fenced", default="")
    ap.add_argument("--layers", default="14,16")
    ap.add_argument("--offset", type=int, default=9)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    o = int(args.offset)

    def parse_list(s):
        d = {}
        for part in s.split(","):
            if part.strip():
                k, v = part.split("=", 1)
                d[int(k)] = v
        return d

    eval_joint = parse_list(args.eval_caches)
    eval_fenced = parse_list(args.eval_caches_fenced) if args.eval_caches_fenced else {}

    arms = [("raw", args.train_cache, eval_joint, "raw"),
            ("massnorm", args.train_cache, eval_joint, "massnorm")]
    if args.train_cache_fenced and eval_fenced:
        arms.append(("fenced", args.train_cache_fenced, eval_fenced, "raw"))

    rows = ["arm,layer,N,n_samples,fn,fp,auc,tally_exact,tally_mae,tally_bias,pred_bias"]
    lines = [f"=== B2 GATE CALIBRATION vs N  (offset -{o}, gate = logistic fit at N=8, frozen) ==="]
    for arm, train_path, eval_map, kind in arms:
        ctr = load_cache(train_path)
        for L in layers:
            Xtr = feats(ctr, L, o, kind)
            ytr = np.asarray(ctr["labels"], dtype=int)
            n8 = Xtr.shape[0]
            rng = np.random.RandomState(0)
            perm = rng.permutation(n8)
            ntr = int(args.train_frac * n8)
            tr = perm[:ntr]
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr[tr].reshape(-1, Xtr.shape[-1]), ytr[tr].reshape(-1))
            lines.append(f"\n--- arm={arm} L{L} (gate trained on {ntr} samples x 8 frames) ---")
            for N in sorted(eval_map):
                ce = load_cache(eval_map[N])
                Xe = feats(ce, L, o, kind)
                ye = np.asarray(ce["labels"], dtype=int)
                gold = np.asarray(ce["gold"], dtype=int)
                if N == 8 and eval_map[N] == train_path:   # holdout split
                    te = perm[ntr:]
                    Xe, ye, gold = Xe[te], ye[te], gold[te]
                nf = Xe.shape[1]
                Xf = Xe.reshape(-1, Xe.shape[-1]); yf = ye.reshape(-1)
                pred = clf.predict(Xf)
                prob = clf.predict_proba(Xf)[:, 1]
                ev = yf == 1
                fn = float((pred[ev] == 0).mean()) if ev.any() else float("nan")
                fp = float((pred[~ev] == 1).mean()) if (~ev).any() else float("nan")
                try:
                    auc = float(roc_auc_score(yf, prob))
                except ValueError:
                    auc = float("nan")
                tally = pred.reshape(ye.shape).sum(1)
                ex = float((tally == gold).mean())
                mae = float(np.abs(tally - gold).mean())
                bias = float((tally - gold).mean())
                pred_bias = float(np.mean(nf * fp - gold * (fn + fp)))
                lines.append(f"  N={N:<4d} n={len(gold):<4d} FN={fn:.3f} FP={fp:.3f} "
                             f"AUC={auc:.3f}  tally: exact={ex:.3f} MAE={mae:.2f} "
                             f"bias={bias:+.2f} (pred {pred_bias:+.2f})")
                rows.append(f"{arm},{L},{N},{len(gold)},{fn:.4f},{fp:.4f},{auc:.4f},"
                            f"{ex:.4f},{mae:.4f},{bias:.4f},{pred_bias:.4f}")

    (out / "results.csv").write_text("\n".join(rows) + "\n")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import csv as _csv
        data = list(_csv.DictReader((out / "results.csv").open()))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        colors = {"raw": "#c53030", "massnorm": "#2b6cb0", "fenced": "#2f855a"}
        Lp = max(layers)
        for arm in sorted({r["arm"] for r in data}):
            rr = sorted((int(r["N"]), float(r["fn"]), float(r["fp"])) for r in data
                        if r["arm"] == arm and int(r["layer"]) == Lp)
            if not rr:
                continue
            Ns = [x[0] for x in rr]
            axes[0].plot(Ns, [x[1] for x in rr], "o-", color=colors.get(arm), label=f"{arm} FN")
            axes[0].plot(Ns, [x[2] for x in rr], "s--", color=colors.get(arm), alpha=0.5,
                         label=f"{arm} FP")
            rr2 = sorted((int(r["N"]), float(r["tally_bias"])) for r in data
                         if r["arm"] == arm and int(r["layer"]) == Lp)
            axes[1].plot([x[0] for x in rr2], [x[1] for x in rr2], "o-",
                         color=colors.get(arm), label=arm)
        axes[0].set_xscale("log", base=2); axes[1].set_xscale("log", base=2)
        axes[0].set_xlabel("N"); axes[0].set_ylabel("rate"); axes[0].legend(fontsize=7)
        axes[0].set_title(f"Gate FN/FP vs N (train@8 frozen, L{Lp})")
        axes[1].set_xlabel("N"); axes[1].set_ylabel("tally bias"); axes[1].legend(fontsize=8)
        axes[1].axhline(0, color="gray", lw=0.5)
        axes[1].set_title("Tally bias vs N")
        fig.tight_layout(); fig.savefig(out / "fig_b2.png", dpi=130)
    except Exception as e:
        print("plot skipped:", e)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

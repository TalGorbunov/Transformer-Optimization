#!/usr/bin/env python3
"""B1 analysis: deployed carrier d′ vs N per arm (joint / fenced / multipass), CPU.

Loads the B1 messages caches (arm × N), computes the room-token (default off −9) whitened d′
(shrinkage-LDA held-out, 3 sample-disjoint splits — same estimator as block_read_completeness)
per layer, plus the zero-parameter law prediction 2Φ(d′/2√N)−1 (prior-mixed, boundary-aware)
at each (d′, N). Writes results.csv + fig_b1.png (d′ vs N, crush line 6.3, law panel).

Usage: --caches "joint:8=path,joint:16=path,...,multipass:128=path" --output DIR
"""
from __future__ import annotations
import argparse
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

ND = NormalDist()


def dprime_pair(X, y, seeds=(0, 1, 2), max_lda=4000):
    """X [n,NF,H], y [n,NF] -> (d'_w mean, std, d'_auc mean) held-out, group-split."""
    n, NF, H = X.shape
    yf = y.reshape(-1).astype(int)
    samp = np.repeat(np.arange(n), NF)
    dws, das = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        perm = rng.permutation(n)
        tr_s = set(perm[: int(0.6 * n)].tolist())
        trf = np.array([i for i in range(len(yf)) if samp[i] in tr_s])
        tef = np.array([i for i in range(len(yf)) if samp[i] not in tr_s])
        Xf = X.reshape(-1, H).astype(np.float64)
        sub = rng.permutation(len(trf))[:max_lda]
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(Xf[trf][sub], yf[trf][sub])
        w = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-12)
        p = Xf[tef] @ w
        yt = yf[tef]
        pE, pN = p[yt == 1], p[yt == 0]
        dws.append(abs(pE.mean() - pN.mean()) / (0.5 * (pE.std() + pN.std()) + 1e-12))
        try:
            auc = min(max(roc_auc_score(yt, p), 1e-4), 1 - 1e-4)
            das.append(np.sqrt(2) * ND.inv_cdf(auc))
        except ValueError:
            pass
    return float(np.mean(dws)), float(np.std(dws)), float(np.mean(das)) if das else float("nan")


def law_pred(dprime, N, gold):
    d_n = dprime / np.sqrt(N)
    p_int = max(2 * ND.cdf(d_n / 2.0) - 1.0, 0.0)
    p_bnd = ND.cdf(d_n / 2.0)
    return float(np.mean([p_bnd if g in (0, N) else p_int for g in gold]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", required=True, help="comma list arm:N=path")
    ap.add_argument("--layers", default="14,16")
    ap.add_argument("--offset", type=int, default=9)
    ap.add_argument("--crush", type=float, default=6.3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    o = int(args.offset)
    entries = []
    for part in args.caches.split(","):
        if not part.strip():
            continue
        key, path = part.split("=", 1)
        arm, N = key.split(":")
        entries.append((arm, int(N), path))

    rows = ["arm,N,layer,n,dprime_w,dprime_w_std,dprime_auc,law_pred_exact,model_acc"]
    lines = [f"=== B1 d' vs N (offset -{o}) ==="]
    for arm, N, path in sorted(entries):
        try:
            c = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            lines.append(f"  {arm} N={N}: LOAD FAILED {e}")
            continue
        y = np.asarray(c["labels"])
        gold = np.asarray(c["gold"])
        macc = float(np.mean(np.asarray(c["model_correct"])))
        for L in layers:
            if L not in c["msgs"] or o not in c["msgs"][L]:
                continue
            X = np.asarray(c["msgs"][L][o], dtype=np.float32)
            dw, ds, da = dprime_pair(X, y)
            lp = law_pred(dw, N, gold)
            lines.append(f"  {arm:<9s} N={N:<4d} L{L}: d'_w {dw:.2f}±{ds:.2f} (auc {da:.2f})  "
                         f"law-pred {lp:.3f}  model/mp-sum {macc:.3f}  n={len(gold)}")
            rows.append(f"{arm},{N},{L},{len(gold)},{dw:.4f},{ds:.4f},{da:.4f},{lp:.4f},{macc:.4f}")

    (out / "results.csv").write_text("\n".join(rows) + "\n")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import csv as _csv
        data = list(_csv.DictReader((out / "results.csv").open()))
        Lp = max(layers)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        colors = {"joint": "#c53030", "fenced": "#dd6b20", "multipass": "#2b6cb0"}
        for arm in sorted({r["arm"] for r in data}):
            rr = sorted((int(r["N"]), float(r["dprime_w"]), float(r["dprime_w_std"]),
                         float(r["law_pred_exact"]))
                        for r in data if r["arm"] == arm and int(r["layer"]) == Lp)
            if not rr:
                continue
            Ns = [x[0] for x in rr]
            axes[0].errorbar(Ns, [x[1] for x in rr], yerr=[x[2] for x in rr], marker="o",
                             capsize=3, color=colors.get(arm), label=arm)
            axes[1].plot(Ns, [x[3] for x in rr], "o-", color=colors.get(arm), label=arm)
        axes[0].axhline(args.crush, ls="--", c="gray", label=f"128-crush ({args.crush})")
        for ax in axes:
            ax.set_xscale("log", base=2); ax.set_xlabel("N (frames)"); ax.legend(fontsize=8)
        axes[0].set_ylabel("carrier d'_w (room token)"); axes[0].set_title(f"Fig B1 — d' vs N (L{Lp})")
        axes[1].set_ylabel("law-predicted exact match"); axes[1].set_title("law 2Φ(d'/2√N)−1 at each (d',N)")
        fig.tight_layout(); fig.savefig(out / "fig_b1.png", dpi=130)
        print("wrote fig_b1.png")
    except Exception as e:
        print("plot skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

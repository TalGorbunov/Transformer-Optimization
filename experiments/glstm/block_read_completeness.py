#!/usr/bin/env python3
"""Locus-completeness / block-read d' on distributed carriers (A1 follow-up, RESULTS [2026-07-08c]).

Pure CPU post-processing on a `messages_cache.pt` written by
`probe_frame_to_carrier_message.py --carrier per_token --decode-offsets ... --save-messages`.

Question: on TEXT-MMRED the frame->carrier evidence is spread over ~6 question tokens, so any
single-token whitened d' is a LOWER BOUND on the true supply. This script measures how much
separability the loci carry JOINTLY, and whether adding more loci keeps growing d' (distributed
carrier, previously mismeasured) or saturates near the single-token value (genuine write cap).

Per layer L, over 3 sample-disjoint 60/40 splits (group-split by sample, no frame leakage):
  (1) SINGLE-LOCUS d'   : shrinkage-LDA (Ledoit-Wolf) held-out; gap/width d'_w + sqrt(2)*Phi^-1(AUC).
  (2) BLOCK d'          : two estimators on the K concatenated loci —
        score-concat : per-locus held-out LDA score -> K-dim -> LDA (well-powered, honest lower bd)
        pca-concat   : joint PCA(256) on the K*hidden concat -> shrinkage LDA (captures cross-locus)
  (3) INCREMENTAL curve : greedily add loci ranked by single-locus d'; block d' (score-concat) vs K.
  (4) dtc (charitable)  : per-frame logistic on the block -> sum predictions -> round vs gold.
  (5) LAW PARITY        : predict 2*Phi(d'_block/2sqrt(N))-1 (prior-mixed) vs measured ridge-on-
                          (sum of block-projected scores) -> round. Zero fitted parameters.
  (6) E4 adequacy       : skew / excess-kurtosis of block matched-filter projections per class,
                          per-class std ratio (equal-covariance check). Same battery as the parity engine.

Outputs: results.csv, incremental.csv, report.txt, block_read.png under --output/<ts>/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Tuple

import numpy as np
import torch

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score

ND = NormalDist()


def phi(x: float) -> float:
    return ND.cdf(x)


def phi_inv(p: float) -> float:
    return ND.inv_cdf(min(max(p, 1e-6), 1 - 1e-6))


def dprime_gap(pE: np.ndarray, pN: np.ndarray) -> float:
    s = 0.5 * (pE.std() + pN.std())
    return float(abs(pE.mean() - pN.mean()) / (s + 1e-12))


def dprime_auc(scores: np.ndarray, y: np.ndarray) -> float:
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        return float("nan")
    auc = min(max(auc, 1e-4), 1 - 1e-4)
    return float(np.sqrt(2.0) * phi_inv(auc))


def predict_acc(dprime: float, N: int, gold_test: np.ndarray) -> float:
    """Closed-form exact-count accuracy of matched-filter + nearest-integer, prior-mixed."""
    d_n = dprime / np.sqrt(N)
    p_int = max(2 * phi(d_n / 2.0) - 1.0, 0.0)
    p_bnd = phi(d_n / 2.0)
    ps = [p_bnd if g in (0, N) else p_int for g in gold_test]
    return float(np.mean(ps)) if len(ps) else float("nan")


def moments(p: np.ndarray) -> Tuple[float, float]:
    z = (p - p.mean()) / (p.std() + 1e-12)
    return float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)


def lda_dir(Xtr: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(Xtr, ytr)
    w = lda.coef_[0].astype(np.float64)
    return w / (np.linalg.norm(w) + 1e-12)


def analyze_layer(L: int, msgs_L: Dict[int, np.ndarray], labels: np.ndarray, gold: np.ndarray,
                  offsets: List[int], seeds: List[int], max_lda: int, pca_dim: int) -> dict:
    """msgs_L[o] = [n_samples, NF, hidden]; labels = [n_samples, NF]; gold = [n_samples]."""
    n_samp, NF, H = msgs_L[offsets[0]].shape
    K = len(offsets)
    res: Dict[str, list] = {"single_dw": {o: [] for o in offsets}, "single_dauc": {o: [] for o in offsets},
                            "block_score_dw": [], "block_score_dauc": [], "block_pca_dw": [],
                            "dtc_acc": [], "pred_law": [], "meas_ridge": [], "model_acc": [],
                            "adq_skewE": [], "adq_kurtE": [], "adq_skewN": [], "adq_kurtN": [], "adq_stdratio": [],
                            "incr": []}
    # frame-flat labels
    yf = labels.reshape(-1).astype(int)                       # [n_samp*NF]
    samp_of_frame = np.repeat(np.arange(n_samp), NF)

    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_samp)
        n_tr = int(0.6 * n_samp)
        tr_s, te_s = set(perm[:n_tr].tolist()), set(perm[n_tr:].tolist())
        trf = np.array([i for i in range(len(yf)) if samp_of_frame[i] in tr_s])
        tef = np.array([i for i in range(len(yf)) if samp_of_frame[i] in te_s])
        ytr, yte = yf[trf], yf[tef]

        # ---- (1) single-locus d' + store held-out & train scores for score-concat ----
        s_tr = np.zeros((len(trf), K)); s_te = np.zeros((len(tef), K))
        single_dw = {}
        for j, o in enumerate(offsets):
            Xo = msgs_L[o].reshape(n_samp * NF, H).astype(np.float64)
            sub = rng.permutation(len(trf))[:max_lda]
            w = lda_dir(Xo[trf][sub], ytr[sub])
            ptr, pte = Xo[trf] @ w, Xo[tef] @ w
            s_tr[:, j], s_te[:, j] = ptr, pte
            dw = dprime_gap(pte[yte == 1], pte[yte == 0])
            da = dprime_auc(pte, yte)
            res["single_dw"][o].append(dw); res["single_dauc"][o].append(da)
            single_dw[o] = dw

        # ---- (2a) block d' — score-concat (two-stage) ----
        wsc = lda_dir(s_tr, ytr)
        bsc_tr, bsc_te = s_tr @ wsc, s_te @ wsc
        res["block_score_dw"].append(dprime_gap(bsc_te[yte == 1], bsc_te[yte == 0]))
        res["block_score_dauc"].append(dprime_auc(bsc_te, yte))

        # ---- (2b) block d' — joint PCA(pca_dim) on K*hidden concat ----
        Xcat = np.concatenate([msgs_L[o].reshape(n_samp * NF, H) for o in offsets], axis=1).astype(np.float32)
        pca = PCA(n_components=min(pca_dim, Xcat.shape[1], len(trf) - 1), svd_solver="randomized", random_state=seed)
        Zt = pca.fit_transform(Xcat[trf]); Ze = pca.transform(Xcat[tef])
        wp = lda_dir(Zt.astype(np.float64), ytr)
        bpc_te = Ze @ wp
        res["block_pca_dw"].append(dprime_gap(bpc_te[yte == 1], bpc_te[yte == 0]))

        # ---- E4 adequacy on the block matched-filter (score-concat) projections ----
        skE, kuE = moments(bsc_te[yte == 1]); skN, kuN = moments(bsc_te[yte == 0])
        res["adq_skewE"].append(skE); res["adq_kurtE"].append(kuE)
        res["adq_skewN"].append(skN); res["adq_kurtN"].append(kuN)
        res["adq_stdratio"].append(float(bsc_te[yte == 1].std() / (bsc_te[yte == 0].std() + 1e-12)))

        # ---- (3) incremental curve: greedy add by single-locus d' ----
        order = sorted(offsets, key=lambda o: single_dw[o], reverse=True)
        incr = []
        for k in range(1, K + 1):
            cols = [offsets.index(o) for o in order[:k]]
            wk = lda_dir(s_tr[:, cols], ytr)
            bk = s_te[:, cols] @ wk
            incr.append(dprime_gap(bk[yte == 1], bk[yte == 0]))
        res["incr"].append(incr)

        # ---- (4) dtc: per-frame logistic on the block (score features) -> sum -> round ----
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(s_tr, ytr)
        # aggregate predictions per TEST sample
        pred_frame = clf.predict(s_te)
        te_samp = samp_of_frame[tef]
        dtc_pred = {}
        for i, si in enumerate(te_samp):
            dtc_pred[si] = dtc_pred.get(si, 0) + int(pred_frame[i])
        te_list = sorted(te_s)
        dtc_acc = np.mean([1.0 if dtc_pred.get(si, 0) == gold[si] else 0.0 for si in te_list])
        res["dtc_acc"].append(float(dtc_acc))

        # ---- (5) law parity: predicted vs measured ridge-on-sum-of-block-scores ----
        gold_te = np.array([gold[si] for si in te_list])
        d_block = np.median([res["block_score_dw"][-1], res["block_score_dauc"][-1]])
        res["pred_law"].append(predict_acc(d_block, NF, gold_te))
        # measured: sum block score per sample -> ridge -> round
        sum_tr = {}; sum_te = {}
        for i, si in enumerate(samp_of_frame[trf]):
            sum_tr[si] = sum_tr.get(si, 0.0) + bsc_tr[i]
        for i, si in enumerate(te_samp):
            sum_te[si] = sum_te.get(si, 0.0) + bsc_te[i]
        tr_list = sorted(tr_s)
        Xr_tr = np.array([[sum_tr[si]] for si in tr_list]); yr_tr = np.array([gold[si] for si in tr_list])
        Xr_te = np.array([[sum_te[si]] for si in te_list])
        ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(Xr_tr, yr_tr)
        pr = np.clip(np.round(ridge.predict(Xr_te)), 0, NF)
        res["meas_ridge"].append(float(np.mean(pr == gold_te)))

    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="path to messages_cache.pt")
    ap.add_argument("--layers", default="", help="comma layers to analyze (default: all in cache)")
    ap.add_argument("--offsets", default="", help="comma offsets to use (default: all in cache)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-lda", type=int, default=4000)
    ap.add_argument("--pca-dim", type=int, default=256)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    c = torch.load(args.cache, map_location="cpu", weights_only=False)
    all_layers = c["layers"]; all_off = c["offsets"]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else all_layers
    layers = [L for L in layers if L in all_layers]
    offsets = [int(x) for x in args.offsets.split(",")] if args.offsets else all_off
    offsets = [o for o in offsets if o in all_off]
    seeds = [int(x) for x in args.seeds.split(",")]
    labels = c["labels"].astype(int) if not isinstance(c["labels"], np.ndarray) else c["labels"]
    labels = np.asarray(labels)
    gold = np.asarray(c["gold"]); NF = int(c["n_frames"])
    model_acc = float(np.mean(c["model_correct"])) if "model_correct" in c else float("nan")
    n_samp = len(gold)

    tokmap = {}  # offset -> decoded token if present (best-effort; not stored, so leave labels generic)
    lines = [f"=== BLOCK-READ / LOCUS-COMPLETENESS  (task={c['task']}, n_samples={n_samp}, N={NF}) ===",
             f"cache={args.cache}", f"layers={layers}  offsets(from-end)={offsets}  seeds={seeds}",
             f"model own-answer acc = {model_acc:.3f}", ""]
    rows = ["layer,quantity,offset_or_k,mean,std"]
    incr_rows = ["layer,K,block_dw_mean,block_dw_std,added_offset"]

    best = None
    for L in layers:
        msgs_L = {o: np.asarray(c["msgs"][L][o]) for o in offsets}
        r = analyze_layer(L, msgs_L, labels, gold, offsets, seeds, args.max_lda, args.pca_dim)

        def ms(key):
            v = np.array(r[key]); return float(v.mean()), float(v.std())

        lines.append(f"--- L{L} ---")
        sd = {o: (np.mean(r['single_dw'][o]), np.mean(r['single_dauc'][o])) for o in offsets}
        ranked = sorted(offsets, key=lambda o: sd[o][0], reverse=True)
        lines.append("  single-locus d'  (offset: d'_w / d'_auc):")
        for o in ranked:
            lines.append(f"     off -{o:<3d}: {sd[o][0]:.2f} / {sd[o][1]:.2f}")
            rows.append(f"{L},single_dw,{o},{np.mean(r['single_dw'][o]):.4f},{np.std(r['single_dw'][o]):.4f}")
            rows.append(f"{L},single_dauc,{o},{np.mean(r['single_dauc'][o]):.4f},{np.std(r['single_dauc'][o]):.4f}")
        bs_m, bs_s = ms("block_score_dw"); ba_m, _ = ms("block_score_dauc"); bp_m, bp_s = ms("block_pca_dw")
        best_single = max(sd[o][0] for o in offsets)
        lines.append(f"  BLOCK d'  score-concat = {bs_m:.2f}±{bs_s:.2f} (d'_auc {ba_m:.2f}) | "
                     f"pca{args.pca_dim}-concat = {bp_m:.2f}±{bp_s:.2f}   [best single = {best_single:.2f}]")
        gain = bs_m - best_single
        lines.append(f"    -> block gain over best single locus: {gain:+.2f} d'  "
                     f"({'COMPLEMENTARY (distributed carrier)' if gain > 0.4 else 'REDUNDANT (write-capped)'})")
        for k in ("block_score_dw", "block_score_dauc", "block_pca_dw"):
            m, s = ms(k); rows.append(f"{L},{k},all,{m:.4f},{s:.4f}")

        # incremental
        incr = np.array(r["incr"])   # [seeds, K]
        order = sorted(offsets, key=lambda o: sd[o][0], reverse=True)
        lines.append("  incremental block d' (greedy add by single d'):")
        for k in range(len(offsets)):
            m, s = float(incr[:, k].mean()), float(incr[:, k].std())
            added = order[k]
            lines.append(f"     K={k+1:<2d} (+off-{added:<3d}): d'_w {m:.2f}±{s:.2f}")
            incr_rows.append(f"{L},{k+1},{m:.4f},{s:.4f},{added}")

        dtc_m, dtc_s = ms("dtc_acc"); pl_m, _ = ms("pred_law"); mr_m, mr_s = ms("meas_ridge")
        lines.append(f"  LADDER @L{L}:  model {model_acc:.3f} < law-pred(block d') {pl_m:.3f} "
                     f"~ ridge-on-sum {mr_m:.3f}±{mr_s:.3f} < dtc(block) {dtc_m:.3f}±{dtc_s:.3f}")
        for k in ("dtc_acc", "pred_law", "meas_ridge"):
            m, s = ms(k); rows.append(f"{L},{k},all,{m:.4f},{s:.4f}")
        skE, _ = ms("adq_skewE"); kuE, _ = ms("adq_kurtE"); sr, _ = ms("adq_stdratio")
        adq_ok = abs(skE) <= 0.5 and abs(kuE) <= 1.0 and 0.7 <= sr <= 1.4
        lines.append(f"  E4 adequacy (block): skewE {skE:+.2f} kurtE {kuE:+.2f} std-ratio {sr:.2f} "
                     f"-> {'PASS (law licensed)' if adq_ok else 'FAIL (quote with caveat)'}")
        rows.append(f"{L},adq_skewE,all,{skE:.4f},nan"); rows.append(f"{L},adq_kurtE,all,{kuE:.4f},nan")
        rows.append(f"{L},adq_stdratio,all,{sr:.4f},nan")
        lines.append("")
        if best is None or bs_m > best[1]:
            best = (L, bs_m, best_single, dtc_m, pl_m, mr_m, adq_ok)

    if best:
        L, bd, bsng, dtc, pl, mr, ok = best
        lines.append("=== VERDICT ===")
        lines.append(f"Peak block d' = {bd:.2f} @L{L}  (best single locus {bsng:.2f}; gain {bd-bsng:+.2f}).")
        verdict = ("DISTRIBUTED carrier: single-token d-prime was an underestimate."
                   if bd - bsng > 0.4 else
                   "WRITE-CAPPED: combining loci does NOT recover much; single-token d-prime ~ true supply.")
        lines.append(f"  {verdict}")
        lines.append(f"  dtc(block) charitable ceiling = {dtc:.3f}; law@block predicts {pl:.3f} ~ measured {mr:.3f}; adequacy {'PASS' if ok else 'FAIL'}.")

    (out / "report.txt").write_text("\n".join(lines))
    (out / "results.csv").write_text("\n".join(rows))
    (out / "incremental.csv").write_text("\n".join(incr_rows))

    # plot: incremental d' vs K (best layer) + single-vs-block bars
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Lp = best[0]
        inc = np.loadtxt(out / "incremental.csv", delimiter=",", skiprows=1)
        inc = inc[inc[:, 0] == Lp]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.errorbar(inc[:, 1], inc[:, 2], yerr=inc[:, 3], marker="o", capsize=3, color="#2b6cb0")
        ax.axhline(best[2], ls="--", c="gray", label=f"best single locus ({best[2]:.2f})")
        ax.set_xlabel("# loci combined (greedy by single d')"); ax.set_ylabel("block whitened d'")
        ax.set_title(f"Locus completeness — {c['task']} L{Lp} (N={NF}, n={n_samp})")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "block_read.png", dpi=130)
    except Exception as e:
        print("plot skipped:", e)

    print("\n".join(lines))
    print(f"\n[done {time.time()-t0:.0f}s] -> {out}")


if __name__ == "__main__":
    main()

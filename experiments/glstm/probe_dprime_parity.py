#!/usr/bin/env python3
"""E1/E2/E3/E4 — zero-free-parameter validation of the d'/sqrt(N) aggregation law (RESULTS [2026-07-03b]).

Pure CPU post-processing on cached per-frame reps (cache_minimal_frame_reps.py format). No model, no GPU.

Per cache regime (binary-evidence tasks: steps_in_room / co_occupancy):
  E1 (the law): measure per-frame whitened discriminability d' three ways —
        d'_naive : along the raw class-mean-difference axis (the old "SNR"; known to understate)
        d'_w     : along the shrinkage-LDA (Ledoit-Wolf) direction, gap/std on HELD-OUT frames
        d'_auc   : sqrt(2)*Phi^-1(AUC) of the same held-out scores (threshold-free; agrees with d'_w iff ~Gaussian)
      then PREDICT exact-count accuracy of the best linear readout of the sum:
        P(exact | interior g) = 2*Phi(d'/(2*sqrt(N))) - 1 ;  g in {0,N}: Phi(d'/(2*sqrt(N)))
      mixed over the test count prior — and independently MEASURE it (ridge-on-sum -> round, held out).
      Parity = predicted vs measured, zero fitted parameters.
  E2 (iid check): cross-frame noise correlation rho along the readout direction (residuals after
      train-class-mean removal; mean off-diagonal of the frame-slot correlation matrix). Refined
      prediction uses sqrt(N*(1+(N-1)*rho)) in place of sqrt(N)  [information-limiting correlations].
  E3 (sufficiency): MLP-on-sum vs linear-on-sum — the model predicts NO MLP headroom.
  E4 (adequacy): skew/excess-kurtosis of held-out matched-filter projections per class + QQ points
      (Gaussianity), and per-class std ratio (equal-covariance check).

Outputs: results.csv, qq_<label>.csv, parity.png, README.md under --output/<ts>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.glstm.probe_message_sum_decodability import Example, load_cache  # noqa: E402

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression, RidgeCV  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ND = NormalDist()


def phi(x: float) -> float:
    return ND.cdf(x)


def phi_inv(p: float) -> float:
    return ND.inv_cdf(min(max(p, 1e-6), 1 - 1e-6))


def dprime_from_proj(pE: np.ndarray, pN: np.ndarray) -> float:
    s = 0.5 * (pE.std() + pN.std())
    return float(abs(pE.mean() - pN.mean()) / (s + 1e-12))


def predict_acc(dprime: float, N: int, gold_test: np.ndarray, rho: float = 0.0) -> float:
    """Closed-form exact-count accuracy of matched-filter + nearest-integer, mixed over the
    empirical count prior. rho>0 inflates the accumulated noise (information-limiting corr.)."""
    infl = max(1.0 + (N - 1) * rho, 0.05)
    d_n = dprime / np.sqrt(N * infl)
    p_int = max(2 * phi(d_n / 2.0) - 1.0, 0.0)
    p_bnd = phi(d_n / 2.0)
    ps = [p_bnd if g in (0, N) else p_int for g in gold_test]
    return float(np.mean(ps))


def frame_matrices(exs: List[Example]) -> tuple:
    X = np.concatenate([e.reps for e in exs])              # [n_ex*N, H]
    y = np.concatenate([e.labels for e in exs]).astype(int)
    return X, y


def cross_frame_rho(exs: List[Example], w: np.ndarray, mu_E: np.ndarray, mu_N: np.ndarray) -> float:
    """Mean off-diagonal correlation across frame slots of residual projections."""
    P = []
    for e in exs:
        mus = np.where(e.labels[:, None] == 1, mu_E[None, :], mu_N[None, :])
        P.append((e.reps - mus) @ w)
    P = np.stack(P)                                        # [n_ex, N]
    C = np.corrcoef(P.T)                                   # [N, N]
    off = C[~np.eye(C.shape[0], dtype=bool)]
    return float(np.nanmean(off))


def moments(p: np.ndarray) -> tuple:
    z = (p - p.mean()) / (p.std() + 1e-12)
    return float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)


def qq_points(p: np.ndarray, n: int = 99) -> np.ndarray:
    z = np.sort((p - p.mean()) / (p.std() + 1e-12))
    qs = (np.arange(1, n + 1)) / (n + 1)
    theo = np.array([phi_inv(q) for q in qs])
    emp = np.quantile(z, qs)
    return np.stack([theo, emp], axis=1)


def analyze(label: str, path: Path, seq_len: int, seeds: List[int], max_lda_frames: int,
            out_dir: Path) -> Optional[dict]:
    exs = load_cache(path, seq_len)
    return analyze_examples(label, exs, seeds, max_lda_frames, out_dir)


def analyze_examples(label: str, exs: List[Example], seeds: List[int], max_lda_frames: int,
                     out_dir: Path) -> Optional[dict]:
    if len(exs) < 60:
        print(f"  [{label}] only {len(exs)} usable examples; skipping")
        return None
    N = exs[0].reps.shape[0]
    rows = []
    qq_saved = False
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(exs))
        n_tr = int(0.6 * len(exs))
        tr = [exs[i] for i in idx[:n_tr]]
        te = [exs[i] for i in idx[n_tr:]]
        Xtr, ytr = frame_matrices(tr)
        Xte, yte = frame_matrices(te)
        mu_E, mu_N = Xtr[ytr == 1].mean(0), Xtr[ytr == 0].mean(0)

        # --- naive axis (the old "SNR") ---
        u = (mu_E - mu_N)
        u = u / (np.linalg.norm(u) + 1e-12)
        d_naive = dprime_from_proj(Xte[yte == 1] @ u, Xte[yte == 0] @ u)

        # --- whitened (shrinkage-LDA / Ledoit-Wolf) direction, evaluated held-out ---
        sub = rng.permutation(len(Xtr))[:max_lda_frames]
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(Xtr[sub], ytr[sub])
        w = lda.coef_[0].astype(np.float64)
        w = w / (np.linalg.norm(w) + 1e-12)
        pE, pN = Xte[yte == 1] @ w, Xte[yte == 0] @ w
        d_w = dprime_from_proj(pE, pN)
        scores = Xte @ w
        auc = roc_auc_score(yte, scores if pE.mean() > pN.mean() else -scores)
        d_auc = float(np.sqrt(2) * phi_inv(auc))
        balacc = balanced_accuracy_score(yte, lda.predict(Xte))

        # --- E2: cross-frame residual correlation along w ---
        rho = cross_frame_rho(te, w, mu_E, mu_N)

        # --- E4: adequacy along the matched filter ---
        skew_E, kurt_E = moments(pE)
        skew_N, kurt_N = moments(pN)
        std_ratio = float(pE.std() / (pN.std() + 1e-12))
        if not qq_saved:
            qq = np.concatenate([qq_points(pE), qq_points(pN)], axis=1)  # theoE,empE,theoN,empN
            np.savetxt(out_dir / f"qq_{label}.csv", qq, delimiter=",",
                       header="theo_E,emp_E,theo_N,emp_N", comments="")
            qq_saved = True

        # --- measured: linear (ridge->round) and logistic and MLP on the SUM, held out ---
        Str = np.stack([e.reps.sum(0) for e in tr]); gtr = np.array([e.gold for e in tr])
        Ste = np.stack([e.reps.sum(0) for e in te]); gte = np.array([e.gold for e in te])
        sc = StandardScaler().fit(Str)
        Ztr, Zte = sc.transform(Str), sc.transform(Ste)
        ridge = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(Ztr, gtr)
        pred_r = np.clip(np.round(ridge.predict(Zte)), 0, N).astype(int)
        acc_ridge = float(np.mean(pred_r == gte))
        logit = LogisticRegression(max_iter=3000, C=0.1).fit(Ztr, gtr)
        acc_logit = float(np.mean(logit.predict(Zte) == gte))
        mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=600, early_stopping=True,
                            random_state=seed).fit(Ztr, gtr)
        acc_mlp = float(np.mean(mlp.predict(Zte) == gte))

        # --- predictions (zero fitted parameters) ---
        row = dict(label=label, seed=seed, N=N, n_ex=len(exs),
                   d_naive=d_naive, d_w=d_w, d_auc=d_auc, frame_balacc=float(balacc), rho=rho,
                   pred_iid=predict_acc(d_w, N, gte), pred_rho=predict_acc(d_w, N, gte, rho=rho),
                   pred_naive=predict_acc(d_naive, N, gte),
                   acc_ridge=acc_ridge, acc_logit=acc_logit, acc_mlp=acc_mlp,
                   mlp_minus_linear=acc_mlp - max(acc_ridge, acc_logit),
                   skew_E=skew_E, kurt_E=kurt_E, skew_N=skew_N, kurt_N=kurt_N,
                   std_ratio=std_ratio, majority=float(np.mean(gte == np.bincount(gte).argmax())))
        rows.append(row)
        print(f"  [{label}] seed{seed}: d'_naive={d_naive:.2f} d'_w={d_w:.2f} d'_auc={d_auc:.2f} "
              f"rho={rho:+.3f} | pred_iid={row['pred_iid']:.3f} pred_rho={row['pred_rho']:.3f} "
              f"| meas ridge={acc_ridge:.3f} logit={acc_logit:.3f} mlp={acc_mlp:.3f}", flush=True)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], (int, float))}
    agg.update(label=label, N=N, n_ex=len(exs),
               acc_ridge_std=float(np.std([r["acc_ridge"] for r in rows])))
    return {"rows": rows, "agg": agg}


def analyze_kchannel(label: str, msgs: np.ndarray, labels_raw: list, gold: np.ndarray,
                     seeds: List[int], max_lda_frames: int, out_dir: Path) -> Optional[dict]:
    """Rooms-style K-channel parity: per-room one-vs-rest d'_r on per-frame messages, closed-form
    prediction of distinct-count accuracy for the threshold-per-channel (soft-OR) readout, measured
    against (a) ridge-linear-on-sum [structurally capped], (b) the theory's own hard threshold readout,
    (c) decode-then-count (per-frame multiclass LDA -> distinct), (d) MLP-on-sum [PREDICTED to beat
    linear here, unlike counting: support-size is nonlinear in the tallies]."""
    n, NF, H = msgs.shape
    lab = np.array([[str(x) for x in row] for row in labels_raw])   # [n, NF] strings; 'None' = absent
    rows = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n); ntr = int(0.6 * n)
        tr, te = idx[:ntr], idx[ntr:]
        Xtr = msgs[tr].reshape(-1, H); ytr = lab[tr].reshape(-1)
        Xte = msgs[te].reshape(-1, H); yte = lab[te].reshape(-1)
        room_list = sorted(c for c in set(ytr.tolist()) if c != "None")
        # per-room one-vs-rest whitened direction + held-out d'_r and per-frame threshold params
        dirs, dprimes, thr = {}, {}, {}
        for c in room_list:
            m = ytr == c
            if m.sum() < 30 or (~m).sum() < 30:
                continue
            sub = rng.permutation(len(Xtr))[:max_lda_frames]
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(Xtr[sub], (ytr[sub] == c).astype(int))
            w = lda.coef_[0].astype(np.float64); w /= (np.linalg.norm(w) + 1e-12)
            pE, pN = Xte[yte == c] @ w, Xte[yte != c] @ w
            if pE.mean() < pN.mean():
                w = -w; pE, pN = -pE, -pN
            dirs[c] = w
            dprimes[c] = dprime_from_proj(pE, pN)
            # per-frame step and base along w from TRAIN (for the hard threshold readout on the sum)
            muE = (Xtr[ytr == c] @ w).mean(); muN = (Xtr[ytr != c] @ w).mean()
            thr[c] = (muN, muE - muN)          # base per non-c frame, step per c frame
        if not dirs:
            print(f"  [{label}] no usable rooms; skipping"); return None
        # test-sample visit counts per room
        S_te = msgs[te].sum(1)                                       # [n_te, H]
        g_te = gold[te]
        nvis = {c: (lab[te] == c).sum(1) for c in dirs}              # [n_te] per room
        # --- closed-form prediction: independent per-channel presence detection, DP over channels ---
        preds = []
        for i in range(len(te)):
            probs = []
            for c in dirs:
                dN = dprimes[c] / np.sqrt(NF)
                k = nvis[c][i]
                p_detect = phi((k - 0.5) * dN) if k >= 1 else 1 - phi(0.5 * dN)  # P(channel says visited)
                probs.append((p_detect, k >= 1))
            # P(sum of channel-votes == true distinct)
            true_d = sum(1 for _, v in probs if v)
            dp = np.zeros(len(probs) + 1); dp[0] = 1.0
            for p, _ in probs:
                dp[1:] = dp[1:] * (1 - p) + dp[:-1] * p
                dp[0] *= (1 - p)
            preds.append(dp[true_d])
        pred_acc = float(np.mean(preds))
        # --- measured (a): ridge-linear on the sum ---
        sc = StandardScaler().fit(msgs[tr].sum(1))
        ridge = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(sc.transform(msgs[tr].sum(1)), gold[tr])
        acc_lin = float(np.mean(np.clip(np.round(ridge.predict(sc.transform(S_te))), 0, len(dirs)) == g_te))
        # --- measured (b): the theory's own hard threshold-per-channel readout (no fit on the sum) ---
        hard = np.zeros(len(te))
        for c in dirs:
            base, step = thr[c]
            score = S_te @ dirs[c] - NF * base                      # ≈ n_c * step + noise
            hard += (score > 0.5 * step).astype(float)
        acc_hard = float(np.mean(hard == g_te))
        # --- measured (c): decode-then-count (per-frame multiclass LDA -> #unique non-None) ---
        sub = rng.permutation(len(Xtr))[:max_lda_frames]
        mlda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr[sub], ytr[sub])
        pf = mlda.predict(Xte).reshape(len(te), NF)
        dtc = np.array([len(set(r.tolist()) - {"None"}) for r in pf])
        acc_dtc = float(np.mean(dtc == g_te))
        # --- measured (d): MLP on the sum (predicted to BEAT linear here) ---
        mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=600, early_stopping=True,
                            random_state=seed).fit(sc.transform(msgs[tr].sum(1)), gold[tr])
        acc_mlp = float(np.mean(mlp.predict(sc.transform(S_te)) == g_te))
        row = dict(label=label, seed=seed, N=NF, n_ex=n, n_rooms=len(dirs),
                   dprime_mean=float(np.mean(list(dprimes.values()))),
                   dprime_min=float(np.min(list(dprimes.values()))),
                   dprime_max=float(np.max(list(dprimes.values()))),
                   pred_threshold=pred_acc, acc_linear=acc_lin, acc_hard_threshold=acc_hard,
                   acc_dtc=acc_dtc, acc_mlp=acc_mlp,
                   majority=float(np.mean(g_te == np.bincount(g_te).argmax())))
        rows.append(row)
        print(f"  [{label}] seed{seed}: rooms={len(dirs)} d'_r mean={row['dprime_mean']:.2f} "
              f"[{row['dprime_min']:.2f}..{row['dprime_max']:.2f}] | pred(thresh)={pred_acc:.3f} | "
              f"lin={acc_lin:.3f} hard={acc_hard:.3f} dtc={acc_dtc:.3f} mlp={acc_mlp:.3f} "
              f"maj={row['majority']:.3f}  d'_r: " +
              " ".join(f"{c}:{dprimes[c]:.2f}" for c in dirs), flush=True)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], (int, float))}
    agg.update(label=label)
    return {"rows": rows, "agg": agg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", default=[],
                    help="spec: label:seq_len:path  (cache_minimal_frame_reps format, binary labels)")
    ap.add_argument("--carrier-caches", nargs="+", default=[],
                    help="spec: label:layer:offset:path — messages_cache.pt from "
                         "probe_frame_to_carrier_message.py --save-messages (deployed locus)")
    ap.add_argument("--kchannel-caches", nargs="+", default=[],
                    help="spec: label:layer:offset:path — messages_cache.pt with labels_raw "
                         "(rooms_visited): K-channel distinct-count parity")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-lda-frames", type=int, default=4000)
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "outputs" / "frame_axis" / "probes" / "dprime_parity")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, aggs = [], []
    for spec in args.caches:
        label, sl, path = spec.split(":", 2)
        print(f"== {label} (seq_len={sl}) {path}", flush=True)
        res = analyze(label, Path(path), int(sl), seeds, args.max_lda_frames, out_dir)
        if res:
            all_rows.extend(res["rows"]); aggs.append(res["agg"])
    for spec in args.carrier_caches:
        label, L, off, path = spec.split(":", 3)
        print(f"== {label} (carrier msgs L{L} off{off}) {path}", flush=True)
        import torch
        c = torch.load(Path(path), map_location="cpu", weights_only=False)  # our own cache (numpy arrays)
        msgs = c["msgs"][int(L)][int(off)]                     # [n, NF, H] float16
        exs = [Example(msgs[i].astype(np.float32), c["labels"][i].astype(int),
                       int(c["gold"][i]), msgs[i].sum(0).astype(np.float32))
               for i in range(len(c["gold"]))]
        res = analyze_examples(label, exs, seeds, args.max_lda_frames, out_dir)
        if res:
            all_rows.extend(res["rows"]); aggs.append(res["agg"])
    kch_rows = []
    for spec in args.kchannel_caches:
        label, L, off, path = spec.split(":", 3)
        print(f"== {label} (K-channel msgs L{L} off{off}) {path}", flush=True)
        import torch
        c = torch.load(Path(path), map_location="cpu", weights_only=False)
        msgs = c["msgs"][int(L)][int(off)].astype(np.float32)
        res = analyze_kchannel(label, msgs, c["labels_raw"], c["gold"], seeds,
                               args.max_lda_frames, out_dir)
        if res:
            kch_rows.extend(res["rows"])
    if kch_rows:
        keys = list(kch_rows[0])
        with open(out_dir / "kchannel_results.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for r in kch_rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    if not aggs and not kch_rows:
        print("no usable caches"); return 1
    if not aggs:
        print(f"\nwrote {out_dir} (K-channel only)"); return 0
    keys = list(all_rows[0])
    with open(out_dir / "results.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, pk, title in [(axes[0], "pred_iid", "iid prediction  2Φ(d'/2√N)−1"),
                              (axes[1], "pred_rho", "ρ-refined prediction")]:
            for a in aggs:
                ax.errorbar(a[pk], a["acc_ridge"], yerr=a["acc_ridge_std"], fmt="o", ms=7, capsize=3)
                ax.annotate(f"{a['label']}(N={a['N']})", (a[pk], a["acc_ridge"]), fontsize=7,
                            xytext=(4, 4), textcoords="offset points")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("predicted exact-count acc (zero fitted params)")
            ax.set_ylabel("measured ridge-on-sum acc (held out)")
            ax.set_title(title); ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
        fig.tight_layout(); fig.savefig(out_dir / "parity.png", dpi=150)

    lines = ["# d'-parity validation (E1/E2/E3/E4 of RESULTS [2026-07-03b])", "",
             "| label | N | n | d'_naive | d'_w | d'_auc | frame balacc | rho | pred_iid | pred_rho | "
             "meas ridge | meas logit | meas MLP | MLP−linear | skewE | kurtE | stdE/stdN |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in aggs:
        lines.append(
            f"| {a['label']} | {a['N']} | {a['n_ex']} | {a['d_naive']:.2f} | {a['d_w']:.2f} | "
            f"{a['d_auc']:.2f} | {a['frame_balacc']:.3f} | {a['rho']:+.3f} | {a['pred_iid']:.3f} | "
            f"{a['pred_rho']:.3f} | {a['acc_ridge']:.3f}±{a['acc_ridge_std']:.3f} | {a['acc_logit']:.3f} | "
            f"{a['acc_mlp']:.3f} | {a['mlp_minus_linear']:+.3f} | {a['skew_E']:+.2f} | {a['kurt_E']:+.2f} | "
            f"{a['std_ratio']:.2f} |")
    lines += ["", "Readout: parity holds if measured ≈ predicted (diagonal in parity.png) with the "
              "ρ-refined column at least as close as iid. MLP−linear ≈ 0 is the sufficiency signature "
              "(E3). |skew|,|kurt| small and stdE/stdN ≈ 1 certify the Gaussian equal-covariance model "
              "(E4). d'_w vs d'_auc agreement is a second Gaussianity check.",
              f"caches: {args.caches}", f"seeds: {seeds}"]
    (out_dir / "README.md").write_text("\n".join(lines))
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

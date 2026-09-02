#!/usr/bin/env python3
"""ARMOR Exp A — assemble the Hahn figure + verdicts from probe_hahn run dirs.

Panels (the theory-to-model figure):
  (a) log-log single-frame sensitivity vs N: joint final-position ||dh|| (plain
      arm, L20/L28) with fitted slope alpha, the matched answer-preserving ctrl
      flip, the measured bf16 reduction floor (fenced perm control), and the
      fenced per-frame supply curve (rep_t) that should NOT decay.
  (b) A2: digit-logit margin at the answer position vs N (median +- IQR), with
      joint accuracy-vs-N and the margin noise floor (|margin jitter| under the
      answer-preserving ctrl flip); the accuracy-chance crossing N marked.

Pre-registered bands (CAMPAIGN_BRIEF): joint alpha in [0.7,1.3]; fenced
|alpha| < 0.2; margin crosses its floor within 2x of the accuracy-chance N.

Usage:
  python scripts/armor/fig_hahn.py --runs outputs/armor/hahn/<ts>_N* \
      --output outputs/armor/hahn/<ts>_fig
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANCE = 1.0 / 9.0     # argmax over digits, gold ~uniform on 0..8


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fit_slope(ns, meds):
    """log-log LS fit Delta ~ N^-alpha -> (alpha, r2)."""
    x, y = np.log(np.array(ns, float)), np.log(np.array(meds, float))
    A = np.vstack([x, np.ones_like(x)]).T
    (m, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ [m, b]
    ss = 1 - ((y - yhat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return -m, float(ss)


def boot_slope(ns, groups, iters=2000, seed=0):
    """bootstrap CI on alpha by resampling pairs within each N."""
    rng = np.random.default_rng(seed)
    alphas = []
    for _ in range(iters):
        meds = [np.median(rng.choice(g, size=len(g))) for g in groups]
        if min(meds) <= 0:
            continue
        a, _ = fit_slope(ns, meds)
        alphas.append(a)
    return np.percentile(alphas, [2.5, 97.5])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(set(sum([glob.glob(r) for r in args.runs], [])))
    pairs, logits, controls = [], [], []
    for rd in run_dirs:
        pairs += read_csv(Path(rd) / "pairs.csv")
        logits += read_csv(Path(rd) / "logits.csv")
        controls += read_csv(Path(rd) / "controls.csv")
    print(f"{len(run_dirs)} runs: {len(pairs)} pair rows, {len(logits)} logit "
          f"rows, {len(controls)} control rows")

    # dnorm samples per (arm, layer, locus, N)
    d = defaultdict(list)
    for r in pairs:
        d[(r["arm"], int(r["layer"]), r["locus"], int(r["n_frames"]))].append(
            float(r["dnorm"]))
    dc = defaultdict(list)
    for r in controls:
        dc[(r["kind"], r["arm"], int(r["layer"]), r["locus"],
            int(r["n_frames"]))].append(float(r["dnorm"]))
    ns = sorted({int(r["n_frames"]) for r in pairs})

    # ------------------------------------------------------------- slope table
    slopes = {}
    rows_csv = []
    for key in [("plain", 20, "final"), ("plain", 28, "final"),
                ("plain", 16, "final"),
                ("repjoint", 20, "final"), ("repjoint", 28, "final"),
                ("repjoint", 20, "rep_t"), ("repjoint", 28, "rep_t"),
                ("fenced", 16, "rep_t"), ("fenced", 20, "rep_t"),
                ("fenced", 28, "rep_t")]:
        arm, L, loc = key
        groups = [np.array(d[(arm, L, loc, n)]) for n in ns
                  if d[(arm, L, loc, n)]]
        ns_k = [n for n in ns if d[(arm, L, loc, n)]]
        if len(ns_k) < 3:
            continue
        meds = [float(np.median(g)) for g in groups]
        a, r2 = fit_slope(ns_k, meds)
        lo, hi = boot_slope(ns_k, groups)
        slopes[key] = {"alpha": a, "r2": r2, "ci": [float(lo), float(hi)],
                       "medians": dict(zip(ns_k, meds))}
        for n, m in zip(ns_k, meds):
            rows_csv.append({"arm": arm, "layer": L, "locus": loc, "N": n,
                             "median_dnorm": m})

    # ---------------------------------------------------------------- A2 table
    marg = defaultdict(list)
    acc = defaultdict(list)
    for r in logits:
        if r["flip_kind"] != "evid":
            continue
        n = int(r["n_frames"])
        marg[n].append(float(r["margin_base"]))
        acc[n].append(int(int(r["pred_base"]) == int(r["gold"])))
    mfloor = defaultdict(list)   # margin jitter under answer-preserving flip
    for r in logits:
        if r["flip_kind"] == "ctrl":
            mfloor[int(r["n_frames"])].append(
                abs(float(r["margin_flip"]) - float(r["margin_base"])))

    # accuracy chance crossing: first N with acc <= CHANCE + 1.96*se
    n_chance = None
    for n in ns:
        a_ = np.array(acc[n], float)
        se = a_.std() / max(len(a_) ** 0.5, 1)
        if a_.mean() <= CHANCE + 1.96 * se:
            n_chance = n
            break
    # margin-floor crossing: first N with median margin <= median floor
    n_cross = None
    for n in ns:
        if mfloor[n] and np.median(marg[n]) <= np.median(mfloor[n]):
            n_cross = n
            break

    # ------------------------------------------------------------------ verdicts
    a_plain = {f"{k[0]}_L{k[1]}_{k[2]}": v for k, v in slopes.items()}
    v_joint = {k: v for k, v in a_plain.items()
               if k.startswith("plain") and not k.endswith("L16_final")}
    joint_ok = {k: bool(0.7 <= v["alpha"] <= 1.3) for k, v in v_joint.items()}
    fen = {k: v for k, v in a_plain.items() if k.startswith("fenced")}
    fen_ok = {k: bool(abs(v["alpha"]) < 0.2) for k, v in fen.items()}
    if n_cross is not None and n_chance is not None:
        band_a2 = bool(0.5 <= n_cross / n_chance <= 2.0)
    else:
        band_a2 = None
    verdict = {"slopes": {k: {"alpha": float(v["alpha"]), "r2": float(v["r2"]),
                              "ci": v["ci"],
                              "medians": {int(n): float(m)
                                          for n, m in v["medians"].items()}}
                          for k, v in a_plain.items()},
               "joint_band_0.7-1.3": joint_ok,
               "fenced_band_|a|<0.2": fen_ok,
               "n_margin_cross_floor": n_cross,
               "n_accuracy_chance": n_chance,
               "a2_band_within_2x": band_a2,
               "acc_by_n": {n: float(np.mean(acc[n])) for n in ns},
               "margin_median_by_n": {n: float(np.median(marg[n])) for n in ns},
               "margin_floor_by_n": {n: (float(np.median(mfloor[n]))
                                         if mfloor[n] else None) for n in ns}}

    # -------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    styles = {("plain", 20, "final"): ("tab:red", "o-", "joint answer-pos ‖Δh‖ L20"),
              ("plain", 28, "final"): ("tab:orange", "s-", "joint answer-pos ‖Δh‖ L28"),
              ("fenced", 20, "rep_t"): ("tab:blue", "o-", "fenced per-frame supply L20"),
              ("fenced", 16, "rep_t"): ("tab:cyan", "s--", "fenced per-frame supply L16")}
    for key, (c, st, lab) in styles.items():
        if key not in slopes:
            continue
        v = slopes[key]
        nsk = sorted(v["medians"])
        med = [v["medians"][n] for n in nsk]
        q1 = [np.percentile(d[(key[0], key[1], key[2], n)], 25) for n in nsk]
        q3 = [np.percentile(d[(key[0], key[1], key[2], n)], 75) for n in nsk]
        ax.plot(nsk, med, st, color=c,
                label=f"{lab}  (α={v['alpha']:.2f} [{v['ci'][0]:.2f},{v['ci'][1]:.2f}])")
        ax.fill_between(nsk, q1, q3, color=c, alpha=0.15)
    ctrl_meds = [np.median(dc[("ctrl", "plain", 20, "final", n)]) for n in ns
                 if dc[("ctrl", "plain", 20, "final", n)]]
    ctrl_ns = [n for n in ns if dc[("ctrl", "plain", 20, "final", n)]]
    if ctrl_ns:
        ax.plot(ctrl_ns, ctrl_meds, "^:", color="tab:pink",
                label="answer-preserving flip (ctrl) L20")
    perm_meds = [np.median(dc[("perm", "fenced", 20, "final", n)]) for n in ns
                 if dc[("perm", "fenced", 20, "final", n)]]
    perm_ns = [n for n in ns if dc[("perm", "fenced", 20, "final", n)]]
    if perm_ns:
        ax.plot(perm_ns, perm_meds, "--", color="gray",
                label="measured bf16 reduction floor L20")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (frames)")
    ax.set_ylabel("median ‖Δ hidden state‖ (one frame flipped)")
    ax.set_title("(a) Hahn O(1/N): single-frame sensitivity, joint vs fenced")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    med_m = [np.median(marg[n]) for n in ns]
    q1m = [np.percentile(marg[n], 25) for n in ns]
    q3m = [np.percentile(marg[n], 75) for n in ns]
    ax.plot(ns, med_m, "o-", color="tab:red", label="margin logit(gold) − max other")
    ax.fill_between(ns, q1m, q3m, color="tab:red", alpha=0.15)
    fl_ns = [n for n in ns if mfloor[n]]
    if fl_ns:
        ax.plot(fl_ns, [np.median(mfloor[n]) for n in fl_ns], "--", color="gray",
                label="margin noise floor (ctrl-flip jitter)")
    ax.axhline(0, color="k", lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(ns, [np.mean(acc[n]) for n in ns], "s-", color="tab:blue",
             label="joint accuracy (digit argmax)")
    ax2.axhline(CHANCE, color="tab:blue", ls=":", lw=1, label="chance (1/9)")
    ax2.set_ylabel("accuracy", color="tab:blue")
    ax2.set_ylim(0, 1)
    if n_chance:
        ax.axvline(n_chance, color="tab:blue", ls=":", alpha=0.6)
    if n_cross:
        ax.axvline(n_cross, color="gray", ls=":", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (frames)")
    ax.set_ylabel("answer-position digit margin")
    ax.set_title("(b) margin collapse vs accuracy collapse")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "hahn_figure.png", dpi=160)

    # full grid: every (arm, layer, locus) median curve
    fig2, ax = plt.subplots(figsize=(9, 6))
    for key, v in slopes.items():
        nsk = sorted(v["medians"])
        ax.plot(nsk, [v["medians"][n] for n in nsk], "o-",
                label=f"{key[0]} L{key[1]} {key[2]} (α={v['alpha']:.2f})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("all loci/layers: median ‖Δh‖ vs N")
    fig2.tight_layout()
    fig2.savefig(out / "hahn_grid.png", dpi=160)

    with open(out / "medians.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "layer", "locus", "N",
                                           "median_dnorm"])
        w.writeheader()
        w.writerows(rows_csv)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2)[:3000])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

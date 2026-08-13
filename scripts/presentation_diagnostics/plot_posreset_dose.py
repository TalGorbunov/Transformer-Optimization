#!/usr/bin/env python3
"""Posreset dose-response, gate-accuracy edition (Tal: acc, not d').

Loads the 10 posreset_sweep caches (N=4,8,16,32,64 x {reset, noreset} — Q-first block
fence in both arms, only --reset-positions differs), fits the standard held-out
gate->tally (5 seeds, L16) on each, and plots exact-count accuracy vs N for the two
arms. The gap is the POSITION TAX in task currency; the logged d' version is the
[2026-07-27] dose-response (reset 8.05/9.70/12.06/12.57 vs no-reset 7.77/8.45/9.05/8.51
at N=2/4/16/32 — different probe batches, same direction).

Usage: python plot_posreset_dose.py [<output_dir>]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

# d' cross-check comes from each run's own report.txt (probe_supply prints it);
# no dprime_pair here — the shrinkage-LDA is too slow for a login-node fit.

SWEEP = Path("outputs/presentation/posreset_sweep")
NS = [4, 8, 16, 32, 64]
ARM_COLOR = {"reset": "#2a78d6", "noreset": "#eb6834"}
ARM_LABEL = {"reset": "with position reset", "noreset": "without position reset"}


def gate_fit(cache_path, layer=16, seeds=5):
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    X = np.asarray(c["rep"][layer], dtype=np.float32)
    Y = np.asarray(c["labels"], dtype=int)
    G = np.asarray(c["gold"], dtype=int)
    n, NF, H = X.shape
    accs, ferrs = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        tr, ev = idx[: n // 2], idx[n // 2:]
        clf = LogisticRegression(max_iter=2000).fit(X[tr].reshape(-1, H), Y[tr].reshape(-1))
        pr = clf.predict(X[ev].reshape(-1, H)).reshape(len(ev), NF)
        ferrs.append(float((pr != Y[ev]).mean()))
        accs.append(float((pr.sum(1) == G[ev]).mean()))
    return (float(np.mean(accs)), float(np.std(accs)), float(np.mean(ferrs)), float("nan"), n)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               "outputs/presentation/posreset_dose/local")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    res = {}
    for n_frames in NS:
        for arm in ("reset", "noreset"):
            caches = sorted((SWEEP / f"N{n_frames}_{arm}").glob("2*/messages_cache.pt"))
            if not caches:
                print(f"[warn] missing cache N{n_frames}_{arm}")
                continue
            acc, sd, ferr, dp, n = gate_fit(caches[-1])
            res[(n_frames, arm)] = (acc, sd)
            rows.append([n_frames, arm, f"{acc:.4f}", f"{sd:.4f}", f"{ferr:.4f}",
                         f"{dp:.2f}", n, str(caches[-1].parent)])
            print(f"N={n_frames} {arm}: tally {acc:.3f}±{sd:.3f} ferr {ferr:.4f} d' {dp:.2f}")
    with open(out / "posreset_dose.csv", "w", newline="") as f:
        csv.writer(f).writerows([["N", "arm", "tally", "std", "ferr", "dprime", "n", "run"],
                                 *rows])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for arm in ("reset", "noreset"):
        pts = [(n, *res[(n, arm)]) for n in NS if (n, arm) in res]
        xs, ac, sd = zip(*pts)
        ax.errorbar(xs, ac, yerr=sd, color=ARM_COLOR[arm], marker="o", ms=6, lw=2,
                    capsize=3, label=ARM_LABEL[arm])
        ax.annotate(ARM_LABEL[arm].split(" position")[0], (xs[-1], ac[-1]),
                    xytext=(6, 0), textcoords="offset points", color=ARM_COLOR[arm],
                    fontsize=9, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS, NS)
    ax.set_xlim(3.5, 130)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("N frames in context")
    ax.set_ylabel("exact-count accuracy (gate→tally, held-out)")
    ax.set_title("The position tax: same fence, only the position IDs differ", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e6e5e2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"posreset_dose.{ext}", dpi=300)
    (out / "ABOUT.md").write_text(
        "# Position-tax dose-response — gate->tally accuracy vs N, with/without posreset\n\n"
        "Ten fresh A3-config probe cells (Q-first + block fence in BOTH arms; the ONLY\n"
        "difference is --reset-positions), N=4..64, park/longN_park, L16 read, standard\n"
        "held-out gate->tally. The widening gap is the RoPE position tax in task currency\n"
        "— every island identical in position space (reset) vs islands drifting far from\n"
        "the question (no reset). Complements the d' dose-response logged [2026-07-27]\n"
        "(monotone growth, late-copy decay fingerprint) and the behavioral collapse cells\n"
        "(caption winner without reset: N=32 0.313, N=64 0.000). Sources: run dirs in\n"
        "posreset_dose.csv (jobs 127736/127737).\n"
        "Artifacts: posreset_dose.png/pdf, posreset_dose.csv.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

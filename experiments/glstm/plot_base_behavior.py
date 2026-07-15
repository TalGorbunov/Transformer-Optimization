#!/usr/bin/env python3
"""Two base-model behavior plots (steps task) from the B3 runs — all numbers trace to
outputs/ladder/image_longN/behavior/N{8,16,32,64,128}/*/{summary,rows}.json.
  (1) exact-match accuracy vs sequence length
  (2) mean prediction vs gold count, with the y=x ideal (pooled over the runs)
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("outputs/ladder/image_longN/behavior")
OUT = BASE / "plots"; OUT.mkdir(exist_ok=True)
Ns = [8, 16, 32, 64, 128]

acc, rows_all = {}, []
for N in Ns:
    sd = sorted(glob.glob(str(BASE / f"N{N}" / "*")))[-1]
    acc[N] = json.load(open(Path(sd) / "summary.json"))["exact_match"]
    for r in json.load(open(Path(sd) / "rows.json")):
        rows_all.append((int(r["gold"]), int(r["pred"]), N))

INK, ACC, MUT = "#1d2330", "#c0392b", "#7a8394"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#c9cfd8",
                     "axes.linewidth": 0.9, "figure.dpi": 140})

# ---- Plot 1: accuracy vs sequence length ----
fig, ax = plt.subplots(figsize=(6.2, 4.2))
xs = Ns; ys = [acc[N] for N in Ns]
ax.plot(xs, ys, "-o", color=ACC, lw=2.2, ms=7, zorder=3)
for x, y in zip(xs, ys):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=10, color=INK)
ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
ax.set_xlabel("sequence length  N  (number of frames)")
ax.set_ylabel("exact-match accuracy")
ax.set_ylim(-0.02, 0.28); ax.grid(True, alpha=0.25, lw=0.7)
ax.set_title("Base model on steps — accuracy collapses with N", fontsize=12.5, color=INK)
ax.text(0.98, 0.95, "frozen Qwen2.5-VL-7B · 392px · n=150/N\nsource: B3 [2026-07-10f]",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUT)
fig.tight_layout(); fig.savefig(OUT / "acc_vs_seqlen.png", bbox_inches="tight")
print("wrote", OUT / "acc_vs_seqlen.png")

# ---- Plot 2: mean prediction vs gold (pooled), with y=x ideal ----
g = np.array([r[0] for r in rows_all]); p = np.array([r[1] for r in rows_all])
golds = sorted(set(g.tolist()))
mean_p = np.array([p[g == gg].mean() for gg in golds])
std_p = np.array([p[g == gg].std() for gg in golds])
n_pg = np.array([int((g == gg).sum()) for gg in golds])
keep = n_pg >= 3
golds = np.array(golds)[keep]; mean_p = mean_p[keep]; std_p = std_p[keep]

fig, ax = plt.subplots(figsize=(6.2, 4.2))
lim = max(golds.max(), 12) + 3
ax.plot([0, lim], [0, lim], "--", color=MUT, lw=1.5, label="ideal (prediction = gold)")
ax.fill_between(golds, mean_p - std_p, mean_p + std_p, color=ACC, alpha=0.15, zorder=2)
ax.plot(golds, mean_p, "-o", color=ACC, lw=2, ms=5, zorder=3, label="model mean prediction")
ax.axhline(np.median(mean_p), color=INK, lw=0.8, ls=":", alpha=0.5)
ax.set_xlabel("actual count  (gold answer)")
ax.set_ylabel("model's mean predicted count")
ax.set_xlim(-1, lim); ax.set_ylim(-1, lim)
ax.grid(True, alpha=0.25, lw=0.7); ax.legend(loc="upper left", fontsize=10, frameon=False)
ax.set_title("The readout clamp — predictions saturate near ~3", fontsize=12.5, color=INK)
ax.text(0.98, 0.06, "pooled over N=8–128 steps runs · frozen Qwen2.5-VL-7B\nsource: B3 rows.json",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=MUT)
fig.tight_layout(); fig.savefig(OUT / "meanpred_vs_gold.png", bbox_inches="tight")
print("wrote", OUT / "meanpred_vs_gold.png")

print("\nacc vs N:", {N: round(acc[N], 3) for N in Ns})
print("mean pred by gold:", {int(x): round(float(m), 2) for x, m in zip(golds, mean_p)})

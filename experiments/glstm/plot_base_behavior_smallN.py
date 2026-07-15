#!/usr/bin/env python3
"""Two simple base-model behavior plots (steps), seq-len 1-16.
Data: outputs/ladder/image_smallN/behavior/N{1..8} + outputs/ladder/image_longN/behavior/N16.
All numbers trace to those runs' summary/rows json.
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SMALL = Path("outputs/ladder/image_smallN/behavior")
LONG = Path("outputs/ladder/image_longN/behavior")
OUT = SMALL / "plots"; OUT.mkdir(parents=True, exist_ok=True)

def latest(p): return sorted(glob.glob(str(p / "*")))[-1]

acc, rows_all = {}, []
srcs = [(n, SMALL / f"N{n}") for n in range(1, 9)] + [(16, LONG / "N16")]
for N, d in srcs:
    sd = latest(d)
    acc[N] = json.load(open(Path(sd) / "summary.json"))["exact_match"]
    for r in json.load(open(Path(sd) / "rows.json")):
        rows_all.append((int(r["gold"]), int(r["pred"])))

Ns = sorted(acc)
ACC = "#c0392b"
plt.rcParams.update({"font.size": 13, "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---- Plot 1: accuracy vs seq len ----
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(Ns, [acc[n] for n in Ns], "-o", color=ACC, lw=2.4, ms=6)
ax.set_xlabel("sequence length (frames)")
ax.set_ylabel("accuracy")
ax.set_xticks([1, 4, 8, 12, 16])
ax.set_ylim(0, 1.0)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "acc_vs_seqlen_1to16.png", bbox_inches="tight")
print("wrote", OUT / "acc_vs_seqlen_1to16.png")

# ---- Plot 2: mean prediction vs gold ----
g = np.array([r[0] for r in rows_all]); p = np.array([r[1] for r in rows_all])
golds = np.array([gg for gg in sorted(set(g.tolist())) if (g == gg).sum() >= 3])
mean_p = np.array([p[g == gg].mean() for gg in golds])

fig, ax = plt.subplots(figsize=(5.6, 3.8))
lim = 16.5
ax.plot([0, lim], [0, lim], "--", color="#888", lw=1.6, label="ideal")
ax.plot(golds, mean_p, "-o", color=ACC, lw=2.4, ms=6, label="model")
ax.set_xlabel("actual count")
ax.set_ylabel("mean predicted count")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xticks([0, 4, 8, 12, 16]); ax.set_yticks([0, 4, 8, 12, 16])
ax.grid(True, alpha=0.3); ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "meanpred_vs_gold_1to16.png", bbox_inches="tight")
print("wrote", OUT / "meanpred_vs_gold_1to16.png")

print("acc:", {n: round(acc[n], 3) for n in Ns})
print("mean_pred_by_gold:", {int(x): round(float(m), 2) for x, m in zip(golds, mean_p)})

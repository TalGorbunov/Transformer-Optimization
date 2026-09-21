#!/usr/bin/env python3
"""LEARNMASK headline figures (CPU, seconds).

Figure 1  acc_vs_n:   exact-match vs sequence length, five mask regimes — the
                      length-transfer money figure. The learned (pruned) mask is
                      drawn dashed so its exact overlap with full-open at the
                      training lengths is visible rather than hidden.
Figure 2  survivors:  relation x layer P(open) after prune-from-open, sequential
                      single-hue; the hand design's opening (R4/R7 at layers >=
                      L_OPEN) outlined for the agree/disagree comparison.

Colors: the dataviz reference palette (validated); text in ink tokens, never
series colors. Output: outputs/learnmask/figures/*.{png,pdf}.

Run: .venv/bin/python scripts/learnmask/plot_results.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import L_OPEN
from gnnformer.learnmask import CHANNELS, arm_learn_mask

# dataviz reference palette (light mode) — color follows the entity, fixed order
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
C_LEARNED, C_HAND, C_OPEN, C_NOFENCE, C_INIT = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")

SERIES = [  # (csv regime, label, color, style)
    ("gates",   "learned (pruned) mask", C_LEARNED, dict(ls="--", marker="o")),
    ("s2open",  "full open",             C_OPEN,    dict(ls="-",  marker="s")),
    ("hand",    "hand fence",            C_HAND,    dict(ls="-",  marker="^")),
    ("nofence", "no fence",              C_NOFENCE, dict(ls="-",  marker="D")),
    ("init",    "fence init",            C_INIT,    dict(ls="-",  marker="v")),
]
ROOT_N = {"seq_len_8_val_steps_in_room": 8, "seq_len_16_test": 16,
          "seq_len_32_test": 32, "seq_len_64_test": 64}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="y", color="#e6e5e1", lw=0.8, zorder=0)


def fig_acc_vs_n(results_csv: Path, out: Path):
    rows = list(csv.DictReader(results_csv.open()))
    data = {}
    for r in rows:
        n = ROOT_N.get(r["root"])
        if n:
            data.setdefault(r["regime"], {})[n] = float(r["em"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ns = [8, 16, 32, 64]
    for reg, label, color, kw in SERIES:
        ys = [data[reg][n] for n in ns]
        # learned drawn ON TOP: its dashes must show over full-open where they overlap
        z = 5 if reg == "gates" else 3
        ax.plot(ns, ys, color=color, lw=2, ms=6, zorder=z,
                markeredgecolor=SURFACE, markeredgewidth=1.2, **kw)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(7, 72)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("sequence length (frames) — gates trained at 8+16 only",
                  fontsize=9.5, color=INK2)
    ax.set_ylabel("exact match", fontsize=9.5, color=INK)
    ax.set_title("Learned mask transfers like the full-open topology it was pruned from\n",
                 fontsize=11.5, color=INK, loc="left", pad=14)
    ax.text(0, 1.02, "MMReD-HF steps_in_room, frozen 7B + fence-agnostic readout; "
                     "50 samples per length; 32/64 are zero-shot",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.legend([plt.Line2D([], [], color=c, lw=2, ls=k.get("ls", "-"),
                          marker=k.get("marker"), ms=5, markeredgecolor=SURFACE)
               for _, _, c, k in SERIES],
              [l for _, l, _, _ in SERIES], frameon=False, fontsize=8.5,
              labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out.with_suffix(".png"))


def fig_survivors(heatmap_csv: Path, out: Path):
    h = np.loadtxt(heatmap_csv, delimiter=",")
    learn = arm_learn_mask("s2").numpy()
    rows = [i for i in range(len(CHANNELS)) if learn[i]]
    # drop structurally empty bucket rows (no such block pairs at these lengths)
    rows = [i for i in rows if "17+" not in CHANNELS[i].name]
    names = [CHANNELS[i].name for i in rows]
    M = h[rows]
    n_layers = M.shape[1]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f2f6fc", "#123c6b"])
    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.pcolormesh(np.arange(n_layers + 1) - 0.5, np.arange(len(rows) + 1) - 0.5,
                       M, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, lw=1.2)
    # hand-design opening (R4 buckets + R7 at layers >= L_OPEN), outlined for contrast
    for y, i in enumerate(rows):
        if CHANNELS[i].rel in ("R4", "R7"):
            ax.add_patch(Rectangle((L_OPEN - 0.5, y - 0.5), n_layers - L_OPEN, 1,
                                   fill=False, ec=C_HAND, lw=1.4, ls=(0, (3, 2)),
                                   zorder=4))
    ax.set_xticks(range(0, n_layers, 4))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=8)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("layer", fontsize=9.5, color=INK2)
    ax.invert_yaxis()
    ax.set_title("Which edges survive pruning: aggregation earns its keep only in "
                 "layers 0–17\n", fontsize=11.5, color=INK, loc="left", pad=16)
    ax.text(0, 1.04, "P(open) after prune-from-open (blue = kept); dashed orange = "
                     "the hand design's opening (R4/R7, layers ≥ 12)",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("P(open)", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out.with_suffix(".png"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transfer-csv", default=None)
    ap.add_argument("--heatmap-csv", default=None)
    ap.add_argument("--outdir", default="outputs/learnmask/figures")
    args = ap.parse_args()
    tr = Path(args.transfer_csv) if args.transfer_csv else sorted(
        (_REPO / "outputs/learnmask/pruned_transfer").glob("2*/results.csv"))[-1]
    hm = Path(args.heatmap_csv) if args.heatmap_csv else sorted(
        (_REPO / "outputs/learnmask/prune_s2").glob("2*/heatmap_ep4.csv"))[-1]
    out = _REPO / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    fig_acc_vs_n(tr, out / "acc_vs_n")
    fig_survivors(hm, out / "survivor_heatmap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

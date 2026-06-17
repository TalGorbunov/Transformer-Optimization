#!/usr/bin/env python3
"""Plot the token-group restoration/corruption ablation using the ROBUST metric.

The runner used --disable_plots, and the built-in plot uses mean-of-ratios (inflated by small per-sample
denominators). This reads per_sample_patch_importance.csv for each task and plots MEDIAN normalized rescue
(with 25-75 IQR band), denom-filtered (|denom|>thr), per token group (frames/last_token/question) by layer,
plus a combined FRAMES contrast across tasks.
"""
from __future__ import annotations
import csv, os, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs" / "token_group_corruption_new_tasks"
TASKS = ["count", "rooms_visited", "co_occupancy"]
TARGETS = ["frames", "last_token", "question"]
COLORS = {"frames": "#1f77b4", "last_token": "#d62728", "question": "#ff7f0e"}
TASK_COLORS = {"count": "#2ca02c", "rooms_visited": "#1f77b4", "co_occupancy": "#9467bd"}
DENOM_THR = 0.5


def load(task: str):
    f = BASE / task / "per_sample_patch_importance.csv"
    if not f.exists():
        return None
    # data[(target, layer)] -> list of normalized_rescue (denom-filtered)
    data = defaultdict(list)
    for r in csv.DictReader(open(f)):
        try:
            nr = float(r["normalized_rescue"]); dn = float(r["denominator"]); L = int(r["layer"])
        except (ValueError, KeyError):
            continue
        if abs(dn) < DENOM_THR:
            continue
        data[(r["patch_target"], L)].append(nr)
    return data


def med_iqr(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    med = st.median(vals)
    lo = vals[max(0, int(0.25 * (n - 1)))]
    hi = vals[min(n - 1, int(0.75 * (n - 1)))]
    return med, lo, hi


def per_task_plot(task: str, data):
    layers = sorted({L for (_t, L) in data})
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    for tgt in TARGETS:
        xs, ys, los, his = [], [], [], []
        for L in layers:
            v = data.get((tgt, L), [])
            if not v:
                continue
            m, lo, hi = med_iqr(v)
            xs.append(L); ys.append(m); los.append(lo); his.append(hi)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", color=COLORS[tgt], lw=2, label=f"restore {tgt}")
        ax.fill_between(xs, los, his, color=COLORS[tgt], alpha=0.15)
    ax.axhline(1.0, color="#444", ls=":", lw=1, label="full rescue (=clean)")
    ax.axhline(0.0, color="#888", ls="-", lw=0.8)
    ax.set_xlabel("layer (restoration site)"); ax.set_ylabel("median normalized rescue (denom-filtered)")
    ax.set_title(f"Token-group restoration on frame-blanked input — {task} (n per pt, |denom|>{DENOM_THR})")
    ax.legend(); ax.grid(alpha=0.25, ls="--")
    out = BASE / "plots" / f"restoration_by_token_group_{task}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def frames_contrast_plot(all_data):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    for task in TASKS:
        data = all_data.get(task)
        if not data:
            continue
        layers = sorted({L for (t, L) in data if t == "frames"})
        xs, ys, los, his = [], [], [], []
        for L in layers:
            v = data.get(("frames", L), [])
            if not v:
                continue
            m, lo, hi = med_iqr(v)
            xs.append(L); ys.append(m); los.append(lo); his.append(hi)
        if xs:
            ax.plot(xs, ys, marker="o", color=TASK_COLORS[task], lw=2, label=task)
            ax.fill_between(xs, los, his, color=TASK_COLORS[task], alpha=0.12)
    ax.axhline(1.0, color="#444", ls=":", lw=1)
    ax.set_xlabel("layer (restoration site)"); ax.set_ylabel("median frame-restore normalized rescue")
    ax.set_title("FRAME-token restoration by depth: count vs rooms_visited vs co_occupancy")
    ax.legend(); ax.grid(alpha=0.25, ls="--")
    out = BASE / "plots" / "frames_restoration_contrast.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    all_data = {}
    written = []
    for task in TASKS:
        d = load(task)
        if d is None:
            print(f"  {task}: no per_sample csv, skip"); continue
        all_data[task] = d
        written.append(per_task_plot(task, d))
    written.append(frames_contrast_plot(all_data))
    for w in written:
        print(f"wrote {w}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot BOTH families from the surgical dual-direction ablation (patch_metrics_long.csv):
  - restoration            (patch clean into corrupted run)
  - clean_ablation_damage  (patch corrupted into clean run)
Robust median normalized value (denom-filtered) by layer, per token group, plus a frames contrast
across tasks for each family. Matches the canonical normalized_restoration_curves.png methodology.
"""
from __future__ import annotations
import csv, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs" / "token_group_corruption_new_tasks"
TASKS = ["count", "rooms_visited", "co_occupancy"]
TARGETS = ["frames", "last_token", "question"]
METRICS = ["restoration", "clean_ablation_damage"]
COLORS = {"frames": "#1f77b4", "last_token": "#d62728", "question": "#ff7f0e"}
TASK_COLORS = {"count": "#2ca02c", "rooms_visited": "#1f77b4", "co_occupancy": "#9467bd"}
DENOM_THR = 0.5


def load(task):
    f = BASE / f"{task}_surgical" / "patch_metrics_long.csv"
    if not f.exists():
        return None
    data = defaultdict(list)  # (metric, target, layer) -> [normalized_value]
    for r in csv.DictReader(open(f)):
        try:
            nv = float(r["normalized_value"]); dn = float(r["denominator"]); L = int(r["layer"])
        except (ValueError, KeyError):
            continue
        if abs(dn) < DENOM_THR:
            continue
        data[(r["metric_type"], r["patch_target"], L)].append(nv)
    return data


def med_iqr(vals):
    vals = sorted(vals); n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return st.median(vals), vals[max(0, int(0.25*(n-1)))], vals[min(n-1, int(0.75*(n-1)))]


def per_task_metric_plot(task, data, metric):
    layers = sorted({L for (m, _t, L) in data if m == metric})
    if not layers:
        return None
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    for tgt in TARGETS:
        xs, ys, lo, hi = [], [], [], []
        for L in layers:
            v = data.get((metric, tgt, L), [])
            if not v:
                continue
            m, l, h = med_iqr(v); xs.append(L); ys.append(m); lo.append(l); hi.append(h)
        if xs:
            ax.plot(xs, ys, marker="o", color=COLORS[tgt], lw=2, label=tgt)
            ax.fill_between(xs, lo, hi, color=COLORS[tgt], alpha=0.15)
    ax.axhline(1.0, color="#444", ls=":", lw=1, label="full effect (=clean−corrupted gap)")
    ax.axhline(0.0, color="#888", lw=0.8)
    lab = "restoration (patch clean→corrupted)" if metric == "restoration" else "clean-ablation damage (patch corrupted→clean)"
    ax.set_xlabel("layer"); ax.set_ylabel(f"median normalized {('rescue' if metric=='restoration' else 'damage')} (denom-filtered)")
    ax.set_title(f"{task} — {lab}")
    ax.legend(); ax.grid(alpha=0.25, ls="--")
    out = BASE / "plots_surgical" / f"{metric}_{task}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def frames_contrast(all_data, metric):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    any_pts = False
    for task in TASKS:
        data = all_data.get(task)
        if not data:
            continue
        layers = sorted({L for (m, t, L) in data if m == metric and t == "frames"})
        xs, ys, lo, hi = [], [], [], []
        for L in layers:
            v = data.get((metric, "frames", L), [])
            if not v:
                continue
            m, l, h = med_iqr(v); xs.append(L); ys.append(m); lo.append(l); hi.append(h)
        if xs:
            any_pts = True
            ax.plot(xs, ys, marker="o", color=TASK_COLORS[task], lw=2, label=task)
            ax.fill_between(xs, lo, hi, color=TASK_COLORS[task], alpha=0.12)
    if not any_pts:
        plt.close(fig); return None
    ax.axhline(1.0, color="#444", ls=":", lw=1)
    lab = "restoration" if metric == "restoration" else "clean-ablation damage"
    ax.set_xlabel("layer"); ax.set_ylabel(f"median FRAME normalized {('rescue' if metric=='restoration' else 'damage')}")
    ax.set_title(f"FRAME {lab} by depth: count vs rooms_visited vs co_occupancy")
    ax.legend(); ax.grid(alpha=0.25, ls="--")
    out = BASE / "plots_surgical" / f"frames_contrast_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    all_data = {}
    written = []
    for task in TASKS:
        d = load(task)
        if d is None:
            print(f"  {task}: no patch_metrics_long.csv yet"); continue
        all_data[task] = d
        for metric in METRICS:
            p = per_task_metric_plot(task, d, metric)
            if p:
                written.append(p)
    for metric in METRICS:
        p = frames_contrast(all_data, metric)
        if p:
            written.append(p)
    for w in written:
        print(f"wrote {w}")
    if not written:
        print("no plots written (results not ready)")


if __name__ == "__main__":
    main()

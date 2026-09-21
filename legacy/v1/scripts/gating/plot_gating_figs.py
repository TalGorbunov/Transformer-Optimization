#!/usr/bin/env python3
"""P5 of the gating campaign: the three verdict figures.

  fig1  held-out accuracy per arm vs the LoRA control
  fig2  the P3.5 discriminator — gain vs distractor count (N at fixed evidence) and
        gain vs evidence count (at fixed N), one line per arm, zero line marked
  fig3  mean gate score per layer per arm — the proof the gates were alive. An arm whose
        curve sits on 1.0 learned nothing and is VOID, not a null.

Inputs are run dirs, given as LABEL=PATH so the legend is explicit:
  --arms  lora=<p3 run dir> g1_headwise=<...> ...      (each has report.txt)
  --grid  lora=<p3.5 run dir> g1_headwise=<...> ...    (each has cells.csv)
The first --arms entry is the control every gain is measured against.

Usage:
  python scripts/gating/plot_gating_figs.py --arms lora=<dir> g2_literal=<dir> \
      --grid lora=<dir> g2_literal=<dir> --output outputs/gating/p5_figs/<stamp>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_EP = re.compile(r"^ep(\d+) loss ([\d.]+) acc ([\d.]+) mae ([\d.]+) \[(.*)\]$")
_GATE = re.compile(r"^ep(\d+) gate mean/layer (.+?)(?: \| min|$)")
_BEST = re.compile(r"^BEST acc ([\d.]+) \(tf-exact ([\d.]+)\) @ ep (\d+)")
_TFEX = re.compile(r"tf-exact ([\d.]+)")


def parse_arm(run_dir: Path) -> Dict[str, Any]:
    """report.txt -> {header, epochs:[{ep,loss,acc,tf_exact}], gate:{ep:{layer:score}},
    best_acc, best_tf_exact, best_ep}."""
    txt = (run_dir / "report.txt").read_text().splitlines()
    out: Dict[str, Any] = {"header": txt[0] if txt else "", "epochs": [], "gate": {},
                           "best_acc": None, "best_tf_exact": None, "best_ep": None,
                           "run_dir": str(run_dir)}
    for ln in txt:
        m = _EP.match(ln.strip())
        if m:
            tf = _TFEX.search(m.group(5))
            out["epochs"].append({"ep": int(m.group(1)), "loss": float(m.group(2)),
                                  "acc": float(m.group(3)),
                                  "tf_exact": float(tf.group(1)) if tf else float("nan")})
            continue
        m = _GATE.match(ln.strip())
        if m:
            out["gate"][int(m.group(1))] = {
                int(k[1:]): float(v)
                for k, v in (p.split(":") for p in m.group(2).split() if ":" in p)
            }
            continue
        m = _BEST.match(ln.strip())
        if m:
            out["best_acc"], out["best_tf_exact"], out["best_ep"] = (
                float(m.group(1)), float(m.group(2)), int(m.group(3)))
    return out


METRIC_IDX = {"tf_acc": 1, "tf_exact": 2, "dec_acc": 3}


def parse_grid(run_dir: Path, metric: str = "tf_exact") -> Dict[str, Any]:
    """cells.csv -> {(N, gold): (hits, n)} for `metric`, plus per-root headline accuracy.

    per_gold rows are [n, count-token ok, tf-exact ok, decoded ok]. The count metric
    saturates at 1.000 for every arm at N<=8, so it cannot discriminate there — default to
    tf_exact and prefer dec_acc when the run decoded."""
    idx = METRIC_IDX[metric]
    grid: Dict[Tuple[int, int], Tuple[int, int]] = {}
    roots: Dict[int, float] = {}
    for r in csv.DictReader((run_dir / "cells.csv").open()):
        m = re.search(r"seq_len_(\d+)", r["root"])
        if not m:
            continue
        N = int(m.group(1))
        if r.get(metric, "") not in ("", "nan"):
            roots[N] = float(r[metric])
        for g, v in json.loads(r["per_gold"]).items():
            if len(v) < 4:
                raise SystemExit(
                    f"{run_dir}/cells.csv uses the pre-2026-08-07 per_gold format "
                    f"{v} ([hits, n]); it carries only the count metric, which saturates. "
                    "Re-run scripts/gating/eval_gated.py to get [n, count, tf_exact, dec].")
            n, hits = v[0], v[idx]
            k = (N, int(g))
            prev = grid.get(k, (0, 0))
            grid[k] = (prev[0] + hits, prev[1] + n)
    return {"grid": grid, "roots": roots, "run_dir": str(run_dir), "metric": metric}


def acc(grid: Dict[Tuple[int, int], Tuple[int, int]], N: int, g: int) -> Optional[float]:
    h, n = grid.get((N, g), (0, 0))
    return h / n if n else None


def kv(pairs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"expected LABEL=PATH, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = Path(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=[], metavar="LABEL=DIR",
                    help="P3 trainer run dirs; the FIRST is the control")
    ap.add_argument("--grid", nargs="+", default=[], metavar="LABEL=DIR",
                    help="P3.5 eval run dirs; the FIRST is the control")
    ap.add_argument("--evidence", type=int, nargs="+", default=[1, 2],
                    help="evidence counts held fixed for the distractor axis")
    ap.add_argument("--capacity-n", type=int, default=8, help="N held fixed for the capacity axis")
    ap.add_argument("--metric", choices=tuple(METRIC_IDX), default="tf_exact",
                    help="grid metric. tf_acc (count token) SATURATES at 1.000 for every "
                         "arm at N<=8 and cannot discriminate there")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    arms = {k: parse_arm(v) for k, v in kv(args.arms).items()} if args.arms else {}
    grids = ({k: parse_grid(v, args.metric) for k, v in kv(args.grid).items()}
             if args.grid else {})
    summary: List[str] = []

    # ---------------------------------------------------------------- fig 1: per arm
    if arms:
        ctrl = list(arms)[0]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        labels = list(arms)
        base = arms[ctrl]["best_acc"]
        for ax, key, ttl in ((axes[0], "best_acc", "held-out TF-count accuracy"),
                             (axes[1], "best_tf_exact", "held-out tf-exact (full transcript)")):
            vals = [arms[k][key] if arms[k][key] is not None else np.nan for k in labels]
            cols = ["0.45" if k == ctrl else "C0" for k in labels]
            ax.bar(range(len(labels)), vals, color=cols)
            ax.axhline(arms[ctrl][key] or np.nan, color="k", ls="--", lw=1,
                       label=f"{ctrl} control")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            ax.set_ylim(0, 1.02)
            ax.set_title(ttl, fontsize=11)
            ax.legend(fontsize=8)
            for i, v in enumerate(vals):
                if v == v:
                    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
        fig.suptitle("Fig 1 — gated arms vs the LoRA control (best epoch, held-out half)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig1_arms.{ext}", dpi=300)
        summary.append("--- Fig 1: best held-out accuracy per arm")
        for k in labels:
            a = arms[k]
            d = (a["best_acc"] - base) if (a["best_acc"] is not None and base is not None) else float("nan")
            summary.append(f"  {k:<18} acc {a['best_acc']} tf-exact {a['best_tf_exact']} "
                           f"@ep {a['best_ep']}  (delta vs {ctrl}: {d:+.3f})")

    # ------------------------------------------------- fig 2: the P3.5 discriminator
    if grids:
        ctrl = list(grids)[0]
        cg = grids[ctrl]["grid"]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        Ns = sorted({N for N, _ in cg})
        for lab in grids:
            g = grids[lab]["grid"]
            if lab == ctrl:
                continue
            for e in args.evidence:
                xs, ys = [], []
                for N in Ns:
                    a, b = acc(g, N, e), acc(cg, N, e)
                    if a is not None and b is not None:
                        xs.append(N)
                        ys.append(a - b)
                if xs:
                    axes[0].plot(xs, ys, marker="o", label=f"{lab}, evidence={e}")
            xs, ys = [], []
            for gd in sorted({gg for N, gg in cg if N == args.capacity_n}):
                a, b = acc(g, args.capacity_n, gd), acc(cg, args.capacity_n, gd)
                if a is not None and b is not None:
                    xs.append(gd)
                    ys.append(a - b)
            if xs:
                axes[1].plot(xs, ys, marker="s", label=lab)
        for ax, xl, ttl in ((axes[0], "N (frames — distractors, evidence held fixed)",
                             "DISTRACTOR axis"),
                            (axes[1], f"evidence count (N fixed at {args.capacity_n})",
                             "CAPACITY axis")):
            ax.axhline(0, color="k", lw=1)
            ax.set_xlabel(xl)
            ax.set_ylabel(f"{args.metric} gain vs {ctrl}")
            ax.set_title(ttl, fontsize=11)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        fig.suptitle("Fig 2 — capacity vs interference: where (if anywhere) gating helps")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig2_discriminator.{ext}", dpi=300)
        summary.append(f"\n--- Fig 2: (N x evidence) {args.metric} grid, control = {ctrl}")
        for lab in grids:
            g = grids[lab]["grid"]
            sat = [v for v in (acc(g, N, e) for N, e in g) if v is not None]
            flag = "  [SATURATED — cannot discriminate]" if sat and min(sat) >= 0.999 else ""
            summary.append(f"  {lab}{flag}")
            hdr = "    N \\ e " + " ".join(f"{e:>6}" for e in
                                          sorted({gg for _, gg in g}))
            summary.append(hdr)
            for N in Ns:
                row = " ".join(
                    (f"{acc(g, N, e):>6.3f}" if acc(g, N, e) is not None else "     -")
                    for e in sorted({gg for _, gg in g}))
                summary.append(f"    {N:>5} {row}")

    # ------------------------------------------------------ fig 3: gate liveness proof
    gated = {k: a for k, a in arms.items() if a["gate"]}
    if gated:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for lab, a in gated.items():
            last = max(a["gate"])
            d = a["gate"][last]
            ax.plot(sorted(d), [d[k] for k in sorted(d)], marker="o", label=f"{lab} (ep {last})")
        ax.axhline(1.0, color="k", ls="--", lw=1, label="identity init (VOID if flat here)")
        ax.set_xlabel("layer")
        ax.set_ylabel("mean gate score")
        ax.set_title("Fig 3 — were the gates alive?", fontsize=11)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig3_gate_scores.{ext}", dpi=300)
        summary.append("\n--- Fig 3: mean gate score per layer (last epoch)")
        for lab, a in gated.items():
            last = max(a["gate"])
            v = list(a["gate"][last].values())
            verdict = "VOID (gate never moved)" if max(abs(x - 1.0) for x in v) < 0.005 else "alive"
            summary.append(f"  {lab:<18} ep{last} min {min(v):.4f} max {max(v):.4f} "
                           f"mean {np.mean(v):.4f} -> {verdict}")
    elif arms:
        summary.append("\n--- Fig 3: no arm reported gate scores (all ungated)")

    (out / "report.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    (out / "ABOUT.md").write_text(
        "# P5 — gating campaign verdict figures\n\n"
        "fig1_arms: best held-out TF-count accuracy and tf-exact per P3 arm, LoRA control\n"
        "marked. fig2_discriminator: P3.5 gains against the control along the distractor\n"
        "axis (N grows, evidence count held fixed) and the capacity axis (N fixed,\n"
        "evidence count grows), zero line marked — a gain confined to the distractor axis\n"
        "means interference, flat/capacity-only means a capacity wall. fig3_gate_scores:\n"
        "mean gate score per layer, with the identity line — an arm sitting on 1.0 learned\n"
        "nothing and is VOID rather than a null result.\n\n"
        f"Arms: {args.arms}\nGrids: {args.grid}\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

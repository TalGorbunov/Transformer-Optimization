#!/usr/bin/env python3
"""qkv-swap N-scaling figure: what each repair buys, as the graph grows.

Parses the four canonical qkv_swap run reports (N=8/16/32/64) and plots, per N,
the d' GAIN over the joint (dirty-dirty) corner for three interventions:
clean queries only, clean values only, both (the fence). The value-only repair
collapsing to ~0 (N=32) and negative (N=64) while query-only stays ~+1 is the
headline; the gap between 'both' and the sum of singles is the interaction.

CPU-only, seconds; reads report.txt files, no caches.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = {  # N -> canonical qkv_swap run dir
    8: "outputs/presentation/qkv_swap/20260731_220735",
    16: "outputs/presentation/qkv_swap/20260731_222936_127589",
    32: "outputs/presentation/qkv_swap/20260731_221639",
    64: "outputs/presentation/qkv_swap/20260731_222127",
}
CELL_RE = re.compile(r"^(q[CD]_kv[CD]): d' ([\d.]+)±([\d.]+).*gate->tally ([\d.]+)±([\d.]+)")


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/presentation/qkv_scaling/local")
    out.mkdir(parents=True, exist_ok=True)
    grids = {}
    for n, rd in RUNS.items():
        cells = {}
        for line in (Path(rd) / "report.txt").read_text().splitlines():
            m = CELL_RE.match(line.strip())
            if m:
                cells[m.group(1)] = (float(m.group(2)), float(m.group(3)),
                                     float(m.group(4)), float(m.group(5)))
        assert len(cells) == 4, (rd, cells)
        grids[n] = cells

    ns = sorted(grids)
    q_only, kv_only, both, inter, rows = [], [], [], [], []
    for n in ns:
        c = grids[n]
        dd = c["qD_kvD"][0]
        q1, k1, cc = c["qC_kvD"][0] - dd, c["qD_kvC"][0] - dd, c["qC_kvC"][0] - dd
        q_only.append(q1), kv_only.append(k1), both.append(cc)
        inter.append(cc - q1 - k1)
        rows.append([n, f"{q1:.2f}", f"{k1:.2f}", f"{cc:.2f}", f"{cc - q1 - k1:.2f}",
                     f"{c['qD_kvD'][2]:.3f}", f"{c['qD_kvC'][2]:.3f}",
                     f"{c['qC_kvD'][2]:.3f}", f"{c['qC_kvC'][2]:.3f}", RUNS[n]])
    with open(out / "qkv_scaling.csv", "w", newline="") as f:
        csv.writer(f).writerows(
            [["N", "q_only_gain", "kv_only_gain", "both_gain", "interaction",
              "acc_joint", "acc_kv_only", "acc_q_only", "acc_both", "run"], *rows])

    # ---- accuracy version: absolute gate->tally per cell (the understandable one) ----
    # true joint baseline (single read locus, full-context softmax, plain prompt) from
    # the curves run — NOT protocol-comparable to the 2x2 cells; shown as reference dashes
    CURVES_CSV = "outputs/presentation/curves/20260731_202940/curves.csv"
    joint_base = {}
    with open(CURVES_CSV) as f:
        for row in csv.DictReader(f):
            if row["arm"] == "joint":
                joint_base[int(row["N"])] = float(row["tally_exact"])
    x = np.arange(len(ns))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    order = [("qD_kvD", "joint attention", "#9a9891"),
             ("qD_kvC", "+ clean values only", "#eb6834"),
             ("qC_kvD", "+ clean queries only", "#2a78d6"),
             ("qC_kvC", "both (the fence)", "#1baf7a")]
    for i, (cell, lab, col) in enumerate(order):
        vals = [grids[n][cell][2] for n in ns]
        errs = [grids[n][cell][3] for n in ns]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, capsize=2, label=lab, color=col)
        for b, v, e in zip(bars, vals, errs):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v + e + 0.012),
                        ha="center", va="bottom", fontsize=7.5)
    for xi, n in enumerate(ns):
        if n in joint_base:
            ax.hlines(joint_base[n], xi - 2 * w, xi + 2 * w, color="#52514e",
                      linestyle="--", linewidth=1.4,
                      label="deployed joint baseline\n(single read locus, full softmax)"
                      if xi == 0 else None)
            ax.annotate(f"{joint_base[n]:.2f}", (xi + 2 * w, joint_base[n]),
                        fontsize=7.5, color="#52514e", va="center", ha="left")
    ax.set_xticks(x, [f"N={n}" for n in ns])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("exact-count accuracy (gate→tally, held-out)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color="#e6e5e2", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.suptitle("Count accuracy under each repair, as the graph grows", fontsize=12, y=0.99)
    fig.text(0.5, 0.905, "2×2 cells use per-frame readers + own-frame softmax (a controlled "
             "decomposition) — the dashed line is the deployed joint readout",
             ha="center", fontsize=8.5, color="#52514e")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"qkv_scaling_acc.{ext}", dpi=300)

    x = np.arange(len(ns))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(0, color="#52514e", lw=1)
    for off, vals, lab, col in [(-w, q_only, "clean queries only", "#2a78d6"),
                                (0, kv_only, "clean values only", "#eb6834"),
                                (w, both, "both (the fence)", "#1baf7a")]:
        bars = ax.bar(x + off, vals, w, label=lab, color=col)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:+.2f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(x, [f"N={n}" for n in ns])
    ax.set_ylabel("d′ gain over joint attention")
    ax.set_title("What each repair buys as the graph grows (q/kv swap 2×2)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color="#e6e5e2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"qkv_scaling.{ext}", dpi=300)

    (out / "ABOUT.md").write_text(
        "# q/kv repair gains vs N — the query side is the binding constraint at scale\n\n"
        "Bars = d' gain over the joint (dirty-dirty) corner of the q/kv swap 2x2, per N,\n"
        "for: cleaning only the read-position queries, cleaning only the frame keys/values,\n"
        "and cleaning both (what the fence does). Value-only repair decays with N and goes\n"
        "NEGATIVE at N=64; query-only stays ~+1 d' at every N; the gap between 'both' and\n"
        "the sum of the singles is the interaction (~+2 at long N) — most of the tax is\n"
        "recoverable only by fixing both sides at once.\n\n"
        "PROTOCOL CAVEAT (qkv_scaling_acc): the 2x2 cells are NOT the deployed joint\n"
        "readout — they use per-frame replica read positions, own-frame-restricted softmax\n"
        "(competition excluded from ALL cells by design), and the Q-first replica prompt.\n"
        "That is what makes the four cells comparable to each other, and why even the\n"
        "dirty-dirty corner (0.55 @N=64) sits far above the deployed joint baseline\n"
        "(single read locus, full-context softmax: 0.47/0.23/0.15/0.09 @N=8/16/32/64,\n"
        "dashed line, from curves/20260731_202940/curves.csv). The gap between the dashed\n"
        "line and the gray bar is itself informative: it is the part of the joint failure\n"
        "attributable to the READ PROTOCOL (single-locus addressing + softmax competition)\n"
        "rather than to state contamination. Sources (canonical run reports):\n"
        + "\n".join(f"- N={n}: `{RUNS[n]}`" for n in ns) +
        "\n\nArtifacts: qkv_scaling.png/pdf, qkv_scaling.csv. Generated by\n"
        "`scripts/presentation_diagnostics/plot_qkv_scaling.py` (CPU, parses report.txt).\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

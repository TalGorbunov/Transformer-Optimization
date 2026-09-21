#!/usr/bin/env python3
"""Repair waterfall @N=32: from the joint failure to the method, one named fix at a time.

One figure, one readout (held-out gate->tally), five measured rungs:
  1. joint model readout        (curves joint N=32: single reader, full softmax)
  2. + a reader per frame       (qkv qD_kvD: query side 1 — addressing/competition)
  3. + clean the readers        (qkv qC_kvD: query side 2 — query contamination)
  4. + clean the frames         (qkv qC_kvC: value noise)
  5. + posreset & Q-first       (curves fenced N=32: the full deployed fence)
Annotation: the learned 1-token carrier matches rung 5 (curves carrier N=32).

CPU, seconds; parses the canonical run artifacts, no caches.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVES_CSV = "outputs/presentation/curves/20260731_202940/curves.csv"
QKV_N32_REPORT = "outputs/presentation/qkv_swap/20260731_221639/report.txt"
CELL_RE = re.compile(r"^(q[CD]_kv[CD]): .*gate->tally ([\d.]+)±([\d.]+)")


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/presentation/waterfall/local")
    out.mkdir(parents=True, exist_ok=True)
    cv = {}
    with open(CURVES_CSV) as f:
        for row in csv.DictReader(f):
            if int(row["N"]) == 32:
                cv[row["arm"]] = (float(row["tally_exact"]), float(row["std"]))
    qk = {}
    for line in Path(QKV_N32_REPORT).read_text().splitlines():
        m = CELL_RE.match(line.strip())
        if m:
            qk[m.group(1)] = (float(m.group(2)), float(m.group(3)))

    steps = [
        ("joint model\nreadout", "one reader,\nfull competition", cv["joint"], "#9a9891"),
        ("+ a reader\nper frame", "query side 1:\naddressing", qk["qD_kvD"], "#86b6ef"),
        ("+ clean the\nreaders", "query side 2:\nreader states", qk["qC_kvD"], "#2a78d6"),
        ("+ clean the\nframes", "value noise", qk["qC_kvC"], "#eb6834"),
        ("+ posreset\n& Q-first", "the full fence\n(deployed supply)", cv["fenced"], "#1baf7a"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = range(len(steps))
    for i, (lab, sub, (v, e), col) in enumerate(steps):
        ax.bar(i, v, 0.62, yerr=e, capsize=3, color=col)
        ax.annotate(f"{v:.2f}", (i, v + e + 0.015), ha="center", fontsize=10)
        ax.annotate(sub, (i, 0.02), ha="center", va="bottom", fontsize=7.5,
                    color="#3d3c39" if col == "#86b6ef" else "white")
        if i:
            prev = steps[i - 1][2][0]
            ax.annotate(f"+{v - prev:.2f}", (i - 0.5, (v + prev) / 2), ha="center",
                        fontsize=8.5, color="#52514e", style="italic")
    ax.set_xticks(list(xs), [s[0] for s in steps], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("exact-count accuracy (gate→tally, held-out)")
    ax.set_title("From the joint failure to the method, one repair at a time (N=32)",
                 fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color="#e6e5e2", linewidth=0.7)
    ax.set_axisbelow(True)
    cr = cv["carrier"]
    ax.annotate(f"learned 1-token carrier replaces the 20-token replicas: {cr[0]:.2f}",
                (len(steps) - 1, 1.06), ha="right", fontsize=8.5, color="#52514e")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"repair_waterfall.{ext}", dpi=300)

    with open(out / "waterfall.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "repairs", "acc", "std", "source"])
        for (lab, sub, (v, e), _), src in zip(
                steps, ["curves joint", "qkv qD_kvD", "qkv qC_kvD", "qkv qC_kvC",
                        "curves fenced"]):
            w.writerow([lab.replace("\n", " "), sub.replace("\n", " "), v, e, src])
        w.writerow(["learned carrier", "1 token replaces 20-token replica", cr[0], cr[1],
                    "curves carrier"])
    (out / "ABOUT.md").write_text(
        "# Repair waterfall @N=32 — the joint failure decomposed against the method\n\n"
        "One readout (held-out gate->tally) at every rung; each bar adds ONE named repair.\n"
        "Rungs 2-4 are the q/kv-swap cells (per-frame readers + own-frame softmax, no\n"
        "posreset); rungs 1 and 5 are the curves cells (rung 5 = full A3: fence + posreset\n"
        "+ Q-first). The two query-side rungs (+0.56, +0.13) vs the value rung (+0.10)\n"
        "show the query half dominates at this N; the final rung (+0.06) is the posreset/\n"
        "Q-first amplifier. The learned 1-token carrier matches rung 5 (1.00). Sources:\n"
        f"- `{CURVES_CSV}`\n- `{QKV_N32_REPORT}`\n\n"
        "Caveat: rungs 2-4 use a different read protocol than rungs 1/5 (per-frame probe\n"
        "readers vs anchor/replica reads) — the waterfall is a narrative decomposition\n"
        "stitched from two instruments, stated as such, not a single ablation ladder.\n"
        "Artifacts: repair_waterfall.png/pdf, waterfall.csv. Generated by\n"
        "`scripts/presentation_diagnostics/plot_repair_waterfall.py`.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

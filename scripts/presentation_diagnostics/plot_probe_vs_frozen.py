#!/usr/bin/env python3
"""Linear probe on the fenced replicas vs the frozen model's own answer, per N.

Both arms are the counting (steps) task on park / longN_park renders, layer 16.

  probe  = held-out logistic gate per replica message + sum of verdicts (the
           gate_tally protocol) on Q-first + block-fence + posreset messages.
  frozen = the same frozen model answering the same question with all N frames in
           one context (its own decoded answer, exact match).

Every cell carries its source run in PROV below and in the emitted CSV — the two
arms come from different run batches (noted on the figure), so this script is the
single place where that bookkeeping lives.

Usage: python plot_probe_vs_frozen.py [<output_dir>]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# N -> (value, std or None, source)
PROBE = {
    8:   (0.997, 0.005, "posreset_sweep/N8_reset (job 127736, n=150)"),
    16:  (1.000, 0.000, "posreset_sweep/N16_reset (job 127736, n=120)"),
    32:  (0.996, 0.008, "posreset_sweep/N32_reset (job 127737, n=100)"),
    64:  (0.950, 0.077, "posreset_sweep/N64_reset (job 127737, n=40)"),
    128: (0.984, 0.008, "curves/20260731_202940 fenced N=128 (replica_blockfence_qfirst_N128, n=100)"),
}
FROZEN = {
    8:   (0.207, None, "B1 length sweep, jobs 120013-120022 (n=300)"),
    16:  (0.127, None, "B1 (n=300)"),
    32:  (0.053, None, "B1 (n=300)"),
    64:  (0.040, None, "B1 (n=200)"),
    128: (0.013, None, "B1 (n=150)"),
}
NS = [8, 16, 32, 64, 128]


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               "outputs/presentation/probe_vs_frozen/local")
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    pv = [PROBE[n][0] for n in NS]
    pe = [PROBE[n][1] or 0 for n in NS]
    fv = [FROZEN[n][0] for n in NS]

    ax.fill_between(NS, fv, pv, color="#2a78d6", alpha=0.07, lw=0)
    ax.errorbar(NS, pv, yerr=pe, color="#2a78d6", lw=2.2, marker="o", ms=7, capsize=3,
                label="linear probe on the per-frame readers", zorder=3)
    ax.plot(NS, fv, color="#9a9891", lw=2.2, marker="s", ms=6,
            label="the frozen model's own answer", zorder=3)

    for n, v, e in zip(NS, pv, pe):
        ax.annotate(f"{v:.3f}", (n, v + e), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#2a78d6")
    for n, v in zip(NS, fv):
        ax.annotate(f"{v:.3f}", (n, v), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#6f6c66")

    ax.annotate("same model · same frames · same question",
                (0.5, 0.40), xycoords="axes fraction", ha="center",
                fontsize=9.5, color="#52514e", style="italic")

    ax.set_xscale("log", base=2)
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.set_xlim(7, 150)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("frames in context")
    ax.set_ylabel("exact-count accuracy (held-out)")
    ax.set_title("What the states know vs what the model says", fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e6e5e2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="center left", bbox_to_anchor=(0.03, 0.62))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"probe_vs_frozen.{ext}", dpi=300)

    with open(out / "probe_vs_frozen.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "arm", "exact", "std", "source"])
        for n in NS:
            for arm, d in (("probe", PROBE), ("frozen", FROZEN)):
                v, s, src = d[n]
                w.writerow([n, arm, v, "" if s is None else s, src])

    (out / "ABOUT.md").write_text(
        "# Linear probe on the fenced readers vs the frozen model's own answer\n\n"
        "Counting task, park / longN_park renders, layer 16. PROBE = held-out logistic gate\n"
        "per replica message + sum of verdicts (gate_tally protocol) on Q-first + block-fence\n"
        "+ posreset messages. FROZEN = the same frozen model answering with all N frames in\n"
        "one context, exact match on its decoded answer.\n\n"
        "HONEST NOTE: the two arms come from different run batches (see the source column in\n"
        "probe_vs_frozen.csv), not from one matched-sample experiment — same task, model,\n"
        "renders and layer, different sample draws and n. The gap is two orders of magnitude,\n"
        "far larger than that discrepancy, but a single matched sweep would be the airtight\n"
        "version if this figure goes into the thesis rather than a talk.\n\n"
        "Artifacts: probe_vs_frozen.png/pdf, probe_vs_frozen.csv.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

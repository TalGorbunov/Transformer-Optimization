#!/usr/bin/env python3
"""VNBench collapse figure (plan docs/VNBENCH_2026-10-09.md §7): accuracy vs N on Qwen2.5-VL-7B, from the
evaluate.py run dirs that experiments/curve_table.py reads. CPU only.

  python experiments/figs/vnbench_fig.py [--out outputs/scaleup/vnbench/qwen2.5-vl-7b/fig] [--rtag native_mp151200]

Panels: retrieval / ordering / counting (task means over the subtasks; cnt_edit2 is excluded from the counting
panel as a perception wall, E = 0.21) and the planted deployed read by true k. Three series, fixed across the set
(experiments/figs/diag_fig.py COL): the deployed read (frames then question, no fence) with the needles PLANTED
in view (grey, solid) and under the benchmark's own UNIFORM sampling (grey, dashed), and the fenced oracle-GATED
read on the planted frames (blue). Chance = 0.25 (4 options) as a light band. Every plotted number -> <out>/<name>.csv;
one ABOUT line in <out>/README.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.curve_table import K_BUCKETS, acc, load_rows, newest_eval  # noqa: E402

COL = dict(plain="#8a8f96", gated="#2f6fd6", ink="#1c1e21", muted="#6b6f75", grid="#e4e6e9", chance="#eceef0")
KCOL = ["#aeb2b8", "#868b92", "#5a5f66", "#1c1e21"]           # one-hue ramp, light -> dark = k small -> large
FAMILY = {"retrieval": ("ret_edit1", "ret_insert1", "ret_insert2"), "ordering": ("ord_edit1", "ord_insert1", "ord_insert2"),
          "counting (cnt_edit1, cnt_insert1)": ("cnt_edit1", "cnt_insert1")}
SERIES = [  # (label, cell pattern, colour, marker, linestyle)
    ("deployed read, needles in view (planted)", "s1_eval/{rtag}/planted_N{N}/plain", COL["plain"], "o", "-"),
    ("deployed read, official sampling (uniform)", "s1_eval/{rtag}/uniform_N{N}/plain", COL["plain"], "s", "--"),
    ("fenced + oracle gate, needles in view", "s3_oracle/{rtag}/planted_N{N}/gated", COL["gated"], "^", "-"),
]
N_LIST = (8, 16, 32, 64, 128, 256)


def family_mean(rows: List[Dict[str, str]], qtypes: Tuple[str, ...]) -> Tuple[float, int]:
    """Mean over the subtasks of their accuracies (the benchmark's own family rule), n = rows used."""
    per = [acc([r for r in rows if r["qtype"] == q]) for q in qtypes]
    per = [a for a in per if a[1]]
    return (sum(a for a, _ in per) / len(per) if per else float("nan")), sum(n for _, n in per)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="vnbench")
    ap.add_argument("--backbone", default="qwen2.5-vl-7b")
    ap.add_argument("--rtag", default="native_mp151200")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    base = REPO / "outputs" / "scaleup" / args.dataset / args.backbone
    out = args.out or base / "fig"
    out.mkdir(parents=True, exist_ok=True)
    k_of = {r["qid"]: len(r["evidence"]) for r in json.loads((REPO / "data" / args.dataset / "json" / "test.json").read_text())}
    data: Dict[str, Dict[int, List[Dict[str, str]]]] = {}
    for label, pat, *_ in SERIES:
        data[label] = {}
        for n in N_LIST:
            p = newest_eval(base / pat.format(rtag=args.rtag, N=n))
            if p is not None:
                data[label][n] = load_rows(p)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COL["muted"], "axes.labelcolor": COL["ink"], "xtick.color": COL["ink"],
                         "ytick.color": COL["ink"], "axes.grid": True, "grid.color": COL["grid"], "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    rows_csv: List[Dict[str, object]] = []

    def end_labels(ax, ends: List[Tuple[float, float, str]], min_gap: float = 0.06) -> None:
        """Direct labels at each series' last point, pushed apart vertically when they would collide."""
        ends = sorted(ends, key=lambda e: e[1])
        ys = [y for _, y, _ in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + min_gap)
        for (x, y, text), y_lab in zip(ends, ys):
            ax.annotate(text, (x, y), xytext=(5, (y_lab - y) * 72 * 3.0 / 1.02), textcoords="offset points", va="center",
                        fontsize=8, color=COL["ink"])

    for ax, (fam, qtypes) in zip(axes[:3], FAMILY.items()):
        ax.axhspan(0.0, 0.25, color=COL["chance"], lw=0, zorder=0)
        ends: List[Tuple[float, float, str]] = []
        for label, pat, colour, marker, ls in SERIES:
            pts = [(n, *family_mean(rows, qtypes)) for n, rows in sorted(data[label].items())]
            pts = [(n, a, m) for n, a, m in pts if m]
            if not pts:
                continue
            ax.plot([n for n, _, _ in pts], [a for _, a, _ in pts], color=colour, marker=marker, ls=ls, lw=2, ms=6,
                    mec="white", mew=1, label=label, zorder=3)
            ends.append((pts[-1][0], pts[-1][1], f"{pts[-1][1]:.2f}"))
            rows_csv += [{"panel": fam, "series": label, "N": n, "acc": round(a, 4), "n_rows": m} for n, a, m in pts]
        end_labels(ax, ends)
        ax.set_title(fam, fontsize=10, loc="left", color=COL["ink"])
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_LIST)
        ax.set_xticklabels([str(n) for n in N_LIST])
        ax.set_xlabel("frames N")
    ax = axes[3]
    ax.axhspan(0.0, 0.25, color=COL["chance"], lw=0, zorder=0)
    planted = data[SERIES[0][0]]
    for (lo, hi), colour in zip(K_BUCKETS, KCOL):
        pts = [(n, *acc([r for r in rows if lo <= k_of.get(r["qid"], -1) <= hi])) for n, rows in sorted(planted.items())]
        pts = [(n, a, m) for n, a, m in pts if m]
        if not pts:
            continue
        lab = f"k={lo}" if lo == hi else f"k={lo}-{hi}"
        ax.plot([n for n, _, _ in pts], [a for _, a, _ in pts], color=colour, marker="o", lw=2, ms=6, mec="white", mew=1, zorder=3)
        ax.annotate(lab, pts[-1][:2], xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=COL["ink"])
        rows_csv += [{"panel": "planted deployed read by k", "series": lab, "N": n, "acc": round(a, 4), "n_rows": m} for n, a, m in pts]
    ax.set_title("deployed read, needles in view, by needle count k", fontsize=10, loc="left", color=COL["ink"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_LIST[:-1])
    ax.set_xticklabels([str(n) for n in N_LIST[:-1]])
    ax.set_xlabel("frames N")
    axes[0].set_ylabel("accuracy (4-way MCQ)")
    axes[0].set_ylim(0.0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("VNBench, frozen Qwen2.5-VL-7B: with the needles in view, retrieval holds and aggregation collapses with N; "
                 "the fenced oracle gate is flat but cannot order or count the kept blocks", fontsize=10.5, x=0.01, ha="left", color=COL["ink"])
    fig.text(0.01, -0.09, "1,350 try-0 rows; lo resolution (max_pixels 151,200); chance = 0.25 (grey band); family = mean of its subtasks; "
             "planted N = k needle frames + N-k fillers; uniform = the benchmark's own grid (needles often not sampled); "
             "oracle gate keeps exactly the k needle blocks.", fontsize=7.5, color=COL["muted"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    name = "F_vnbench_collapse"
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["panel", "series", "N", "acc", "n_rows"])
        w.writeheader()
        w.writerows(rows_csv)
    about = (f"- `{name}`: accuracy vs N per family (retrieval / ordering / counting) for the deployed read with the needles planted, "
             f"the deployed read under uniform sampling, and the fenced oracle-gated read; plus the planted read by true k. "
             f"Source: the newest eval.csv per cell under s1_eval/{args.rtag} and s3_oracle/{args.rtag}.\n")
    readme = out / "README.md"
    lines = [ln for ln in (readme.read_text().splitlines(keepends=True) if readme.exists() else []) if not ln.startswith(f"- `{name}`")]
    readme.write_text("".join(lines) + about)
    print(f"wrote {out / name}.png/.pdf/.csv ({len(rows_csv)} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

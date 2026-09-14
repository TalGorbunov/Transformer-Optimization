"""Summarize every registered P1 cell and plot fixed-epoch results on CPU Slurm."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_native_aggregation import paired, summarize


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Submit the report/plot job through Slurm")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = REPO / "outputs/native_aggregation/p1"
    labels = ("lora", "raw64", "centered64", "raw16", "centered16")
    runs, cells, rows_by_cell = {}, {}, {}
    for label in labels:
        runs[label], = sorted((root / label).glob("*/summary.json"))
        run = runs[label].parent
        cells[label] = {}
        for epoch in (3, 9):
            data = json.loads((run / f"test_epoch{epoch}.json").read_text())
            rows = data["predictions"]
            rows_by_cell[label, epoch] = rows
            cells[label][epoch] = {
                "per_n": data["metrics"],
                "in_range": summarize([r for r in rows if r["n_frames"] in (8, 32)]),
                "out_of_range": summarize([r for r in rows if r["n_frames"] in (64, 128)]),
            }
    comparisons = {}
    for epoch in (3, 9):
        for left, right in (("lora", "raw64"), ("lora", "centered64"),
                            ("lora", "raw16"), ("lora", "centered16"),
                            ("raw64", "centered64"), ("raw16", "centered16"),
                            ("raw64", "raw16"), ("centered64", "centered16")):
            for lengths, ns in (("in_range", (8, 32)), ("out_of_range", (64, 128))):
                comparisons[f"epoch{epoch}:{left}->{right}:{lengths}"] = paired(
                    [r for r in rows_by_cell[left, epoch] if r["n_frames"] in ns],
                    [r for r in rows_by_cell[right, epoch] if r["n_frames"] in ns])

    def metric(label, region, key="strict_exact", epoch=9):
        return cells[label][epoch][region][key]

    def gain(left, right, region="out_of_range", key="strict_exact"):
        return metric(right, region, key) - metric(left, region, key)

    criteria = {
        "P1.1_center64": gain("raw64", "centered64") >= .10
            and gain("raw64", "centered64", key="gold_first_token_nll") <= -.30
            and gain("raw64", "centered64", region="in_range") >= -.05,
        "P1.2_block16": any(gain(a, b) >= .10 for a, b in
                               (("raw64", "raw16"), ("centered64", "centered16"))),
        "P1.3_duration_raw64": metric("raw64", "in_range")
            - metric("raw64", "in_range", epoch=3) >= .10,
        "any_branch_meets_native_gate": any(
            gain("lora", label) >= .10 and gain("lora", label, region="in_range") >= -.05
            for label in labels[1:]),
    }
    report = dict(scope="Exploratory; P0 samples reused; fixed epochs, one seed",
                  runs={k: str(v.parent) for k, v in runs.items()}, cells=cells,
                  paired_comparisons=comparisons, registered_thresholds=criteria)
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    colors = {"lora": "#222222", "raw64": "#D55E00", "centered64": "#D55E00",
              "raw16": "#0072B2", "centered16": "#0072B2"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for row, epoch in enumerate((3, 9)):
        for label in labels:
            records = sorted(cells[label][epoch]["per_n"], key=lambda x: x["n_frames"])
            xs = [record["n_frames"] for record in records]
            for col, key in enumerate(("exact", "gold_first_token_nll")):
                axes[row, col].plot(xs, [record[key] for record in records],
                    label=label, color=colors[label], marker="o", markersize=4,
                    linestyle="--" if label.startswith("raw") else "-")
                axes[row, col].set_xscale("log", base=2)
                axes[row, col].set_xticks([8, 32, 64, 128], [8, 32, 64, 128])
                axes[row, col].grid(alpha=.2)
                axes[row, col].set_title(f"Epoch {epoch}: {'native exact match' if col == 0 else 'first-answer-token NLL'}")
        axes[row, 0].set_ylim(0, 1)
    for axis in axes[-1]:
        axis.set_xlabel("Number of frames (training N ≤ 32)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("P1 diagnostic · Qwen2.5-3B · 72 examples per length · one seed", y=.945, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(root / "fixed_epochs.png", dpi=180)
    fig.savefig(root / "fixed_epochs.pdf")
    print(json.dumps(criteria, indent=2))
    for label in labels:
        print(label, "epoch9 exact counts", [m["correct"] for m in cells[label][9]["per_n"]],
              "OOD", metric(label, "out_of_range"))


if __name__ == "__main__":
    main()

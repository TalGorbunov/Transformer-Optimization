#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_NAME = "message_memory_carrier_update_seq8_7b"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
BASELINE = "baseline_no_injection"
LAYER_LOCAL = "layer_local_message_memory"
CUMULATIVE = "cumulative_sum_message_memory"
METHOD_ORDER = [BASELINE, LAYER_LOCAL, CUMULATIVE]
METHOD_LABELS = {
    BASELINE: "baseline",
    LAYER_LOCAL: "layer-local",
    CUMULATIVE: "cumulative-sum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate message-memory carrier update seq_len=8 7B runs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--include-smoke", action="store_true", default=False)
    parser.add_argument("--no-plots", action="store_true", default=False)
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(leading)
    for key in sorted({key for row in rows for key in row.keys()}):
        if key not in fields:
            fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_mean(values: Iterable[Any], default: float = math.nan) -> float:
    vals = [float(v) for v in values if finite_float(v) is not None]
    return float(np.mean(vals)) if vals else float(default)


def method_from(row: Dict[str, Any], config: Dict[str, Any], run_dir: Path) -> str:
    method = str(row.get("method") or row.get("memory_variant") or config.get("memory_variant") or "")
    if method:
        return method
    name = run_dir.name
    if "cumulative" in name:
        return CUMULATIVE
    if "layer_local" in name or "layer-local" in name:
        return LAYER_LOCAL
    if "baseline" in name:
        return BASELINE
    return "unknown"


def enrich(row: Dict[str, Any], path: Path) -> Dict[str, Any]:
    run_dir = path.parent
    config = load_json(run_dir / "run_config.json")
    out = dict(row)
    method = method_from(out, config, run_dir)
    out["method"] = method
    out["memory_variant"] = method
    out["method_label"] = METHOD_LABELS.get(method, method)
    out["run_dir"] = os.fspath(run_dir)
    out["run_name"] = run_dir.name
    out["source_file"] = os.fspath(path)
    out["seed"] = config.get("seed", out.get("seed", ""))
    out["layer_start"] = config.get("layer_start", out.get("layer_start", ""))
    out["layer_end"] = config.get("layer_end", out.get("layer_end", ""))
    out["d_mem"] = config.get("d_mem", out.get("d_mem", ""))
    out["message_mode"] = config.get("message_mode", out.get("message_mode", ""))
    return out


def collect(root: Path, filename: str, *, include_smoke: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob(filename)):
        if path.parent == root:
            continue
        if not include_smoke and "smoke" in path.parts:
            continue
        try:
            rows.extend(enrich(row, path) for row in read_csv_rows(path))
        except Exception as exc:
            print(f"Skipping {path}: {type(exc).__name__}: {exc}")
    return rows


def grouped_mean(rows: Sequence[Dict[str, Any]], group_keys: Sequence[str], value_keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key, "") for key in group_keys)].append(row)
    out: List[Dict[str, Any]] = []
    for key, bucket in sorted(buckets.items(), key=lambda item: tuple(str(x) for x in item[0])):
        row = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        row["num_runs"] = len({item.get("run_dir", "") for item in bucket})
        row["n_rows"] = len(bucket)
        for value_key in value_keys:
            row[value_key] = finite_mean(item.get(value_key) for item in bucket)
        out.append(row)
    return out


def method_accuracy(summary_rows: Sequence[Dict[str, Any]], method: str) -> Optional[float]:
    vals = [finite_float(row.get("accuracy")) for row in summary_rows if row.get("method") == method]
    clean = [float(v) for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def method_count_means(per_count_rows: Sequence[Dict[str, Any]], value_key: str) -> Dict[Tuple[str, int], float]:
    buckets: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in per_count_rows:
        method = str(row.get("method"))
        count = finite_float(row.get("evidence_count"))
        value = finite_float(row.get(value_key))
        if count is not None and value is not None:
            buckets[(method, int(count))].append(float(value))
    return {key: float(np.mean(values)) for key, values in buckets.items() if values}


def compare_counts(per_count_rows: Sequence[Dict[str, Any]], a: str, b: str) -> Tuple[List[int], List[int]]:
    means = method_count_means(per_count_rows, "accuracy")
    improved: List[int] = []
    worsened: List[int] = []
    for count in range(9):
        if (a, count) not in means or (b, count) not in means:
            continue
        delta = means[(b, count)] - means[(a, count)]
        if delta > 1e-9:
            improved.append(count)
        elif delta < -1e-9:
            worsened.append(count)
    return improved, worsened


def save_combined_line(
    plots_dir: Path,
    filename: str,
    per_count_rows: Sequence[Dict[str, Any]],
    key: str,
    ylabel: str,
    title: str,
) -> None:
    means = method_count_means(per_count_rows, key)
    counts = sorted({count for _method, count in means})
    if not counts:
        counts = list(range(9))
    plt.figure(figsize=(7.6, 4.8))
    for method in METHOD_ORDER:
        ys = [means.get((method, count), math.nan) for count in counts]
        if any(math.isfinite(float(y)) for y in ys):
            plt.plot(counts, ys, marker="o", linewidth=1.9, label=METHOD_LABELS.get(method, method))
    plt.xlabel("Evidence count")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(counts)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=180, bbox_inches="tight")
    plt.close()


def make_plots(root: Path, summary_rows: Sequence[Dict[str, Any]], per_count_rows: Sequence[Dict[str, Any]]) -> None:
    plots_dir = root
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_combined_line(
        plots_dir,
        "combined_accuracy_vs_evidence_count.png",
        per_count_rows,
        "accuracy",
        "Accuracy",
        "Combined Accuracy vs Evidence Count",
    )
    save_combined_line(
        plots_dir,
        "combined_margin_vs_evidence_count.png",
        per_count_rows,
        "mean_margin",
        "Mean margin",
        "Combined Margin vs Evidence Count",
    )
    save_combined_line(
        plots_dir,
        "combined_gate_sum_vs_evidence_count.png",
        per_count_rows,
        "mean_gate_sum",
        "Mean gate sum",
        "Combined Gate Sum vs Evidence Count",
    )
    save_combined_line(
        plots_dir,
        "combined_update_l2_vs_evidence_count.png",
        per_count_rows,
        "mean_update_norm",
        "Mean update L2",
        "Combined Update L2 vs Evidence Count",
    )

    accs = [method_accuracy(summary_rows, method) for method in METHOD_ORDER]
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    plt.figure(figsize=(6.2, 4.5))
    plt.bar(np.arange(len(labels)), [0.0 if value is None else float(value) for value in accs])
    plt.xticks(np.arange(len(labels)), labels, rotation=15, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Combined Accuracy")
    plt.ylim(0, max(1.0, max([value for value in accs if value is not None] or [0.0]) * 1.1))
    plt.tight_layout()
    plt.savefig(plots_dir / "combined_accuracy_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("accuracy", "Accuracy"),
        ("mean_margin", "Margin"),
        ("mean_gate_sum", "Gate sum"),
        ("mean_update_norm", "Update L2"),
    ]
    means_by_key = {key: method_count_means(per_count_rows, key) for key, _label in panels}
    counts = sorted({count for means in means_by_key.values() for _method, count in means})
    if not counts:
        counts = list(range(9))
    for ax, (key, label) in zip(axes.reshape(-1), panels):
        means = means_by_key[key]
        for method in METHOD_ORDER:
            ys = [means.get((method, count), math.nan) for count in counts]
            if any(math.isfinite(float(y)) for y in ys):
                ax.plot(counts, ys, marker="o", linewidth=1.6, label=METHOD_LABELS.get(method, method))
        ax.set_title(label)
        ax.set_xlabel("Evidence count")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Message Memory Carrier Update Diagnostic Dashboard")
    fig.tight_layout()
    fig.savefig(plots_dir / "combined_diagnostic_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_readme(root: Path, summary_rows: Sequence[Dict[str, Any]], per_count_rows: Sequence[Dict[str, Any]], metrics_rows: Sequence[Dict[str, Any]]) -> None:
    baseline_acc = method_accuracy(summary_rows, BASELINE)
    local_acc = method_accuracy(summary_rows, LAYER_LOCAL)
    cumulative_acc = method_accuracy(summary_rows, CUMULATIVE)
    local_imp, local_worse = compare_counts(per_count_rows, BASELINE, LAYER_LOCAL)
    cum_imp, cum_worse = compare_counts(per_count_rows, LAYER_LOCAL, CUMULATIVE)

    gate_corrs = {
        method: finite_mean(row.get("corr_gate_sum_evidence_count") for row in summary_rows if row.get("method") == method)
        for method in METHOD_ORDER
    }
    update_l2 = {
        method: finite_mean(
            (row.get("approx_total_update_l2") for row in summary_rows if row.get("method") == method),
            default=0.0,
        )
        for method in METHOD_ORDER
    }
    token_fail = sum(int(finite_float(row.get("token_selection_failures")) or 0) for row in summary_rows)
    frame_fail = sum(int(finite_float(row.get("frame_grouping_failures")) or 0) for row in summary_rows)
    wrong_rows = [row for row in metrics_rows if int(finite_float(row.get("correct")) or 0) == 0]
    under = sum(1 for row in wrong_rows if int(finite_float(row.get("pred_answer")) or 0) < int(finite_float(row.get("gold_answer")) or 0))
    over = sum(1 for row in wrong_rows if int(finite_float(row.get("pred_answer")) or 0) > int(finite_float(row.get("gold_answer")) or 0))
    if under > over * 1.25:
        failure_shape = "undercounting"
    elif over > under * 1.25:
        failure_shape = "overcounting"
    else:
        failure_shape = "mixed/random confusion"

    def acc_text(value: Optional[float]) -> str:
        return "unavailable" if value is None else f"{value:.4f}"

    def update_text(value: float) -> str:
        if value < 1e-3:
            return "near zero"
        if value > 10.0:
            return "huge"
        return "reasonable"

    lines = [
        "# Combined Message Memory Carrier Update seq_len=8 7B",
        "",
        f"- Baseline accuracy: {acc_text(baseline_acc)}",
        f"- Layer-local accuracy: {acc_text(local_acc)}",
        f"- Cumulative-sum accuracy: {acc_text(cumulative_acc)}",
        (
            f"- Did layer-local improve over baseline? {local_acc > baseline_acc}"
            if baseline_acc is not None and local_acc is not None
            else "- Did layer-local improve over baseline? unavailable"
        ),
        (
            f"- Did cumulative-sum improve over layer-local? {cumulative_acc > local_acc}"
            if cumulative_acc is not None and local_acc is not None
            else "- Did cumulative-sum improve over layer-local? unavailable"
        ),
        f"- Layer-local vs baseline evidence counts: improved={local_imp}, worsened={local_worse}",
        f"- Cumulative vs layer-local evidence counts: improved={cum_imp}, worsened={cum_worse}",
        "- Gate sum correlations: "
        + ", ".join(f"{METHOD_LABELS[m]}={gate_corrs[m]:.4f}" if math.isfinite(gate_corrs[m]) else f"{METHOD_LABELS[m]}=NA" for m in METHOD_ORDER),
        "- Update norms: "
        + ", ".join(f"{METHOD_LABELS[m]}={update_text(update_l2[m])} ({update_l2[m]:.6f})" for m in METHOD_ORDER),
        f"- Failures look like: {failure_shape} (wrong={len(wrong_rows)}, under={under}, over={over})",
        f"- Localization failures: token_selection={token_fail}, frame_grouping={frame_fail}",
        "",
        "## Interpretation Guide",
        "",
        "- If accuracy is low but gate sum correlates with evidence count, aggregation is working, but residual update translation into Qwen is weak.",
        "- If accuracy is low and gate sum does not correlate with evidence count, message extraction/querying is not identifying useful frame evidence.",
        "- If update norm is near zero, gamma/init/lr/reg may be too conservative.",
        "- If update norm is huge but accuracy low, the adapter is blasting residuals in wrong directions; add stronger structure or pretrain against oracle codebook direction.",
        "- If layer-local works but cumulative does not, cumulative memory update is washing out layer-specific messages.",
        "- If cumulative works better, cross-layer memory accumulation is helping reduce carrier saturation.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).resolve()
    summary_rows = collect(root, "summary.csv", include_smoke=bool(args.include_smoke))
    per_count_rows = collect(root, "accuracy_by_evidence_count.csv", include_smoke=bool(args.include_smoke))
    metrics_rows = collect(root, "metrics.csv", include_smoke=bool(args.include_smoke))
    if not summary_rows and not per_count_rows and not metrics_rows:
        raise FileNotFoundError(f"No message-memory run CSVs found under {root}")

    combined_summary = grouped_mean(
        summary_rows,
        group_keys=["method", "method_label"],
        value_keys=[
            "accuracy",
            "mean_margin",
            "mean_gold_logit",
            "mean_update_energy",
            "approx_total_update_l2",
            "mean_gate_sum",
            "corr_gate_sum_evidence_count",
            "corr_update_norm_evidence_count",
            "token_selection_failures",
            "frame_grouping_failures",
        ],
    )
    combined_per_count = grouped_mean(
        per_count_rows,
        group_keys=["method", "method_label", "evidence_count"],
        value_keys=[
            "accuracy",
            "mean_margin",
            "mean_predicted_count",
            "mean_gate_sum",
            "mean_update_norm",
            "mean_message_norm",
            "mean_memory_norm",
        ],
    )
    write_csv_rows(
        root / "combined_summary.csv",
        combined_summary,
        ["method", "method_label", "num_runs", "accuracy", "mean_margin", "mean_gate_sum", "approx_total_update_l2"],
    )
    write_csv_rows(
        root / "combined_accuracy_by_evidence_count.csv",
        combined_per_count,
        ["method", "method_label", "evidence_count", "num_runs", "accuracy", "mean_margin", "mean_gate_sum", "mean_update_norm"],
    )
    write_csv_rows(
        root / "combined_metrics.csv",
        metrics_rows,
        ["method", "method_label", "run_name", "sample_id", "evidence_count", "gold_answer", "pred_answer", "correct", "margin"],
    )
    if not bool(args.no_plots):
        make_plots(root, summary_rows, per_count_rows)
    write_readme(root, summary_rows, per_count_rows, metrics_rows)
    print(f"Wrote combined outputs under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

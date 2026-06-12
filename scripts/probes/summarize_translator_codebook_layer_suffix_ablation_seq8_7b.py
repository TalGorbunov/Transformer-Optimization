#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
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


EXPERIMENT_NAME = "translator_codebook_layer_suffix_ablation_seq8_7b"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "outputs" / "translator_ablation_gold_count_seq8_7b"
WINDOW_ORDER = ["14-17", "15-17", "16-17", "17-17"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate translator codebook layer suffix ablation outputs."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument(
        "--include-baseline-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If no baseline exists in output-root, include one no-injection baseline from baseline-root.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(leading)
    for key in sorted({key for row in rows for key in row.keys()}):
        if key not in fields:
            fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        return {}
    with config_path.open("r") as handle:
        return json.load(handle)


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_int(value: Any) -> Optional[int]:
    number = finite_float(value)
    if number is None:
        return None
    return int(number)


def config_layer_window(config_name: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"_l(\d+)(?:_(\d+))?(?:_|$)", str(config_name))
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) is not None else start
    return start, end


def normalize_window_label(start: Optional[int], end: Optional[int], raw: Any = "") -> str:
    if start is not None and end is not None:
        return f"{int(start)}-{int(end)}"
    text = str(raw or "").strip()
    if not text:
        return "baseline"
    if "-" not in text and text.isdigit():
        return f"{int(text)}-{int(text)}"
    return text


def method_short(method: str) -> str:
    text = str(method)
    if text in {"static", "static_count_codebook"}:
        return "static"
    if text in {"layer", "layer_count_codebook"}:
        return "layer"
    if text in {"baseline", "baseline_no_injection"}:
        return "baseline"
    return text


def alpha_regime_from(config_name: str, method: str) -> str:
    if method_short(method) == "baseline":
        return "baseline"
    if "energy_norm" in str(config_name):
        return "energy_norm"
    if "fixed_alpha" in str(config_name):
        return "fixed_alpha"
    return "unknown"


def coerce_energy_fields(row: Dict[str, Any], source_key: str = "mean_injection_norm") -> None:
    raw_energy = row.get("mean_injection_energy", row.get(source_key, ""))
    energy = finite_float(raw_energy)
    row["mean_injection_energy"] = "" if energy is None else energy
    row["approx_total_l2"] = "" if energy is None else math.sqrt(max(0.0, energy))


def enrich_row(row: Dict[str, Any], *, path: Path, root_role: str) -> Dict[str, Any]:
    run_dir = path.parent
    config = load_run_config(run_dir)
    out = dict(row)
    out["source_file"] = str(path)
    out["run_dir"] = str(run_dir)
    out["source_root_role"] = str(root_role)

    config_name = str(out.get("config_name") or config.get("config_name") or run_dir.name)
    method = str(out.get("method") or config.get("method") or "")
    short = method_short(method)
    parsed_window = config_layer_window(config_name)
    layer_start = finite_int(out.get("layer_start")) or finite_int(config.get("layer_start"))
    layer_end = finite_int(out.get("layer_end")) or finite_int(config.get("layer_end"))
    if parsed_window is not None and short != "baseline":
        layer_start, layer_end = parsed_window
    if short == "baseline":
        layer_start, layer_end = None, None

    out["config_name"] = config_name
    out["method"] = method
    out["method_short"] = short
    out["token_group"] = str(out.get("token_group") or config.get("token_group") or "")
    out["resolved_token_group"] = str(out.get("resolved_token_group") or config.get("resolved_token_group") or out["token_group"])
    out["alpha"] = finite_float(out.get("alpha")) if finite_float(out.get("alpha")) is not None else config.get("alpha", "")
    out["alpha_regime"] = alpha_regime_from(config_name, method)
    out["layer_start"] = "" if layer_start is None else int(layer_start)
    out["layer_end"] = "" if layer_end is None else int(layer_end)
    out["window"] = normalize_window_label(layer_start, layer_end, out.get("layer_window"))
    out["window_start"] = "" if layer_start is None else int(layer_start)
    out["window_end"] = "" if layer_end is None else int(layer_end)
    out["is_baseline"] = int(short == "baseline")
    coerce_energy_fields(out)
    return out


def collect_named_csv(root: Path, filename: str, root_role: str) -> List[Dict[str, Any]]:
    if not root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob(filename)):
        for row in read_csv_rows(path):
            rows.append(enrich_row(row, path=path, root_role=root_role))
    return rows


def pick_baseline_run(summary_rows: Sequence[Dict[str, Any]]) -> Optional[Path]:
    candidates = [row for row in summary_rows if row.get("method_short") == "baseline"]
    if not candidates:
        return None

    def score(row: Dict[str, Any]) -> Tuple[int, str]:
        token_group = str(row.get("token_group", ""))
        config_name = str(row.get("config_name", ""))
        preferred = int(token_group == "room_char") + int(config_name.startswith("A_room_char"))
        return preferred, str(row.get("run_dir", ""))

    best = sorted(candidates, key=score)[-1]
    return Path(str(best["run_dir"]))


def append_baseline_fallback(
    *,
    summary_rows: List[Dict[str, Any]],
    accuracy_rows: List[Dict[str, Any]],
    baseline_root: Path,
) -> None:
    if any(row.get("method_short") == "baseline" for row in summary_rows):
        return
    baseline_summaries = collect_named_csv(baseline_root, "summary.csv", "baseline_source")
    baseline_run = pick_baseline_run(baseline_summaries)
    if baseline_run is None:
        return
    summary_path = baseline_run / "summary.csv"
    accuracy_path = baseline_run / "accuracy_by_evidence_count.csv"
    for row in read_csv_rows(summary_path):
        enriched = enrich_row(row, path=summary_path, root_role="baseline_source")
        enriched["config_name"] = "baseline_no_injection_reference"
        enriched["alpha_regime"] = "baseline"
        summary_rows.append(enriched)
    if accuracy_path.is_file():
        for row in read_csv_rows(accuracy_path):
            enriched = enrich_row(row, path=accuracy_path, root_role="baseline_source")
            enriched["config_name"] = "baseline_no_injection_reference"
            enriched["alpha_regime"] = "baseline"
            accuracy_rows.append(enriched)


def collect_train_history(root: Path) -> List[Dict[str, Any]]:
    rows = collect_named_csv(root, "train_history.csv", "output_root")
    for row in rows:
        raw_energy = row.get("mean_val_injection_energy", row.get("mean_val_injection_norm", ""))
        energy = finite_float(raw_energy)
        row["mean_val_injection_energy"] = "" if energy is None else energy
        row["mean_val_approx_total_l2"] = "" if energy is None else math.sqrt(max(0.0, energy))
    return rows


def latest_by_key(rows: Sequence[Dict[str, Any]], key_fields: Sequence[str]) -> List[Dict[str, Any]]:
    selected: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        previous = selected.get(key)
        if previous is None or str(row.get("run_dir", "")) > str(previous.get("run_dir", "")):
            selected[key] = row
    return list(selected.values())


def row_float(row: Dict[str, Any], key: str) -> float:
    value = finite_float(row.get(key))
    return float(value) if value is not None else math.nan


def plot_metric_by_window(rows: Sequence[Dict[str, Any]], plots_dir: Path, metric: str, ylabel: str, filename: str) -> None:
    data_rows = [row for row in rows if row.get("method_short") in {"static", "layer"} and row.get("window") in WINDOW_ORDER]
    if not data_rows:
        return
    labels = [
        ("static", "fixed_alpha"),
        ("layer", "fixed_alpha"),
        ("static", "energy_norm"),
        ("layer", "energy_norm"),
    ]
    colors = {
        ("static", "fixed_alpha"): "#1f77b4",
        ("layer", "fixed_alpha"): "#ff7f0e",
        ("static", "energy_norm"): "#2ca02c",
        ("layer", "energy_norm"): "#d62728",
    }
    by_key = {(row["method_short"], row["alpha_regime"], row["window"]): row for row in data_rows}
    plt.figure(figsize=(8.0, 4.8))
    for method, regime in labels:
        ys = [row_float(by_key[(method, regime, window)], metric) if (method, regime, window) in by_key else math.nan for window in WINDOW_ORDER]
        if all(math.isnan(y) for y in ys):
            continue
        plt.plot(WINDOW_ORDER, ys, marker="o", linewidth=2.0, label=f"{method} / {regime}", color=colors[(method, regime)])
    baseline_rows = [row for row in rows if row.get("method_short") == "baseline"]
    if baseline_rows and metric in {"overall_acc", "mean_margin"}:
        baseline = row_float(baseline_rows[0], metric)
        if math.isfinite(baseline):
            plt.axhline(baseline, color="#666666", linestyle="--", linewidth=1.2, label="baseline")
    plt.xlabel("Layer window")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / filename, dpi=180, bbox_inches="tight")
    plt.close()


def plot_accuracy_heatmap(rows: Sequence[Dict[str, Any]], plots_dir: Path) -> None:
    labels = [
        ("static / fixed", "static", "fixed_alpha"),
        ("layer / fixed", "layer", "fixed_alpha"),
        ("static / energy norm", "static", "energy_norm"),
        ("layer / energy norm", "layer", "energy_norm"),
    ]
    by_key = {(row.get("method_short"), row.get("alpha_regime"), row.get("window")): row for row in rows}
    matrix = np.full((len(labels), len(WINDOW_ORDER)), np.nan, dtype=float)
    for i, (_label, method, regime) in enumerate(labels):
        for j, window in enumerate(WINDOW_ORDER):
            row = by_key.get((method, regime, window))
            if row is not None:
                matrix[i, j] = row_float(row, "overall_acc")

    baseline_rows = [row for row in rows if row.get("method_short") == "baseline"]
    ylabels = [label for label, _method, _regime in labels]
    if baseline_rows:
        baseline = row_float(baseline_rows[0], "overall_acc")
        if math.isfinite(baseline):
            matrix = np.vstack([matrix, np.full((1, len(WINDOW_ORDER)), baseline, dtype=float)])
            ylabels.append("baseline")

    if not np.isfinite(matrix).any():
        return
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.48 * len(ylabels) + 1.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(WINDOW_ORDER)))
    ax.set_xticklabels(WINDOW_ORDER)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Layer window")
    ax.set_title("Accuracy by Method and Window")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if math.isfinite(float(matrix[i, j])):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "accuracy_heatmap_method_x_window.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_acc_by_count(row: Dict[str, Any]) -> Dict[int, float]:
    raw = row.get("acc_by_evidence_count", "")
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    out: Dict[int, float] = {}
    for key, value in payload.items():
        number = finite_float(value)
        if number is not None:
            out[int(key)] = float(number)
    return out


def format_acc_sequence(rows: Sequence[Dict[str, Any]], method: str, regime: str) -> str:
    by_window = {row.get("window"): row for row in rows if row.get("method_short") == method and row.get("alpha_regime") == regime}
    pieces = []
    for window in WINDOW_ORDER:
        row = by_window.get(window)
        pieces.append(f"{window}: {'NA' if row is None else row_float(row, 'overall_acc'):.3g}" if row is not None else f"{window}: NA")
    return ", ".join(pieces)


def first_failing_counts(rows: Sequence[Dict[str, Any]], method: str, regime: str) -> str:
    by_window = {row.get("window"): row for row in rows if row.get("method_short") == method and row.get("alpha_regime") == regime}
    pieces = []
    for window in WINDOW_ORDER:
        row = by_window.get(window)
        if row is None:
            continue
        failures = [count for count, acc in sorted(parse_acc_by_count(row).items()) if acc < 1.0 - 1e-9]
        pieces.append(f"{window}: {'none' if not failures else ','.join(str(x) for x in failures)}")
    return "; ".join(pieces) if pieces else "NA"


def write_readme(output_root: Path, rows: Sequence[Dict[str, Any]], accuracy_rows: Sequence[Dict[str, Any]]) -> None:
    summary_rows = latest_by_key(rows, ["method_short", "alpha_regime", "window"])
    baseline_rows = [row for row in summary_rows if row.get("method_short") == "baseline"]
    baseline_text = "not included"
    if baseline_rows:
        baseline_text = f"{row_float(baseline_rows[0], 'overall_acc'):.3f} accuracy"

    def acc(method: str, regime: str, window: str) -> Optional[float]:
        for row in summary_rows:
            if row.get("method_short") == method and row.get("alpha_regime") == regime and row.get("window") == window:
                return row_float(row, "overall_acc")
        return None

    lines = [
        f"# {EXPERIMENT_NAME}",
        "",
        "Automatically aggregated translator gold-count layer suffix ablation results.",
        "",
        "## Files",
        "",
        "- `combined_summary.csv`: run-level summaries with `mean_injection_energy` and `approx_total_l2 = sqrt(mean_injection_energy)`.",
        "- `combined_accuracy_by_evidence_count.csv`: per-count accuracy/margin/energy rows.",
        "- `combined_train_history.csv`: training histories when present.",
        "- `plots/`: accuracy, margin, injection energy, and heatmap views.",
        "",
        "## Current Readout",
        "",
        f"- Baseline no-injection reference: {baseline_text}.",
    ]
    for regime in ("fixed_alpha", "energy_norm"):
        for method in ("static", "layer"):
            value = acc(method, regime, "17-17")
            text = "NA" if value is None else f"{value:.3f}"
            lines.append(f"- {method} / {regime} at 17-17 accuracy: {text}.")
    lines.extend(
        [
            "",
            "## Window Trends",
            "",
        ]
    )
    for regime in ("fixed_alpha", "energy_norm"):
        for method in ("static", "layer"):
            lines.append(f"- {method} / {regime}: {format_acc_sequence(summary_rows, method, regime)}")
    lines.extend(
        [
            "",
            "## First Per-Count Failures",
            "",
        ]
    )
    for regime in ("fixed_alpha", "energy_norm"):
        for method in ("static", "layer"):
            lines.append(f"- {method} / {regime}: {first_failing_counts(summary_rows, method, regime)}")
    lines.extend(
        [
            "",
            "## Questions To Check",
            "",
            "- Does static/layer still reach 100% with only 17-17?",
            "- Does performance degrade smoothly from 14-17 to 15-17 to 16-17 to 17-17?",
            "- Does layer-codebook need multiple layers less than static?",
            "- Under energy-normalized alpha, is the late-only layer still enough?",
            "- Per evidence count, which counts fail first when layers are removed?",
            "",
        ]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    summary_rows = collect_named_csv(output_root, "summary.csv", "output_root")
    accuracy_rows = collect_named_csv(output_root, "accuracy_by_evidence_count.csv", "output_root")
    if bool(args.include_baseline_source):
        append_baseline_fallback(summary_rows=summary_rows, accuracy_rows=accuracy_rows, baseline_root=baseline_root)

    summary_plot_rows = latest_by_key(summary_rows, ["method_short", "alpha_regime", "window"])
    train_history_rows = collect_train_history(output_root)

    write_csv_rows(
        output_root / "combined_summary.csv",
        summary_rows,
        [
            "config_name",
            "method",
            "method_short",
            "alpha_regime",
            "window",
            "overall_acc",
            "mean_margin",
            "mean_injection_energy",
            "approx_total_l2",
            "train_acc",
            "eval_acc",
            "acc_by_evidence_count",
            "source_root_role",
            "run_dir",
        ],
    )
    write_csv_rows(
        output_root / "combined_accuracy_by_evidence_count.csv",
        accuracy_rows,
        [
            "config_name",
            "method",
            "method_short",
            "alpha_regime",
            "window",
            "evidence_count",
            "n",
            "accuracy",
            "mean_margin",
            "mean_injection_energy",
            "approx_total_l2",
            "source_root_role",
            "run_dir",
        ],
    )
    if train_history_rows:
        write_csv_rows(
            output_root / "combined_train_history.csv",
            train_history_rows,
            [
                "config_name",
                "method",
                "method_short",
                "alpha_regime",
                "window",
                "epoch",
                "train_ce",
                "train_loss",
                "val_ce",
                "val_acc",
                "mean_val_injection_energy",
                "mean_val_approx_total_l2",
                "run_dir",
            ],
        )

    plots_dir = output_root / "plots"
    plot_metric_by_window(
        summary_plot_rows,
        plots_dir,
        "overall_acc",
        "Accuracy",
        "accuracy_by_window_method_alpha_regime.png",
    )
    plot_metric_by_window(
        summary_plot_rows,
        plots_dir,
        "mean_margin",
        "Mean margin",
        "margin_by_window_method_alpha_regime.png",
    )
    plot_metric_by_window(
        summary_plot_rows,
        plots_dir,
        "mean_injection_energy",
        "Mean injection energy",
        "injection_energy_by_window_method_alpha_regime.png",
    )
    plot_accuracy_heatmap(summary_plot_rows, plots_dir)
    write_readme(output_root, summary_rows, accuracy_rows)

    print(f"Wrote combined outputs under {output_root}")
    print(f"summary_rows={len(summary_rows)} accuracy_rows={len(accuracy_rows)} train_history_rows={len(train_history_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

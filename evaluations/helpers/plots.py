import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def plot_layer_metric_lines(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    selected_layers: List[int],
    metric_key: str,
    seq_len_label: Optional[str] = None,
    title_override: Optional[str] = None,
    filename_stem: str = "layer_metric_lines",
    line_label: Optional[str] = None,
    y_label: Optional[str] = None,
    x_label: str = "Layer",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Path]:
    if not sample_metrics or not selected_layers:
        return None

    per_layer_values: Dict[int, List[float]] = {layer_idx: [] for layer_idx in selected_layers}
    for sample in sample_metrics:
        for layer_metrics in sample.get("layer_metrics", {}).get("layers", []):
            layer_idx = int(layer_metrics["layer"])
            if layer_idx not in per_layer_values:
                continue
            value = layer_metrics.get(metric_key)
            if value is None:
                continue
            per_layer_values[layer_idx].append(float(value))

    if not any(per_layer_values[layer_idx] for layer_idx in selected_layers):
        return None

    rng = random.Random(seed)
    mean_vals: List[float] = []
    lo_vals: List[float] = []
    hi_vals: List[float] = []
    for layer_idx in selected_layers:
        values = per_layer_values[layer_idx]
        center, lo_value, hi_value = _bootstrap_center_and_ci(values, _mean, n_bootstrap, rng)
        mean_vals.append(center)
        lo_vals.append(lo_value)
        hi_vals.append(hi_value)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    label = line_label or metric_key
    ax.plot(selected_layers, mean_vals, linewidth=2.0, label=label)
    ax.fill_between(selected_layers, lo_vals, hi_vals, alpha=0.16)
    title = title_override or f"Mean {metric_key} by Layer"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label or f"Mean {metric_key}")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    tick_step = max(1, math.ceil(len(selected_layers) / 32))
    xticks = selected_layers[::tick_step]
    if selected_layers[-1] not in xticks:
        xticks.append(selected_layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{seq_len_label}" if seq_len_label else ""
    plot_path = output_dir / f"{filename_stem}{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_single_sample_layer_metric(
    sample_payload: Dict[str, Any],
    output_dir: Path,
    metric_key: str,
    seq_len_label: Optional[str] = None,
    title_override: Optional[str] = None,
    filename_stem: str = "sample_layer_metric",
    y_label: Optional[str] = None,
    file_prefix: Optional[str] = None,
    threshold: Optional[float] = None,
    peak_layer: Optional[int] = None,
    first_significant_layer: Optional[int] = None,
    x_label: str = "Layer",
) -> Optional[Path]:
    layer_metrics = list(sample_payload.get("layer_metrics", {}).get("layers", []))
    if not layer_metrics:
        return None

    layers: List[int] = []
    values: List[float] = []
    value_by_layer: Dict[int, float] = {}
    for layer_metric in layer_metrics:
        value = layer_metric.get(metric_key)
        if value is None:
            continue
        layer_idx = int(layer_metric["layer"])
        layers.append(layer_idx)
        values.append(float(value))
        value_by_layer[layer_idx] = float(value)

    if not layers:
        return None

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.plot(layers, values, marker="o", linewidth=2.0, label=metric_key)
    if threshold is not None:
        ax.axhline(float(threshold), color="#ff7f0e", linestyle="--", linewidth=1.4, label="threshold")
    if peak_layer is not None and int(peak_layer) in value_by_layer:
        ax.scatter(
            [int(peak_layer)],
            [value_by_layer[int(peak_layer)]],
            color="#d62728",
            s=70,
            zorder=3,
            label="peak",
        )
    if first_significant_layer is not None and int(first_significant_layer) in value_by_layer:
        ax.scatter(
            [int(first_significant_layer)],
            [value_by_layer[int(first_significant_layer)]],
            color="#2ca02c",
            s=70,
            zorder=3,
            label="first significant",
        )

    title = title_override or f"{sample_payload.get('sample_id', 'sample')} {metric_key} by Layer"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label or metric_key)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    tick_step = max(1, math.ceil(len(layers) / 32))
    xticks = layers[::tick_step]
    if layers[-1] not in xticks:
        xticks.append(layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{seq_len_label}" if seq_len_label else ""
    prefix = f"{file_prefix}_" if file_prefix else ""
    plot_path = output_dir / f"{prefix}{filename_stem}{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_last_token_importance_lines(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    selected_layers: List[int],
    seq_len_label: Optional[str] = None,
    title_override: Optional[str] = None,
    filename_stem: str = "last_token_importance_lines",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Path]:
    if not sample_metrics or not selected_layers:
        return None

    per_layer_values: Dict[int, List[float]] = {layer_idx: [] for layer_idx in selected_layers}
    for sample in sample_metrics:
        for layer_metrics in sample["layer_metrics"]["layers"]:
            layer_idx = int(layer_metrics["layer"])
            if layer_idx in per_layer_values:
                per_layer_values[layer_idx].append(float(layer_metrics["importance"]))

    rng = random.Random(seed)
    mean_vals: List[float] = []
    lo_vals: List[float] = []
    hi_vals: List[float] = []
    for layer_idx in selected_layers:
        values = per_layer_values[layer_idx]
        n = len(values)
        mean_value = (sum(values) / n) if n > 0 else 0.0
        if n <= 1:
            lo_value = hi_value = mean_value
        else:
            boot_means: List[float] = []
            for _ in range(n_bootstrap):
                sample = [values[rng.randrange(n)] for _ in range(n)]
                boot_means.append(sum(sample) / n)
            boot_means.sort()
            lo_idx = int(0.025 * (n_bootstrap - 1))
            hi_idx = int(0.975 * (n_bootstrap - 1))
            lo_value = boot_means[lo_idx]
            hi_value = boot_means[hi_idx]
        mean_vals.append(mean_value)
        lo_vals.append(lo_value)
        hi_vals.append(hi_value)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.plot(selected_layers, mean_vals, linewidth=2.0, label="last_token")
    ax.fill_between(selected_layers, lo_vals, hi_vals, alpha=0.16)
    title = title_override or "Mean Last-Token Importance by Layer"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean importance")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{seq_len_label}" if seq_len_label else ""
    plot_path = output_dir / f"{filename_stem}{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_group_importance_lines(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    num_layers: int,
    selected_layers: List[int],
    group_order: List[str],
    seq_len_label: Optional[str] = None,
    title_override: Optional[str] = None,
    filename_stem: str = "group_importance_lines",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Path]:
    if num_layers <= 0 or not sample_metrics or not selected_layers or not group_order:
        return None

    per_group_per_layer_values: Dict[str, Dict[int, List[float]]] = {
        group: {layer_idx: [] for layer_idx in selected_layers} for group in group_order
    }
    for sample in sample_metrics:
        for layer_metrics in sample["layer_metrics"]["layers"]:
            layer_idx = int(layer_metrics["layer"])
            if layer_idx not in per_group_per_layer_values[group_order[0]]:
                continue
            groups = list(layer_metrics["groups"])
            values = [float(x) for x in layer_metrics["r"]]
            by_group = {groups[idx]: values[idx] for idx in range(min(len(groups), len(values)))}
            for group_name in group_order:
                if group_name in by_group:
                    per_group_per_layer_values[group_name][layer_idx].append(by_group[group_name])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    rng = random.Random(seed)
    for group_name in group_order:
        mean_vals = []
        lo_vals = []
        hi_vals = []
        for layer_idx in selected_layers:
            values = per_group_per_layer_values[group_name][layer_idx]
            n = len(values)
            mean_value = (sum(values) / n) if n > 0 else 0.0
            if n <= 1:
                lo_value = hi_value = mean_value
            else:
                boot_means: List[float] = []
                for _ in range(n_bootstrap):
                    sample = [values[rng.randrange(n)] for _ in range(n)]
                    boot_means.append(sum(sample) / n)
                boot_means.sort()
                lo_idx = int(0.025 * (n_bootstrap - 1))
                hi_idx = int(0.975 * (n_bootstrap - 1))
                lo_value = boot_means[lo_idx]
                hi_value = boot_means[hi_idx]
            mean_vals.append(mean_value)
            lo_vals.append(lo_value)
            hi_vals.append(hi_value)
        line, = ax.plot(selected_layers, mean_vals, linewidth=2.0, label=group_name)
        ax.fill_between(selected_layers, lo_vals, hi_vals, color=line.get_color(), alpha=0.16)

    title = title_override or "Mean Group Importance by Layer"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean importance")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    tick_step = max(1, math.ceil(len(selected_layers) / 32))
    xticks = selected_layers[::tick_step]
    if selected_layers[-1] not in xticks:
        xticks.append(selected_layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{seq_len_label}" if seq_len_label else ""
    plot_path = output_dir / f"{filename_stem}{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_entropy_summary(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    n_bootstrap: int = 1000,
    seed: int = 0,
    seq_len_label: Optional[str] = None,
) -> Optional[Path]:
    entropy_by_layer: Dict[int, List[float]] = {}
    for sm in sample_metrics:
        for lmtr in sm["layer_metrics"]["layers"]:
            layer_idx = int(lmtr["layer"])
            entropy_by_layer.setdefault(layer_idx, []).append(float(lmtr["entropy"]))

    if not entropy_by_layer:
        return None

    rng = random.Random(seed)
    layers = sorted(entropy_by_layer.keys())
    means: List[float] = []
    medians: List[float] = []
    mean_lo: List[float] = []
    mean_hi: List[float] = []
    med_lo: List[float] = []
    med_hi: List[float] = []

    for layer_idx in layers:
        vals = entropy_by_layer[layer_idx]
        n = len(vals)
        sorted_vals = sorted(vals)
        mean_value = sum(vals) / n
        median_value = sorted_vals[n // 2] if n % 2 == 1 else 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])

        boot_mean: List[float] = []
        boot_median: List[float] = []
        for _ in range(n_bootstrap):
            sample = [vals[rng.randrange(n)] for _ in range(n)]
            sample_sorted = sorted(sample)
            boot_mean.append(sum(sample) / n)
            boot_median.append(
                sample_sorted[n // 2] if n % 2 == 1 else 0.5 * (sample_sorted[n // 2 - 1] + sample_sorted[n // 2])
            )

        boot_mean.sort()
        boot_median.sort()
        lo_idx = int(0.025 * (n_bootstrap - 1))
        hi_idx = int(0.975 * (n_bootstrap - 1))

        means.append(mean_value)
        medians.append(median_value)
        mean_lo.append(boot_mean[lo_idx])
        mean_hi.append(boot_mean[hi_idx])
        med_lo.append(boot_median[lo_idx])
        med_hi.append(boot_median[hi_idx])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.plot(layers, means, color="#1f77b4", linewidth=2.2, label="Mean H(l)/N")
    ax.fill_between(layers, mean_lo, mean_hi, color="#1f77b4", alpha=0.2, label="Mean 95% CI")
    ax.plot(layers, medians, color="#d62728", linewidth=2.2, label="Median H(l)/N")
    ax.fill_between(layers, med_lo, med_hi, color="#d62728", alpha=0.2, label="Median 95% CI")

    title = "Normalized Entropy by Layer (H/N)"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer l", fontsize=11)
    ax.set_ylabel("H(l)/N", fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    tick_step = max(1, math.ceil(len(layers) / 32))
    xticks = layers[::tick_step]
    if layers[-1] not in xticks:
        xticks.append(layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "entropy_summary.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _bootstrap_center_and_ci(
    values: List[float],
    reducer,
    n_bootstrap: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    n = len(values)
    center = reducer(values)
    if n <= 1:
        return center, center, center

    boot_values: List[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_values.append(reducer(sample))

    boot_values.sort()
    lo_idx = int(0.025 * (n_bootstrap - 1))
    hi_idx = int(0.975 * (n_bootstrap - 1))
    return center, boot_values[lo_idx], boot_values[hi_idx]


def _mean(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _sample_mode_title_suffix(sample_payload: Dict[str, Any]) -> str:
    mode = str(sample_payload.get("sample_mode", "")).strip()
    policy = str(sample_payload.get("patch_target_policy", "")).strip()
    parts = []
    if mode:
        parts.append(f"mode={mode}")
    if policy:
        parts.append(f"target={policy}")
    return " | ".join(parts)


def _plot_attention_importance_subplot(
    ax: Any,
    summary: Dict[str, List[float]],
    stat_name: str,
) -> None:
    layers = summary["layers"]
    if not layers:
        raise RuntimeError(f"No data available for {stat_name} subplot.")

    ax.plot(layers, summary["importance"], color="#1f77b4", linewidth=2.2, label=f"{stat_name} importance")
    ax.fill_between(
        layers,
        summary["importance_lo"],
        summary["importance_hi"],
        color="#1f77b4",
        alpha=0.18,
        label=f"{stat_name} importance 95% CI",
    )
    ax.plot(layers, summary["attention"], color="#d62728", linewidth=2.2, label=f"{stat_name} attention ratio")
    ax.fill_between(
        layers,
        summary["attention_lo"],
        summary["attention_hi"],
        color="#d62728",
        alpha=0.18,
        label=f"{stat_name} attention 95% CI",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Value")
    ax.set_title(f"Per-layer {stat_name.lower()}: importance vs attention-share")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    tick_step = max(1, math.ceil(len(layers) / 32))
    xticks = layers[::tick_step]
    if layers[-1] not in xticks:
        xticks.append(layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)


def plot_attention_importance_summary(
    importance_by_layer: Dict[int, List[float]],
    attention_by_layer: Dict[int, List[float]],
    output_path: Path,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Path]:
    candidate_layers = sorted(set(importance_by_layer) & set(attention_by_layer))
    if not candidate_layers:
        return None

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals)

    def _median(vals: List[float]) -> float:
        ordered = sorted(vals)
        n = len(ordered)
        mid = n // 2
        return ordered[mid] if n % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])

    def _build_summary(reducer, rng_seed: int) -> Dict[str, List[float]]:
        rng = random.Random(rng_seed)
        summary: Dict[str, List[float]] = {
            "layers": [],
            "importance": [],
            "importance_lo": [],
            "importance_hi": [],
            "attention": [],
            "attention_lo": [],
            "attention_hi": [],
        }
        for layer_idx in candidate_layers:
            importance_values = importance_by_layer.get(layer_idx, [])
            attention_values = attention_by_layer.get(layer_idx, [])
            if not importance_values or not attention_values:
                continue

            imp_center, imp_lo, imp_hi = _bootstrap_center_and_ci(importance_values, reducer, n_bootstrap, rng)
            att_center, att_lo, att_hi = _bootstrap_center_and_ci(attention_values, reducer, n_bootstrap, rng)
            summary["layers"].append(layer_idx)
            summary["importance"].append(imp_center)
            summary["importance_lo"].append(imp_lo)
            summary["importance_hi"].append(imp_hi)
            summary["attention"].append(att_center)
            summary["attention_lo"].append(att_lo)
            summary["attention_hi"].append(att_hi)
        return summary

    mean_summary = _build_summary(_mean, seed)
    median_summary = _build_summary(_median, seed + 1)
    if not mean_summary["layers"] or not median_summary["layers"]:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=140, sharey=False)
    _plot_attention_importance_subplot(axes[0], mean_summary, "Mean")
    _plot_attention_importance_subplot(axes[1], median_summary, "Median")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_token_recovery_heatmap(
    sample_payload: Dict[str, Any],
    output_dir: Path,
    seq_len_label: Optional[str] = None,
    use_clamped: bool = True,
    file_prefix: Optional[str] = None,
) -> Optional[Path]:
    token_metadata = list(sample_payload.get("token_metadata", []))
    layer_metrics = list(sample_payload.get("layer_metrics", {}).get("layers", []))
    if not token_metadata or not layer_metrics:
        return None

    value_key = "clamped_recovery_by_token" if use_clamped else "raw_recovery_by_token"
    matrix = [
        [float(value) for value in layer_metric.get(value_key, [])]
        for layer_metric in layer_metrics
    ]
    if not matrix or not matrix[0]:
        return None

    split_points = [(0, math.ceil(len(token_metadata) / 2))]
    if len(token_metadata) > 1:
        split_points.append((split_points[0][1], len(token_metadata)))

    fig_width = max(14, 0.7 * max(end - start for start, end in split_points))
    fig_height = 5.5 * len(split_points)
    fig, axes = plt.subplots(len(split_points), 1, figsize=(fig_width, fig_height), dpi=140, squeeze=False)
    axes_flat = [ax for row in axes for ax in row]
    image = None
    for subplot_idx, (ax, (start_idx, end_idx)) in enumerate(zip(axes_flat, split_points), start=1):
        slice_matrix = [row[start_idx:end_idx] for row in matrix]
        image = ax.imshow(
            slice_matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0 if use_clamped else None,
        )
        xticks = list(range(end_idx - start_idx))
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [
                f"{token_metadata[token_idx]['token_index']}:{token_metadata[token_idx]['token_text']}"
                for token_idx in range(start_idx, end_idx)
            ],
            rotation=60,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(list(range(len(layer_metrics))))
        ax.set_yticklabels([str(int(layer_metric["layer"])) for layer_metric in layer_metrics], fontsize=9)
        ax.set_xlabel(f"Token index and token text (part {subplot_idx})")
        ax.set_ylabel("Layer")

    title = "Clamped Token Recovery Heatmap" if use_clamped else "Raw Token Recovery Heatmap"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    mode_suffix = _sample_mode_title_suffix(sample_payload)
    if mode_suffix:
        title = f"{title}\n{mode_suffix}"
    axes_flat[0].set_title(title, fontsize=13, pad=10)
    fig.subplots_adjust(right=0.88, hspace=0.55)
    cbar_ax = fig.add_axes([0.91, 0.18, 0.018, 0.68])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Recovery")

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{seq_len_label}" if seq_len_label else ""
    filename = "token_recovery_heatmap" if use_clamped else "token_recovery_heatmap_raw"
    prefix = f"{file_prefix}_" if file_prefix else ""
    output_path = output_dir / f"{prefix}{filename}{suffix}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_topk_token_mass(
    sample_payload: Dict[str, Any],
    output_dir: Path,
    top_ks: List[int],
    seq_len_label: Optional[str] = None,
    file_prefix: Optional[str] = None,
) -> List[Path]:
    token_statistics = list(sample_payload.get("token_min_clamped_summary", []))
    if not token_statistics:
        raise ValueError(
            "Missing token_min_clamped_summary in aggregate payload. "
            "Re-run the experiment with the updated script so the aggregate JSON includes per-sample min clamped recovery."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []
    rng = random.Random(0)
    for top_k in top_ks:
        top_rows = token_statistics[: min(int(top_k), len(token_statistics))]
        if not top_rows:
            continue

        labels = [
            f"{row['token_index']}:{row['token_text']}\n[{row['word_label']}]"
            for row in top_rows
        ]
        centers: List[float] = []
        lo_values: List[float] = []
        hi_values: List[float] = []
        for row in top_rows:
            values = [float(value) for value in row.get("per_sample_min_clamped", [])]
            center, lo_value, hi_value = _bootstrap_center_and_ci(values, _mean, 1000, rng)
            centers.append(center)
            lo_values.append(lo_value)
            hi_values.append(hi_value)

        x_positions = list(range(len(top_rows)))
        lower_err = [max(0.0, center - lo) for center, lo in zip(centers, lo_values)]
        upper_err = [max(0.0, hi - center) for center, hi in zip(centers, hi_values)]

        fig_width = max(12, 1.25 * len(top_rows))
        fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=140)
        ax.bar(
            x_positions,
            centers,
            yerr=[lower_err, upper_err],
            capsize=4,
            color="#1f77b4",
            alpha=0.85,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean minimum clamped recovery")
        title = f"Top-{top_k} Tokens by Minimum Clamped Recovery"
        if seq_len_label:
            title = f"{title} ({seq_len_label})"
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)

        fig.tight_layout()
        prefix = f"{file_prefix}_" if file_prefix else ""
        output_path = output_dir / f"{prefix}top_{top_k}_token_mass{'_' + seq_len_label if seq_len_label else ''}.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def plot_first_last_token_importance_lines(
    sample_payload: Dict[str, Any],
    output_dir: Path,
    seq_len_label: Optional[str] = None,
    file_prefix: Optional[str] = None,
) -> Optional[Path]:
    token_metadata = list(sample_payload.get("token_metadata", []))
    layer_metrics = list(sample_payload.get("layer_metrics", {}).get("layers", []))
    first_last_stats = sample_payload.get("first_last_token_layer_summary", {})
    if len(token_metadata) < 1 or not layer_metrics:
        return None

    first_token_idx = 0
    last_token_idx = len(token_metadata) - 1
    layers = [int(layer_metric["layer"]) for layer_metric in layer_metrics]
    rng = random.Random(0)
    first_vals: List[float] = []
    first_lo: List[float] = []
    first_hi: List[float] = []
    last_vals: List[float] = []
    last_lo: List[float] = []
    last_hi: List[float] = []
    for layer_metric in layer_metrics:
        layer_idx = int(layer_metric["layer"])
        first_dist = [float(value) for value in first_last_stats.get("first", {}).get(str(layer_idx), [])]
        last_dist = [float(value) for value in first_last_stats.get("last", {}).get(str(layer_idx), [])]

        if first_dist:
            center, lo_value, hi_value = _bootstrap_center_and_ci(first_dist, _mean, 1000, rng)
        else:
            center = lo_value = hi_value = float(layer_metric["clamped_recovery_by_token"][first_token_idx])
        first_vals.append(center)
        first_lo.append(lo_value)
        first_hi.append(hi_value)

        if last_dist:
            center, lo_value, hi_value = _bootstrap_center_and_ci(last_dist, _mean, 1000, rng)
        else:
            center = lo_value = hi_value = float(layer_metric["clamped_recovery_by_token"][last_token_idx])
        last_vals.append(center)
        last_lo.append(lo_value)
        last_hi.append(hi_value)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    first_line, = ax.plot(layers, first_vals, linewidth=2.2, label=f"first: {token_metadata[first_token_idx]['token_text']}")
    ax.fill_between(layers, first_lo, first_hi, color=first_line.get_color(), alpha=0.18, label="first 95% CI")
    last_line, = ax.plot(layers, last_vals, linewidth=2.2, label=f"last: {token_metadata[last_token_idx]['token_text']}")
    ax.fill_between(layers, last_lo, last_hi, color=last_line.get_color(), alpha=0.18, label="last 95% CI")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Clamped recovery")
    title = "First vs Last Token Recovery by Layer"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    mode_suffix = _sample_mode_title_suffix(sample_payload)
    if mode_suffix:
        title = f"{title}\n{mode_suffix}"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    tick_step = max(1, math.ceil(len(layers) / 32))
    xticks = layers[::tick_step]
    if layers[-1] not in xticks:
        xticks.append(layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{file_prefix}_" if file_prefix else ""
    output_path = output_dir / f"{prefix}first_last_token_lines{'_' + seq_len_label if seq_len_label else ''}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path

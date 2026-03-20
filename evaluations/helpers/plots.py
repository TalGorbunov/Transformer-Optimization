import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


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

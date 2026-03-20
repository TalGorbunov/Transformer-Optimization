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

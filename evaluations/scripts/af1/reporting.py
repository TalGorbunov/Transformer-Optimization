"""AF1 CSV, markdown, heatmap, and table reporting helpers."""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from evaluations.helpers import utils as eval_utils
from evaluations.scripts.af1.common import SUMMARY_FIELDS
from evaluations.scripts.af1.kernel import instruction_mask_mode_summary


def format_summary_table(rows: Sequence[Dict[str, Any]]) -> str:
    values = [
        [
            str(row["model"]),
            str(row["mode"]),
            str(row["seq_len"]),
            str(row["wait_layer"]),
            str(row["transfer_layers"]),
            str(row["n_total"]),
            str(row["n_used"]),
            str(row["n_clean_correct"]),
            f"{float(row['clean_acc']):.4f}",
            f"{float(row['af1_acc']):.4f}",
            f"{float(row['af1_faith']):.4f}",
            f"{float(row['mean_clean_top1_score_drop']):.4f}",
            f"{float(row['mean_gold_answer_score_drop']):.4f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(value[col_idx]) for value in values)) if values else len(header)
        for col_idx, header in enumerate(SUMMARY_FIELDS)
    ]
    header_row = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(SUMMARY_FIELDS)) + " |"
    sep_row = "|-" + "-|-".join("-" * widths[idx] for idx in range(len(SUMMARY_FIELDS))) + "-|"
    data_rows = [
        "| " + " | ".join(value[idx].ljust(widths[idx]) for idx in range(len(SUMMARY_FIELDS))) + " |"
        for value in values
    ]
    return "\n".join([header_row, sep_row] + data_rows)


def row_for_fieldnames(row: Dict[str, Any], fieldnames: Sequence[str]) -> Dict[str, Any]:
    return {field: row.get(field) for field in fieldnames}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row_for_fieldnames(row, fieldnames))


def csv_header_line(fieldnames: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    return buffer.getvalue().rstrip("\n")


def plot_metric_heatmap(
    summary_rows: Sequence[Dict[str, Any]],
    wait_layers: Sequence[int],
    transfer_layers_grid: Sequence[int],
    value_key: str,
    output_path: Path,
    title: str,
    seq_len: int,
) -> Optional[Path]:
    if not wait_layers or not transfer_layers_grid:
        return None
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is required to write AF1 grid heatmaps; install matplotlib in the active environment."
        )

    matrix = np.full((len(wait_layers), len(transfer_layers_grid)), np.nan, dtype=float)
    wait_layer_to_index = {int(wait_layer): idx for idx, wait_layer in enumerate(wait_layers)}
    transfer_layers_to_index = {
        int(transfer_layers): idx for idx, transfer_layers in enumerate(transfer_layers_grid)
    }
    for row in summary_rows:
        wait_layer = int(row["wait_layer"])
        transfer_layers = int(row["transfer_layers"])
        if wait_layer in wait_layer_to_index and transfer_layers in transfer_layers_to_index:
            matrix[wait_layer_to_index[wait_layer], transfer_layers_to_index[transfer_layers]] = float(
                row[value_key]
            )

    fig_width = max(6.0, 1.4 * len(transfer_layers_grid))
    fig_height = max(5.0, 1.0 * len(wait_layers))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=140)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    masked_matrix = np.ma.masked_invalid(matrix)
    image = ax.imshow(
        masked_matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )

    finite_values = np.isfinite(matrix)
    if bool(finite_values.any()):
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"{title} (seq_len={seq_len})")
    ax.set_xlabel("transfer_layers")
    ax.set_ylabel("wait_layer")
    ax.set_xticks(range(len(transfer_layers_grid)))
    ax.set_xticklabels([str(value) for value in transfer_layers_grid])
    ax.set_yticks(range(len(wait_layers)))
    ax.set_yticklabels([str(value) for value in wait_layers])
    ax.set_xticks(np.arange(-0.5, len(transfer_layers_grid), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(wait_layers), 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    if bool(finite_values.any()):
        for wait_idx in range(len(wait_layers)):
            for transfer_idx in range(len(transfer_layers_grid)):
                value = matrix[wait_idx, transfer_idx]
                if np.isfinite(value):
                    text_color = "black" if float(image.norm(value)) > 0.5 else "white"
                    ax.text(
                        transfer_idx,
                        wait_idx,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path if output_path.exists() else None


def write_markdown_summary(
    path: Path,
    config: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    validation_notes: Sequence[str],
    donor_notes: Sequence[str],
    cache_notes: Sequence[str],
    output_notes: Sequence[str],
    elapsed_seconds: float,
) -> None:
    lines = [
        "# AF1 Qwen-VL Frame CAMA",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
        "## Results",
        "",
        format_summary_table(summary_rows),
        "",
        "## Method",
        "",
        f"- Mode run: `{config['mode']}` across a `(wait_layer, transfer_layers)` grid for one `seq_len={config['seq_len']}` dataset.",
        "- `full_af1` = wait-boundary frame-group plus non-frame prompt patching, then ABP masking afterward.",
        "- `wait_only` = wait-boundary frame-group plus non-frame prompt patching only, with later attention left clean.",
        "- `mask_only` = ABP masking only, with no wait-boundary patching.",
        "- This script implements AF1 with frame-group conditional means plus one all-non-frame prompt conditional mean at the wait boundary.",
        "- Donor hybrids keep the target prompt text fixed while changing only the frame inputs.",
        "- Hallway samples are skipped by default because their room tokenization differs.",
        "- Transfer uses an ABP-style policy where the prompt carrier token can read earlier prompt tokens only during the configured transfer layers, then becomes self-only.",
        f"- `instruction_mask_mode={config['instruction_mask_mode']}` controls instruction-token rows during transfer: {instruction_mask_mode_summary(config['instruction_mask_mode'])}.",
        "- Faithfulness is defined as intervention accuracy on the subset of used samples that the clean model got correct.",
        "- `max_samples` caps eligible evaluation targets; skipped samples do not count toward that cap, so `n_total` can exceed `max_samples` when extra skipped rows were needed to find enough usable targets.",
        "- `n_total` counts selected samples, `n_used` counts samples that passed compatibility and donor checks plus any enabled clean-top1 filter, and `clean_acc`/`af1_acc` are computed on the used subset.",
        "- `mean_clean_top1_score_drop` averages `clean_best_score - intervention_score(clean top-1)` over used samples, freezing the clean top-1 answer separately for each sample.",
        "- `mean_gold_answer_score_drop` averages `clean_score(gold answer) - intervention_score(gold answer)` over used samples.",
        "",
        "## Validation",
        "",
    ]
    lines.extend(f"- {note}" for note in validation_notes)
    lines.extend(["", "## Donor Policy", ""])
    lines.extend(f"- {note}" for note in donor_notes)
    lines.extend(["", "## Cache Notes", ""])
    lines.extend(f"- {note}" for note in cache_notes)
    lines.extend(["", "## Outputs", ""])
    lines.extend(f"- {note}" for note in output_notes)
    lines.extend(
        [
            "",
            "## Method Notes / Limitations",
            "",
            "- This is a multimodal adaptation of CAMA, not the paper's exact token-level text-token formulation.",
            "- Conditional-mean replacement is applied to frame token groups plus one all-non-frame prompt token set.",
            "- The conditional mean is estimated from compatible hybrid contexts, so donor/layout compatibility is required before a sample is used.",
            "- Cached conditional means are target-sample-specific and donor-set-specific; frame-group and non-frame caches are not reusable global means.",
            "- In `mask_only`, the `af1_*` output columns still mean 'intervention result' even though no wait-boundary patch is applied.",
            "",
            "## Runtime",
            "",
            f"- {eval_utils.format_runtime(elapsed_seconds)}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

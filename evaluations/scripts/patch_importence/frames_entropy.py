import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as core
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from models.model import get_default_runtime, get_layers, image_token_groups


def _model() -> Any:
    return get_default_runtime().model


def _processor() -> Any:
    return get_default_runtime().processor

_STEPS_IN_ROOM_RE = re.compile(
    r"How many steps did\s+([A-Za-z]+)\s+spend in\s+the\s+([A-Za-z]+)",
    flags=re.IGNORECASE,
)

token_ids_of_answer = core.token_ids_of_answer
build_prompt = core.build_prompt
build_inputs = core.build_inputs
move_inputs_to_model_device = core.move_inputs_to_model_device
repeat_inputs_for_batch = core.repeat_inputs_for_batch


def parse_target_character_room(question_text: str) -> Optional[Tuple[str, str]]:
    return eval_utils.parse_target_character_room(question_text)


def rooms_to_room2chars(rooms: Dict[str, Any]) -> Dict[str, List[str]]:
    return eval_utils.rooms_to_room2chars(rooms)


def collect_evidence_frame_indices(question: str, states: List[Dict[str, Any]]) -> List[int]:
    return eval_utils.collect_evidence_frame_indices(question, states)


_materialize_saved = core._materialize_saved
_to_hidden_tensor = core._to_hidden_tensor


def infer_corrupted_data_root(clean_data_root: Path) -> Path:
    return eval_utils.infer_corrupted_data_root(clean_data_root)


def resolve_corrupted_sample_dir(
    corrupted_data_root: Path,
    sample_id: str,
    frame_idx: int,
) -> Path:
    return eval_utils.resolve_corrupted_sample_dir(corrupted_data_root, sample_id, frame_idx)


append_answer_tokens_for_scoring = core.append_answer_tokens_for_scoring
sequence_logprob_from_logits = core.sequence_logprob_from_logits
run_clean_sequence_logprob = core.run_clean_sequence_logprob


def run_layer_multi_frame_corrupted_sequence_logprob(
    lm: LanguageModel,
    layers: Any,
    clean_batched_scoring_inputs: Dict[str, torch.Tensor],
    corrupted_batched_scoring_inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    token_positions_by_batch: List[List[int]],
    prompt_len: int,
    answer_token_ids: List[int],
    corruption_mode: str = "patch_from_corrupted",
) -> torch.Tensor:
    with torch.no_grad():
        # Corrupted run: this is a normal forward pass on corrupted input frames.
        # We save layer-l activations here and later patch them into the clean run.
        with lm.trace(corrupted_batched_scoring_inputs):
            corrupted_layer_saved = _to_hidden_tensor(layers[layer_idx].output).save()

        # Clean run: this starts from the original clean sample. Corruption is injected
        # only at layer l by patching evidence-frame token activations.
        with lm.trace(clean_batched_scoring_inputs):
            clean_layer_out = _to_hidden_tensor(layers[layer_idx].output)
            if corruption_mode == "patch_from_corrupted":
                corrupted_layer_out = _materialize_saved(corrupted_layer_saved)
                for batch_idx, token_positions in enumerate(token_positions_by_batch):
                    if token_positions:
                        # Layerwise patching site:
                        # copy corrupted evidence-frame token states into the clean run.
                        clean_layer_out[batch_idx, token_positions, :] = corrupted_layer_out[
                            batch_idx, token_positions, :
                        ]
            elif corruption_mode == "zero":
                # Kept for easy future re-enable, but not used by default.
                for batch_idx, token_positions in enumerate(token_positions_by_batch):
                    if token_positions:
                        clean_layer_out[batch_idx, token_positions, :] = 0
            else:
                raise ValueError(f"Unsupported corruption_mode={corruption_mode!r}")
            saved_logits = lm.output.logits.save()
    logits = _materialize_saved(saved_logits)
    return sequence_logprob_from_logits(logits, prompt_len=prompt_len, answer_token_ids=answer_token_ids)


concatenate_inputs_for_batch = core.concatenate_inputs_for_batch


def compute_ld(answer_score: float, competitor_score: float) -> float:
    return float(answer_score - competitor_score)


def best_competing_answer(
    lm: LanguageModel,
    inputs: Dict[str, torch.Tensor],
    prompt_len: int,
    correct_answer_text: str,
    max_answer_value: int,
) -> Tuple[str, List[int], float]:
    normalized_correct = str(correct_answer_text).strip()

    best_text: Optional[str] = None
    best_token_ids: Optional[List[int]] = None
    best_score: Optional[float] = None

    for value in range(max_answer_value + 1):
        candidate_text = str(value)
        if candidate_text == normalized_correct:
            continue

        candidate_token_ids = token_ids_of_answer(candidate_text)
        candidate_scoring_inputs = append_answer_tokens_for_scoring(inputs, candidate_token_ids)
        candidate_score = run_clean_sequence_logprob(
            lm=lm,
            scoring_inputs=candidate_scoring_inputs,
            prompt_len=prompt_len,
            answer_token_ids=candidate_token_ids,
        )

        if best_score is None or candidate_score > best_score:
            best_text = candidate_text
            best_token_ids = candidate_token_ids
            best_score = candidate_score

    if best_text is None or best_token_ids is None or best_score is None:
        raise RuntimeError("Failed to select competing answer sequence.")

    return best_text, best_token_ids, float(best_score)


normalize_to_probabilities = core.normalize_to_probabilities
entropy_from_probabilities = core.entropy_from_probabilities


def normalize_entropy(entropy: float, num_evidence_frames: int) -> float:
    if num_evidence_frames <= 1:
        return 0.0
    return float(entropy) / math.log(float(num_evidence_frames))


def format_corrupted_ld_table(
    evidence_frames: List[int],
    layer_rows: List[Tuple[int, List[float]]],
) -> str:
    headers = ["layer"] + [f"frame_{frame_idx}" for frame_idx in evidence_frames]

    table_rows: List[List[str]] = []
    for layer_idx, corrupted_lds in layer_rows:
        table_rows.append([str(layer_idx)] + [f"{float(ld):.4f}" for ld in corrupted_lds])

    col_widths = [
        max(
            len(headers[col_idx]),
            *(len(row[col_idx]) for row in table_rows),
        )
        for col_idx in range(len(headers))
    ]

    def _fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(
            cell.rjust(col_widths[col_idx]) for col_idx, cell in enumerate(row)
        ) + " |"

    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    lines = [separator, _fmt_row(headers), separator]
    lines.extend(_fmt_row(row) for row in table_rows)
    lines.append(separator)
    return "\n".join(lines)


def plot_layer_invalidity_rates(
    sampled_counts: List[int],
    invalid_counts: List[int],
    output_dir: Path,
    seq_len_label: Optional[str] = None,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping invalidity plot: matplotlib is not available ({exc})")
        return None

    layers = list(range(len(sampled_counts)))
    invalid_rates = [
        (100.0 * float(invalid_counts[idx]) / float(sampled_counts[idx]))
        if sampled_counts[idx] > 0 else 0.0
        for idx in layers
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)
    ax.bar(layers, invalid_rates, color="#ff7f0e", width=0.8)
    title = "Layer Invalidity Rate (All Importances Zero)"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Invalid rate (%)")
    ax.set_ylim(0.0, 100.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)

    if layers:
        tick_step = max(1, math.ceil(len(layers) / 32))
        xticks = layers[::tick_step]
        if layers[-1] not in xticks:
            xticks.append(layers[-1])
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelrotation=45, labelsize=9)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{seq_len_label}" if seq_len_label else ""
    plot_path = output_dir / f"layer_invalidity_rate{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_entropy_report(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    def _fmt_float_list(vals: List[float]) -> str:
        return "[" + ", ".join(f"{v:.8f}" for v in vals) + "]"

    lines: List[str] = []
    for sample in sample_metrics:
        lines.append(f"sample_id={sample['sample_id']}")
        lines.append(
            f"clean_ld={sample['clean_ld']:.8f} "
            f"clean_answer_score={sample['clean_answer_score']:.8f} "
            f"clean_competing_score={sample['clean_competing_score']:.8f} "
            f"a_star_text={sample['a_star_text']!r} a_hat_text={sample['a_hat_text']!r} "
            f"a_star_ids={sample['a_star_ids']} a_hat_ids={sample['a_hat_ids']}"
        )
        lines.append(f"evidence_frames={sample['evidence_frames']}")
        for layer_metrics in sample["layer_metrics"]["layers"]:
            lines.append(
                f"layer={layer_metrics['layer']} "
                f"r={_fmt_float_list(layer_metrics['r'])} "
                f"p={_fmt_float_list(layer_metrics['p'])} "
                f"H_norm={layer_metrics['entropy']:.8f} "
                f"R_total={layer_metrics['total_importance']:.8f}"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_metrics.txt"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def plot_entropy_means_medians(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    seq_len_label: Optional[str] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot: matplotlib is not available ({exc})")
        return None

    entropy_by_layer: Dict[int, List[float]] = {}
    for sample in sample_metrics:
        for layer_metrics in sample["layer_metrics"]["layers"]:
            layer_idx = int(layer_metrics["layer"])
            entropy_by_layer.setdefault(layer_idx, []).append(float(layer_metrics["entropy"]))

    if not entropy_by_layer:
        return None

    rng = random.Random(seed)
    layers = sorted(entropy_by_layer.keys())
    mean_values: List[float] = []
    median_values: List[float] = []
    mean_lo_values: List[float] = []
    mean_hi_values: List[float] = []

    for layer_idx in layers:
        values = entropy_by_layer[layer_idx]
        n = len(values)
        sorted_values = sorted(values)

        mean_value = sum(values) / n
        if n % 2 == 1:
            median_value = sorted_values[n // 2]
        else:
            median_value = 0.5 * (sorted_values[n // 2 - 1] + sorted_values[n // 2])

        if n <= 1:
            mean_lo = mean_hi = mean_value
        else:
            boot_means: List[float] = []
            for _ in range(n_bootstrap):
                sample = [values[rng.randrange(n)] for _ in range(n)]
                boot_means.append(sum(sample) / n)

            boot_means.sort()
            lo_idx = int(0.025 * (n_bootstrap - 1))
            hi_idx = int(0.975 * (n_bootstrap - 1))
            mean_lo = boot_means[lo_idx]
            mean_hi = boot_means[hi_idx]

        mean_values.append(mean_value)
        median_values.append(median_value)
        mean_lo_values.append(mean_lo)
        mean_hi_values.append(mean_hi)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.plot(layers, mean_values, color="#1f77b4", linewidth=2.2, label="Mean entropy")
    ax.fill_between(
        layers,
        mean_lo_values,
        mean_hi_values,
        color="#1f77b4",
        alpha=0.2,
        label="Mean 95% CI",
    )
    ax.plot(layers, median_values, color="#d62728", linewidth=2.2, label="Median entropy")

    title = "Entropy by Layer"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy")
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
    plot_path = output_dir / f"entropy_summary{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_total_importance_mean(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    num_layers: int,
    seq_len_label: Optional[str] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping total-importance plot: matplotlib is not available ({exc})")
        return None

    if num_layers <= 0 or not sample_metrics:
        return None

    per_layer_values: Dict[int, List[float]] = {layer_idx: [] for layer_idx in range(num_layers)}
    for sample in sample_metrics:
        sample_layer_totals = {
            int(layer_metrics["layer"]): float(layer_metrics.get("total_importance", sum(layer_metrics["r"])))
            for layer_metrics in sample["layer_metrics"]["layers"]
        }
        for layer_idx in range(num_layers):
            # Layers missing from sample_layer_totals are zero-importance/invalid for that sample.
            per_layer_values[layer_idx].append(sample_layer_totals.get(layer_idx, 0.0))

    rng = random.Random(seed)
    layers = list(range(num_layers))
    mean_values: List[float] = []
    mean_lo_values: List[float] = []
    mean_hi_values: List[float] = []

    for layer_idx in layers:
        values = per_layer_values[layer_idx]
        n = len(values)
        mean_value = (sum(values) / n) if n > 0 else 0.0

        if n <= 1:
            mean_lo = mean_hi = mean_value
        else:
            boot_means: List[float] = []
            for _ in range(n_bootstrap):
                sample = [values[rng.randrange(n)] for _ in range(n)]
                boot_means.append(sum(sample) / n)
            boot_means.sort()
            lo_idx = int(0.025 * (n_bootstrap - 1))
            hi_idx = int(0.975 * (n_bootstrap - 1))
            mean_lo = boot_means[lo_idx]
            mean_hi = boot_means[hi_idx]

        mean_values.append(mean_value)
        mean_lo_values.append(mean_lo)
        mean_hi_values.append(mean_hi)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.plot(layers, mean_values, color="#2ca02c", linewidth=2.2, label="Mean total importance")
    ax.fill_between(
        layers,
        mean_lo_values,
        mean_hi_values,
        color="#2ca02c",
        alpha=0.2,
        label="Mean 95% CI",
    )

    title = "Total Importance by Layer"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Total importance R(l)")
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
    plot_path = output_dir / f"total_importance_summary{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def resolve_lambda_threshold(args: argparse.Namespace) -> float:
    lambda_arg = args.lambda_threshold
    min_clean_ld = args.min_clean_ld

    if lambda_arg is None and min_clean_ld is None:
        return 1.0
    if lambda_arg is None:
        return float(min_clean_ld)
    if min_clean_ld is None:
        return float(lambda_arg)

    if abs(float(lambda_arg) - float(min_clean_ld)) > 1e-12:
        raise ValueError(
            "Conflicting thresholds: --lambda and --min_clean_ld differ. "
            "Please pass only one or keep them equal."
        )
    return float(lambda_arg)


def load_clean_ld_cache(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to parse clean LD cache at {path}: {exc}. Starting with empty cache.")
        return {}

    if not isinstance(payload, dict):
        print(f"[WARN] Invalid clean LD cache format at {path}. Expected JSON object; starting empty.")
        return {}

    cache: Dict[str, float] = {}
    for sample_id, value in payload.items():
        if not isinstance(sample_id, str):
            continue
        try:
            cache[sample_id] = float(value)
        except Exception:
            continue
    return cache


def save_clean_ld_cache(path: Path, cache: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {sample_id: float(ld) for sample_id, ld in sorted(cache.items())}
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    # This is the original experiment: corrupt each evidence frame, patch the
    # corrupted activations layer-by-layer, and summarize the resulting entropy.
    ap = argparse.ArgumentParser(
        description=(
            "Compute per-layer evidence importances and entropies. "
            "For each sample: score the full GT answer sequence, choose a^ as the best non-a* full-answer "
            "competitor over numeric candidates, skip if LD_clean < lambda, then patch corrupted-run "
            "evidence-frame activations into the clean run at each layer and compute "
            "importance=max(LD_clean-LD_corrupted,0)."
        )
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--corrupted_root",
        type=str,
        default=None,
        help=(
            "Root directory for corrupted samples. If omitted, inferred from --data_root "
            "(e.g., .../mmred_images/... -> .../mmred_corrupted/...)."
        ),
    )
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--output", type=str, default="outputs")
    ap.add_argument(
        "--clean_ld_cache_dir",
        type=str,
        default=None,
        help=(
            "Directory containing clean_lds.json for loading/updating clean LD cache. "
            "Defaults to --output when omitted."
        ),
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of evidence-frame corruption runs to execute together in one forward pass.",
    )
    ap.add_argument("--lambda", dest="lambda_threshold", type=float, default=None)
    ap.add_argument(
        "--min_clean_ld",
        type=float,
        default=None,
        help="Alias for --lambda (kept for backward compatibility).",
    )
    args = ap.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer")
    return args


def process_sample(
    sample_dir: Path,
    sample_index: int,
    total_samples: int,
    args: argparse.Namespace,
    lm: LanguageModel,
    layers: Any,
    corrupted_data_root: Path,
    clean_ld_cache: Dict[str, float],
    lambda_threshold: float,
) -> Tuple[Optional[Dict[str, Any]], int, List[int], List[int]]:
    zero_counts = [0 for _ in range(len(layers))]

    try:
        sample_id, frames, question, states, answer = load_mmred_sample(sample_dir)
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_dir.name} skipped: load failure ({exc})")
        return None, 0, zero_counts, zero_counts.copy()

    evidence_frames = collect_evidence_frame_indices(question, states)
    if len(evidence_frames) < 2:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} "
            f"skipped: evidence frames={len(evidence_frames)} < 2"
        )
        return None, 0, zero_counts, zero_counts.copy()

    cached_clean_ld = clean_ld_cache.get(sample_id)
    if cached_clean_ld is not None and cached_clean_ld < lambda_threshold:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped from cache: "
            f"LD_clean={cached_clean_ld:.4f} < lambda={lambda_threshold:.4f}"
        )
        return None, 0, zero_counts, zero_counts.copy()

    inputs = move_inputs_to_model_device(build_inputs(frames, question))

    try:
        prompt_len = int(inputs["input_ids"].shape[1])
        a_star_text = str(answer).strip()
        a_star_ids = token_ids_of_answer(a_star_text)

        clean_answer_scoring_inputs = append_answer_tokens_for_scoring(inputs, a_star_ids)
        clean_answer_score = run_clean_sequence_logprob(
            lm=lm,
            scoring_inputs=clean_answer_scoring_inputs,
            prompt_len=prompt_len,
            answer_token_ids=a_star_ids,
        )
        a_hat_text, a_hat_ids, clean_competing_score = best_competing_answer(
            lm=lm,
            inputs=inputs,
            prompt_len=prompt_len,
            correct_answer_text=a_star_text,
            max_answer_value=len(frames),
        )
        clean_ld = compute_ld(clean_answer_score, clean_competing_score)
    except Exception as exc:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} "
            f"skipped: sequence LD setup failed ({exc})"
        )
        return None, 0, zero_counts, zero_counts.copy()

    cache_updates = 0
    cached_before = clean_ld_cache.get(sample_id)
    if cached_before is None or abs(cached_before - clean_ld) > 1e-12:
        clean_ld_cache[sample_id] = clean_ld
        cache_updates += 1

    if clean_ld < lambda_threshold:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"LD_clean={clean_ld:.4f} < lambda={lambda_threshold:.4f}"
        )
        return None, cache_updates, zero_counts, zero_counts.copy()

    frame_groups = image_token_groups(inputs["input_ids"][0], expected_num_frames=len(frames))
    if not frame_groups:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} "
            f"skipped: no image token groups found in tokenized input"
        )
        return None, cache_updates, zero_counts, zero_counts.copy()

    valid_evidence_frames: List[int] = []
    missing_corruption_dirs = 0
    for frame_idx in evidence_frames:
        if frame_idx >= len(frame_groups):
            continue
        if not frame_groups[frame_idx]:
            continue
        corrupted_sample_dir = resolve_corrupted_sample_dir(
            corrupted_data_root=corrupted_data_root,
            sample_id=sample_id,
            frame_idx=frame_idx,
        )
        if not corrupted_sample_dir.is_dir():
            missing_corruption_dirs += 1
            continue
        valid_evidence_frames.append(frame_idx)

    if not valid_evidence_frames:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} "
            f"skipped: evidence frames exist but none map to image token spans"
        )
        return None, cache_updates, zero_counts, zero_counts.copy()

    if missing_corruption_dirs > 0:
        print(
            f"  missing corrupted inputs for {missing_corruption_dirs} evidence frame(s); "
            "using available corrupted_frame_* directories only"
        )

    print(
        f"[{sample_index}/{total_samples}] sample_id={sample_id} "
        f"LD_clean={clean_ld:.4f} lambda={lambda_threshold:.4f} "
        f"a*={a_star_text} a^={a_hat_text} "
        f"evidence_frames={valid_evidence_frames} "
        f"batch_size={args.batch_size} "
        f"corrupted_root={corrupted_data_root}"
    )

    # Frame groups come from the clean prompt tokenization, so every corruption
    # run reuses the same token span for that evidence frame.
    evidence_token_positions_by_frame: List[Tuple[int, List[int]]] = [
        (frame_idx, frame_groups[frame_idx]) for frame_idx in valid_evidence_frames
    ]
    chunk_size = min(args.batch_size, len(evidence_token_positions_by_frame))
    evidence_chunks: List[List[Tuple[int, List[int]]]] = [
        evidence_token_positions_by_frame[start:start + chunk_size]
        for start in range(0, len(evidence_token_positions_by_frame), chunk_size)
    ]

    batched_answer_scoring_inputs_chunks: List[Dict[str, torch.Tensor]] = []
    batched_competing_scoring_inputs_chunks: List[Dict[str, torch.Tensor]] = []
    batched_corrupted_answer_scoring_inputs_chunks: List[Dict[str, torch.Tensor]] = []
    batched_corrupted_competing_scoring_inputs_chunks: List[Dict[str, torch.Tensor]] = []
    try:
        for evidence_chunk in evidence_chunks:
            repeated_inputs = repeat_inputs_for_batch(inputs, batch_size=len(evidence_chunk))
            batched_answer_scoring_inputs_chunks.append(
                append_answer_tokens_for_scoring(repeated_inputs, a_star_ids)
            )
            batched_competing_scoring_inputs_chunks.append(
                append_answer_tokens_for_scoring(repeated_inputs, a_hat_ids)
            )

            corrupted_chunk_inputs: List[Dict[str, torch.Tensor]] = []
            for frame_idx, _ in evidence_chunk:
                corrupted_sample_dir = resolve_corrupted_sample_dir(
                    corrupted_data_root=corrupted_data_root,
                    sample_id=sample_id,
                    frame_idx=frame_idx,
                )
                _, corrupted_frames, corrupted_question, _, _ = load_mmred_sample(corrupted_sample_dir)
                corrupted_input = move_inputs_to_model_device(build_inputs(corrupted_frames, corrupted_question))
                corrupted_prompt_len = int(corrupted_input["input_ids"].shape[1])
                if corrupted_prompt_len != prompt_len:
                    raise ValueError(
                        f"Prompt length mismatch for corrupted frame {frame_idx}: "
                        f"clean={prompt_len} corrupted={corrupted_prompt_len}"
                    )
                corrupted_chunk_inputs.append(corrupted_input)

            corrupted_batched_inputs = concatenate_inputs_for_batch(corrupted_chunk_inputs)
            batched_corrupted_answer_scoring_inputs_chunks.append(
                append_answer_tokens_for_scoring(corrupted_batched_inputs, a_star_ids)
            )
            batched_corrupted_competing_scoring_inputs_chunks.append(
                append_answer_tokens_for_scoring(corrupted_batched_inputs, a_hat_ids)
            )
    except Exception as exc:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} "
            f"skipped: failed to build batched inputs ({exc})"
        )
        return None, cache_updates, zero_counts, zero_counts.copy()

    per_layer_metrics: List[Dict[str, Any]] = []
    all_layer_corrupted_ld_rows: List[Tuple[int, List[float]]] = []
    layer_sampled_deltas = [0 for _ in range(len(layers))]
    layer_invalid_deltas = [0 for _ in range(len(layers))]
    skipped_zero_importance_layers = 0

    for layer_idx in range(len(layers)):
        layer_sampled_deltas[layer_idx] += 1
        layer_corrupted_lds: List[float] = []
        layer_importances: List[float] = []

        for chunk_idx, (
            evidence_chunk,
            batched_answer_scoring_inputs_chunk,
            batched_competing_scoring_inputs_chunk,
            batched_corrupted_answer_scoring_inputs_chunk,
            batched_corrupted_competing_scoring_inputs_chunk,
        ) in enumerate(
            zip(
                evidence_chunks,
                batched_answer_scoring_inputs_chunks,
                batched_competing_scoring_inputs_chunks,
                batched_corrupted_answer_scoring_inputs_chunks,
                batched_corrupted_competing_scoring_inputs_chunks,
            ),
            start=1,
        ):
            try:
                token_positions_by_batch = [token_positions for _, token_positions in evidence_chunk]
                corrupted_answer_scores = run_layer_multi_frame_corrupted_sequence_logprob(
                    lm=lm,
                    layers=layers,
                    clean_batched_scoring_inputs=batched_answer_scoring_inputs_chunk,
                    corrupted_batched_scoring_inputs=batched_corrupted_answer_scoring_inputs_chunk,
                    layer_idx=layer_idx,
                    token_positions_by_batch=token_positions_by_batch,
                    prompt_len=prompt_len,
                    answer_token_ids=a_star_ids,
                    corruption_mode="patch_from_corrupted",
                )
                corrupted_competing_scores = run_layer_multi_frame_corrupted_sequence_logprob(
                    lm=lm,
                    layers=layers,
                    clean_batched_scoring_inputs=batched_competing_scoring_inputs_chunk,
                    corrupted_batched_scoring_inputs=batched_corrupted_competing_scoring_inputs_chunk,
                    layer_idx=layer_idx,
                    token_positions_by_batch=token_positions_by_batch,
                    prompt_len=prompt_len,
                    answer_token_ids=a_hat_ids,
                    corruption_mode="patch_from_corrupted",
                )
            except Exception as exc:
                print(
                    f"  layer={layer_idx} failed batched corruption forward "
                    f"(chunk {chunk_idx}/{len(evidence_chunks)}, {exc}); "
                    "using importance=0 for this chunk"
                )
                chunk_count = len(evidence_chunk)
                layer_corrupted_lds.extend([clean_ld] * chunk_count)
                layer_importances.extend([0.0] * chunk_count)
                continue

            for batch_idx in range(len(token_positions_by_batch)):
                corrupted_ld = compute_ld(
                    float(corrupted_answer_scores[batch_idx].item()),
                    float(corrupted_competing_scores[batch_idx].item()),
                )
                importance = max(clean_ld - corrupted_ld, 0.0)
                layer_corrupted_lds.append(corrupted_ld)
                layer_importances.append(importance)

        all_layer_corrupted_ld_rows.append((layer_idx, list(layer_corrupted_lds)))

        if sum(layer_importances) <= 0.0:
            layer_invalid_deltas[layer_idx] += 1
            skipped_zero_importance_layers += 1
            continue

        layer_probabilities = normalize_to_probabilities(layer_importances)
        layer_entropy = normalize_entropy(
            entropy_from_probabilities(layer_probabilities),
            num_evidence_frames=len(valid_evidence_frames),
        )
        per_layer_metrics.append({
            "layer": layer_idx,
            "evidence_frames": list(valid_evidence_frames),
            "corrupted_ld": layer_corrupted_lds,
            "r": layer_importances,
            "p": layer_probabilities,
            "entropy": layer_entropy,
            "total_importance": float(sum(layer_importances)),
        })

    if all_layer_corrupted_ld_rows:
        print("  Corrupted LD table (rows=layers, columns=evidence frames):")
        print(format_corrupted_ld_table(valid_evidence_frames, all_layer_corrupted_ld_rows))

    if not per_layer_metrics:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            "all layers had zero total importance"
        )
        return None, cache_updates, layer_sampled_deltas, layer_invalid_deltas

    if skipped_zero_importance_layers > 0:
        print(f"  skipped zero-importance layers: {skipped_zero_importance_layers}")

    return {
        "sample_id": sample_id,
        "answer": answer,
        "clean_ld": clean_ld,
        "clean_answer_score": clean_answer_score,
        "clean_competing_score": clean_competing_score,
        "a_star_text": a_star_text,
        "a_hat_text": a_hat_text,
        "a_star_ids": a_star_ids,
        "a_hat_ids": a_hat_ids,
        "evidence_frames": list(valid_evidence_frames),
        "layer_metrics": {"layers": per_layer_metrics},
    }, cache_updates, layer_sampled_deltas, layer_invalid_deltas


def finalize_outputs(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    clean_ld_cache_path: Path,
    clean_ld_cache: Dict[str, float],
    cache_updates: int,
    processed_samples: int,
    target_processed_samples: int,
    lambda_threshold: float,
    num_layers: int,
    layer_sampled_counts: List[int],
    layer_invalid_counts: List[int],
    seq_len_label: Optional[str],
) -> None:
    if cache_updates > 0:
        save_clean_ld_cache(clean_ld_cache_path, clean_ld_cache)
        print(f"Updated clean LD cache at {clean_ld_cache_path} ({cache_updates} new/changed entries).")
    elif not clean_ld_cache_path.exists():
        save_clean_ld_cache(clean_ld_cache_path, clean_ld_cache)
        print(f"Wrote empty clean LD cache to: {clean_ld_cache_path}")
    else:
        print(f"No clean LD cache updates. Reused existing cache at: {clean_ld_cache_path}")

    output_path = write_entropy_report(sample_metrics, output_dir)
    print(f"Wrote sample metrics to: {output_path}")
    print(
        f"Processed {processed_samples} samples "
        f"(target limit={target_processed_samples}, lambda={lambda_threshold:.4f})."
    )

    plot_path = plot_entropy_means_medians(sample_metrics, output_dir, seq_len_label=seq_len_label)
    if plot_path is not None:
        print(f"Wrote entropy plot to: {plot_path}")
    else:
        print("Skipped entropy plot: no layer metrics available.")

    total_importance_plot_path = plot_total_importance_mean(
        sample_metrics,
        output_dir,
        num_layers=num_layers,
        seq_len_label=seq_len_label,
    )
    if total_importance_plot_path is not None:
        print(f"Wrote total-importance plot to: {total_importance_plot_path}")
    else:
        print("Skipped total-importance plot: no layer metrics available.")

    invalidity_plot_path = plot_layer_invalidity_rates(
        layer_sampled_counts,
        layer_invalid_counts,
        output_dir,
        seq_len_label=seq_len_label,
    )
    if invalidity_plot_path is not None:
        print(f"Wrote layer invalidity plot to: {invalidity_plot_path}")
    else:
        print("Skipped layer invalidity plot: no matplotlib available.")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    lambda_threshold = resolve_lambda_threshold(args)

    data_root = Path(args.data_root)
    corrupted_data_root = Path(args.corrupted_root) if args.corrupted_root is not None else infer_corrupted_data_root(data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_ld_cache_dir = Path(args.clean_ld_cache_dir) if args.clean_ld_cache_dir is not None else output_dir
    clean_ld_cache_dir.mkdir(parents=True, exist_ok=True)
    clean_ld_cache_path = clean_ld_cache_dir / "clean_lds.json"
    clean_ld_cache = load_clean_ld_cache(clean_ld_cache_path)
    cache_updates = 0
    if clean_ld_cache:
        print(f"Loaded {len(clean_ld_cache)} cached clean LD values from: {clean_ld_cache_path}")

    seq_len_label = eval_utils.resolve_seq_len_label(data_root)

    lm = LanguageModel(_model(), tokenizer=_processor().tokenizer)
    layers = get_layers(lm.model)
    num_layers = len(layers)
    layer_sampled_counts = [0 for _ in range(num_layers)]
    layer_invalid_counts = [0 for _ in range(num_layers)]

    sample_dirs = iter_sample_dirs(data_root)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under: {data_root}")

    sample_metrics: List[Dict[str, Any]] = []
    processed_samples = 0
    target_processed_samples = max(int(args.limit), 0)

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= target_processed_samples:
            break
        sample_metrics_row, cache_delta, sampled_deltas, invalid_deltas = process_sample(
            sample_dir=sample_dir,
            sample_index=idx,
            total_samples=len(sample_dirs),
            args=args,
            lm=lm,
            layers=layers,
            corrupted_data_root=corrupted_data_root,
            clean_ld_cache=clean_ld_cache,
            lambda_threshold=lambda_threshold,
        )
        cache_updates += cache_delta
        for layer_idx, delta in enumerate(sampled_deltas):
            layer_sampled_counts[layer_idx] += delta
        for layer_idx, delta in enumerate(invalid_deltas):
            layer_invalid_counts[layer_idx] += delta
        if sample_metrics_row is None:
            continue
        sample_metrics.append(sample_metrics_row)
        processed_samples += 1

    finalize_outputs(
        sample_metrics=sample_metrics,
        output_dir=output_dir,
        clean_ld_cache_path=clean_ld_cache_path,
        clean_ld_cache=clean_ld_cache,
        cache_updates=cache_updates,
        processed_samples=processed_samples,
        target_processed_samples=target_processed_samples,
        lambda_threshold=lambda_threshold,
        num_layers=num_layers,
        layer_sampled_counts=layer_sampled_counts,
        layer_invalid_counts=layer_invalid_counts,
        seq_len_label=seq_len_label,
    )

    elapsed_seconds = time.perf_counter() - start_time
    print(eval_utils.format_runtime(elapsed_seconds))


if __name__ == "__main__":
    main()

"""
Stage-wise evidence survival by patching corrupted-donor activations into the
clean run and measuring clean top-1 score drop.

For each MMRed sample, this script:
1. runs the clean sample once, determines the clean run's top-1 numeric answer,
   and caches clean activations for the requested phase/token-set + layer
   combinations,
2. loads the existing single-frame corrupted donor for each evidence frame and
   caches donor activations for the same phase/token-set + layer combinations,
3. optionally filters donors by a minimum clean-vs-donor corruption gap:

       corruption_gap = clean_top1_score - donor_score_before_patch

4. patches donor activations into the clean run at one layer for one token set,
5. scores the clean run's own top-1 answer after patching and defines:

       importance_raw = clean_top1_score - patched_score
       importance_normalized_raw = importance_raw / corruption_gap
       importance_normalized_clipped = clip(importance_normalized_raw, 0.0, 1.0)

6. normalizes clipped donor importances across evidence-frame donors and computes
   entropy / normalized entropy per phase and layer.

The implementation intentionally reuses the current MMRed patching helpers so
model loading, prompt construction, answer scoring, and token indexing stay
aligned with the existing experiments.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import DEFAULT_MODEL_ID, find_subsequence, get_layers, image_token_groups

NORMALIZATION_EPS = 1e-8
PHASE_ORDER = ("frame_phase", "question_phase", "last_token_phase")
PHASE_LABELS = {
    "frame_phase": "frame phase",
    "question_phase": "question phase",
    "last_token_phase": "last-token phase",
}
PHASE_TOKEN_SETS = {
    "frame_phase": "evidence_frame_tokens",
    "question_phase": "question_tokens",
    "last_token_phase": "last_token",
}
PHASE_COLORS = {
    "frame_phase": "#1f77b4",
    "question_phase": "#ff7f0e",
    "last_token_phase": "#d62728",
}


def _phase_sort_index(phase: str) -> int:
    phase = str(phase)
    if phase not in PHASE_ORDER:
        raise ValueError(f"Unsupported phase={phase!r}; expected one of {PHASE_ORDER!r}")
    return PHASE_ORDER.index(phase)


def _runtime() -> Any:
    return gri._runtime()


def _model() -> Any:
    return _runtime().model


def _processor() -> Any:
    return _runtime().processor


def configure_runtime(model_name: str) -> None:
    gri.configure_runtime(model_name)


def _mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return (sum(values) / len(values)) if values else float("nan")


def _bootstrap_center_and_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int,
    rng: random.Random,
) -> Tuple[float, float, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        nan = float("nan")
        return nan, nan, nan

    center = _mean(finite_values)
    if len(finite_values) <= 1:
        return center, center, center

    boot_values: List[float] = []
    sample_count = len(finite_values)
    for _ in range(int(n_bootstrap)):
        sample = [finite_values[rng.randrange(sample_count)] for _ in range(sample_count)]
        boot_values.append(_mean(sample))
    boot_values.sort()
    lo_idx = int(0.025 * (n_bootstrap - 1))
    hi_idx = int(0.975 * (n_bootstrap - 1))
    return center, boot_values[lo_idx], boot_values[hi_idx]


def _summary_triplet(
    values: Sequence[float],
    *,
    n_bootstrap: int,
    rng: random.Random,
) -> Tuple[float, float, float]:
    return _bootstrap_center_and_ci(values, n_bootstrap=n_bootstrap, rng=rng)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _resolve_split_label(clean_data_dir: Path, split_arg: Optional[str]) -> str:
    if split_arg is not None and str(split_arg).strip():
        return str(split_arg).strip()
    return clean_data_dir.name


def _resolve_seq_len_display(
    clean_data_dir: Path,
    seq_len_arg: Optional[int],
    fallback_seq_len: Optional[int] = None,
) -> str:
    if seq_len_arg is not None:
        return f"seq_len={int(seq_len_arg)}"
    seq_len_label = eval_utils.resolve_seq_len_label(clean_data_dir)
    if seq_len_label and seq_len_label.startswith("seq_len_"):
        return f"seq_len={seq_len_label.split('_')[-1]}"
    if fallback_seq_len is not None:
        return f"seq_len={int(fallback_seq_len)}"
    return "seq_len=unknown"


def parse_inclusive_layer_selection(
    raw: Optional[str],
    *,
    num_layers: int,
    arg_name: str,
) -> List[int]:
    if raw is None or not str(raw).strip():
        return []

    selected: set[int] = set()
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{arg_name} must not be empty when provided")

    for part in parts:
        if ":" in part:
            fields = [field.strip() for field in part.split(":")]
            if len(fields) not in {2, 3}:
                raise ValueError(
                    f"Invalid range in {arg_name}: {part!r}. Expected start:end or start:end:step."
                )
            try:
                start = int(fields[0])
                end = int(fields[1])
                step = int(fields[2]) if len(fields) == 3 else 1
            except ValueError as exc:
                raise ValueError(f"Invalid integer in {arg_name}: {part!r}") from exc
            if step <= 0:
                raise ValueError(f"{arg_name} step must be positive: {part!r}")
            if end < start:
                raise ValueError(f"{arg_name} inclusive range end must be >= start: {part!r}")
            for layer_idx in range(start, end + 1, step):
                selected.add(int(layer_idx))
            continue

        if "-" in part:
            fields = [field.strip() for field in part.split("-")]
            if len(fields) != 2:
                raise ValueError(
                    f"Invalid range in {arg_name}: {part!r}. Expected start-end or start:end."
                )
            try:
                start = int(fields[0])
                end = int(fields[1])
            except ValueError as exc:
                raise ValueError(f"Invalid integer in {arg_name}: {part!r}") from exc
            if end < start:
                raise ValueError(f"{arg_name} inclusive range end must be >= start: {part!r}")
            for layer_idx in range(start, end + 1):
                selected.add(int(layer_idx))
            continue

        try:
            selected.add(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid layer index in {arg_name}: {part!r}") from exc

    selected_layers = sorted(selected)
    invalid = [layer_idx for layer_idx in selected_layers if layer_idx < 0 or layer_idx >= num_layers]
    if invalid:
        raise ValueError(
            f"{arg_name} contains out-of-bounds layers: {invalid}. Valid range is [0, {num_layers - 1}]."
        )
    return selected_layers


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Measure stage-wise evidence survival by patching corrupted-donor activations "
            "into the clean run and measuring clean top-1 score drop."
        )
    )
    ap.add_argument(
        "--clean_data_dir",
        "--data_root",
        dest="clean_data_dir",
        type=str,
        required=True,
        help="Root directory of clean MMRed samples.",
    )
    ap.add_argument(
        "--corrupted_data_dir",
        "--corrupted_root",
        dest="corrupted_data_dir",
        type=str,
        default=None,
        help=(
            "Root directory of per-frame corrupted MMRed donor samples. "
            "If omitted, infer it from --clean_data_dir."
        ),
    )
    ap.add_argument(
        "--output_dir",
        "--output",
        dest="output_dir",
        type=str,
        default="outputs",
        help="Directory where CSVs and plots will be written.",
    )
    ap.add_argument(
        "--sample_limit",
        "--limit",
        "--max_samples",
        dest="sample_limit",
        type=int,
        default=1,
        help="Maximum number of valid samples to process.",
    )
    ap.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help=(
            "Optional seed for randomized sample order. If omitted, sample order is "
            "randomized from system entropy on each run."
        ),
    )
    ap.add_argument(
        "--model",
        "--model_name",
        dest="model_name",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Model name to load. Defaults to the repo's current default runtime model.",
    )
    ap.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Optional seq_len label for logging and plot titles.",
    )
    ap.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split label for logging and plot titles.",
    )
    ap.add_argument(
        "--sample_mode",
        type=str,
        choices=("success_only", "failure_only", "all"),
        default="success_only",
        help="Whether to keep only clean successes, only clean failures, or all samples.",
    )
    ap.add_argument(
        "--lambda_threshold",
        "--min_score_diff_lambda",
        dest="min_score_diff_lambda",
        type=float,
        default=0.0,
        help=(
            "Minimum clean-vs-donor corruption gap required to use a donor, where "
            "corruption_gap = clean_top1_score - donor_score_before_patch."
        ),
    )
    ap.add_argument(
        "--frame_phase_layers",
        type=str,
        default=None,
        help="Inclusive layer selection for the frame phase. Examples: 0, 0:10, 0:10:2.",
    )
    ap.add_argument(
        "--question_phase_layers",
        type=str,
        default=None,
        help="Inclusive layer selection for the question phase. Examples: 17, 16:18.",
    )
    ap.add_argument(
        "--last_phase_layers",
        type=str,
        default=None,
        help="Inclusive layer selection for the last-token phase. Examples: 21, 21:28.",
    )
    ap.add_argument(
        "--clean_top1_must_match_gold",
        type=lambda raw: str(raw).strip().lower() in {"1", "true", "yes", "y", "on"},
        default=False,
        help=(
            "Optional compatibility flag. If true, keep only samples where the clean run's top-1 valid "
            "numeric answer matches the gold answer."
        ),
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed used for bootstrap confidence intervals.")
    ap.add_argument("--disable_plots", action="store_true")
    args = ap.parse_args()

    if int(args.sample_limit) <= 0:
        raise ValueError("--sample_limit must be a positive integer")
    if bool(args.clean_top1_must_match_gold) and str(args.sample_mode) != "success_only":
        raise ValueError("--clean_top1_must_match_gold is only compatible with --sample_mode success_only")
    return args


def _token_span_from_char_span(text: str, char_span: Tuple[int, int]) -> Tuple[int, int]:
    return gri._token_span_from_char_span(text, char_span)


def _positions_from_token_span(base_start: int, token_span: Tuple[int, int]) -> List[int]:
    return gri._positions_from_token_span(base_start, token_span)


def extract_token_metadata(
    *,
    inputs: Dict[str, Any],
    question: str,
    num_frames: int,
    sample_id: str,
    log_context: str,
) -> Optional[Dict[str, Any]]:
    try:
        input_ids = [int(token_id) for token_id in inputs["input_ids"][0].detach().cpu().tolist()]
        frame_groups = image_token_groups(
            inputs["input_ids"][0].detach().cpu(),
            expected_num_frames=num_frames,
            processor=_processor(),
        )
    except Exception as exc:
        print(f"{log_context} sample_id={sample_id} warning: failed to extract frame token groups ({exc})")
        return None

    if num_frames > 0 and len(frame_groups) != num_frames:
        print(
            f"{log_context} sample_id={sample_id} warning: expected {num_frames} frame token groups "
            f"but found {len(frame_groups)}"
        )
        return None

    all_frame_positions = [int(position) for group in frame_groups for position in group]
    if not all_frame_positions:
        print(f"{log_context} sample_id={sample_id} warning: no image token groups found in tokenized input")
        return None

    question_positions: List[int] = []
    try:
        prompt_text = tgi.build_prompt(question, num_frames=num_frames)
        prompt_text_ids = _processor().tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_text_start = find_subsequence(input_ids, [int(token_id) for token_id in prompt_text_ids])
        if prompt_text_start is None:
            raise RuntimeError("failed to locate prompt text in multimodal prompt")

        question_fragment = f"Question: {question}\n"
        question_fragment_start_in_prompt = prompt_text.find(question_fragment)
        if question_fragment_start_in_prompt < 0:
            raise RuntimeError(f"failed to locate question fragment in prompt: {question_fragment!r}")

        question_text_start_in_prompt = question_fragment_start_in_prompt + len("Question: ")
        question_span_in_prompt = (
            question_text_start_in_prompt,
            question_text_start_in_prompt + len(question) + len("\n"),
        )
        question_token_span = _token_span_from_char_span(prompt_text, question_span_in_prompt)
        question_positions = _positions_from_token_span(prompt_text_start, question_token_span)
        if not question_positions:
            raise RuntimeError("question span tokenized to an empty position set")
    except Exception as exc:
        print(f"{log_context} sample_id={sample_id} warning: failed to locate question tokens ({exc})")

    carrier_index = int(inputs["input_ids"].shape[1] - 1)
    return {
        "frame_groups": [[int(position) for position in group] for group in frame_groups],
        "frame_group_sizes": [int(len(group)) for group in frame_groups],
        "all_frame_positions": [int(position) for position in all_frame_positions],
        "question_positions": [int(position) for position in question_positions],
        "carrier_index": carrier_index,
    }


def _normalize_positions(token_positions: Sequence[int], *, prompt_len: int) -> List[int]:
    normalized: List[int] = []
    for position in token_positions:
        position_int = int(position)
        if position_int < 0:
            position_int = int(prompt_len) + position_int
        normalized.append(position_int)
    return normalized


def _validate_prompt_positions(
    token_positions: Sequence[int],
    *,
    prompt_len: int,
    label: str,
) -> Optional[List[int]]:
    normalized = _normalize_positions(token_positions, prompt_len=prompt_len)
    invalid = [position for position in normalized if position < 0 or position >= int(prompt_len)]
    if invalid:
        print(f"{label} warning: prompt positions out of bounds for prompt_len={prompt_len}: {invalid}")
        return None
    return normalized


def cache_clean_phase_activations(
    *,
    lm: LanguageModel,
    layers: Any,
    scoring_inputs: Dict[str, Any],
    prompt_len: int,
    answer_token_ids: Sequence[int],
    selected_layers_by_phase: Dict[str, List[int]],
    metadata: Dict[str, Any],
    evidence_frame_indices: Sequence[int],
) -> Tuple[float, Dict[Tuple[str, int, Optional[int]], torch.Tensor]]:
    all_selected_layers = sorted({
        int(layer_idx)
        for layer_indices in selected_layers_by_phase.values()
        for layer_idx in layer_indices
    })

    frame_positions_by_frame: Dict[int, List[int]] = {}
    for frame_idx in evidence_frame_indices:
        if int(frame_idx) >= len(metadata["frame_groups"]):
            continue
        normalized = _validate_prompt_positions(
            metadata["frame_groups"][int(frame_idx)],
            prompt_len=prompt_len,
            label=f"[cache] frame_idx={frame_idx}",
        )
        if normalized:
            frame_positions_by_frame[int(frame_idx)] = normalized

    question_positions = _validate_prompt_positions(
        metadata.get("question_positions", []),
        prompt_len=prompt_len,
        label="[cache] question",
    ) or []
    last_token_positions = _validate_prompt_positions(
        [-1],
        prompt_len=prompt_len,
        label="[cache] last_token",
    ) or []

    saved_cache: Dict[Tuple[str, int, Optional[int]], Any] = {}
    with torch.no_grad():
        with lm.trace(scoring_inputs):
            for layer_idx in all_selected_layers:
                layer_out = tgi._to_hidden_tensor(layers[int(layer_idx)].output)
                if int(layer_idx) in selected_layers_by_phase["frame_phase"]:
                    for frame_idx, positions in frame_positions_by_frame.items():
                        if positions:
                            saved_cache[("frame_phase", int(layer_idx), int(frame_idx))] = (
                                layer_out[:, positions, :].save()
                            )
                if int(layer_idx) in selected_layers_by_phase["question_phase"] and question_positions:
                    saved_cache[("question_phase", int(layer_idx), None)] = (
                        layer_out[:, question_positions, :].save()
                    )
                if int(layer_idx) in selected_layers_by_phase["last_token_phase"] and last_token_positions:
                    saved_cache[("last_token_phase", int(layer_idx), None)] = (
                        layer_out[:, last_token_positions, :].save()
                    )
            saved_logits = lm.output.logits.save()

    logits = tgi._materialize_saved(saved_logits)
    score = float(
        tgi.sequence_logprob_from_logits(
            logits,
            prompt_len=prompt_len,
            answer_token_ids=list(answer_token_ids),
        )[0].item()
    )
    cache: Dict[Tuple[str, int, Optional[int]], torch.Tensor] = {}
    for cache_key, saved_tensor in saved_cache.items():
        tensor = tgi._materialize_saved(saved_tensor)
        if not torch.is_tensor(tensor):
            raise TypeError(f"Clean cache entry {cache_key!r} did not materialize to a tensor")
        cache[cache_key] = tensor.detach().cpu()
    return score, cache


def run_patched_score_from_cached_activations(
    *,
    lm: LanguageModel,
    layers: Any,
    target_scoring_inputs: Dict[str, Any],
    target_prompt_len: int,
    answer_token_ids: Sequence[int],
    layer_idx: int,
    target_token_positions: Sequence[int],
    source_cache_tensor: torch.Tensor,
) -> float:
    normalized_target_positions = _normalize_positions(target_token_positions, prompt_len=target_prompt_len)
    if not normalized_target_positions:
        raise ValueError("target token positions are empty")
    if int(source_cache_tensor.shape[1]) != len(normalized_target_positions):
        raise ValueError(
            "source/target token count mismatch: "
            f"source={int(source_cache_tensor.shape[1])} target={len(normalized_target_positions)}"
        )

    with torch.no_grad():
        with lm.trace(target_scoring_inputs):
            target_layer_out = tgi._to_hidden_tensor(layers[int(layer_idx)].output)
            source_tensor = source_cache_tensor.to(device=target_layer_out.device, dtype=target_layer_out.dtype)
            target_layer_out[0, normalized_target_positions, :] = source_tensor[0, :, :]
            saved_logits = lm.output.logits.save()

    logits = tgi._materialize_saved(saved_logits)
    return float(
        tgi.sequence_logprob_from_logits(
            logits,
            prompt_len=target_prompt_len,
            answer_token_ids=list(answer_token_ids),
        )[0].item()
    )


def _phase_token_positions(
    *,
    phase: str,
    metadata: Dict[str, Any],
    frame_idx: int,
) -> List[int]:
    if phase == "frame_phase":
        if int(frame_idx) >= len(metadata["frame_groups"]):
            return []
        return [int(position) for position in metadata["frame_groups"][int(frame_idx)]]
    if phase == "question_phase":
        return [int(position) for position in metadata.get("question_positions", [])]
    if phase == "last_token_phase":
        return [-1]
    raise ValueError(f"Unsupported phase={phase!r}")


def format_phase_layer_importance_summary(
    *,
    phase: str,
    layer_idx: int,
    importance_rows: Sequence[Dict[str, Any]],
    entropy_row: Dict[str, Any],
) -> str:
    per_frame_items = [
        (
            int(row["evidence_frame_index"]),
            int(row["evidence_frame_rank"]),
            float(row["importance_normalized_clipped"]),
        )
        for row in importance_rows
    ]
    per_frame_items.sort(key=lambda item: item[1])
    importance_text = ", ".join(
        f"frame_{frame_idx}[rank={rank}]={importance:.4f}"
        for frame_idx, rank, importance in per_frame_items
    ) or "<none>"
    return (
        f"  phase={phase} layer={layer_idx} "
        f"used_count={int(entropy_row['used_evidence_count'])} "
        f"d_i_normalized_clipped=[{importance_text}] "
        f"entropy={float(entropy_row['entropy']):.6f} "
        f"normalized_entropy={float(entropy_row['normalized_entropy']):.6f} "
        f"importance_sum_normalized_clipped={float(entropy_row['importance_sum_normalized_clipped']):.6f}"
    )


def process_sample(
    *,
    sample_dir: Path,
    sample_index: int,
    total_samples: int,
    split: str,
    lm: LanguageModel,
    layers: Any,
    selected_layers_by_phase: Dict[str, List[int]],
    corrupted_data_dir: Path,
    selected_layers_spec_by_phase: Dict[str, Optional[str]],
    sample_mode: str,
    clean_top1_must_match_gold: bool,
    min_score_diff_lambda: float,
) -> Optional[Dict[str, Any]]:
    try:
        sample_id, frames, question, states, answer = eval_utils.load_mmred_sample(sample_dir)
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_dir.name} skipped: load failure ({exc})")
        return None

    evidence_frame_indices = eval_utils.collect_evidence_frame_indices(question, states)
    if len(evidence_frame_indices) < 1:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: no evidence frames")
        return None

    clean_input_result = gri.build_clean_inputs(
        sample_id=sample_id,
        frames=frames,
        question=question,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if clean_input_result is None:
        return None
    clean_inputs, clean_prompt_len = clean_input_result

    gold_answer = str(answer).strip()
    clean_answer_metrics = gri.score_clean_sample(
        sample_id=sample_id,
        frames=frames,
        answer_text=gold_answer,
        clean_inputs=clean_inputs,
        prompt_len=clean_prompt_len,
        clean_top1_must_match_gold=False,
        lm=lm,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if clean_answer_metrics is None:
        return None

    clean_pred_answer = str(clean_answer_metrics["best_answer_text"]).strip()
    clean_top1_answer = str(clean_pred_answer)
    clean_is_correct = bool(clean_answer_metrics["clean_top1_correct"])

    if clean_top1_must_match_gold and not clean_is_correct:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: clean top-1 is "
            f"{clean_pred_answer!r}, not correct answer {gold_answer!r}"
        )
        return None
    if sample_mode == "success_only" and not clean_is_correct:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"sample_mode=success_only but clean top-1 is {clean_pred_answer!r}, not gold answer {gold_answer!r}"
        )
        return None
    if sample_mode == "failure_only" and clean_is_correct:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"sample_mode=failure_only but clean top-1 already matches gold answer {gold_answer!r}"
        )
        return None

    clean_top1_ids = gri.build_answer_token_ids(
        sample_id=sample_id,
        answer_text=clean_top1_answer,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if clean_top1_ids is None:
        return None

    clean_metadata = extract_token_metadata(
        inputs=clean_inputs,
        question=question,
        num_frames=len(frames),
        sample_id=sample_id,
        log_context=f"[{sample_index}/{total_samples}]",
    )
    if clean_metadata is None:
        return None

    patchable_evidence_frames = [
        int(frame_idx)
        for frame_idx in evidence_frame_indices
        if int(frame_idx) < len(clean_metadata["frame_groups"]) and clean_metadata["frame_groups"][int(frame_idx)]
    ]
    if selected_layers_by_phase["frame_phase"] and not patchable_evidence_frames:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
            "frame phase selected but no evidence frames map to clean frame token groups"
        )
    if selected_layers_by_phase["question_phase"] and not clean_metadata.get("question_positions"):
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
            "question phase selected but clean question token set is empty"
        )

    has_any_requested_target = bool(selected_layers_by_phase["last_token_phase"])
    if selected_layers_by_phase["frame_phase"] and patchable_evidence_frames:
        has_any_requested_target = True
    if selected_layers_by_phase["question_phase"] and clean_metadata.get("question_positions"):
        has_any_requested_target = True
    if not has_any_requested_target:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            "none of the requested phases have usable clean target token sets"
        )
        return None

    if len(evidence_frame_indices) <= 1:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
            f"E={len(evidence_frame_indices)} <= 1, entropy across evidence frames will be NaN"
        )

    clean_answer_inputs = tgi.append_answer_tokens_for_scoring(clean_inputs, clean_top1_ids)
    try:
        clean_top1_score, _clean_cache = cache_clean_phase_activations(
            lm=lm,
            layers=layers,
            scoring_inputs=clean_answer_inputs,
            prompt_len=clean_prompt_len,
            answer_token_ids=clean_top1_ids,
            selected_layers_by_phase=selected_layers_by_phase,
            metadata=clean_metadata,
            evidence_frame_indices=evidence_frame_indices,
        )
    except Exception as exc:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"failed to cache clean activations / score clean top-1 answer ({exc})"
        )
        return None

    print(
        f"[{sample_index}/{total_samples}] sample_id={sample_id} sample_mode={sample_mode} "
        f"gold_answer={gold_answer!r} clean_pred_answer={clean_pred_answer!r} "
        f"clean_is_correct={clean_is_correct} clean_top1_answer={clean_top1_answer!r} "
        f"clean_top1_score={clean_top1_score:.4f} evidence_frames={evidence_frame_indices} "
        f"frame_phase_patchable_evidence_frames={patchable_evidence_frames}"
    )

    importance_rows: List[Dict[str, Any]] = []
    entropy_rows: List[Dict[str, Any]] = []

    evidence_rank_lookup = {
        int(frame_idx): rank
        for rank, frame_idx in enumerate([int(frame_idx) for frame_idx in evidence_frame_indices], start=1)
    }

    for frame_idx in evidence_frame_indices:
        donor_sample_dir = eval_utils.resolve_corrupted_sample_dir(
            corrupted_data_root=corrupted_data_dir,
            sample_id=sample_id,
            frame_idx=int(frame_idx),
        )
        if not donor_sample_dir.is_dir():
            print(
                f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
                f"missing donor for frame_idx={frame_idx} at {donor_sample_dir}"
            )
            continue

        try:
            _, donor_frames, donor_question, _, _ = eval_utils.load_mmred_sample(donor_sample_dir)
        except Exception as exc:
            print(
                f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
                f"failed to load donor frame_idx={frame_idx} ({exc})"
            )
            continue

        donor_input_result = gri.build_clean_inputs(
            sample_id=f"{sample_id}/corrupted_frame_{frame_idx}",
            frames=donor_frames,
            question=donor_question,
            sample_index=sample_index,
            total_samples=total_samples,
        )
        if donor_input_result is None:
            print(
                f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
                f"failed to build donor inputs for frame_idx={frame_idx}"
            )
            continue
        donor_inputs, donor_prompt_len = donor_input_result

        donor_metadata = extract_token_metadata(
            inputs=donor_inputs,
            question=donor_question,
            num_frames=len(donor_frames),
            sample_id=f"{sample_id}/corrupted_frame_{frame_idx}",
            log_context=f"[{sample_index}/{total_samples}]",
        )
        if donor_metadata is None:
            print(
                f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
                f"failed to locate donor token metadata for frame_idx={frame_idx}"
            )
            continue

        donor_answer_inputs = tgi.append_answer_tokens_for_scoring(donor_inputs, clean_top1_ids)
        try:
            donor_score_before_patch, donor_cache = cache_clean_phase_activations(
                lm=lm,
                layers=layers,
                scoring_inputs=donor_answer_inputs,
                prompt_len=donor_prompt_len,
                answer_token_ids=clean_top1_ids,
                selected_layers_by_phase=selected_layers_by_phase,
                metadata=donor_metadata,
                evidence_frame_indices=[int(frame_idx)],
            )
        except Exception as exc:
            print(
                f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
                f"failed to cache donor activations / score donor frame_idx={frame_idx} ({exc})"
            )
            continue

        corruption_gap = float(clean_top1_score - donor_score_before_patch)
        donor_passed_lambda = bool(corruption_gap >= float(min_score_diff_lambda))
        common_importance_row = {
            "sample_id": str(sample_id),
            "seq_len": int(len(frames)),
            "split": str(split),
            "sample_mode": str(sample_mode),
            "gold_answer": str(gold_answer),
            "clean_pred_answer": str(clean_pred_answer),
            "clean_is_correct": bool(clean_is_correct),
            "clean_top1_answer": str(clean_top1_answer),
            "evidence_count": int(len(evidence_frame_indices)),
            "patchable_evidence_count": int(len(patchable_evidence_frames)),
            "evidence_frame_index": int(frame_idx),
            "evidence_frame_rank": int(evidence_rank_lookup[int(frame_idx)]),
            "clean_top1_score": float(clean_top1_score),
            "donor_sample_dir": str(donor_sample_dir),
            "donor_score_before_patch": float(donor_score_before_patch),
            "corruption_gap": float(corruption_gap),
            "min_score_diff_lambda": float(min_score_diff_lambda),
            "donor_passed_lambda": bool(donor_passed_lambda),
            "clean_prompt_len": int(clean_prompt_len),
            "donor_prompt_len": int(donor_prompt_len),
            "model_name": str(_runtime().model_name),
        }
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} donor frame_i={int(frame_idx)} "
            f"clean_top1_answer={clean_top1_answer!r} clean_top1_score={clean_top1_score:.6f} "
            f"donor_score_before_patch={donor_score_before_patch:.6f} "
            f"corruption_gap={corruption_gap:.6f} min_score_diff_lambda={float(min_score_diff_lambda):.6f} "
            f"donor_passed_lambda={donor_passed_lambda}"
        )
        if not donor_passed_lambda:
            print(
                f"[{sample_index}/{total_samples}] sample_id={sample_id} donor frame_i={int(frame_idx)} "
                "skipped for all phases/layers because corruption_gap is below lambda"
            )
            for phase in PHASE_ORDER:
                phase_layers = selected_layers_by_phase[phase]
                if not phase_layers:
                    continue
                for layer_idx in phase_layers:
                    importance_rows.append({
                        **common_importance_row,
                        "phase": str(phase),
                        "phase_label": PHASE_LABELS[str(phase)],
                        "token_set": PHASE_TOKEN_SETS[str(phase)],
                        "layer": int(layer_idx),
                        "patched_score": float("nan"),
                        "importance_raw": float("nan"),
                        "importance_clipped_rawdrop": float("nan"),
                        "importance_normalized_raw": float("nan"),
                        "importance_normalized_clipped": float("nan"),
                        "phase_layers_spec": selected_layers_spec_by_phase[str(phase)],
                    })
            continue

        if corruption_gap <= NORMALIZATION_EPS:
            print(
                f"[{sample_index}/{total_samples}] sample_id={sample_id} donor frame_i={int(frame_idx)} warning: "
                f"corruption_gap={corruption_gap:.8e} <= {NORMALIZATION_EPS:.1e}; "
                "normalized importance values will be NaN and excluded from entropy/normalized aggregates"
            )

        for phase in PHASE_ORDER:
            phase_layers = selected_layers_by_phase[phase]
            if not phase_layers:
                continue

            cache_key_suffix: Optional[int] = int(frame_idx) if phase == "frame_phase" else None
            if phase == "question_phase" and not clean_metadata.get("question_positions"):
                continue

            raw_target_positions = _phase_token_positions(
                phase=phase,
                metadata=clean_metadata,
                frame_idx=int(frame_idx),
            )
            target_positions = _validate_prompt_positions(
                raw_target_positions,
                prompt_len=clean_prompt_len,
                label=(
                    f"[{sample_index}/{total_samples}] sample_id={sample_id} clean target frame_idx={frame_idx} "
                    f"phase={phase}"
                ),
            )
            if target_positions is None or not target_positions:
                print(
                    f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
                    f"empty/invalid target positions for frame_idx={frame_idx} phase={phase}; skipping"
                )
                continue

            for layer_idx in phase_layers:
                cache_key = (phase, int(layer_idx), cache_key_suffix)
                source_cache_tensor = donor_cache.get(cache_key)
                if source_cache_tensor is None:
                    print(
                        f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
                        f"donor cache missing for phase={phase} layer={layer_idx} "
                        f"frame_idx={cache_key_suffix}; skipping"
                    )
                    continue

                if int(source_cache_tensor.shape[1]) != len(target_positions):
                    print(
                        f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
                        f"phase={phase} layer={layer_idx} frame_idx={frame_idx} token-count mismatch "
                        f"(donor={int(source_cache_tensor.shape[1])}, clean={len(target_positions)}); skipping"
                    )
                    continue

                try:
                    patched_score = run_patched_score_from_cached_activations(
                        lm=lm,
                        layers=layers,
                        target_scoring_inputs=clean_answer_inputs,
                        target_prompt_len=clean_prompt_len,
                        answer_token_ids=clean_top1_ids,
                        layer_idx=int(layer_idx),
                        target_token_positions=target_positions,
                        source_cache_tensor=source_cache_tensor,
                    )
                except Exception as exc:
                    print(
                        f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
                        f"failed patching frame_idx={frame_idx} phase={phase} layer={layer_idx} ({exc})"
                    )
                    continue

                importance_raw = float(clean_top1_score - patched_score)
                importance_clipped_rawdrop = float(max(importance_raw, 0.0))
                if corruption_gap <= NORMALIZATION_EPS:
                    importance_normalized_raw = float("nan")
                    importance_normalized_clipped = float("nan")
                else:
                    importance_normalized_raw = float(importance_raw / corruption_gap)
                    importance_normalized_clipped = float(
                        min(max(importance_normalized_raw, 0.0), 1.0)
                    )
                importance_rows.append({
                    **common_importance_row,
                    "phase": str(phase),
                    "phase_label": PHASE_LABELS[str(phase)],
                    "token_set": PHASE_TOKEN_SETS[str(phase)],
                    "layer": int(layer_idx),
                    "patched_score": float(patched_score),
                    "importance_raw": float(importance_raw),
                    "importance_clipped_rawdrop": float(importance_clipped_rawdrop),
                    "importance_normalized_raw": float(importance_normalized_raw),
                    "importance_normalized_clipped": float(importance_normalized_clipped),
                    "phase_layers_spec": selected_layers_spec_by_phase[str(phase)],
                })

    for phase in PHASE_ORDER:
        for layer_idx in selected_layers_by_phase[phase]:
            phase_layer_rows = [
                row
                for row in importance_rows
                if str(row["phase"]) == str(phase)
                and int(row["layer"]) == int(layer_idx)
                and bool(row.get("donor_passed_lambda", False))
                and math.isfinite(float(row.get("importance_normalized_clipped", float("nan"))))
            ]
            used_evidence_count = len(phase_layer_rows)
            importance_sum_raw = float(
                sum(float(row["importance_raw"]) for row in phase_layer_rows)
            )
            importance_sum_normalized_clipped = float(
                sum(float(row["importance_normalized_clipped"]) for row in phase_layer_rows)
            )

            entropy = float("nan")
            normalized_entropy = float("nan")
            used_evidence_frames = [
                int(row["evidence_frame_index"]) for row in phase_layer_rows
            ]

            if used_evidence_count <= 1:
                print(
                    f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
                    f"phase={phase} layer={layer_idx} has used_evidence_count={used_evidence_count} <= 1; "
                    "entropy and normalized_entropy will be NaN"
                )
            elif importance_sum_normalized_clipped <= NORMALIZATION_EPS:
                print(
                    f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
                    f"phase={phase} layer={layer_idx} has near-zero normalized/clipped importance sum "
                    f"({importance_sum_normalized_clipped:.8e}); entropy and normalized_entropy will be NaN"
                )
            else:
                probabilities = [
                    float(row["importance_normalized_clipped"]) / importance_sum_normalized_clipped
                    for row in phase_layer_rows
                ]
                entropy = float(tgi.entropy_from_probabilities(probabilities))
                normalized_entropy = float(entropy / math.log(float(used_evidence_count)))

            entropy_row = {
                "sample_id": str(sample_id),
                "seq_len": int(len(frames)),
                "split": str(split),
                "sample_mode": str(sample_mode),
                "gold_answer": str(gold_answer),
                "clean_pred_answer": str(clean_pred_answer),
                "clean_is_correct": bool(clean_is_correct),
                "clean_top1_answer": str(clean_top1_answer),
                "evidence_count": int(len(evidence_frame_indices)),
                "patchable_evidence_count": int(len(patchable_evidence_frames)),
                "used_evidence_count": int(used_evidence_count),
                "used_evidence_frames_json": json.dumps(
                    [int(frame_idx) for frame_idx in used_evidence_frames]
                ),
                "phase": str(phase),
                "phase_label": PHASE_LABELS[str(phase)],
                "token_set": PHASE_TOKEN_SETS[str(phase)],
                "layer": int(layer_idx),
                "clean_top1_score": float(clean_top1_score),
                "importance_sum_normalized_clipped": float(importance_sum_normalized_clipped),
                "importance_sum_raw": float(importance_sum_raw),
                "entropy": float(entropy),
                "normalized_entropy": float(normalized_entropy),
                "min_score_diff_lambda": float(min_score_diff_lambda),
                "phase_layers_spec": selected_layers_spec_by_phase[str(phase)],
                "model_name": str(_runtime().model_name),
            }
            entropy_rows.append(entropy_row)
            print(
                format_phase_layer_importance_summary(
                    phase=phase,
                    layer_idx=int(layer_idx),
                    importance_rows=phase_layer_rows,
                    entropy_row=entropy_row,
                )
            )

    return {
        "sample_id": str(sample_id),
        "seq_len": int(len(frames)),
        "split": str(split),
        "sample_mode": str(sample_mode),
        "question": str(question),
        "answer": str(answer),
        "gold_answer": str(gold_answer),
        "clean_pred_answer": str(clean_pred_answer),
        "clean_top1_answer": str(clean_top1_answer),
        "clean_is_correct": bool(clean_is_correct),
        "clean_top1_score": float(clean_top1_score),
        "evidence_frames": [int(frame_idx) for frame_idx in evidence_frame_indices],
        "patchable_evidence_frames": [int(frame_idx) for frame_idx in patchable_evidence_frames],
        "importance_rows": importance_rows,
        "entropy_rows": entropy_rows,
    }


def flatten_sample_rows(sample_payloads: Sequence[Dict[str, Any]], row_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample_payload in sample_payloads:
        rows.extend(sample_payload.get(row_key, []))
    return rows


def build_sum_importance_aggregate_rows(
    entropy_rows: Sequence[Dict[str, Any]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    grouped_rows: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in entropy_rows:
        grouped_rows.setdefault((str(row["phase"]), int(row["layer"])), []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for phase, layer_idx in sorted(grouped_rows, key=lambda item: (_phase_sort_index(item[0]), item[1])):
        bucket_rows = grouped_rows[(phase, layer_idx)]
        clipped_values = [float(row["importance_sum_normalized_clipped"]) for row in bucket_rows]
        clipped_center, clipped_lo, clipped_hi = _summary_triplet(
            clipped_values,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        aggregate_rows.append({
            "phase": str(phase),
            "phase_label": PHASE_LABELS[str(phase)],
            "token_set": PHASE_TOKEN_SETS[str(phase)],
            "layer": int(layer_idx),
            "n": int(len(bucket_rows)),
            "model_name": str(bucket_rows[0]["model_name"]),
            "min_score_diff_lambda": float(bucket_rows[0]["min_score_diff_lambda"]),
            "mean_importance_sum_normalized_clipped": clipped_center,
            "importance_sum_normalized_clipped_ci_lower": clipped_lo,
            "importance_sum_normalized_clipped_ci_upper": clipped_hi,
        })
    return aggregate_rows


def build_entropy_aggregate_rows(
    entropy_rows: Sequence[Dict[str, Any]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    grouped_rows: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in entropy_rows:
        grouped_rows.setdefault((str(row["phase"]), int(row["layer"])), []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for phase, layer_idx in sorted(grouped_rows, key=lambda item: (_phase_sort_index(item[0]), item[1])):
        bucket_rows = grouped_rows[(phase, layer_idx)]
        entropy_values = [float(row["entropy"]) for row in bucket_rows]
        normalized_values = [float(row["normalized_entropy"]) for row in bucket_rows]
        entropy_center, entropy_lo, entropy_hi = _summary_triplet(
            entropy_values,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        norm_center, norm_lo, norm_hi = _summary_triplet(
            normalized_values,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        aggregate_rows.append({
            "phase": str(phase),
            "phase_label": PHASE_LABELS[str(phase)],
            "token_set": PHASE_TOKEN_SETS[str(phase)],
            "layer": int(layer_idx),
            "n": int(len(bucket_rows)),
            "model_name": str(bucket_rows[0]["model_name"]),
            "min_score_diff_lambda": float(bucket_rows[0]["min_score_diff_lambda"]),
            "n_entropy": int(sum(math.isfinite(float(value)) for value in entropy_values)),
            "n_normalized_entropy": int(sum(math.isfinite(float(value)) for value in normalized_values)),
            "mean_entropy": entropy_center,
            "entropy_ci_lower": entropy_lo,
            "entropy_ci_upper": entropy_hi,
            "mean_normalized_entropy": norm_center,
            "normalized_entropy_ci_lower": norm_lo,
            "normalized_entropy_ci_upper": norm_hi,
        })
    return aggregate_rows


def build_per_rank_importance_aggregate_rows(
    importance_rows: Sequence[Dict[str, Any]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    per_sample_rank_values: Dict[Tuple[str, str, int], List[float]] = {}
    default_model_name = ""
    default_min_score_diff_lambda = float("nan")
    for row in importance_rows:
        if not default_model_name and row.get("model_name") is not None:
            default_model_name = str(row["model_name"])
        if math.isnan(default_min_score_diff_lambda) and row.get("min_score_diff_lambda") is not None:
            default_min_score_diff_lambda = float(row["min_score_diff_lambda"])
        importance_normalized_clipped = float(row.get("importance_normalized_clipped", float("nan")))
        if not bool(row.get("donor_passed_lambda", False)):
            continue
        if not math.isfinite(importance_normalized_clipped):
            continue
        key = (
            str(row["sample_id"]),
            str(row["phase"]),
            int(row["evidence_frame_rank"]),
        )
        per_sample_rank_values.setdefault(key, []).append(importance_normalized_clipped)

    grouped_values: Dict[Tuple[str, int], List[float]] = {}
    for (_, phase, rank), values in per_sample_rank_values.items():
        grouped_values.setdefault((phase, rank), []).append(_mean(values))

    aggregate_rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for phase, rank in sorted(grouped_values, key=lambda item: (_phase_sort_index(item[0]), item[1])):
        values = grouped_values[(phase, rank)]
        center, lo_value, hi_value = _summary_triplet(values, n_bootstrap=n_bootstrap, rng=rng)
        aggregate_rows.append({
            "phase": str(phase),
            "phase_label": PHASE_LABELS[str(phase)],
            "token_set": PHASE_TOKEN_SETS[str(phase)],
            "evidence_frame_rank": int(rank),
            "n": int(len(values)),
            "model_name": default_model_name,
            "min_score_diff_lambda": float(default_min_score_diff_lambda),
            "mean_importance_normalized_clipped": center,
            "importance_normalized_clipped_ci_lower": lo_value,
            "importance_normalized_clipped_ci_upper": hi_value,
        })
    return aggregate_rows


def _plot_title(main_title: str, *, seq_len_display: str, split: str, sample_mode: str, n_samples: int) -> str:
    return f"{main_title}\n{seq_len_display} | split={split} | sample_mode={sample_mode} | n={n_samples}"


def _prepare_plot_axes(ax: Any, *, x_label: str, y_label: str, title: str) -> None:
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_sum_importance_lines(
    *,
    aggregate_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    seq_len_display: str,
    split: str,
    sample_mode: str,
    n_samples: int,
) -> Optional[Path]:
    if not aggregate_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot {output_path.name}: matplotlib is not available ({exc})")
        return None

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    any_line = False
    for phase in PHASE_ORDER:
        phase_rows = sorted(
            [row for row in aggregate_rows if str(row["phase"]) == str(phase)],
            key=lambda row: int(row["layer"]),
        )
        if not phase_rows:
            continue
        x = [int(row["layer"]) for row in phase_rows]
        y = [float(row["mean_importance_sum_normalized_clipped"]) for row in phase_rows]
        lo = [float(row["importance_sum_normalized_clipped_ci_lower"]) for row in phase_rows]
        hi = [float(row["importance_sum_normalized_clipped_ci_upper"]) for row in phase_rows]
        line, = ax.plot(
            x,
            y,
            linewidth=2.2,
            color=PHASE_COLORS[str(phase)],
            label=PHASE_LABELS[str(phase)],
        )
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.18)
        any_line = True

    if not any_line:
        plt.close(fig)
        return None

    _prepare_plot_axes(
        ax,
        x_label="Layer",
        y_label="Mean sum normalized/clipped importance",
        title=_plot_title(
            "Stage-Wise Sum Importance Across Phases",
            seq_len_display=seq_len_display,
            split=split,
            sample_mode=sample_mode,
            n_samples=n_samples,
        ),
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_normalized_entropy_all_phases(
    *,
    aggregate_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    seq_len_display: str,
    split: str,
    sample_mode: str,
    n_samples: int,
) -> Optional[Path]:
    if not aggregate_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot {output_path.name}: matplotlib is not available ({exc})")
        return None

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    any_line = False
    for phase in PHASE_ORDER:
        phase_rows = sorted(
            [row for row in aggregate_rows if str(row["phase"]) == str(phase)],
            key=lambda row: int(row["layer"]),
        )
        if not phase_rows:
            continue
        x = [int(row["layer"]) for row in phase_rows]
        y = [float(row["mean_normalized_entropy"]) for row in phase_rows]
        lo = [float(row["normalized_entropy_ci_lower"]) for row in phase_rows]
        hi = [float(row["normalized_entropy_ci_upper"]) for row in phase_rows]
        line, = ax.plot(
            x,
            y,
            linewidth=2.2,
            color=PHASE_COLORS[str(phase)],
            label=f"{PHASE_LABELS[str(phase)]} / {PHASE_TOKEN_SETS[str(phase)]}",
        )
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.18)
        any_line = True

    if not any_line:
        plt.close(fig)
        return None

    _prepare_plot_axes(
        ax,
        x_label="Layer",
        y_label="Normalized entropy",
        title=_plot_title(
            "Normalized Entropy Across Phases",
            seq_len_display=seq_len_display,
            split=split,
            sample_mode=sample_mode,
            n_samples=n_samples,
        ),
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_rank_importance_lines(
    *,
    aggregate_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    seq_len_display: str,
    split: str,
    sample_mode: str,
    n_samples: int,
) -> Optional[Path]:
    if not aggregate_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot {output_path.name}: matplotlib is not available ({exc})")
        return None

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    any_line = False
    for phase in PHASE_ORDER:
        phase_rows = sorted(
            [row for row in aggregate_rows if str(row["phase"]) == str(phase)],
            key=lambda row: int(row["evidence_frame_rank"]),
        )
        if not phase_rows:
            continue
        x = [int(row["evidence_frame_rank"]) for row in phase_rows]
        y = [float(row["mean_importance_normalized_clipped"]) for row in phase_rows]
        lo = [float(row["importance_normalized_clipped_ci_lower"]) for row in phase_rows]
        hi = [float(row["importance_normalized_clipped_ci_upper"]) for row in phase_rows]
        line, = ax.plot(
            x,
            y,
            linewidth=2.2,
            color=PHASE_COLORS[str(phase)],
            label=PHASE_LABELS[str(phase)],
        )
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.18)
        any_line = True

    if not any_line:
        plt.close(fig)
        return None

    _prepare_plot_axes(
        ax,
        x_label="Evidence-frame rank within sample (1-based)",
        y_label="Mean normalized/clipped importance",
        title=_plot_title(
            "Mean Per-Frame Normalized/Clipped Importance by Evidence Rank",
            seq_len_display=seq_len_display,
            split=split,
            sample_mode=sample_mode,
            n_samples=n_samples,
        ),
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def finalize_outputs(
    *,
    sample_payloads: Sequence[Dict[str, Any]],
    output_dir: Path,
    seq_len_display: str,
    split: str,
    sample_mode: str,
    seed: int,
    disable_plots: bool,
) -> None:
    importance_rows = flatten_sample_rows(sample_payloads, "importance_rows")
    entropy_rows = flatten_sample_rows(sample_payloads, "entropy_rows")

    sum_importance_aggregate_rows = build_sum_importance_aggregate_rows(
        entropy_rows,
        seed=seed,
    )
    entropy_aggregate_rows = build_entropy_aggregate_rows(
        entropy_rows,
        seed=seed,
    )
    per_rank_aggregate_rows = build_per_rank_importance_aggregate_rows(
        importance_rows,
        seed=seed,
    )

    per_sample_importance_csv = output_dir / "per_sample_importance.csv"
    per_sample_entropy_csv = output_dir / "per_sample_entropy.csv"
    aggregate_sum_importance_csv = output_dir / "aggregate_sum_importance.csv"
    aggregate_entropy_csv = output_dir / "aggregate_entropy.csv"
    aggregate_per_rank_csv = output_dir / "aggregate_per_rank_mean_importance.csv"

    write_csv(
        per_sample_importance_csv,
        importance_rows,
        fieldnames=[
            "sample_id",
            "seq_len",
            "split",
            "sample_mode",
            "gold_answer",
            "clean_pred_answer",
            "clean_is_correct",
            "clean_top1_answer",
            "evidence_count",
            "patchable_evidence_count",
            "evidence_frame_index",
            "evidence_frame_rank",
            "phase",
            "phase_label",
            "token_set",
            "layer",
            "clean_top1_score",
            "donor_sample_dir",
            "donor_score_before_patch",
            "corruption_gap",
            "min_score_diff_lambda",
            "donor_passed_lambda",
            "patched_score",
            "importance_raw",
            "importance_clipped_rawdrop",
            "importance_normalized_raw",
            "importance_normalized_clipped",
            "clean_prompt_len",
            "donor_prompt_len",
            "phase_layers_spec",
            "model_name",
        ],
    )
    write_csv(
        per_sample_entropy_csv,
        entropy_rows,
        fieldnames=[
            "sample_id",
            "seq_len",
            "split",
            "sample_mode",
            "gold_answer",
            "clean_pred_answer",
            "clean_is_correct",
            "clean_top1_answer",
            "evidence_count",
            "patchable_evidence_count",
            "used_evidence_count",
            "used_evidence_frames_json",
            "phase",
            "phase_label",
            "token_set",
            "layer",
            "clean_top1_score",
            "importance_sum_normalized_clipped",
            "importance_sum_raw",
            "entropy",
            "normalized_entropy",
            "min_score_diff_lambda",
            "phase_layers_spec",
            "model_name",
        ],
    )
    write_csv(
        aggregate_sum_importance_csv,
        [
            {
                **row,
                "sample_mode": str(sample_mode),
            }
            for row in sum_importance_aggregate_rows
        ],
        fieldnames=[
            "sample_mode",
            "model_name",
            "min_score_diff_lambda",
            "phase",
            "phase_label",
            "token_set",
            "layer",
            "n",
            "mean_importance_sum_normalized_clipped",
            "importance_sum_normalized_clipped_ci_lower",
            "importance_sum_normalized_clipped_ci_upper",
        ],
    )
    write_csv(
        aggregate_entropy_csv,
        [
            {
                **row,
                "sample_mode": str(sample_mode),
            }
            for row in entropy_aggregate_rows
        ],
        fieldnames=[
            "sample_mode",
            "model_name",
            "min_score_diff_lambda",
            "phase",
            "phase_label",
            "token_set",
            "layer",
            "n",
            "n_entropy",
            "n_normalized_entropy",
            "mean_entropy",
            "entropy_ci_lower",
            "entropy_ci_upper",
            "mean_normalized_entropy",
            "normalized_entropy_ci_lower",
            "normalized_entropy_ci_upper",
        ],
    )
    write_csv(
        aggregate_per_rank_csv,
        [
            {
                **row,
                "sample_mode": str(sample_mode),
            }
            for row in per_rank_aggregate_rows
        ],
        fieldnames=[
            "sample_mode",
            "model_name",
            "min_score_diff_lambda",
            "phase",
            "phase_label",
            "token_set",
            "evidence_frame_rank",
            "n",
            "mean_importance_normalized_clipped",
            "importance_normalized_clipped_ci_lower",
            "importance_normalized_clipped_ci_upper",
        ],
    )

    print(f"Wrote per-sample importance CSV to: {per_sample_importance_csv}")
    print(f"Wrote per-sample entropy CSV to: {per_sample_entropy_csv}")
    print(f"Wrote aggregate sum-importance CSV to: {aggregate_sum_importance_csv}")
    print(f"Wrote aggregate entropy CSV to: {aggregate_entropy_csv}")
    print(f"Wrote aggregate per-rank CSV to: {aggregate_per_rank_csv}")

    if disable_plots:
        print("Plot generation disabled via --disable_plots.")
        return

    plot_paths = [
        plot_sum_importance_lines(
            aggregate_rows=sum_importance_aggregate_rows,
            output_path=output_dir / "plot_stagewise_sum_importance.png",
            seq_len_display=seq_len_display,
            split=split,
            sample_mode=sample_mode,
            n_samples=len(sample_payloads),
        ),
        plot_normalized_entropy_all_phases(
            aggregate_rows=entropy_aggregate_rows,
            output_path=output_dir / "plot_normalized_entropy_all_phases.png",
            seq_len_display=seq_len_display,
            split=split,
            sample_mode=sample_mode,
            n_samples=len(sample_payloads),
        ),
        plot_per_rank_importance_lines(
            aggregate_rows=per_rank_aggregate_rows,
            output_path=output_dir / "plot_per_rank_mean_importance.png",
            seq_len_display=seq_len_display,
            split=split,
            sample_mode=sample_mode,
            n_samples=len(sample_payloads),
        ),
    ]
    for plot_path in plot_paths:
        if plot_path is not None:
            print(f"Wrote plot to: {plot_path}")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    clean_data_dir = Path(args.clean_data_dir)
    corrupted_data_dir = (
        Path(args.corrupted_data_dir)
        if args.corrupted_data_dir is not None
        else eval_utils.infer_corrupted_data_root(clean_data_dir)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_runtime(args.model_name)
    lm = LanguageModel(_model(), tokenizer=_processor().tokenizer)
    layers = get_layers(lm.model)
    num_layers = len(layers)

    selected_layers_by_phase = {
        "frame_phase": parse_inclusive_layer_selection(
            args.frame_phase_layers,
            num_layers=num_layers,
            arg_name="--frame_phase_layers",
        ),
        "question_phase": parse_inclusive_layer_selection(
            args.question_phase_layers,
            num_layers=num_layers,
            arg_name="--question_phase_layers",
        ),
        "last_token_phase": parse_inclusive_layer_selection(
            args.last_phase_layers,
            num_layers=num_layers,
            arg_name="--last_phase_layers",
        ),
    }
    if not any(selected_layers_by_phase.values()):
        raise ValueError(
            "At least one phase layer selection must be provided via "
            "--frame_phase_layers, --question_phase_layers, or --last_phase_layers."
        )

    selected_layers_spec_by_phase = {
        "frame_phase": args.frame_phase_layers,
        "question_phase": args.question_phase_layers,
        "last_token_phase": args.last_phase_layers,
    }

    sample_dirs = eval_utils.iter_sample_dirs(clean_data_dir)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under: {clean_data_dir}")
    sample_rng = random.Random(args.sample_seed)
    sample_rng.shuffle(sample_dirs)
    sample_seed_label = "<system>" if args.sample_seed is None else str(int(args.sample_seed))

    split = _resolve_split_label(clean_data_dir, args.split)
    seq_len_display = _resolve_seq_len_display(clean_data_dir, args.seq_len)
    print(
        "Running stage-wise evidence survival analysis with "
        f"model={_runtime().model_name} split={split} {seq_len_display} "
        f"sample_mode={str(args.sample_mode)} sample_limit={int(args.sample_limit)} "
        f"sample_seed={sample_seed_label} "
        f"scoring_target=clean_top1 min_score_diff_lambda={float(args.min_score_diff_lambda):.6f} "
        f"clean_data_dir={clean_data_dir} "
        f"corrupted_data_dir={corrupted_data_dir}"
    )
    for phase in PHASE_ORDER:
        if selected_layers_by_phase[phase]:
            print(
                f"  {phase}: layers={selected_layers_by_phase[phase]} "
                f"(spec={selected_layers_spec_by_phase[phase]!r}) token_set={PHASE_TOKEN_SETS[phase]}"
            )

    sample_payloads: List[Dict[str, Any]] = []
    processed_samples = 0
    for sample_index, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= int(args.sample_limit):
            break
        try:
            sample_payload = process_sample(
                sample_dir=sample_dir,
                sample_index=sample_index,
                total_samples=len(sample_dirs),
                split=split,
                lm=lm,
                layers=layers,
                selected_layers_by_phase=selected_layers_by_phase,
                corrupted_data_dir=corrupted_data_dir,
                selected_layers_spec_by_phase=selected_layers_spec_by_phase,
                sample_mode=str(args.sample_mode),
                clean_top1_must_match_gold=bool(args.clean_top1_must_match_gold),
                min_score_diff_lambda=float(args.min_score_diff_lambda),
            )
        except Exception as exc:
            print(
                f"[{sample_index}/{len(sample_dirs)}] sample_id={sample_dir.name} warning: "
                f"unhandled sample failure ({exc})"
            )
            continue
        if sample_payload is None:
            continue
        sample_payloads.append(sample_payload)
        processed_samples += 1

    if sample_payloads:
        seq_len_display = _resolve_seq_len_display(
            clean_data_dir,
            args.seq_len,
            fallback_seq_len=int(sample_payloads[0]["seq_len"]),
        )
    print(
        f"Processed {processed_samples} valid samples "
        f"(target limit={int(args.sample_limit)}, sample_mode={str(args.sample_mode)}, "
        f"sample_seed={sample_seed_label}, "
        f"min_score_diff_lambda={float(args.min_score_diff_lambda):.6f})."
    )

    finalize_outputs(
        sample_payloads=sample_payloads,
        output_dir=output_dir,
        seq_len_display=seq_len_display,
        split=split,
        sample_mode=str(args.sample_mode),
        seed=int(args.seed),
        disable_plots=bool(args.disable_plots),
    )
    print(f"All CSVs and plots were saved under: {output_dir}")
    print(eval_utils.format_runtime(time.perf_counter() - start_time))


if __name__ == "__main__":
    main()

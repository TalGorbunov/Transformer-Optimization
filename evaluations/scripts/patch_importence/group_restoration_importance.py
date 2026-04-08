"""
Group restoration importance experiment.

For each MMRed sample, this script reuses the already-existing corrupted-frame assets
to build the fully corrupted counterpart where all evidence frames are corrupted. It
then patches clean activations back into that corrupted run at each selected layer for
three patch targets:

1. frame tokens selected by --frame_patch_mode (default: evidence_only)
2. the last prompt token / carrier token
3. the question-text span
The main reported metric is normalized rescue:

    normalized_rescue = (patched_score - corrupted_score) / (clean_score - corrupted_score)

The raw score diff is still saved alongside it:

    raw_score_diff = patched_score - corrupted_score
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from models import model as model_utils
from models.model import DEFAULT_MODEL_ID, find_subsequence, get_default_runtime, get_layers, image_token_groups, load_model_runtime

_RUNTIME: Optional[Any] = None
NORMALIZED_RESCUE_EPS = 1e-8


def _runtime() -> Any:
    return get_default_runtime() if _RUNTIME is None else _RUNTIME


def _model() -> Any:
    return _runtime().model


def _processor() -> Any:
    return _runtime().processor


def configure_runtime(model_name: str) -> None:
    global _RUNTIME
    if str(model_name).strip() == str(DEFAULT_MODEL_ID):
        _RUNTIME = get_default_runtime()
    else:
        _RUNTIME = load_model_runtime(model_name)
    model_utils._DEFAULT_RUNTIME = _RUNTIME


def _token_span_from_char_span(text: str, char_span: tuple[int, int]) -> tuple[int, int]:
    start_char, end_char = int(char_span[0]), int(char_span[1])
    if start_char > 0 and text[start_char - 1].isspace():
        start_char -= 1
    start_token = len(_processor().tokenizer(text[:start_char], add_special_tokens=False)["input_ids"])
    end_token = len(_processor().tokenizer(text[:end_char], add_special_tokens=False)["input_ids"])
    return start_token, end_token


def _positions_from_token_span(base_start: int, token_span: tuple[int, int]) -> List[int]:
    return list(range(base_start + int(token_span[0]), base_start + int(token_span[1])))


def _frame_patch_label(frame_patch_mode: str) -> str:
    if frame_patch_mode == "all_frames":
        return "patch all frames"
    if frame_patch_mode == "non_evidence_only":
        return "patch non-evidence frames"
    return "patch evidence frames"


def _compute_normalized_rescue(
    patched_score: float,
    corrupted_score: float,
    denominator: float,
) -> float:
    if abs(float(denominator)) < NORMALIZED_RESCUE_EPS:
        return float("nan")
    return float((float(patched_score) - float(corrupted_score)) / float(denominator))


def _select_frame_patch_positions(
    *,
    frame_groups: Sequence[Sequence[int]],
    evidence_frame_indices: Sequence[int],
    frame_patch_mode: str,
) -> tuple[List[int], List[int]]:
    valid_evidence_indices = sorted({
        int(frame_idx)
        for frame_idx in evidence_frame_indices
        if 0 <= int(frame_idx) < len(frame_groups)
    })
    evidence_index_set = set(valid_evidence_indices)
    non_evidence_indices = [frame_idx for frame_idx in range(len(frame_groups)) if frame_idx not in evidence_index_set]

    if frame_patch_mode == "all_frames":
        selected_frame_indices = list(range(len(frame_groups)))
    elif frame_patch_mode == "non_evidence_only":
        selected_frame_indices = non_evidence_indices
    else:
        selected_frame_indices = valid_evidence_indices

    selected_positions = [
        int(position)
        for frame_idx in selected_frame_indices
        for position in frame_groups[int(frame_idx)]
    ]
    return [int(frame_idx) for frame_idx in selected_frame_indices], selected_positions


def format_patch_importance_table(
    layer_rows: List[tuple[int, float, float, float]]
) -> str:
    if not layer_rows:
        return "<none>"
    header = (
        "layer".ljust(7)
        + "patch_frames".center(18)
        + "patch_last_token".center(20)
        + "patch_question".center(18)
    )
    rows = [header]
    for (
        layer_idx,
        frame_importance,
        last_token_importance,
        question_importance,
    ) in layer_rows:
        rows.append(
            f"{str(layer_idx).ljust(7)}"
            f"{f'{frame_importance:.4f}'.center(18)}"
            f"{f'{last_token_importance:.4f}'.center(20)}"
            f"{f'{question_importance:.4f}'.center(18)}"
        )
    return "\n".join(rows)


def load_sample_components(
    sample_dir: Path,
    sample_index: int,
    total_samples: int,
) -> Optional[tuple[str, List[Any], str, List[Dict[str, Any]], str]]:
    try:
        return load_mmred_sample(sample_dir)
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_dir.name} skipped: load failure ({exc})")
        return None


def build_clean_inputs(
    sample_id: str,
    frames: List[Any],
    question: str,
    sample_index: int,
    total_samples: int,
) -> Optional[tuple[Dict[str, Any], int]]:
    try:
        clean_inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, question))
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to build clean inputs ({exc})")
        return None

    prompt_len = int(clean_inputs["input_ids"].shape[1])
    if prompt_len <= 0:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: prompt has no last token")
        return None
    return clean_inputs, prompt_len


def build_answer_token_ids(
    sample_id: str,
    answer_text: str,
    sample_index: int,
    total_samples: int,
) -> Optional[List[int]]:
    try:
        return tgi.token_ids_of_answer(str(answer_text).strip())
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: invalid answer tokenization ({exc})")
        return None


def score_clean_sample(
    sample_id: str,
    frames: List[Any],
    answer_text: str,
    clean_inputs: Dict[str, Any],
    prompt_len: int,
    clean_top1_must_match_gold: bool,
    lm: LanguageModel,
    sample_index: int,
    total_samples: int,
) -> Optional[Dict[str, Any]]:
    try:
        metrics = tgi.score_valid_numeric_answers(
            lm=lm,
            inputs=clean_inputs,
            prompt_len=prompt_len,
            num_frames=len(frames),
        )
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: clean scoring failed ({exc})")
        return None

    a_star_text = str(answer_text).strip()
    clean_answer_score = float(metrics["scores_by_answer"].get(a_star_text, float("-inf")))
    clean_correct_prob = float(metrics["probs_by_answer"].get(a_star_text, 0.0))
    best_answer_text = str(metrics["best_answer_text"])
    clean_top1_correct = (best_answer_text == a_star_text)

    if clean_top1_must_match_gold and not clean_top1_correct:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: clean top-1 is {best_answer_text!r}, "
            f"not correct answer {a_star_text!r}"
        )
        return None

    return {
        "clean_answer_score": clean_answer_score,
        "clean_correct_prob": clean_correct_prob,
        "clean_top1_correct": clean_top1_correct,
        "best_answer_text": best_answer_text,
    }


def extract_patch_positions(
    sample_id: str,
    clean_inputs: Dict[str, Any],
    question: str,
    num_frames: int,
    sample_index: int,
    total_samples: int,
) -> Optional[Dict[str, Any]]:
    try:
        input_ids = [int(token_id) for token_id in clean_inputs["input_ids"][0].detach().cpu().tolist()]
        frame_groups = image_token_groups(
            clean_inputs["input_ids"][0].detach().cpu(),
            expected_num_frames=num_frames,
            processor=_processor(),
        )
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
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"failed to locate patch token groups ({exc})"
        )
        return None

    if num_frames > 0 and len(frame_groups) != num_frames:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"expected {num_frames} frame token groups but found {len(frame_groups)}"
        )
        return None

    all_frame_positions = [int(position) for group in frame_groups for position in group]
    if not all_frame_positions:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            "no image token groups found in tokenized input"
        )
        return None

    return {
        "frame_groups": [[int(position) for position in group] for group in frame_groups],
        "all_frame_positions": all_frame_positions,
        "frame_group_sizes": [len(group) for group in frame_groups],
        "question_positions": [int(position) for position in question_positions],
    }


def build_corrupted_inputs(
    sample_id: str,
    frames: List[Any],
    question: str,
    evidence_frame_indices: List[int],
    corrupted_data_root: Path,
    prompt_len: int,
    sample_index: int,
    total_samples: int,
) -> Optional[Dict[str, Any]]:
    corrupted_frames, corruption_issues = eval_utils.build_composite_corrupted_frames(
        sample_id=sample_id,
        clean_frames=frames,
        evidence_frame_indices=evidence_frame_indices,
        corrupted_data_root=corrupted_data_root,
    )
    if corrupted_frames is None:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            "missing corrupted counterpart for the fully corrupted sample "
            f"(issues={corruption_issues})"
        )
        return None

    try:
        corrupted_inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(corrupted_frames, question))
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to build corrupted inputs ({exc})")
        return None
    if int(corrupted_inputs["input_ids"].shape[1]) != prompt_len:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: seq_len mismatch "
            f"(clean={prompt_len}, corrupted={int(corrupted_inputs['input_ids'].shape[1])})"
        )
        return None
    return corrupted_inputs


def score_corrupted_sample(
    sample_id: str,
    clean_inputs: Dict[str, Any],
    corrupted_inputs: Dict[str, Any],
    answer_token_ids: List[int],
    prompt_len: int,
    lm: LanguageModel,
    sample_index: int,
    total_samples: int,
) -> Optional[Dict[str, Any]]:
    clean_answer_inputs = tgi.append_answer_tokens_for_scoring(clean_inputs, answer_token_ids)
    corrupted_answer_inputs = tgi.append_answer_tokens_for_scoring(corrupted_inputs, answer_token_ids)
    try:
        corrupted_answer_score = tgi.run_clean_sequence_logprob(
            lm=lm,
            scoring_inputs=corrupted_answer_inputs,
            prompt_len=prompt_len,
            answer_token_ids=answer_token_ids,
        )
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to score corrupted input ({exc})")
        return None

    return {
        "clean_answer_inputs": clean_answer_inputs,
        "corrupted_answer_inputs": corrupted_answer_inputs,
        "corrupted_answer_score": float(corrupted_answer_score),
    }


def _run_patch_target(
    *,
    lm: LanguageModel,
    layers: Any,
    corrupted_answer_inputs: Dict[str, Any],
    clean_answer_inputs: Dict[str, Any],
    corrupted_answer_score: float,
    prompt_len: int,
    answer_token_ids: List[int],
    layer_idx: int,
    patch_target_name: str,
    token_positions: Sequence[int],
) -> tuple[float, float]:
    if not list(token_positions):
        return float(corrupted_answer_score), 0.0
    try:
        patched_score = tgi.run_layer_corrupted_sequence_logprob(
            lm=lm,
            layers=layers,
            target_scoring_inputs=corrupted_answer_inputs,
            source_scoring_inputs=clean_answer_inputs,
            layer_idx=layer_idx,
            target_token_positions=token_positions,
            source_token_positions=token_positions,
            prompt_len=prompt_len,
            answer_token_ids=answer_token_ids,
        )
    except Exception as exc:
        print(f"  layer={layer_idx} patch_target={patch_target_name} failed ({exc}); using corrupted score")
        patched_score = corrupted_answer_score

    patched_score = float(patched_score)
    return patched_score, float(patched_score - corrupted_answer_score)


def compute_patch_target_layer_metrics(
    lm: LanguageModel,
    layers: Any,
    selected_layers: List[int],
    corrupted_answer_inputs: Dict[str, Any],
    clean_answer_inputs: Dict[str, Any],
    corrupted_answer_score: float,
    clean_answer_score: float,
    prompt_len: int,
    answer_token_ids: List[int],
    frame_token_positions: Sequence[int],
    question_positions: Sequence[int],
) -> tuple[List[Dict[str, Any]], List[tuple[int, float, float, float]]]:
    per_layer_metrics: List[Dict[str, Any]] = []
    importance_rows: List[tuple[int, float, float, float]] = []

    denominator = float(clean_answer_score - corrupted_answer_score)
    frame_token_positions = [int(position) for position in frame_token_positions]
    question_positions = [int(position) for position in question_positions]
    for layer_idx in selected_layers:
        frames_patched_score, frames_raw_score_diff = _run_patch_target(
            lm=lm,
            layers=layers,
            corrupted_answer_inputs=corrupted_answer_inputs,
            clean_answer_inputs=clean_answer_inputs,
            corrupted_answer_score=corrupted_answer_score,
            prompt_len=prompt_len,
            answer_token_ids=answer_token_ids,
            layer_idx=layer_idx,
            patch_target_name="frames",
            token_positions=frame_token_positions,
        )
        last_token_patched_score, last_token_raw_score_diff = _run_patch_target(
            lm=lm,
            layers=layers,
            corrupted_answer_inputs=corrupted_answer_inputs,
            clean_answer_inputs=clean_answer_inputs,
            corrupted_answer_score=corrupted_answer_score,
            prompt_len=prompt_len,
            answer_token_ids=answer_token_ids,
            layer_idx=layer_idx,
            patch_target_name="last_token",
            token_positions=[-1],
        )
        question_patched_score, question_raw_score_diff = _run_patch_target(
            lm=lm,
            layers=layers,
            corrupted_answer_inputs=corrupted_answer_inputs,
            clean_answer_inputs=clean_answer_inputs,
            corrupted_answer_score=corrupted_answer_score,
            prompt_len=prompt_len,
            answer_token_ids=answer_token_ids,
            layer_idx=layer_idx,
            patch_target_name="question",
            token_positions=question_positions,
        )

        frame_normalized_rescue = _compute_normalized_rescue(
            frames_patched_score,
            corrupted_answer_score,
            denominator,
        )
        last_token_normalized_rescue = _compute_normalized_rescue(
            last_token_patched_score,
            corrupted_answer_score,
            denominator,
        )
        question_normalized_rescue = _compute_normalized_rescue(
            question_patched_score,
            corrupted_answer_score,
            denominator,
        )

        importance_rows.append(
            (
                int(layer_idx),
                frame_normalized_rescue,
                last_token_normalized_rescue,
                question_normalized_rescue,
            )
        )
        per_layer_metrics.append({
            "layer": int(layer_idx),
            "corrupted_score": float(corrupted_answer_score),
            "clean_score": float(clean_answer_score),
            "denominator": denominator,
            "frames_patched_score": float(frames_patched_score),
            "frames_raw_score_diff": float(frames_raw_score_diff),
            "frames_normalized_rescue": float(frame_normalized_rescue),
            "frames_importance": float(frame_normalized_rescue),
            "last_token_patched_score": float(last_token_patched_score),
            "last_token_raw_score_diff": float(last_token_raw_score_diff),
            "last_token_normalized_rescue": float(last_token_normalized_rescue),
            "last_token_importance": float(last_token_normalized_rescue),
            "question_patched_score": float(question_patched_score),
            "question_raw_score_diff": float(question_raw_score_diff),
            "question_normalized_rescue": float(question_normalized_rescue),
            "question_importance": float(question_normalized_rescue),
        })

    return per_layer_metrics, importance_rows


def parse_layer_selection(raw: Optional[str], num_layers: int) -> List[int]:
    if raw is None:
        return tgi.parse_layer_selection(raw, num_layers=num_layers)

    normalized_parts: List[str] = []
    for part in [field.strip() for field in str(raw).split(",") if field.strip()]:
        if "-" in part and ":" not in part:
            fields = part.split("-")
            if len(fields) != 2:
                raise ValueError(
                    f"Invalid range in --layers: {part!r}. Expected start-end, start:end, or start:end:step."
                )
            try:
                start = int(fields[0])
                end = int(fields[1])
            except ValueError as exc:
                raise ValueError(f"Invalid integer in --layers: {part!r}") from exc
            if end < start:
                raise ValueError(f"--layers range end must be >= start: {part!r}")
            normalized_parts.append(f"{start}:{end + 1}")
            continue
        normalized_parts.append(part)
    normalized_raw = ",".join(normalized_parts)
    return tgi.parse_layer_selection(normalized_raw, num_layers=num_layers)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Measure per-layer restoration on a fully corrupted MMRed input by restoring selected "
            "prompt-token groups (frames, last token, question text), saving both raw score diff "
            "and normalized rescue."
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
    ap.add_argument("--batch_size", type=int, default=1, help="Accepted for interface parity; unused here.")
    ap.add_argument(
        "--clean_top1_must_match_gold",
        type=lambda raw: str(raw).strip().lower() in {"1", "true", "yes", "y", "on"},
        default=True,
        help="If true, keep only samples where the clean run's top-1 answer matches the gold answer.",
    )
    ap.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Optional layer selection. Examples: --layers 0,1,2,10-20,31 or --layers 32:42 or --layers 0:64:2."
        ),
    )
    ap.add_argument(
        "--model_name",
        "--model",
        dest="model_name",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Model name to load. Defaults to the repo's current default runtime model.",
    )
    ap.add_argument(
        "--frame_patch_mode",
        type=str,
        default="evidence_only",
        choices=["evidence_only", "all_frames", "non_evidence_only"],
        help=(
            "Which frame-token groups to restore for the frames patch target. "
            "Default: evidence_only."
        ),
    )
    ap.add_argument("--disable_plots", action="store_true")
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer")
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer")
    return args


def process_sample(
    sample_dir: Path,
    sample_index: int,
    total_samples: int,
    lm: LanguageModel,
    layers: Any,
    selected_layers: List[int],
    corrupted_data_root: Path,
    clean_top1_must_match_gold: bool,
    frame_patch_mode: str,
) -> Optional[Dict[str, Any]]:
    sample_components = load_sample_components(sample_dir, sample_index, total_samples)
    if sample_components is None:
        return None
    sample_id, frames, question, states, answer = sample_components

    evidence_frame_indices = eval_utils.collect_evidence_frame_indices(question, states)
    if len(evidence_frame_indices) < 1:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: no evidence frames")
        return None

    clean_input_result = build_clean_inputs(
        sample_id=sample_id,
        frames=frames,
        question=question,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if clean_input_result is None:
        return None
    clean_inputs, prompt_len = clean_input_result
    carrier_index = int(prompt_len - 1)
    if carrier_index < 0:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: invalid carrier index {carrier_index}")
        return None

    a_star_text = str(answer).strip()
    a_star_ids = build_answer_token_ids(sample_id, a_star_text, sample_index, total_samples)
    if a_star_ids is None:
        return None

    clean_answer_metrics = score_clean_sample(
        sample_id=sample_id,
        frames=frames,
        answer_text=a_star_text,
        clean_inputs=clean_inputs,
        prompt_len=prompt_len,
        clean_top1_must_match_gold=clean_top1_must_match_gold,
        lm=lm,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if clean_answer_metrics is None:
        return None
    clean_answer_score = float(clean_answer_metrics["clean_answer_score"])
    clean_correct_prob = float(clean_answer_metrics["clean_correct_prob"])
    clean_top1_correct = bool(clean_answer_metrics["clean_top1_correct"])
    best_answer_text = str(clean_answer_metrics["best_answer_text"])

    patch_metadata = extract_patch_positions(
        sample_id=sample_id,
        clean_inputs=clean_inputs,
        question=question,
        num_frames=len(frames),
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if patch_metadata is None:
        return None
    frame_groups = patch_metadata["frame_groups"]
    frame_group_sizes = patch_metadata["frame_group_sizes"]
    all_frame_positions = patch_metadata["all_frame_positions"]
    question_positions = patch_metadata["question_positions"]
    selected_frame_indices, frame_patch_positions = _select_frame_patch_positions(
        frame_groups=frame_groups,
        evidence_frame_indices=evidence_frame_indices,
        frame_patch_mode=frame_patch_mode,
    )

    corrupted_inputs = build_corrupted_inputs(
        sample_id=sample_id,
        frames=frames,
        question=question,
        evidence_frame_indices=evidence_frame_indices,
        corrupted_data_root=corrupted_data_root,
        prompt_len=prompt_len,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if corrupted_inputs is None:
        return None

    corrupted_score_result = score_corrupted_sample(
        sample_id=sample_id,
        clean_inputs=clean_inputs,
        corrupted_inputs=corrupted_inputs,
        answer_token_ids=a_star_ids,
        prompt_len=prompt_len,
        lm=lm,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if corrupted_score_result is None:
        return None
    clean_answer_inputs = corrupted_score_result["clean_answer_inputs"]
    corrupted_answer_inputs = corrupted_score_result["corrupted_answer_inputs"]
    corrupted_answer_score = float(corrupted_score_result["corrupted_answer_score"])
    denominator = float(clean_answer_score - corrupted_answer_score)

    if abs(denominator) < NORMALIZED_RESCUE_EPS:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
            f"clean_score - corrupted_score is too small for normalized rescue "
            f"(clean_score={clean_answer_score:.8f}, corrupted_score={corrupted_answer_score:.8f}, "
            f"denominator={denominator:.8e}); normalized_rescue will be NaN"
        )
    if not frame_patch_positions:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
            f"frame_patch_mode={frame_patch_mode} selects zero frame tokens; "
            "frame patching will leave the corrupted score unchanged"
        )

    print(
        f"[{sample_index}/{total_samples}] sample_id={sample_id} clean_answer_score={clean_answer_score:.4f} "
        f"corrupted_answer_score={corrupted_answer_score:.4f} clean_correct_prob={clean_correct_prob:.4f} "
        f"evidence_frames={evidence_frame_indices} num_frame_tokens={len(all_frame_positions)} "
        f"frame_patch_mode={frame_patch_mode} frame_patch_frames={selected_frame_indices} "
        f"frame_patch_token_count={len(frame_patch_positions)} "
        f"question_group_size={len(question_positions)} "
        f"denominator={denominator:.4f} "
        f"carrier_index={carrier_index}"
    )

    per_layer_metrics, importance_rows = compute_patch_target_layer_metrics(
        lm=lm,
        layers=layers,
        selected_layers=selected_layers,
        corrupted_answer_inputs=corrupted_answer_inputs,
        clean_answer_inputs=clean_answer_inputs,
        corrupted_answer_score=corrupted_answer_score,
        clean_answer_score=clean_answer_score,
        prompt_len=prompt_len,
        answer_token_ids=a_star_ids,
        frame_token_positions=frame_patch_positions,
        question_positions=question_positions,
    )

    if importance_rows:
        print(f"  Normalized rescue table (frame_patch_mode={frame_patch_mode}):")
        print(format_patch_importance_table(importance_rows))

    return {
        "sample_id": sample_id,
        "seq_len": int(len(frames)),
        "question": question,
        "answer": answer,
        "a_star_text": a_star_text,
        "a_star_ids": [int(token_id) for token_id in a_star_ids],
        "clean_answer_score": clean_answer_score,
        "corrupted_answer_score": corrupted_answer_score,
        "clean_correct_prob": clean_correct_prob,
        "clean_top1_correct": clean_top1_correct,
        "best_answer_text": best_answer_text,
        "denominator": denominator,
        "evidence_frames": [int(frame_idx) for frame_idx in evidence_frame_indices],
        "frame_patch_mode": str(frame_patch_mode),
        "frame_patch_label": _frame_patch_label(frame_patch_mode),
        "frame_patch_frames": [int(frame_idx) for frame_idx in selected_frame_indices],
        "frame_patch_positions": [int(position) for position in frame_patch_positions],
        "frame_groups": [[int(position) for position in group] for group in frame_groups],
        "frame_group_sizes": [int(size) for size in frame_group_sizes],
        "all_frame_positions": [int(position) for position in all_frame_positions],
        "question_positions": [int(position) for position in question_positions],
        "carrier_index": carrier_index,
        "patched_token_position": -1,
        "selected_layers": [int(layer_idx) for layer_idx in selected_layers],
        "layer_metrics": {"layers": per_layer_metrics},
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _mean(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _bootstrap_center_and_ci(
    values: List[float],
    n_bootstrap: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    n = len(values)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan
    center = _mean(values)
    if n <= 1:
        return center, center, center

    boot_values: List[float] = []
    for _ in range(int(n_bootstrap)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_values.append(_mean(sample))

    boot_values.sort()
    lo_idx = int(0.025 * (n_bootstrap - 1))
    hi_idx = int(0.975 * (n_bootstrap - 1))
    return center, boot_values[lo_idx], boot_values[hi_idx]


def build_per_sample_csv_rows(sample_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in sample_metrics:
        common = {
            "sample_id": str(sample["sample_id"]),
            "seq_len": int(sample["seq_len"]),
            "question": str(sample["question"]),
            "answer": str(sample["answer"]),
            "a_star_text": str(sample["a_star_text"]),
            "a_star_ids_json": json.dumps(sample["a_star_ids"]),
            "clean_score": float(sample["clean_answer_score"]),
            "clean_answer_score": float(sample["clean_answer_score"]),
            "corrupted_score": float(sample["corrupted_answer_score"]),
            "corrupted_answer_score": float(sample["corrupted_answer_score"]),
            "denominator": float(sample["denominator"]),
            "clean_correct_prob": float(sample["clean_correct_prob"]),
            "clean_top1_correct": bool(sample["clean_top1_correct"]),
            "best_answer_text": str(sample["best_answer_text"]),
            "evidence_frames_json": json.dumps(sample["evidence_frames"]),
            "frame_patch_mode": str(sample["frame_patch_mode"]),
            "frame_patch_label": str(sample["frame_patch_label"]),
            "frame_patch_frames_json": json.dumps(sample["frame_patch_frames"]),
            "frame_patch_positions_json": json.dumps(sample["frame_patch_positions"]),
            "frame_patch_token_count": int(len(sample["frame_patch_positions"])),
            "frame_group_sizes_json": json.dumps(sample["frame_group_sizes"]),
            "frame_groups_json": json.dumps(sample["frame_groups"]),
            "all_frame_positions_json": json.dumps(sample["all_frame_positions"]),
            "question_positions_json": json.dumps(sample["question_positions"]),
            "num_frame_tokens": int(len(sample["all_frame_positions"])),
            "carrier_index": int(sample["carrier_index"]),
            "patched_token_position": int(sample["patched_token_position"]),
            "selected_layers_spec": sample.get("selected_layers_spec"),
            "model_name": sample.get("model_name"),
            "corrupted_root": sample.get("control_debug", {}).get("corrupted_root"),
        }
        for layer_metric in sample.get("layer_metrics", {}).get("layers", []):
            layer_idx = int(layer_metric["layer"])
            rows.append({
                **common,
                "layer": layer_idx,
                "patch_target": "frames",
                "raw_score_diff": float(layer_metric["frames_raw_score_diff"]),
                "normalized_rescue": float(layer_metric["frames_normalized_rescue"]),
                "patched_score": float(layer_metric["frames_patched_score"]),
                "importance": float(layer_metric["frames_normalized_rescue"]),
                "score_diff": float(layer_metric["frames_raw_score_diff"]),
            })
            rows.append({
                **common,
                "layer": layer_idx,
                "patch_target": "last_token",
                "raw_score_diff": float(layer_metric["last_token_raw_score_diff"]),
                "normalized_rescue": float(layer_metric["last_token_normalized_rescue"]),
                "patched_score": float(layer_metric["last_token_patched_score"]),
                "importance": float(layer_metric["last_token_normalized_rescue"]),
                "score_diff": float(layer_metric["last_token_raw_score_diff"]),
            })
            rows.append({
                **common,
                "layer": layer_idx,
                "patch_target": "question",
                "raw_score_diff": float(layer_metric["question_raw_score_diff"]),
                "normalized_rescue": float(layer_metric["question_normalized_rescue"]),
                "patched_score": float(layer_metric["question_patched_score"]),
                "importance": float(layer_metric["question_normalized_rescue"]),
                "score_diff": float(layer_metric["question_raw_score_diff"]),
            })
    return rows


def _summary_triplet(values: List[float], rng: random.Random, n_bootstrap: int) -> tuple[float, float, float]:
    return _bootstrap_center_and_ci(values, n_bootstrap=n_bootstrap, rng=rng)


def build_aggregate_csv_rows(
    sample_metrics: List[Dict[str, Any]],
    selected_layers: List[int],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    patch_targets = {
        "frames": ("frames_raw_score_diff", "frames_normalized_rescue", "frames_patched_score"),
        "last_token": ("last_token_raw_score_diff", "last_token_normalized_rescue", "last_token_patched_score"),
        "question": ("question_raw_score_diff", "question_normalized_rescue", "question_patched_score"),
    }

    for patch_target, (raw_score_diff_key, normalized_rescue_key, patched_score_key) in patch_targets.items():
        for layer_idx in selected_layers:
            raw_score_diff_values: List[float] = []
            normalized_rescue_values: List[float] = []
            patched_score_values: List[float] = []
            corrupted_score_values: List[float] = []
            clean_score_values: List[float] = []

            for sample in sample_metrics:
                layer_metric = next(
                    (
                        layer
                        for layer in sample.get("layer_metrics", {}).get("layers", [])
                        if int(layer["layer"]) == int(layer_idx)
                    ),
                    None,
                )
                if layer_metric is None:
                    continue
                raw_score_diff_values.append(float(layer_metric[raw_score_diff_key]))
                normalized_rescue_values.append(float(layer_metric[normalized_rescue_key]))
                patched_score_values.append(float(layer_metric[patched_score_key]))
                corrupted_score_values.append(float(sample["corrupted_answer_score"]))
                clean_score_values.append(float(sample["clean_answer_score"]))

            raw_center, raw_lo, raw_hi = _summary_triplet(raw_score_diff_values, rng, n_bootstrap)
            norm_center, norm_lo, norm_hi = _summary_triplet(normalized_rescue_values, rng, n_bootstrap)
            patched_center, patched_lo, patched_hi = _summary_triplet(patched_score_values, rng, n_bootstrap)
            corrupted_center, corrupted_lo, corrupted_hi = _summary_triplet(corrupted_score_values, rng, n_bootstrap)
            clean_center, clean_lo, clean_hi = _summary_triplet(clean_score_values, rng, n_bootstrap)

            rows.append({
                "layer": int(layer_idx),
                "patch_target": patch_target,
                "frame_patch_mode": str(sample_metrics[0]["frame_patch_mode"]) if sample_metrics else "evidence_only",
                "n_samples": int(len(raw_score_diff_values)),
                "n_normalized_rescue_samples": int(sum(math.isfinite(float(value)) for value in normalized_rescue_values)),
                "mean_importance": norm_center,
                "importance_ci_lo": norm_lo,
                "importance_ci_hi": norm_hi,
                "mean_raw_score_diff": raw_center,
                "raw_score_diff_ci_lo": raw_lo,
                "raw_score_diff_ci_hi": raw_hi,
                "mean_normalized_rescue": norm_center,
                "normalized_rescue_ci_lo": norm_lo,
                "normalized_rescue_ci_hi": norm_hi,
                "mean_patched_score": patched_center,
                "patched_score_ci_lo": patched_lo,
                "patched_score_ci_hi": patched_hi,
                "mean_corrupted_score": corrupted_center,
                "corrupted_score_ci_lo": corrupted_lo,
                "corrupted_score_ci_hi": corrupted_hi,
                "mean_clean_answer_score": clean_center,
                "clean_answer_score_ci_lo": clean_lo,
                "clean_answer_score_ci_hi": clean_hi,
            })
    return rows


def plot_patch_importance_lines(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    selected_layers: List[int],
    frame_patch_mode: str,
    seq_len_label: Optional[str] = None,
    filename_stem: str = "frames_last_token_question_normalized_rescue_lines",
    title_override: Optional[str] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Path]:
    if not sample_metrics or not selected_layers:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot: matplotlib is not available ({exc})")
        return None

    patch_targets = ("frames", "last_token", "question")
    per_target_values: Dict[str, Dict[int, List[float]]] = {
        patch_target: {int(layer_idx): [] for layer_idx in selected_layers}
        for patch_target in patch_targets
    }
    for sample in sample_metrics:
        for layer_metric in sample.get("layer_metrics", {}).get("layers", []):
            layer_idx = int(layer_metric["layer"])
            if layer_idx not in per_target_values["frames"]:
                continue
            per_target_values["frames"][layer_idx].append(float(layer_metric["frames_normalized_rescue"]))
            per_target_values["last_token"][layer_idx].append(float(layer_metric["last_token_normalized_rescue"]))
            per_target_values["question"][layer_idx].append(float(layer_metric["question_normalized_rescue"]))

    rng = random.Random(seed)
    summary: Dict[str, Dict[str, List[float]]] = {
        patch_target: {"center": [], "lo": [], "hi": []}
        for patch_target in patch_targets
    }
    for patch_target in patch_targets:
        for layer_idx in selected_layers:
            center, lo_value, hi_value = _summary_triplet(per_target_values[patch_target][int(layer_idx)], rng, n_bootstrap)
            summary[patch_target]["center"].append(center)
            summary[patch_target]["lo"].append(lo_value)
            summary[patch_target]["hi"].append(hi_value)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    frame_line, = ax.plot(
        selected_layers,
        summary["frames"]["center"],
        linewidth=2.2,
        color="#1f77b4",
        label=_frame_patch_label(frame_patch_mode),
    )
    ax.fill_between(
        selected_layers,
        summary["frames"]["lo"],
        summary["frames"]["hi"],
        color=frame_line.get_color(),
        alpha=0.18,
    )
    last_line, = ax.plot(
        selected_layers,
        summary["last_token"]["center"],
        linewidth=2.2,
        color="#d62728",
        label="patch last token",
    )
    ax.fill_between(
        selected_layers,
        summary["last_token"]["lo"],
        summary["last_token"]["hi"],
        color=last_line.get_color(),
        alpha=0.18,
    )
    question_line, = ax.plot(
        selected_layers,
        summary["question"]["center"],
        linewidth=2.2,
        color="#ff7f0e",
        label="patch question",
    )
    ax.fill_between(
        selected_layers,
        summary["question"]["lo"],
        summary["question"]["hi"],
        color=question_line.get_color(),
        alpha=0.18,
    )
    ax.axhline(0.0, color="#666666", linewidth=1.0, alpha=0.7)
    title = title_override or "Normalized rescue from restoring prompt-token groups on a FULLY corrupted input"
    title = f"{title} (frame_patch_mode={frame_patch_mode})"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized rescue")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    tick_step = max(1, max(len(selected_layers), 1) // 32 + (1 if len(selected_layers) % 32 else 0))
    xticks = selected_layers[::tick_step]
    if selected_layers[-1] not in xticks:
        xticks.append(selected_layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{seq_len_label}" if seq_len_label else ""
    plot_path = output_dir / f"{filename_stem}_{frame_patch_mode}{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def finalize_outputs(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    selected_layers: List[int],
    seq_len_label: Optional[str],
    clean_top1_must_match_gold: bool,
    args: argparse.Namespace,
) -> None:
    print(
        f"Processed {len(sample_metrics)} samples "
        f"(target limit={int(args.limit)}, clean_top1_must_match_gold={clean_top1_must_match_gold}, "
        f"frame_patch_mode={args.frame_patch_mode})."
    )

    per_sample_rows = build_per_sample_csv_rows(sample_metrics)
    per_sample_csv_path = output_dir / "per_sample_patch_importance.csv"
    write_csv(
        per_sample_csv_path,
        per_sample_rows,
        fieldnames=[
            "sample_id",
            "seq_len",
            "layer",
            "patch_target",
            "raw_score_diff",
            "normalized_rescue",
            "clean_score",
            "corrupted_score",
            "patched_score",
            "denominator",
            "frame_patch_mode",
            "importance",
            "score_diff",
            "corrupted_answer_score",
            "clean_answer_score",
            "clean_correct_prob",
            "clean_top1_correct",
            "best_answer_text",
            "a_star_text",
            "a_star_ids_json",
            "answer",
            "question",
            "evidence_frames_json",
            "frame_patch_label",
            "frame_patch_frames_json",
            "frame_patch_positions_json",
            "frame_patch_token_count",
            "frame_group_sizes_json",
            "frame_groups_json",
            "all_frame_positions_json",
            "question_positions_json",
            "num_frame_tokens",
            "carrier_index",
            "patched_token_position",
            "selected_layers_spec",
            "model_name",
            "corrupted_root",
        ],
    )

    aggregate_rows = build_aggregate_csv_rows(sample_metrics, selected_layers=selected_layers)
    aggregate_csv_path = output_dir / "aggregate_patch_importance.csv"
    write_csv(
        aggregate_csv_path,
        aggregate_rows,
        fieldnames=[
            "layer",
            "patch_target",
            "frame_patch_mode",
            "n_samples",
            "n_normalized_rescue_samples",
            "mean_importance",
            "importance_ci_lo",
            "importance_ci_hi",
            "mean_raw_score_diff",
            "raw_score_diff_ci_lo",
            "raw_score_diff_ci_hi",
            "mean_normalized_rescue",
            "normalized_rescue_ci_lo",
            "normalized_rescue_ci_hi",
            "mean_patched_score",
            "patched_score_ci_lo",
            "patched_score_ci_hi",
            "mean_corrupted_score",
            "corrupted_score_ci_lo",
            "corrupted_score_ci_hi",
            "mean_clean_answer_score",
            "clean_answer_score_ci_lo",
            "clean_answer_score_ci_hi",
        ],
    )

    raw_json_path = output_dir / "sample_metrics.json"
    raw_json_path.write_text(json.dumps(sample_metrics, indent=2) + "\n", encoding="utf-8")

    plot_path = None
    if not args.disable_plots:
        plot_path = plot_patch_importance_lines(
            sample_metrics,
            output_dir,
            selected_layers=selected_layers,
            frame_patch_mode=args.frame_patch_mode,
            seq_len_label=seq_len_label,
        )

    print(f"Wrote per-sample CSV to: {per_sample_csv_path}")
    print(f"Wrote aggregate CSV to: {aggregate_csv_path}")
    print(f"Wrote raw sample metrics JSON to: {raw_json_path}")
    if plot_path is not None:
        print(f"Wrote plot to: {plot_path}")
    print(
        "Output summary: "
        f"per-sample CSV={per_sample_csv_path}, aggregate CSV={aggregate_csv_path}, "
        f"raw JSON={raw_json_path}"
        + (f", plot={plot_path}" if plot_path is not None else "")
    )


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    clean_top1_must_match_gold = bool(args.clean_top1_must_match_gold)

    configure_runtime(args.model_name)

    data_root = Path(args.data_root)
    corrupted_data_root = (
        Path(args.corrupted_root) if args.corrupted_root is not None else eval_utils.infer_corrupted_data_root(data_root)
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_len_label = eval_utils.resolve_seq_len_label(data_root)

    lm = LanguageModel(_model(), tokenizer=_processor().tokenizer)
    layers = get_layers(lm.model)
    num_layers = len(layers)
    selected_layers = parse_layer_selection(args.layers, num_layers=num_layers)

    sample_dirs = iter_sample_dirs(data_root)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under: {data_root}")

    processed_samples = 0
    sample_metrics: List[Dict[str, Any]] = []
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= int(args.limit):
            break
        try:
            sample_metrics_row = process_sample(
                sample_dir=sample_dir,
                sample_index=idx,
                total_samples=len(sample_dirs),
                lm=lm,
                layers=layers,
                selected_layers=selected_layers,
                corrupted_data_root=corrupted_data_root,
                clean_top1_must_match_gold=clean_top1_must_match_gold,
                frame_patch_mode=str(args.frame_patch_mode),
            )
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_dir.name} skipped: unexpected failure ({exc})")
            continue
        if sample_metrics_row is None:
            continue
        sample_metrics_row["selected_layers_spec"] = args.layers
        sample_metrics_row["model_name"] = _runtime().model_name
        sample_metrics_row["control_debug"] = {
            "control_type": "all_evidence_frames_corrupted_from_existing_corrupted_samples",
            "corrupted_root": str(corrupted_data_root),
            "frame_patch_mode": str(args.frame_patch_mode),
        }
        sample_metrics.append(sample_metrics_row)
        processed_samples += 1

    finalize_outputs(
        sample_metrics=sample_metrics,
        output_dir=output_dir,
        selected_layers=selected_layers,
        seq_len_label=seq_len_label,
        clean_top1_must_match_gold=clean_top1_must_match_gold,
        args=args,
    )

    elapsed = time.perf_counter() - start_time
    print(eval_utils.format_runtime(elapsed))


if __name__ == "__main__":
    main()

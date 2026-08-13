"""
Unified standalone bottleneck analysis.

This simplified version keeps only two experiment families:
1. restoration on the fully corrupted run
2. clean-ablation damage on the clean run

And it only measures five prompt-token groups:
- evidence frames
- instruction
- question
- character + room
- last token

The implementation intentionally stays close to the existing restoration
experiment so model loading, prompt construction, answer scoring, corrupted
sample assembly, CSV conventions, and logging remain comparable.
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

from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import DEFAULT_MODEL_ID, find_subsequence, get_layers

NORMALIZED_EPS = 1e-8
INSTRUCTION_PROMPT_SPAN = (
    "Respond with a single integer from 0 to {num_frames} "
    "(0 is allowed). Output only the integer.\n"
)
ANALYSIS_GROUPS = (
    "evidence_frames",
    "instruction",
    "question",
    "character_room",
    "last_token",
)
GROUP_LABELS = {
    "evidence_frames": "evidence frames",
    "instruction": "instruction",
    "question": "question",
    "character_room": "character + room",
    "last_token": "last token",
}
GROUP_COLORS = {
    "evidence_frames": "#1f77b4",
    "instruction": "#9467bd",
    "question": "#ff7f0e",
    "character_room": "#2ca02c",
    "last_token": "#d62728",
}


def _runtime() -> Any:
    return gri._runtime()


def _model() -> Any:
    return _runtime().model


def _processor() -> Any:
    return _runtime().processor


def configure_runtime(model_name: str) -> None:
    gri.configure_runtime(model_name)


def _safe_normalize(raw_value: float, denominator: float) -> float:
    if abs(float(denominator)) < NORMALIZED_EPS:
        return float("nan")
    return float(float(raw_value) / float(denominator))


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


def _mean(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else float("nan")


def _question_subspan_positions(
    *,
    question_text: str,
    question_positions: Sequence[int],
    char_span: Tuple[int, int],
) -> List[int]:
    start_char, end_char = int(char_span[0]), int(char_span[1])
    if start_char > 0 and question_text[start_char - 1].isspace():
        start_char -= 1
    start_token = len(_processor().tokenizer(question_text[:start_char], add_special_tokens=False)["input_ids"])
    end_token = len(_processor().tokenizer(question_text[:end_char], add_special_tokens=False)["input_ids"])
    return [int(position) for position in question_positions[start_token:end_token]]


def _prompt_subspan_positions(
    *,
    prompt_text: str,
    prompt_text_start: int,
    char_span: Tuple[int, int],
) -> List[int]:
    start_char, end_char = int(char_span[0]), int(char_span[1])
    if start_char > 0 and prompt_text[start_char - 1].isspace():
        start_char -= 1
    start_token = len(_processor().tokenizer(prompt_text[:start_char], add_special_tokens=False)["input_ids"])
    end_token = len(_processor().tokenizer(prompt_text[:end_char], add_special_tokens=False)["input_ids"])
    return list(range(int(prompt_text_start) + int(start_token), int(prompt_text_start) + int(end_token)))


def _instruction_positions_from_prompt(
    *,
    clean_inputs: Dict[str, Any],
    question: str,
    num_frames: int,
) -> List[int]:
    input_ids = [int(token_id) for token_id in clean_inputs["input_ids"][0].detach().cpu().tolist()]
    prompt_text = tgi.build_prompt(question, num_frames=num_frames)
    prompt_text_ids = _processor().tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_text_start = find_subsequence(input_ids, [int(token_id) for token_id in prompt_text_ids])
    if prompt_text_start is None:
        raise RuntimeError("failed to locate prompt text in multimodal prompt")

    instruction_text = INSTRUCTION_PROMPT_SPAN.format(num_frames=int(num_frames))
    instruction_start = prompt_text.find(instruction_text)
    if instruction_start < 0:
        raise RuntimeError(f"failed to locate instruction span in prompt: {instruction_text!r}")
    instruction_positions = _prompt_subspan_positions(
        prompt_text=prompt_text,
        prompt_text_start=int(prompt_text_start),
        char_span=(instruction_start, instruction_start + len(instruction_text)),
    )
    if not instruction_positions:
        raise RuntimeError(f"instruction span tokenized to an empty position set: {instruction_text!r}")
    return [int(position) for position in instruction_positions]


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


def _summary_triplet(values: Sequence[float], *, n_bootstrap: int, rng: random.Random) -> Tuple[float, float, float]:
    return _bootstrap_center_and_ci(values, n_bootstrap=n_bootstrap, rng=rng)


def extract_unified_group_metadata(
    *,
    sample_id: str,
    clean_inputs: Dict[str, Any],
    question: str,
    num_frames: int,
    evidence_frame_indices: Sequence[int],
    sample_index: int,
    total_samples: int,
) -> Optional[Dict[str, Any]]:
    patch_metadata = gri.extract_patch_positions(
        sample_id=sample_id,
        clean_inputs=clean_inputs,
        question=question,
        num_frames=num_frames,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if patch_metadata is None:
        return None

    try:
        instruction_positions = _instruction_positions_from_prompt(
            clean_inputs=clean_inputs,
            question=question,
            num_frames=num_frames,
        )
    except Exception as exc:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"failed to locate instruction token positions ({exc})"
        )
        return None

    parsed_target = eval_utils.parse_target_character_room_with_spans(question)
    if parsed_target is None:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            "failed to parse target character/room spans from question"
        )
        return None
    target_character, target_room, character_span, room_span = parsed_target

    selected_frame_indices = sorted({
        int(frame_idx)
        for frame_idx in evidence_frame_indices
        if 0 <= int(frame_idx) < len(patch_metadata["frame_groups"])
    })
    evidence_frame_positions = [
        int(position)
        for frame_idx in selected_frame_indices
        for position in patch_metadata["frame_groups"][int(frame_idx)]
    ]
    character_positions = _question_subspan_positions(
        question_text=question,
        question_positions=patch_metadata["question_positions"],
        char_span=character_span,
    )
    room_positions = _question_subspan_positions(
        question_text=question,
        question_positions=patch_metadata["question_positions"],
        char_span=room_span,
    )
    if not character_positions or not room_positions:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            "failed to locate character/room token positions inside question span"
        )
        return None
    character_room_positions = [int(position) for position in [*character_positions, *room_positions]]

    return {
        "frame_groups": [[int(position) for position in group] for group in patch_metadata["frame_groups"]],
        "frame_group_sizes": [int(size) for size in patch_metadata["frame_group_sizes"]],
        "all_frame_positions": [int(position) for position in patch_metadata["all_frame_positions"]],
        "selected_evidence_frames": [int(frame_idx) for frame_idx in selected_frame_indices],
        "evidence_frame_positions": [int(position) for position in evidence_frame_positions],
        "instruction_positions": [int(position) for position in instruction_positions],
        "question_positions": [int(position) for position in patch_metadata["question_positions"]],
        "target_character": str(target_character),
        "target_room": str(target_room),
        "character_positions": [int(position) for position in character_positions],
        "room_positions": [int(position) for position in room_positions],
        "character_room_positions": [int(position) for position in character_room_positions],
        "carrier_index": int(clean_inputs["input_ids"].shape[1] - 1),
        "group_positions": {
            "evidence_frames": [int(position) for position in evidence_frame_positions],
            "instruction": [int(position) for position in instruction_positions],
            "question": [int(position) for position in patch_metadata["question_positions"]],
            "character_room": [int(position) for position in character_room_positions],
            "last_token": [-1],
        },
    }


def _run_patch_group(
    *,
    lm: LanguageModel,
    layers: Any,
    target_scoring_inputs: Dict[str, Any],
    source_scoring_inputs: Dict[str, Any],
    fallback_score: float,
    prompt_len: int,
    answer_token_ids: Sequence[int],
    layer_idx: int,
    group_name: str,
    token_positions: Sequence[int],
    log_label: str,
) -> float:
    if not list(token_positions):
        return float(fallback_score)
    try:
        patched_score = tgi.run_layer_corrupted_sequence_logprob(
            lm=lm,
            layers=layers,
            target_scoring_inputs=target_scoring_inputs,
            source_scoring_inputs=source_scoring_inputs,
            layer_idx=layer_idx,
            target_token_positions=token_positions,
            source_token_positions=token_positions,
            prompt_len=prompt_len,
            answer_token_ids=list(answer_token_ids),
        )
    except Exception as exc:
        print(f"  layer={layer_idx} {log_label} group={group_name} failed ({exc}); using fallback score")
        patched_score = fallback_score
    return float(patched_score)


def run_group_patch_scores(
    *,
    lm: LanguageModel,
    layers: Any,
    layer_idx: int,
    group_positions: Dict[str, Sequence[int]],
    target_scoring_inputs: Dict[str, Any],
    source_scoring_inputs: Dict[str, Any],
    prompt_len: int,
    answer_token_ids: Sequence[int],
    fallback_score: float,
    log_label: str,
) -> Dict[str, float]:
    scores_by_group: Dict[str, float] = {}
    for group_name in ANALYSIS_GROUPS:
        scores_by_group[group_name] = _run_patch_group(
            lm=lm,
            layers=layers,
            target_scoring_inputs=target_scoring_inputs,
            source_scoring_inputs=source_scoring_inputs,
            fallback_score=fallback_score,
            prompt_len=prompt_len,
            answer_token_ids=answer_token_ids,
            layer_idx=layer_idx,
            group_name=group_name,
            token_positions=group_positions[group_name],
            log_label=log_label,
        )
    return scores_by_group


def build_common_sample_fields(
    *,
    sample_id: str,
    seq_len: int,
    split: str,
    question: str,
    answer: str,
    a_star_text: str,
    a_star_ids: Sequence[int],
    clean_score: float,
    corrupted_score: float,
    denominator: float,
    clean_correct_prob: float,
    clean_top1_correct: bool,
    best_answer_text: str,
    evidence_frames: Sequence[int],
    selected_evidence_frames: Sequence[int],
    evidence_frame_positions: Sequence[int],
    frame_groups: Sequence[Sequence[int]],
    frame_group_sizes: Sequence[int],
    all_frame_positions: Sequence[int],
    instruction_positions: Sequence[int],
    question_positions: Sequence[int],
    carrier_index: int,
    selected_layers_spec: Optional[str],
    model_name: str,
    corrupted_data_dir: Path,
) -> Dict[str, Any]:
    return {
        "sample_id": str(sample_id),
        "seq_len": int(seq_len),
        "split": str(split),
        "question": str(question),
        "answer": str(answer),
        "a_star_text": str(a_star_text),
        "a_star_ids_json": json.dumps([int(token_id) for token_id in a_star_ids]),
        "clean_score": float(clean_score),
        "corrupted_score": float(corrupted_score),
        "denominator": float(denominator),
        "clean_correct_prob": float(clean_correct_prob),
        "clean_top1_correct": bool(clean_top1_correct),
        "best_answer_text": str(best_answer_text),
        "evidence_frames_json": json.dumps([int(frame_idx) for frame_idx in evidence_frames]),
        "selected_evidence_frames_json": json.dumps([int(frame_idx) for frame_idx in selected_evidence_frames]),
        "evidence_frame_positions_json": json.dumps([int(position) for position in evidence_frame_positions]),
        "evidence_frame_token_count": int(len(evidence_frame_positions)),
        "frame_groups_json": json.dumps([[int(position) for position in group] for group in frame_groups]),
        "frame_group_sizes_json": json.dumps([int(size) for size in frame_group_sizes]),
        "all_frame_positions_json": json.dumps([int(position) for position in all_frame_positions]),
        "instruction_positions_json": json.dumps([int(position) for position in instruction_positions]),
        "question_positions_json": json.dumps([int(position) for position in question_positions]),
        "carrier_index": int(carrier_index),
        "selected_layers_spec": selected_layers_spec,
        "model_name": str(model_name),
        "corrupted_data_dir": str(corrupted_data_dir),
    }


def _group_positions_for_output(
    *,
    group_name: str,
    positions: Sequence[int],
    carrier_index: int,
) -> List[int]:
    if str(group_name) == "last_token":
        return [int(carrier_index)]
    return [int(position) for position in positions]


def format_unified_summary_table(
    summary_rows: Sequence[Tuple[int, float, float, float, float, float, float, float, float, float, float]]
) -> str:
    if not summary_rows:
        return "<none>"
    header = (
        "layer".ljust(7)
        + "R_ev_frames".center(14)
        + "R_instruction".center(15)
        + "R_question".center(14)
        + "R_char+room".center(14)
        + "R_last".center(12)
        + "D_ev_frames".center(14)
        + "D_instruction".center(15)
        + "D_question".center(14)
        + "D_char+room".center(14)
        + "D_last".center(12)
    )
    rows = [header]
    for (
        layer_idx,
        r_evidence_frames,
        r_instruction,
        r_question,
        r_character_room,
        r_last,
        d_evidence_frames,
        d_instruction,
        d_question,
        d_character_room,
        d_last,
    ) in summary_rows:
        rows.append(
            f"{str(layer_idx).ljust(7)}"
            f"{f'{r_evidence_frames:.4f}'.center(14)}"
            f"{f'{r_instruction:.4f}'.center(15)}"
            f"{f'{r_question:.4f}'.center(14)}"
            f"{f'{r_character_room:.4f}'.center(14)}"
            f"{f'{r_last:.4f}'.center(12)}"
            f"{f'{d_evidence_frames:.4f}'.center(14)}"
            f"{f'{d_instruction:.4f}'.center(15)}"
            f"{f'{d_question:.4f}'.center(14)}"
            f"{f'{d_character_room:.4f}'.center(14)}"
            f"{f'{d_last:.4f}'.center(12)}"
        )
    return "\n".join(rows)


def process_sample(
    *,
    sample_dir: Path,
    sample_index: int,
    total_samples: int,
    split: str,
    lm: LanguageModel,
    layers: Any,
    selected_layers: Sequence[int],
    corrupted_data_dir: Path,
    clean_top1_must_match_gold: bool,
    selected_layers_spec: Optional[str],
) -> Optional[Dict[str, Any]]:
    sample_components = gri.load_sample_components(sample_dir, sample_index, total_samples)
    if sample_components is None:
        return None
    sample_id, frames, question, states, answer = sample_components

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
    clean_inputs, prompt_len = clean_input_result

    a_star_text = str(answer).strip()
    a_star_ids = gri.build_answer_token_ids(
        sample_id=sample_id,
        answer_text=a_star_text,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if a_star_ids is None:
        return None

    clean_answer_metrics = gri.score_clean_sample(
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

    group_metadata = extract_unified_group_metadata(
        sample_id=sample_id,
        clean_inputs=clean_inputs,
        question=question,
        num_frames=len(frames),
        evidence_frame_indices=evidence_frame_indices,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if group_metadata is None:
        return None

    corrupted_inputs = gri.build_corrupted_inputs(
        sample_id=sample_id,
        frames=frames,
        question=question,
        evidence_frame_indices=evidence_frame_indices,
        corrupted_data_root=corrupted_data_dir,
        prompt_len=prompt_len,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if corrupted_inputs is None:
        return None

    corrupted_score_result = gri.score_corrupted_sample(
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

    if abs(denominator) < NORMALIZED_EPS:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
            f"clean_score - corrupted_score is too small for normalization "
            f"(clean_score={clean_answer_score:.8f}, corrupted_score={corrupted_answer_score:.8f}, "
            f"denominator={denominator:.8e}); normalized metrics will be NaN"
        )

    if not group_metadata["evidence_frame_positions"]:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} warning: "
            "evidence frame token selection is empty; evidence-frame interventions will leave scores unchanged"
        )

    print(
        f"[{sample_index}/{total_samples}] sample_id={sample_id} clean_score={clean_answer_score:.4f} "
        f"corrupted_score={corrupted_answer_score:.4f} clean_correct_prob={clean_correct_prob:.4f} "
        f"evidence_frames={evidence_frame_indices} "
        f"selected_evidence_frames={group_metadata['selected_evidence_frames']} "
        f"evidence_frame_token_count={len(group_metadata['evidence_frame_positions'])} "
        f"instruction_group_size={len(group_metadata['instruction_positions'])} "
        f"question_group_size={len(group_metadata['question_positions'])} "
        f"character_room_group_size={len(group_metadata['character_room_positions'])} "
        f"denominator={denominator:.4f} carrier_index={group_metadata['carrier_index']}"
    )

    common_fields = build_common_sample_fields(
        sample_id=sample_id,
        seq_len=len(frames),
        split=split,
        question=question,
        answer=answer,
        a_star_text=a_star_text,
        a_star_ids=a_star_ids,
        clean_score=clean_answer_score,
        corrupted_score=corrupted_answer_score,
        denominator=denominator,
        clean_correct_prob=clean_correct_prob,
        clean_top1_correct=clean_top1_correct,
        best_answer_text=best_answer_text,
        evidence_frames=evidence_frame_indices,
        selected_evidence_frames=group_metadata["selected_evidence_frames"],
        evidence_frame_positions=group_metadata["evidence_frame_positions"],
        frame_groups=group_metadata["frame_groups"],
        frame_group_sizes=group_metadata["frame_group_sizes"],
        all_frame_positions=group_metadata["all_frame_positions"],
        instruction_positions=group_metadata["instruction_positions"],
        question_positions=group_metadata["question_positions"],
        carrier_index=group_metadata["carrier_index"],
        selected_layers_spec=selected_layers_spec,
        model_name=_runtime().model_name,
        corrupted_data_dir=corrupted_data_dir,
    )

    restoration_rows: List[Dict[str, Any]] = []
    damage_rows: List[Dict[str, Any]] = []
    summary_rows: List[Tuple[int, float, float, float, float, float, float, float, float, float, float]] = []

    for layer_idx in selected_layers:
        restored_scores = run_group_patch_scores(
            lm=lm,
            layers=layers,
            layer_idx=int(layer_idx),
            group_positions=group_metadata["group_positions"],
            target_scoring_inputs=corrupted_answer_inputs,
            source_scoring_inputs=clean_answer_inputs,
            prompt_len=prompt_len,
            answer_token_ids=a_star_ids,
            fallback_score=corrupted_answer_score,
            log_label="restoration",
        )
        ablated_scores = run_group_patch_scores(
            lm=lm,
            layers=layers,
            layer_idx=int(layer_idx),
            group_positions=group_metadata["group_positions"],
            target_scoring_inputs=clean_answer_inputs,
            source_scoring_inputs=corrupted_answer_inputs,
            prompt_len=prompt_len,
            answer_token_ids=a_star_ids,
            fallback_score=clean_answer_score,
            log_label="clean_ablation",
        )

        restoration_normalized_by_group: Dict[str, float] = {}
        damage_normalized_by_group: Dict[str, float] = {}

        for group_name in ANALYSIS_GROUPS:
            restored_score = float(restored_scores[group_name])
            raw_restoration = float(restored_score - corrupted_answer_score)
            normalized_restoration = _safe_normalize(raw_restoration, denominator)
            restoration_normalized_by_group[group_name] = normalized_restoration
            restoration_rows.append({
                **common_fields,
                "metric_type": "restoration",
                "layer": int(layer_idx),
                "group": str(group_name),
                "group_label": GROUP_LABELS[str(group_name)],
                "group_positions_json": json.dumps(
                    _group_positions_for_output(
                        group_name=str(group_name),
                        positions=group_metadata["group_positions"][group_name],
                        carrier_index=int(group_metadata["carrier_index"]),
                    )
                ),
                "group_token_count": int(len(group_metadata["group_positions"][group_name])),
                "intervention_score": float(restored_score),
                "patched_score": float(restored_score),
                "raw_value": float(raw_restoration),
                "normalized_value": float(normalized_restoration),
            })

            ablated_score = float(ablated_scores[group_name])
            raw_damage = float(clean_answer_score - ablated_score)
            normalized_damage = _safe_normalize(raw_damage, denominator)
            damage_normalized_by_group[group_name] = normalized_damage
            damage_rows.append({
                **common_fields,
                "metric_type": "clean_ablation_damage",
                "layer": int(layer_idx),
                "group": str(group_name),
                "group_label": GROUP_LABELS[str(group_name)],
                "group_positions_json": json.dumps(
                    _group_positions_for_output(
                        group_name=str(group_name),
                        positions=group_metadata["group_positions"][group_name],
                        carrier_index=int(group_metadata["carrier_index"]),
                    )
                ),
                "group_token_count": int(len(group_metadata["group_positions"][group_name])),
                "intervention_score": float(ablated_score),
                "ablated_score": float(ablated_score),
                "raw_value": float(raw_damage),
                "normalized_value": float(normalized_damage),
            })

        summary_rows.append(
            (
                int(layer_idx),
                float(restoration_normalized_by_group["evidence_frames"]),
                float(restoration_normalized_by_group["instruction"]),
                float(restoration_normalized_by_group["question"]),
                float(restoration_normalized_by_group["character_room"]),
                float(restoration_normalized_by_group["last_token"]),
                float(damage_normalized_by_group["evidence_frames"]),
                float(damage_normalized_by_group["instruction"]),
                float(damage_normalized_by_group["question"]),
                float(damage_normalized_by_group["character_room"]),
                float(damage_normalized_by_group["last_token"]),
            )
        )

    if summary_rows:
        print("  Unified summary table:")
        print(format_unified_summary_table(summary_rows))

    return {
        "sample_id": str(sample_id),
        "seq_len": int(len(frames)),
        "restoration_rows": restoration_rows,
        "damage_rows": damage_rows,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def flatten_sample_rows(sample_payloads: Sequence[Dict[str, Any]], row_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample_payload in sample_payloads:
        rows.extend(sample_payload.get(row_key, []))
    return rows


def build_group_metric_aggregate_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    metric_type: str,
    extra_value_fields: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    grouped_rows: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        layer_idx = int(row["layer"])
        group_name = str(row["group"])
        grouped_rows.setdefault((layer_idx, group_name), []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for layer_idx, group_name in sorted(grouped_rows, key=lambda item: (item[0], item[1])):
        bucket_rows = grouped_rows[(layer_idx, group_name)]
        aggregate_row: Dict[str, Any] = {
            "metric_type": str(metric_type),
            "layer": int(layer_idx),
            "group": str(group_name),
            "n_samples": int(len(bucket_rows)),
            "n_normalized_samples": int(
                sum(math.isfinite(float(row["normalized_value"])) for row in bucket_rows)
            ),
        }

        for field_name in ["raw_value", "normalized_value", *extra_value_fields]:
            center, lo_value, hi_value = _summary_triplet(
                [float(row.get(field_name, float("nan"))) for row in bucket_rows],
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            aggregate_row[f"mean_{field_name}"] = center
            aggregate_row[f"{field_name}_ci_lo"] = lo_value
            aggregate_row[f"{field_name}_ci_hi"] = hi_value

        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _plot_title(main_title: str, *, seq_len_display: str, split: str, n_samples: int) -> str:
    return f"{main_title}\n{seq_len_display} | split={split} | n={n_samples}"


def _prepare_plot_axes(ax: Any, *, selected_layers: Sequence[int], y_label: str, title: str) -> None:
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="#666666", linewidth=1.0, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    tick_step = max(1, math.ceil(len(selected_layers) / 32))
    xticks = list(selected_layers[::tick_step])
    if selected_layers and selected_layers[-1] not in xticks:
        xticks.append(int(selected_layers[-1]))
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)


def plot_group_metric_curves(
    *,
    aggregate_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    selected_layers: Sequence[int],
    metric_field: str,
    metric_field_ci_lo: str,
    metric_field_ci_hi: str,
    y_label: str,
    title: str,
) -> Optional[Path]:
    if not aggregate_rows or not selected_layers:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot {output_path.name}: matplotlib is not available ({exc})")
        return None

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    any_line = False
    for group_name in ANALYSIS_GROUPS:
        group_rows = sorted(
            [row for row in aggregate_rows if str(row["group"]) == str(group_name)],
            key=lambda row: int(row["layer"]),
        )
        if not group_rows:
            continue
        x = [int(row["layer"]) for row in group_rows]
        y = [float(row[metric_field]) for row in group_rows]
        lo = [float(row[metric_field_ci_lo]) for row in group_rows]
        hi = [float(row[metric_field_ci_hi]) for row in group_rows]
        line, = ax.plot(
            x,
            y,
            linewidth=2.2,
            color=GROUP_COLORS[str(group_name)],
            label=GROUP_LABELS[str(group_name)],
        )
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.18)
        any_line = True

    if not any_line:
        plt.close(fig)
        return None

    _prepare_plot_axes(ax, selected_layers=selected_layers, y_label=y_label, title=title)
    ax.legend(frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run a unified bottleneck pipeline that measures restoration and clean-ablation damage "
            "for evidence frames, instruction, question, character + room, and last token on MMRed samples."
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
        help="Root directory for corrupted samples. If omitted, inferred from --clean_data_dir.",
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
        "--clean_top1_must_match_gold",
        type=lambda raw: str(raw).strip().lower() in {"1", "true", "yes", "y", "on"},
        default=True,
        help="Keep only samples where the clean run's top-1 valid numeric answer matches the gold answer.",
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
    ap.add_argument("--disable_plots", action="store_true")
    args = ap.parse_args()

    if int(args.sample_limit) <= 0:
        raise ValueError("--sample_limit must be a positive integer")
    return args


def finalize_outputs(
    *,
    sample_payloads: Sequence[Dict[str, Any]],
    output_dir: Path,
    selected_layers: Sequence[int],
    seq_len_display: str,
    split: str,
    disable_plots: bool,
) -> None:
    restoration_rows = flatten_sample_rows(sample_payloads, "restoration_rows")
    damage_rows = flatten_sample_rows(sample_payloads, "damage_rows")

    restoration_aggregate = build_group_metric_aggregate_rows(
        restoration_rows,
        metric_type="restoration",
        extra_value_fields=("intervention_score", "clean_score", "corrupted_score", "denominator"),
    )
    damage_aggregate = build_group_metric_aggregate_rows(
        damage_rows,
        metric_type="clean_ablation_damage",
        extra_value_fields=("intervention_score", "clean_score", "corrupted_score", "denominator"),
    )

    per_sample_restoration_csv = output_dir / "per_sample_restoration.csv"
    per_sample_clean_ablation_csv = output_dir / "per_sample_clean_ablation.csv"
    aggregate_restoration_csv = output_dir / "aggregate_restoration.csv"
    aggregate_clean_ablation_csv = output_dir / "aggregate_clean_ablation.csv"

    common_per_sample_fields = [
        "sample_id",
        "seq_len",
        "split",
        "layer",
        "group",
        "group_label",
        "group_positions_json",
        "group_token_count",
        "clean_score",
        "corrupted_score",
        "denominator",
        "metric_type",
        "intervention_score",
        "raw_value",
        "normalized_value",
        "clean_correct_prob",
        "clean_top1_correct",
        "best_answer_text",
        "a_star_text",
        "a_star_ids_json",
        "answer",
        "question",
        "evidence_frames_json",
        "selected_evidence_frames_json",
        "evidence_frame_positions_json",
        "evidence_frame_token_count",
        "frame_group_sizes_json",
        "frame_groups_json",
        "all_frame_positions_json",
        "instruction_positions_json",
        "question_positions_json",
        "carrier_index",
        "selected_layers_spec",
        "model_name",
        "corrupted_data_dir",
    ]

    write_csv(
        per_sample_restoration_csv,
        restoration_rows,
        fieldnames=common_per_sample_fields + ["patched_score"],
    )
    write_csv(
        per_sample_clean_ablation_csv,
        damage_rows,
        fieldnames=common_per_sample_fields + ["ablated_score"],
    )
    write_csv(
        aggregate_restoration_csv,
        restoration_aggregate,
        fieldnames=[
            "metric_type",
            "layer",
            "group",
            "n_samples",
            "n_normalized_samples",
            "mean_raw_value",
            "raw_value_ci_lo",
            "raw_value_ci_hi",
            "mean_normalized_value",
            "normalized_value_ci_lo",
            "normalized_value_ci_hi",
            "mean_intervention_score",
            "intervention_score_ci_lo",
            "intervention_score_ci_hi",
            "mean_clean_score",
            "clean_score_ci_lo",
            "clean_score_ci_hi",
            "mean_corrupted_score",
            "corrupted_score_ci_lo",
            "corrupted_score_ci_hi",
            "mean_denominator",
            "denominator_ci_lo",
            "denominator_ci_hi",
        ],
    )
    write_csv(
        aggregate_clean_ablation_csv,
        damage_aggregate,
        fieldnames=[
            "metric_type",
            "layer",
            "group",
            "n_samples",
            "n_normalized_samples",
            "mean_raw_value",
            "raw_value_ci_lo",
            "raw_value_ci_hi",
            "mean_normalized_value",
            "normalized_value_ci_lo",
            "normalized_value_ci_hi",
            "mean_intervention_score",
            "intervention_score_ci_lo",
            "intervention_score_ci_hi",
            "mean_clean_score",
            "clean_score_ci_lo",
            "clean_score_ci_hi",
            "mean_corrupted_score",
            "corrupted_score_ci_lo",
            "corrupted_score_ci_hi",
            "mean_denominator",
            "denominator_ci_lo",
            "denominator_ci_hi",
        ],
    )

    plot_paths: List[Path] = []
    if not disable_plots:
        restoration_plot = plot_group_metric_curves(
            aggregate_rows=restoration_aggregate,
            output_path=output_dir / "normalized_restoration_curves.png",
            selected_layers=selected_layers,
            metric_field="mean_normalized_value",
            metric_field_ci_lo="normalized_value_ci_lo",
            metric_field_ci_hi="normalized_value_ci_hi",
            y_label="Normalized restoration",
            title=_plot_title(
                "Normalized Restoration Curves",
                seq_len_display=seq_len_display,
                split=split,
                n_samples=len(sample_payloads),
            ),
        )
        damage_plot = plot_group_metric_curves(
            aggregate_rows=damage_aggregate,
            output_path=output_dir / "normalized_clean_ablation_damage_curves.png",
            selected_layers=selected_layers,
            metric_field="mean_normalized_value",
            metric_field_ci_lo="normalized_value_ci_lo",
            metric_field_ci_hi="normalized_value_ci_hi",
            y_label="Normalized damage",
            title=_plot_title(
                "Normalized Clean-Ablation Damage Curves",
                seq_len_display=seq_len_display,
                split=split,
                n_samples=len(sample_payloads),
            ),
        )
        for plot_path in (restoration_plot, damage_plot):
            if plot_path is not None:
                plot_paths.append(plot_path)

    print(f"Wrote per-sample restoration CSV to: {per_sample_restoration_csv}")
    print(f"Wrote per-sample clean-ablation CSV to: {per_sample_clean_ablation_csv}")
    print(f"Wrote aggregate restoration CSV to: {aggregate_restoration_csv}")
    print(f"Wrote aggregate clean-ablation CSV to: {aggregate_clean_ablation_csv}")
    for plot_path in plot_paths:
        print(f"Wrote plot to: {plot_path}")
    print(
        "Output summary: "
        f"per_sample_restoration={per_sample_restoration_csv}, "
        f"per_sample_clean_ablation={per_sample_clean_ablation_csv}, "
        f"aggregate_restoration={aggregate_restoration_csv}, "
        f"aggregate_clean_ablation={aggregate_clean_ablation_csv}"
        + (f", plots={[str(path) for path in plot_paths]}" if plot_paths else "")
    )


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    configure_runtime(args.model_name)

    clean_data_dir = Path(args.clean_data_dir)
    corrupted_data_dir = (
        Path(args.corrupted_data_dir)
        if args.corrupted_data_dir is not None
        else eval_utils.infer_corrupted_data_root(clean_data_dir)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = _resolve_split_label(clean_data_dir, args.split)
    seq_len_display = _resolve_seq_len_display(clean_data_dir, args.seq_len)

    lm = LanguageModel(_model(), tokenizer=_processor().tokenizer)
    layers = get_layers(lm.model)
    selected_layers = gri.parse_layer_selection(args.layers, num_layers=len(layers))

    sample_dirs = eval_utils.iter_sample_dirs(clean_data_dir)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under: {clean_data_dir}")
    sample_rng = random.Random(args.sample_seed)
    sample_rng.shuffle(sample_dirs)
    sample_seed_label = "<system>" if args.sample_seed is None else str(int(args.sample_seed))
    print(
        f"Randomized sample order over {len(sample_dirs)} dataset samples "
        f"(sample_seed={sample_seed_label}, target valid sample_limit={int(args.sample_limit)})."
    )

    sample_payloads: List[Dict[str, Any]] = []
    processed_samples = 0
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= int(args.sample_limit):
            break
        try:
            sample_payload = process_sample(
                sample_dir=sample_dir,
                sample_index=idx,
                total_samples=len(sample_dirs),
                split=split,
                lm=lm,
                layers=layers,
                selected_layers=selected_layers,
                corrupted_data_dir=corrupted_data_dir,
                clean_top1_must_match_gold=bool(args.clean_top1_must_match_gold),
                selected_layers_spec=args.layers,
            )
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_dir.name} skipped: unexpected failure ({exc})")
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
        f"Processed {len(sample_payloads)} samples "
        f"(target limit={int(args.sample_limit)}, sample_seed={sample_seed_label}, "
        f"clean_top1_must_match_gold={bool(args.clean_top1_must_match_gold)})."
    )
    if not sample_payloads:
        print(
            "No valid samples survived filtering. "
            "If many samples were skipped because clean top-1 did not match the gold answer, "
            "rerun with --clean_top1_must_match_gold false to inspect the intervention curves anyway."
        )

    finalize_outputs(
        sample_payloads=sample_payloads,
        output_dir=output_dir,
        selected_layers=selected_layers,
        seq_len_display=seq_len_display,
        split=split,
        disable_plots=bool(args.disable_plots),
    )

    elapsed = time.perf_counter() - start_time
    print(eval_utils.format_runtime(elapsed))


if __name__ == "__main__":
    main()

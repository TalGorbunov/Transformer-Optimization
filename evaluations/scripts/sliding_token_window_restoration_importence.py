"""
Sliding token-window recovery experiment.

For each MMRed sample, this script corrupts all evidence frames, measures the clean vs.
corrupted answer-score gap, and then patches one non-frame prompt-side token window from
the clean run back into the corrupted run at each layer. Recovery is measured as:

    (s_patched - s_corr) / (s_clean - s_corr)

Both the raw recovery and the clamped recovery in [0, 1] are recorded. This script uses
window size 1 by default, so it evaluates every non-frame token in the prompt and
assistant prefix individually.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import plots as plot_utils
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from models.model import find_subsequence, get_layers, model as base_model, processor


def _sanitize_token_text(text: str) -> str:
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text if text else "<empty>"


def _build_word_labels(decoded_tokens: List[str]) -> List[str]:
    if not decoded_tokens:
        return []

    # Tokenizers often split a human-readable word across multiple pieces. We
    # recover rough word labels so plots can show a token and the word it belongs to.
    groups: List[List[int]] = []
    current_group: List[int] = []
    for idx, token_text in enumerate(decoded_tokens):
        starts_new_word = bool(current_group) and bool(token_text[:1]) and token_text[:1].isspace()
        if starts_new_word:
            groups.append(current_group)
            current_group = [idx]
        else:
            current_group.append(idx)
    if current_group:
        groups.append(current_group)

    labels = [""] * len(decoded_tokens)
    for token_indices in groups:
        label = "".join(decoded_tokens[idx] for idx in token_indices).strip()
        if not label:
            label = "".join(decoded_tokens[idx] for idx in token_indices)
        label = _sanitize_token_text(label)
        for idx in token_indices:
            labels[idx] = label
    return labels


def _token_span_from_char_span(
    text: str,
    char_span: tuple[int, int],
) -> tuple[int, int]:
    start_char, end_char = int(char_span[0]), int(char_span[1])
    if start_char > 0 and text[start_char - 1].isspace():
        start_char -= 1
    start_token = len(processor.tokenizer(text[:start_char], add_special_tokens=False)["input_ids"])
    end_token = len(processor.tokenizer(text[:end_char], add_special_tokens=False)["input_ids"])
    return start_token, end_token


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
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: prompt has no tokens")
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
    gold_answer_text: str,
    clean_inputs: Dict[str, Any],
    prompt_len: int,
    sample_mode: str,
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

    gold_answer_text = str(gold_answer_text).strip()
    gold_answer_score = float(metrics["scores_by_answer"].get(gold_answer_text, float("-inf")))
    gold_answer_prob = float(metrics["probs_by_answer"].get(gold_answer_text, 0.0))
    clean_predicted_answer_text = str(metrics["best_answer_text"]).strip()
    clean_predicted_answer_score = float(
        metrics["scores_by_answer"].get(clean_predicted_answer_text, float("-inf"))
    )
    clean_predicted_answer_prob = float(
        metrics["probs_by_answer"].get(clean_predicted_answer_text, 0.0)
    )
    clean_top1_correct = (clean_predicted_answer_text == gold_answer_text)

    if sample_mode == "success_only" and not clean_top1_correct:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"sample_mode=success_only but clean top-1 is {clean_predicted_answer_text!r}, "
            f"not gold answer {gold_answer_text!r}"
        )
        return None
    if sample_mode == "failure_only" and clean_top1_correct:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"sample_mode=failure_only but clean top-1 already matches gold answer {gold_answer_text!r}"
        )
        return None

    patch_target_answer_text = gold_answer_text if sample_mode == "success_only" else clean_predicted_answer_text
    patch_target_answer_score = float(
        metrics["scores_by_answer"].get(patch_target_answer_text, float("-inf"))
    )
    patch_target_answer_prob = float(
        metrics["probs_by_answer"].get(patch_target_answer_text, 0.0)
    )

    return {
        "gold_answer_text": gold_answer_text,
        "gold_answer_score": gold_answer_score,
        "gold_answer_prob": gold_answer_prob,
        "patch_target_answer_text": patch_target_answer_text,
        "patch_target_answer_score": patch_target_answer_score,
        "patch_target_answer_prob": patch_target_answer_prob,
        "clean_answer_score": patch_target_answer_score,
        "clean_correct_prob": patch_target_answer_prob,
        "clean_top1_correct": clean_top1_correct,
        "clean_predicted_answer_text": clean_predicted_answer_text,
        "clean_predicted_answer_score": clean_predicted_answer_score,
        "clean_predicted_answer_prob": clean_predicted_answer_prob,
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
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to build corrupted frames "
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


def locate_patchable_token_metadata(
    inputs: Dict[str, Any],
    question: str,
    num_frames: int,
) -> Optional[List[Dict[str, Any]]]:
    prompt = tgi.build_prompt(question, num_frames=num_frames)
    prompt_token_ids = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not prompt_token_ids:
        return None

    full_input_ids = [int(tok) for tok in inputs["input_ids"][0].tolist()]
    prompt_start = find_subsequence(full_input_ids, [int(tok) for tok in prompt_token_ids])
    if prompt_start is None:
        return None

    # We patch every prompt-side token plus the assistant prefix, but exclude
    # frame/image tokens entirely by anchoring on the text prompt subsequence.
    prompt_positions = list(range(prompt_start, prompt_start + len(prompt_token_ids)))
    assistant_positions = list(range(prompt_start + len(prompt_token_ids), len(full_input_ids)))
    patch_positions = prompt_positions + assistant_positions
    if not patch_positions:
        return None

    decoded_tokens = [
        processor.tokenizer.decode([full_input_ids[position]], clean_up_tokenization_spaces=False)
        for position in patch_positions
    ]
    word_labels = _build_word_labels(decoded_tokens)

    parsed_question = eval_utils.parse_target_character_room_with_spans(question)
    prompt_question_start = prompt.find(question)
    prompt_character_token_span: Optional[tuple[int, int]] = None
    prompt_room_token_span: Optional[tuple[int, int]] = None
    if parsed_question is not None and prompt_question_start >= 0:
        _, _, character_span, room_span = parsed_question
        prompt_character_char_span = (
            int(prompt_question_start + character_span[0]),
            int(prompt_question_start + character_span[1]),
        )
        prompt_room_char_span = (
            int(prompt_question_start + room_span[0]),
            int(prompt_question_start + room_span[1]),
        )
        prompt_character_token_span = _token_span_from_char_span(prompt, prompt_character_char_span)
        prompt_room_token_span = _token_span_from_char_span(prompt, prompt_room_char_span)

    character_token_indices: List[int] = []
    room_token_indices: List[int] = []
    for token_idx, position in enumerate(patch_positions):
        relative_prompt_idx = int(position - prompt_start)
        if relative_prompt_idx < 0 or relative_prompt_idx >= len(prompt_token_ids):
            continue
        if (
            prompt_character_token_span is not None
            and prompt_character_token_span[0] <= relative_prompt_idx < prompt_character_token_span[1]
        ):
            character_token_indices.append(token_idx)
        if prompt_room_token_span is not None and prompt_room_token_span[0] <= relative_prompt_idx < prompt_room_token_span[1]:
            room_token_indices.append(token_idx)

    token_metadata: List[Dict[str, Any]] = []
    skip_token_indices = set(character_token_indices[1:] + room_token_indices[1:])
    grouped_token_lookup = {}
    if character_token_indices:
        grouped_token_lookup[int(character_token_indices[0])] = {
            "group_name": "target_character",
            "group_indices": [int(idx) for idx in character_token_indices],
            "word_label": question[parsed_question[2][0]:parsed_question[2][1]] if parsed_question is not None else None,
        }
    if room_token_indices:
        grouped_token_lookup[int(room_token_indices[0])] = {
            "group_name": "target_room",
            "group_indices": [int(idx) for idx in room_token_indices],
            "word_label": question[parsed_question[3][0]:parsed_question[3][1]] if parsed_question is not None else None,
        }

    for token_idx, position in enumerate(patch_positions):
        if token_idx in skip_token_indices:
            continue
        group_info = grouped_token_lookup.get(int(token_idx))
        if group_info is not None:
            group_indices = [int(idx) for idx in group_info["group_indices"]]
            group_positions = [int(patch_positions[idx]) for idx in group_indices]
            group_token_text = "".join(decoded_tokens[idx] for idx in group_indices)
            group_word_label = str(group_info["word_label"]).strip() or _sanitize_token_text(group_token_text)
        else:
            group_indices = [int(token_idx)]
            group_positions = [int(position)]
            group_token_text = decoded_tokens[token_idx]
            group_word_label = word_labels[token_idx]
        token_metadata.append({
            "token_index": len(token_metadata),
            "full_position": int(group_positions[0]),
            "token_id": int(full_input_ids[group_positions[0]]),
            "token_text": _sanitize_token_text(group_token_text),
            "word_label": _sanitize_token_text(group_word_label),
            "token_region": "prompt" if position in prompt_positions else "assistant",
            "patch_positions": group_positions,
            "patch_group": group_info["group_name"] if group_info is not None else "single_token",
        })
    return token_metadata


def _iter_token_chunks(token_metadata: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [
        token_metadata[start:start + batch_size]
        for start in range(0, len(token_metadata), batch_size)
    ]


def compute_token_window_metrics(
    lm: LanguageModel,
    layers: Any,
    selected_layers: List[int],
    token_metadata: List[Dict[str, Any]],
    corrupted_answer_inputs: Dict[str, Any],
    clean_answer_inputs: Dict[str, Any],
    clean_answer_score: float,
    corrupted_answer_score: float,
    prompt_len: int,
    answer_token_ids: List[int],
    batch_size: int,
) -> List[Dict[str, Any]]:
    denominator = float(clean_answer_score - corrupted_answer_score)
    token_chunks = _iter_token_chunks(token_metadata, batch_size=batch_size)
    layer_metrics: List[Dict[str, Any]] = []

    for layer_idx in selected_layers:
        # We keep both the raw recovery and a clamped [0, 1] version. The raw
        # value is faithful to the formula; the clamped value is easier to compare
        # visually across many tokens/layers.
        raw_recovery_by_token = [0.0 for _ in token_metadata]
        clamped_recovery_by_token = [0.0 for _ in token_metadata]
        patched_score_by_token = [corrupted_answer_score for _ in token_metadata]

        for chunk_idx, token_chunk in enumerate(token_chunks, start=1):
            target_inputs_batch = tgi.repeat_inputs_for_batch(corrupted_answer_inputs, batch_size=len(token_chunk))
            source_inputs_batch = tgi.repeat_inputs_for_batch(clean_answer_inputs, batch_size=len(token_chunk))
            target_positions_by_batch = [
                [int(position) for position in token.get("patch_positions", [token["full_position"]])]
                for token in token_chunk
            ]
            source_positions_by_batch = [
                [int(position) for position in token.get("patch_positions", [token["full_position"]])]
                for token in token_chunk
            ]

            try:
                patched_scores = tgi.run_layer_token_patch_logprob_batch(
                    lm=lm,
                    layers=layers,
                    target_batched_scoring_inputs=target_inputs_batch,
                    source_batched_scoring_inputs=source_inputs_batch,
                    layer_idx=layer_idx,
                    target_token_positions_by_batch=target_positions_by_batch,
                    source_token_positions_by_batch=source_positions_by_batch,
                    prompt_len=prompt_len,
                    answer_token_ids=answer_token_ids,
                )
            except Exception as exc:
                print(
                    f"  layer={layer_idx} chunk={chunk_idx}/{len(token_chunks)} failed token patching ({exc}); "
                    "using corrupted scores for this chunk"
                )
                patched_scores = [corrupted_answer_score for _ in token_chunk]

            for local_idx, token in enumerate(token_chunk):
                token_index = int(token["token_index"])
                patched_score = float(
                    patched_scores[local_idx].item() if hasattr(patched_scores[local_idx], "item") else patched_scores[local_idx]
                )
                # Recovery asks how much of the clean-vs-corrupted score gap is
                # recovered by restoring this token at this layer.
                raw_recovery = (patched_score - corrupted_answer_score) / denominator
                clamped_recovery = min(max(raw_recovery, 0.0), 1.0)
                patched_score_by_token[token_index] = patched_score
                raw_recovery_by_token[token_index] = raw_recovery
                clamped_recovery_by_token[token_index] = clamped_recovery

        layer_metrics.append({
            "layer": int(layer_idx),
            "patched_score_by_token": patched_score_by_token,
            "raw_recovery_by_token": raw_recovery_by_token,
            "clamped_recovery_by_token": clamped_recovery_by_token,
        })
    return layer_metrics


def compute_token_mass_summary(
    token_metadata: List[Dict[str, Any]],
    layer_metrics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # "Mass" is the total recovery carried by a token across the selected layers.
    masses = [0.0 for _ in token_metadata]
    raw_masses = [0.0 for _ in token_metadata]
    for layer_metric in layer_metrics:
        for token_index, value in enumerate(layer_metric["clamped_recovery_by_token"]):
            masses[token_index] += float(value)
        for token_index, value in enumerate(layer_metric["raw_recovery_by_token"]):
            raw_masses[token_index] += float(value)

    total_mass = sum(masses)
    summary: List[Dict[str, Any]] = []
    for token in token_metadata:
        token_index = int(token["token_index"])
        summary.append({
            **token,
            "clamped_mass": float(masses[token_index]),
            "raw_mass": float(raw_masses[token_index]),
            "mass_share": (float(masses[token_index]) / total_mass) if total_mass > 0.0 else 0.0,
        })
    summary.sort(key=lambda item: item["clamped_mass"], reverse=True)
    return summary


def build_aggregate_payload(sample_payloads: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not sample_payloads:
        return None

    common_layers = [
        int(layer_idx)
        for layer_idx in sample_payloads[0].get("selected_layers", [])
        if all(int(layer_idx) in set(int(x) for x in sample.get("selected_layers", [])) for sample in sample_payloads)
    ]
    if not common_layers:
        return None

    expected_token_count = len(sample_payloads[0].get("token_metadata", []))
    if expected_token_count <= 0:
        return None
    for sample in sample_payloads[1:]:
        sample_token_count = len(sample.get("token_metadata", []))
        if sample_token_count != expected_token_count:
            raise RuntimeError(
                "Aggregate token count mismatch: "
                f"reference sample_id={sample_payloads[0].get('sample_id')} has {expected_token_count} tokens, "
                f"but sample_id={sample.get('sample_id')} has {sample_token_count} tokens."
            )

    token_metadata = []
    for token in sample_payloads[0]["token_metadata"]:
        token_metadata.append({
            "token_index": int(token["token_index"]),
            "full_position": int(token["full_position"]),
            "token_id": int(token["token_id"]),
            "token_text": str(token["token_text"]),
            "word_label": str(token["word_label"]),
            "token_region": str(token["token_region"]),
            "patch_positions": [int(position) for position in token.get("patch_positions", [token["full_position"]])],
            "patch_group": str(token.get("patch_group", "single_token")),
        })

    layer_metrics: List[Dict[str, Any]] = []
    for layer_idx in common_layers:
        per_sample_layer_metrics = []
        for sample in sample_payloads:
            layer_metric = next(
                (layer for layer in sample["layer_metrics"]["layers"] if int(layer["layer"]) == layer_idx),
                None,
            )
            if layer_metric is None:
                per_sample_layer_metrics = []
                break
            per_sample_layer_metrics.append(layer_metric)
        if not per_sample_layer_metrics:
            continue

        patched_score_by_token: List[float] = []
        raw_recovery_by_token: List[float] = []
        clamped_recovery_by_token: List[float] = []
        for token_idx in range(expected_token_count):
            patched_vals = [float(layer["patched_score_by_token"][token_idx]) for layer in per_sample_layer_metrics]
            raw_vals = [float(layer["raw_recovery_by_token"][token_idx]) for layer in per_sample_layer_metrics]
            clamped_vals = [float(layer["clamped_recovery_by_token"][token_idx]) for layer in per_sample_layer_metrics]
            patched_score_by_token.append(sum(patched_vals) / len(patched_vals))
            raw_recovery_by_token.append(sum(raw_vals) / len(raw_vals))
            clamped_recovery_by_token.append(sum(clamped_vals) / len(clamped_vals))

        layer_metrics.append({
            "layer": int(layer_idx),
            "patched_score_by_token": patched_score_by_token,
            "raw_recovery_by_token": raw_recovery_by_token,
            "clamped_recovery_by_token": clamped_recovery_by_token,
        })

    if not layer_metrics:
        return None

    token_mass_summary = compute_token_mass_summary(token_metadata, layer_metrics)
    first_last_token_layer_summary = {
        "first": {},
        "last": {},
    }
    token_min_clamped_summary: List[Dict[str, Any]] = []
    for layer_idx in common_layers:
        first_values: List[float] = []
        last_values: List[float] = []
        for sample in sample_payloads:
            layer_metric = next(
                (layer for layer in sample["layer_metrics"]["layers"] if int(layer["layer"]) == layer_idx),
                None,
            )
            if layer_metric is None:
                continue
            first_values.append(float(layer_metric["clamped_recovery_by_token"][0]))
            last_values.append(float(layer_metric["clamped_recovery_by_token"][expected_token_count - 1]))
        first_last_token_layer_summary["first"][str(int(layer_idx))] = first_values
        first_last_token_layer_summary["last"][str(int(layer_idx))] = last_values

    for token_idx, token in enumerate(token_metadata):
        per_sample_min_clamped: List[float] = []
        for sample in sample_payloads:
            per_layer_values: List[float] = []
            for layer_idx in common_layers:
                layer_metric = next(
                    (layer for layer in sample["layer_metrics"]["layers"] if int(layer["layer"]) == layer_idx),
                    None,
                )
                if layer_metric is None:
                    continue
                per_layer_values.append(float(layer_metric["clamped_recovery_by_token"][token_idx]))
            if per_layer_values:
                per_sample_min_clamped.append(min(per_layer_values))
        token_min_clamped_summary.append({
            **token,
            "per_sample_min_clamped": per_sample_min_clamped,
            "mean_min_clamped": (sum(per_sample_min_clamped) / len(per_sample_min_clamped)) if per_sample_min_clamped else 0.0,
        })
    token_min_clamped_summary.sort(key=lambda item: float(item["mean_min_clamped"]), reverse=True)

    mean_denominator = sum(float(sample["denominator"]) for sample in sample_payloads) / len(sample_payloads)
    mean_clean_answer_score = sum(float(sample["clean_answer_score"]) for sample in sample_payloads) / len(sample_payloads)
    mean_corrupted_answer_score = (
        sum(float(sample["corrupted_answer_score"]) for sample in sample_payloads) / len(sample_payloads)
    )
    mean_clean_correct_prob = sum(float(sample["clean_correct_prob"]) for sample in sample_payloads) / len(sample_payloads)
    mean_gold_answer_score = sum(float(sample["gold_answer_score"]) for sample in sample_payloads) / len(sample_payloads)
    mean_gold_answer_prob = sum(float(sample["gold_answer_prob"]) for sample in sample_payloads) / len(sample_payloads)
    mean_patch_target_answer_score = (
        sum(float(sample["patch_target_answer_score"]) for sample in sample_payloads) / len(sample_payloads)
    )
    mean_patch_target_answer_prob = (
        sum(float(sample["patch_target_answer_prob"]) for sample in sample_payloads) / len(sample_payloads)
    )
    mean_clean_predicted_answer_score = (
        sum(float(sample["clean_predicted_answer_score"]) for sample in sample_payloads) / len(sample_payloads)
    )
    mean_clean_predicted_answer_prob = (
        sum(float(sample["clean_predicted_answer_prob"]) for sample in sample_payloads) / len(sample_payloads)
    )
    clean_top1_correct_count = sum(1 for sample in sample_payloads if bool(sample.get("clean_top1_correct")))

    return {
        "sample_id": "aggregate_mean",
        "sample_mode": str(sample_payloads[0].get("sample_mode", "success_only")),
        "patch_target_policy": str(sample_payloads[0].get("patch_target_policy", "gold_answer")),
        "question": f"Mean over {len(sample_payloads)} samples",
        "answer": None,
        "a_star_text": None,
        "a_star_ids": [],
        "gold_answer_text": None,
        "patch_target_answer_text": None,
        "clean_answer_score": mean_clean_answer_score,
        "corrupted_answer_score": mean_corrupted_answer_score,
        "clean_correct_prob": mean_clean_correct_prob,
        "gold_answer_score": mean_gold_answer_score,
        "gold_answer_prob": mean_gold_answer_prob,
        "patch_target_answer_score": mean_patch_target_answer_score,
        "patch_target_answer_prob": mean_patch_target_answer_prob,
        "clean_top1_correct": (clean_top1_correct_count == len(sample_payloads)),
        "clean_predicted_answer_text": None,
        "clean_predicted_answer_score": mean_clean_predicted_answer_score,
        "clean_predicted_answer_prob": mean_clean_predicted_answer_prob,
        "denominator": mean_denominator,
        "lambda_threshold": float(sample_payloads[0]["lambda_threshold"]),
        "evidence_frames": [],
        "selected_layers": common_layers,
        "token_metadata": token_metadata,
        "token_mass_summary": token_mass_summary,
        "layer_metrics": {"layers": layer_metrics},
        "first_last_token_layer_summary": first_last_token_layer_summary,
        "token_min_clamped_summary": token_min_clamped_summary,
        "aggregate_sample_count": len(sample_payloads),
        "aggregate_token_count": expected_token_count,
        "aggregate_clean_top1_correct_count": clean_top1_correct_count,
        "aggregate_clean_top1_incorrect_count": len(sample_payloads) - clean_top1_correct_count,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Measure token-level recovery by corrupting all evidence frames, patching one prompt-side "
            "token window from the clean run back into the corrupted run, and scoring the clean answer."
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
    ap.add_argument("--batch_size", type=int, default=16, help="How many token windows to patch together per forward pass.")
    ap.add_argument(
        "--sample_mode",
        type=str,
        choices=("success_only", "failure_only"),
        default="success_only",
        help="Whether to keep only clean successes or only clean failures.",
    )
    ap.add_argument(
        "--lambda",
        dest="lambda_threshold",
        type=float,
        default=0.0,
        help="Only run a sample if (s_clean - s_corr) is strictly greater than this threshold.",
    )
    ap.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Optional layer selection. Examples: --layers 32:42, --layers 0:64:2, "
            "--layers 30,32,34,36,38,40"
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
    args: argparse.Namespace,
    lm: LanguageModel,
    layers: Any,
    selected_layers: List[int],
    corrupted_data_root: Path,
    output_dir: Path,
    seq_len_label: Optional[str],
) -> Optional[Dict[str, Any]]:
    sample_components = load_sample_components(sample_dir, sample_index, total_samples)
    if sample_components is None:
        return None
    sample_id, frames, question, states, answer = sample_components

    evidence_frame_indices = eval_utils.collect_evidence_frame_indices(question, states)
    if len(evidence_frame_indices) < 1:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: no evidence frames")
        return None

    clean_input_result = build_clean_inputs(sample_id, frames, question, sample_index, total_samples)
    if clean_input_result is None:
        return None
    clean_inputs, prompt_len = clean_input_result

    gold_answer_text = str(answer).strip()

    clean_answer_metrics = score_clean_sample(
        sample_id=sample_id,
        frames=frames,
        gold_answer_text=gold_answer_text,
        clean_inputs=clean_inputs,
        prompt_len=prompt_len,
        sample_mode=str(args.sample_mode),
        lm=lm,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if clean_answer_metrics is None:
        return None
    patch_target_answer_text = str(clean_answer_metrics["patch_target_answer_text"]).strip()
    patch_target_answer_ids = build_answer_token_ids(sample_id, patch_target_answer_text, sample_index, total_samples)
    if patch_target_answer_ids is None:
        return None
    clean_answer_score = float(clean_answer_metrics["clean_answer_score"])
    clean_correct_prob = float(clean_answer_metrics["clean_correct_prob"])

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
        answer_token_ids=patch_target_answer_ids,
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
    # Skip weakly affected samples so recovery is only measured when corruption
    # actually creates a meaningful drop in the clean answer score.
    if denominator <= float(args.lambda_threshold):
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: denominator={denominator:.6f} "
            f"<= lambda={float(args.lambda_threshold):.6f}"
        )
        return None

    token_metadata = locate_patchable_token_metadata(clean_inputs, question, num_frames=len(frames))
    if not token_metadata:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to locate patchable tokens")
        return None

    print(
        f"[{sample_index}/{total_samples}] sample_id={sample_id} "
        f"sample_mode={str(args.sample_mode)} "
        f"gold_answer={gold_answer_text!r} "
        f"patch_target={patch_target_answer_text!r} "
        f"clean_top1={str(clean_answer_metrics['clean_predicted_answer_text'])!r} "
        f"clean_top1_correct={bool(clean_answer_metrics['clean_top1_correct'])} "
        f"clean_answer_score={clean_answer_score:.4f} "
        f"corrupted_answer_score={corrupted_answer_score:.4f} "
        f"denominator={denominator:.4f} "
        f"clean_correct_prob={clean_correct_prob:.4f} "
        f"patchable_tokens={len(token_metadata)} "
        f"first_token={token_metadata[0]['token_text']!r} "
        f"last_token={token_metadata[-1]['token_text']!r}"
    )

    layer_metrics = compute_token_window_metrics(
        lm=lm,
        layers=layers,
        selected_layers=selected_layers,
        token_metadata=token_metadata,
        corrupted_answer_inputs=corrupted_answer_inputs,
        clean_answer_inputs=clean_answer_inputs,
        clean_answer_score=clean_answer_score,
        corrupted_answer_score=corrupted_answer_score,
        prompt_len=prompt_len,
        answer_token_ids=patch_target_answer_ids,
        batch_size=int(args.batch_size),
    )
    token_mass_summary = compute_token_mass_summary(token_metadata, layer_metrics)

    print("  Top token masses:")
    for row in token_mass_summary[: min(10, len(token_mass_summary))]:
        print(
            f"    token_index={row['token_index']:>3} "
            f"token={row['token_text']!r:<16} "
            f"word={row['word_label']!r:<20} "
            f"region={row['token_region']:<9} "
            f"clamped_mass={row['clamped_mass']:.4f} "
            f"raw_mass={row['raw_mass']:.4f} "
            f"share={100.0 * row['mass_share']:.2f}%"
        )

    # Keep the full per-token/per-layer payload for later inspection because the
    # plots compress a lot of detail.
    sample_payload = {
        "sample_id": sample_id,
        "sample_mode": str(args.sample_mode),
        "patch_target_policy": (
            "gold_answer" if str(args.sample_mode) == "success_only" else "clean_top1_predicted_answer"
        ),
        "question": question,
        "answer": answer,
        "gold_answer_text": gold_answer_text,
        "patch_target_answer_text": patch_target_answer_text,
        "a_star_text": patch_target_answer_text,
        "a_star_ids": patch_target_answer_ids,
        "clean_answer_score": clean_answer_score,
        "corrupted_answer_score": corrupted_answer_score,
        "clean_correct_prob": clean_correct_prob,
        "gold_answer_score": float(clean_answer_metrics["gold_answer_score"]),
        "gold_answer_prob": float(clean_answer_metrics["gold_answer_prob"]),
        "patch_target_answer_score": float(clean_answer_metrics["patch_target_answer_score"]),
        "patch_target_answer_prob": float(clean_answer_metrics["patch_target_answer_prob"]),
        "clean_top1_correct": bool(clean_answer_metrics["clean_top1_correct"]),
        "clean_predicted_answer_text": str(clean_answer_metrics["clean_predicted_answer_text"]),
        "clean_predicted_answer_score": float(clean_answer_metrics["clean_predicted_answer_score"]),
        "clean_predicted_answer_prob": float(clean_answer_metrics["clean_predicted_answer_prob"]),
        "denominator": denominator,
        "lambda_threshold": float(args.lambda_threshold),
        "evidence_frames": [int(frame_idx) for frame_idx in evidence_frame_indices],
        "selected_layers": list(selected_layers),
        "token_metadata": token_metadata,
        "token_mass_summary": token_mass_summary,
        "layer_metrics": {"layers": layer_metrics},
    }
    return sample_payload


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    data_root = Path(args.data_root)
    corrupted_data_root = (
        Path(args.corrupted_root) if args.corrupted_root is not None else eval_utils.infer_corrupted_data_root(data_root)
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_len_label = eval_utils.resolve_seq_len_label(data_root)

    lm = LanguageModel(base_model, tokenizer=processor.tokenizer)
    layers = get_layers(lm.model)
    selected_layers = tgi.parse_layer_selection(args.layers, num_layers=len(layers))

    sample_dirs = iter_sample_dirs(data_root)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under: {data_root}")

    processed_samples = 0
    sample_payloads: List[Dict[str, Any]] = []
    aggregate_payload: Optional[Dict[str, Any]] = None
    expected_token_count: Optional[int] = None
    expected_token_sample_id: Optional[str] = None
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= int(args.limit):
            break
        sample_payload = process_sample(
            sample_dir=sample_dir,
            sample_index=idx,
            total_samples=len(sample_dirs),
            args=args,
            lm=lm,
            layers=layers,
            selected_layers=selected_layers,
            corrupted_data_root=corrupted_data_root,
            output_dir=output_dir,
            seq_len_label=seq_len_label,
        )
        if sample_payload is None:
            continue
        sample_token_count = len(sample_payload.get("token_metadata", []))
        if expected_token_count is None:
            expected_token_count = sample_token_count
            expected_token_sample_id = str(sample_payload.get("sample_id"))
        elif sample_token_count != expected_token_count:
            mismatch_report = {
                "sample_mode": str(args.sample_mode),
                "reference_sample_id": expected_token_sample_id,
                "reference_token_count": int(expected_token_count),
                "mismatch_sample_id": str(sample_payload.get("sample_id")),
                "mismatch_token_count": int(sample_token_count),
                "processed_sample_count_before_mismatch": int(len(sample_payloads)),
                "data_root": str(data_root),
                "seq_len_label": seq_len_label,
            }
            mismatch_path = output_dir / f"token_count_mismatch{('_' + seq_len_label) if seq_len_label else ''}.json"
            mismatch_path.write_text(json.dumps(mismatch_report, indent=2) + "\n", encoding="utf-8")
            print(
                "Stopping immediately: token-count mismatch across processed samples. "
                f"reference sample_id={expected_token_sample_id} tokens={expected_token_count}; "
                f"mismatch sample_id={sample_payload.get('sample_id')} tokens={sample_token_count}. "
                f"Wrote mismatch report to: {mismatch_path}"
            )
            elapsed = time.perf_counter() - start_time
            print(eval_utils.format_runtime(elapsed))
            return
        sample_payloads.append(sample_payload)
        processed_samples += 1

    if not args.disable_plots:
        aggregate_payload = build_aggregate_payload(sample_payloads)
        if aggregate_payload is None:
            print("Skipped aggregate plotting: no compatible processed samples to aggregate.")
        else:
            aggregate_metrics_path = output_dir / f"aggregate_mean_token_recovery_metrics{('_' + seq_len_label) if seq_len_label else ''}.json"
            aggregate_metrics_path.write_text(json.dumps(aggregate_payload, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote aggregate token recovery metrics JSON to: {aggregate_metrics_path}")

            heatmap_path = plot_utils.plot_token_recovery_heatmap(
                aggregate_payload,
                output_dir,
                seq_len_label=seq_len_label,
                file_prefix="aggregate_mean",
            )
            if heatmap_path is not None:
                print(f"Wrote aggregate token recovery heatmap to: {heatmap_path}")

            first_last_path = plot_utils.plot_first_last_token_importance_lines(
                aggregate_payload,
                output_dir,
                seq_len_label=seq_len_label,
                file_prefix="aggregate_mean",
            )
            if first_last_path is not None:
                print(f"Wrote aggregate first/last token importance lines plot to: {first_last_path}")

    print(
        f"Processed {len(sample_payloads)} samples "
        f"(target limit={int(args.limit)}, sample_mode={str(args.sample_mode)}, "
        f"lambda={float(args.lambda_threshold):.4f})."
    )
    elapsed = time.perf_counter() - start_time
    print(eval_utils.format_runtime(elapsed))


if __name__ == "__main__":
    main()

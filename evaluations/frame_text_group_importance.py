import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations import text_group_importance as tgi
from evaluations import utils as eval_utils
from models.model import get_layers, image_token_groups, model as base_model, processor

try:
    from evaluations.utils import iter_sample_dirs, load_mmred_sample
except ModuleNotFoundError:
    from utils import iter_sample_dirs, load_mmred_sample

_STEPS_IN_ROOM_RE = re.compile(
    r"How many steps did\s+([A-Za-z]+)\s+spend in\s+the\s+([A-Za-z]+)",
    flags=re.IGNORECASE,
)
_GROUP_ORDER = ["frames", "character", "room", "last_token"]
_CORRUPTION_METHOD = "group_specific_aligned_controls"


def parse_include_groups(raw: Optional[str]) -> List[str]:
    if raw is None:
        return list(_GROUP_ORDER)
    values = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not values:
        return list(_GROUP_ORDER)
    invalid = [x for x in values if x not in _GROUP_ORDER]
    if invalid:
        raise ValueError(f"--include_groups has invalid entries: {invalid}. Supported groups: {_GROUP_ORDER}")
    return [group for group in _GROUP_ORDER if group in values]


def parse_target_character_room(question_text: str) -> Optional[Tuple[str, str]]:
    return eval_utils.parse_target_character_room(question_text)


def rooms_to_room2chars(rooms: Dict[str, Any]) -> Dict[str, List[str]]:
    return eval_utils.rooms_to_room2chars(rooms)


def collect_evidence_frame_indices(question: str, states: List[Dict[str, Any]]) -> List[int]:
    return eval_utils.collect_evidence_frame_indices(question, states)


def infer_corrupted_data_root(clean_data_root: Path) -> Path:
    return eval_utils.infer_corrupted_data_root(clean_data_root)


def resolve_corrupted_sample_dir(corrupted_data_root: Path, sample_id: str, frame_idx: int) -> Path:
    return eval_utils.resolve_corrupted_sample_dir(corrupted_data_root, sample_id, frame_idx)


def build_composite_corrupted_frames(
    sample_id: str,
    clean_frames: Sequence[Any],
    evidence_frame_indices: Sequence[int],
    corrupted_data_root: Path,
) -> Tuple[Optional[List[Any]], Dict[str, str]]:
    corrupted_frames = list(clean_frames)
    issues: Dict[str, str] = {}
    for frame_idx in evidence_frame_indices:
        corrupted_sample_dir = resolve_corrupted_sample_dir(corrupted_data_root, sample_id, int(frame_idx))
        if not corrupted_sample_dir.is_dir():
            issues[f"frame_{frame_idx}"] = "missing_corrupted_sample_dir"
            return None, issues
        try:
            _, corrupted_sample_frames, corrupted_question, _, _ = load_mmred_sample(corrupted_sample_dir)
        except Exception as exc:
            issues[f"frame_{frame_idx}"] = f"load_failure({exc})"
            return None, issues
        if len(corrupted_sample_frames) != len(clean_frames):
            issues[f"frame_{frame_idx}"] = (
                f"frame_count_mismatch(clean={len(clean_frames)},corrupted={len(corrupted_sample_frames)})"
            )
            return None, issues
        corrupted_frames[int(frame_idx)] = corrupted_sample_frames[int(frame_idx)]
    return corrupted_frames, issues


def replace_character_in_question(question: str, new_character: str) -> Optional[str]:
    parsed = tgi.parse_target_character_room_with_spans(question)
    if parsed is None:
        return None
    _, _, char_span, _ = parsed
    return question[:char_span[0]] + str(new_character) + question[char_span[1]:]


def replace_room_in_question(question: str, new_room: str) -> Optional[str]:
    parsed = tgi.parse_target_character_room_with_spans(question)
    if parsed is None:
        return None
    _, _, _, room_span = parsed
    return question[:room_span[0]] + str(new_room) + question[room_span[1]:]


def build_prompt_context(
    inputs: Dict[str, torch.Tensor],
    question: str,
    num_frames: int,
) -> Optional[Dict[str, Any]]:
    prompt = tgi.build_prompt(question, num_frames=num_frames)
    prompt_token_ids, _ = tgi.tokenize_with_offsets_if_available(prompt)
    if not prompt_token_ids:
        return None
    full_input_ids = [int(tok) for tok in inputs["input_ids"][0].tolist()]
    prompt_start_in_full = tgi.find_subsequence(full_input_ids, prompt_token_ids)
    if prompt_start_in_full is None:
        return None
    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "prompt_start_in_full": int(prompt_start_in_full),
        "full_input_ids": full_input_ids,
    }


def validate_text_control_alignment(
    clean_inputs: Dict[str, torch.Tensor],
    clean_question: str,
    clean_group_positions: Dict[str, List[int]],
    control_inputs: Dict[str, torch.Tensor],
    control_question: str,
    control_group_positions: Dict[str, List[int]],
    group_name: str,
    num_frames: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    debug: Dict[str, Any] = {"group": group_name}
    clean_context = build_prompt_context(clean_inputs, clean_question, num_frames=num_frames)
    if clean_context is None:
        return False, "clean_prompt_context_unavailable", debug
    control_context = build_prompt_context(control_inputs, control_question, num_frames=num_frames)
    if control_context is None:
        return False, "control_prompt_context_unavailable", debug

    clean_char_spans, _, clean_warnings = tgi.build_prompt_group_char_spans(clean_question, num_frames=num_frames)
    control_char_spans, _, control_warnings = tgi.build_prompt_group_char_spans(control_question, num_frames=num_frames)
    debug["clean_warnings"] = list(clean_warnings)
    debug["control_warnings"] = list(control_warnings)

    clean_positions = [int(pos) for pos in clean_group_positions.get(group_name, [])]
    control_positions = [int(pos) for pos in control_group_positions.get(group_name, [])]
    debug["clean_positions"] = list(clean_positions)
    debug["control_positions"] = list(control_positions)
    if len(clean_positions) != 1:
        return False, f"{group_name}:clean_not_single_token(len={len(clean_positions)})", debug
    if len(control_positions) != 1:
        return False, f"{group_name}:control_not_single_token(len={len(control_positions)})", debug

    clean_span_list = clean_char_spans.get(group_name, [])
    control_span_list = control_char_spans.get(group_name, [])
    if len(clean_span_list) != 1 or len(control_span_list) != 1:
        return False, f"{group_name}:missing_char_span", debug
    clean_span = clean_span_list[0]
    control_span = control_span_list[0]

    clean_prefix = clean_context["prompt"][:clean_span[0]]
    control_prefix = control_context["prompt"][:control_span[0]]
    if clean_prefix != control_prefix:
        return False, f"{group_name}:prefix_mismatch", debug
    clean_suffix = clean_context["prompt"][clean_span[1]:]
    control_suffix = control_context["prompt"][control_span[1]:]
    if clean_suffix != control_suffix:
        return False, f"{group_name}:suffix_mismatch", debug

    clean_prompt_idx = clean_positions[0] - int(clean_context["prompt_start_in_full"])
    control_prompt_idx = control_positions[0] - int(control_context["prompt_start_in_full"])
    debug["clean_prompt_index"] = int(clean_prompt_idx)
    debug["control_prompt_index"] = int(control_prompt_idx)
    if clean_prompt_idx != control_prompt_idx:
        return False, f"{group_name}:prompt_index_mismatch(clean={clean_prompt_idx},control={control_prompt_idx})", debug

    return True, "ok", debug


def choose_aligned_single_field_control(
    clean_inputs: Dict[str, torch.Tensor],
    clean_question: str,
    clean_group_positions: Dict[str, List[int]],
    frames: Sequence[Any],
    states: List[Dict[str, Any]],
    clean_answer_text: str,
    field_name: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    parsed = tgi.parse_target_character_room(clean_question)
    if parsed is None:
        return None, f"{field_name}:question_parse_failed"
    clean_character, clean_room = parsed
    clean_answer = int(clean_answer_text)
    num_frames = len(frames)
    clean_seq_len = int(clean_inputs["input_ids"].shape[1])

    if field_name == "character":
        candidates = [char for char in tgi.extract_characters_from_states(states) if char != clean_character]
        sort_key = lambda value: (
            0 if len(processor.tokenizer(value, add_special_tokens=False)["input_ids"]) == 1 else 1,
            abs(len(str(value)) - len(clean_character)),
            str(value),
        )
    elif field_name == "room":
        candidates = [room for room in tgi.extract_rooms_from_states(states) if room != clean_room]
        sort_key = lambda value: (
            0 if len(processor.tokenizer(value, add_special_tokens=False)["input_ids"]) == 1 else 1,
            abs(len(str(value)) - len(clean_room)),
            str(value),
        )
    else:
        return None, f"{field_name}:unsupported_field"

    if not candidates:
        return None, f"{field_name}:no_candidates"

    failures: List[str] = []
    for candidate_value in sorted(candidates, key=sort_key):
        if field_name == "character":
            control_question = replace_character_in_question(clean_question, str(candidate_value))
            control_answer = tgi.count_steps_for_character_room(states, str(candidate_value), clean_room)
        else:
            control_question = replace_room_in_question(clean_question, str(candidate_value))
            control_answer = tgi.count_steps_for_character_room(states, clean_character, str(candidate_value))
        if control_question is None:
            failures.append(f"{candidate_value}:question_rewrite_failed")
            continue
        if int(control_answer) == clean_answer:
            failures.append(f"{candidate_value}:answer_unchanged")
            continue
        try:
            control_inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, control_question))
        except Exception as exc:
            failures.append(f"{candidate_value}:build_failed({exc})")
            continue
        if int(control_inputs["input_ids"].shape[1]) != clean_seq_len:
            failures.append(
                f"{candidate_value}:seq_len_mismatch(clean={clean_seq_len},control={int(control_inputs['input_ids'].shape[1])})"
            )
            continue
        control_group_positions, control_group_summaries, control_warnings = tgi.locate_group_token_positions(
            inputs=control_inputs,
            question=control_question,
            num_frames=num_frames,
        )
        ok, reason, debug = validate_text_control_alignment(
            clean_inputs=clean_inputs,
            clean_question=clean_question,
            clean_group_positions=clean_group_positions,
            control_inputs=control_inputs,
            control_question=control_question,
            control_group_positions=control_group_positions,
            group_name=field_name,
            num_frames=num_frames,
        )
        if not ok:
            failures.append(f"{candidate_value}:{reason}")
            continue
        return {
            "control_inputs": control_inputs,
            "control_question": control_question,
            "control_answer": str(control_answer),
            "control_value": str(candidate_value),
            "group_positions": control_group_positions,
            "group_summaries": control_group_summaries,
            "group_warnings": control_warnings,
            "alignment_debug": debug,
        }, "ok"

    return None, f"{field_name}:no_aligned_control({' ; '.join(failures[:8])})"


def build_group_specific_controls(
    sample_id: str,
    clean_inputs: Dict[str, torch.Tensor],
    frames: Sequence[Any],
    question: str,
    states: List[Dict[str, Any]],
    clean_answer_text: str,
    clean_group_positions: Dict[str, List[int]],
    corrupted_data_root: Path,
    evidence_frame_indices: Sequence[int],
) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Dict[str, str]]:
    issues: Dict[str, str] = {}
    corrupted_frames, corruption_issues = build_composite_corrupted_frames(
        sample_id=sample_id,
        clean_frames=frames,
        evidence_frame_indices=evidence_frame_indices,
        corrupted_data_root=corrupted_data_root,
    )
    if corrupted_frames is None:
        issues["wrong_frames"] = f"build_failed({corruption_issues})"
        return None, issues
    try:
        wrong_frames_inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(corrupted_frames, question))
    except Exception as exc:
        issues["wrong_frames"] = f"build_inputs_failed({exc})"
        return None, issues
    clean_seq_len = int(clean_inputs["input_ids"].shape[1])
    if int(wrong_frames_inputs["input_ids"].shape[1]) != clean_seq_len:
        issues["wrong_frames"] = (
            f"seq_len_mismatch(clean={clean_seq_len},control={int(wrong_frames_inputs['input_ids'].shape[1])})"
        )
        return None, issues

    character_control, character_reason = choose_aligned_single_field_control(
        clean_inputs=clean_inputs,
        clean_question=question,
        clean_group_positions=clean_group_positions,
        frames=frames,
        states=states,
        clean_answer_text=clean_answer_text,
        field_name="character",
    )
    if character_control is None:
        issues["wrong_character"] = character_reason
        return None, issues

    room_control, room_reason = choose_aligned_single_field_control(
        clean_inputs=clean_inputs,
        clean_question=question,
        clean_group_positions=clean_group_positions,
        frames=frames,
        states=states,
        clean_answer_text=clean_answer_text,
        field_name="room",
    )
    if room_control is None:
        issues["wrong_room"] = room_reason
        return None, issues

    prompt_len = int(clean_inputs["input_ids"].shape[1])
    last_token_position = prompt_len - 1
    if last_token_position < 0:
        issues["wrong_frames_last_token"] = "last_token:prompt_len_zero"
        return None, issues
    last_token_debug = {
        "group": "last_token",
        "clean_positions": [int(last_token_position)],
        "control_positions": [int(last_token_position)],
        "clean_prompt_index": int(last_token_position),
        "control_prompt_index": int(last_token_position),
    }
    character_control["group_positions"]["last_token"] = [int(last_token_position)]
    character_control.setdefault("alignment_debug", {})
    character_control["alignment_debug"]["last_token"] = dict(last_token_debug)
    room_control["group_positions"]["last_token"] = [int(last_token_position)]
    room_control.setdefault("alignment_debug", {})
    room_control["alignment_debug"]["last_token"] = dict(last_token_debug)

    return {
        "wrong_frames": {
            "control_inputs": wrong_frames_inputs,
            "control_question": question,
            "control_answer": None,
            "group_positions": {
                "frames": list(clean_group_positions.get("frames", [])),
                "last_token": [int(last_token_position)],
            },
            "alignment_debug": {"frames": {"aligned": True}, "last_token": dict(last_token_debug)},
        },
        "wrong_character": character_control,
        "wrong_room": room_control,
    }, issues


def compute_token_control_difference(
    lm: LanguageModel,
    layers: Any,
    clean_inputs: Dict[str, torch.Tensor],
    control_inputs: Dict[str, torch.Tensor],
    token_position: int,
    layer_idx: int,
) -> float:
    with torch.no_grad():
        with lm.trace(clean_inputs):
            clean_layer_saved = tgi._to_hidden_tensor(layers[layer_idx].output).save()
        with lm.trace(control_inputs):
            control_layer_saved = tgi._to_hidden_tensor(layers[layer_idx].output).save()
    clean_hidden = tgi._materialize_saved(clean_layer_saved)[0, int(token_position), :]
    control_hidden = tgi._materialize_saved(control_layer_saved)[0, int(token_position), :]
    return float(torch.norm(clean_hidden - control_hidden, p=2).item())


def choose_strongest_last_token_control(
    lm: LanguageModel,
    layers: Any,
    clean_inputs: Dict[str, torch.Tensor],
    control_sources: Dict[str, Dict[str, Any]],
    token_position: int,
) -> Tuple[str, Dict[str, float]]:
    if not layers:
        raise ValueError("layers must be non-empty")
    probe_layer_idx = len(layers) - 1
    scores: Dict[str, float] = {}
    for source_name in ("wrong_frames", "wrong_character", "wrong_room"):
        control_payload = control_sources[source_name]
        scores[source_name] = compute_token_control_difference(
            lm=lm,
            layers=layers,
            clean_inputs=clean_inputs,
            control_inputs=control_payload["control_inputs"],
            token_position=token_position,
            layer_idx=probe_layer_idx,
        )
    best_source = max(scores.items(), key=lambda item: item[1])[0]
    return best_source, scores


def locate_group_positions_and_metadata(
    inputs: Dict[str, torch.Tensor],
    question: str,
    num_frames: int,
    evidence_frame_indices: Sequence[int],
) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, Any]], List[str]]:
    group_positions, _, warnings = tgi.locate_group_token_positions(
        inputs=inputs,
        question=question,
        num_frames=num_frames,
    )
    full_input_ids = [int(tok) for tok in inputs["input_ids"][0].tolist()]

    frame_groups = image_token_groups(inputs["input_ids"][0], expected_num_frames=num_frames)
    metadata: Dict[str, Dict[str, Any]] = {}
    positions: Dict[str, List[int]] = {}

    evidence_positions: List[int] = []
    for frame_idx in evidence_frame_indices:
        if int(frame_idx) >= len(frame_groups):
            warnings.append(f"frames:missing_image_token_group(frame_idx={frame_idx})")
            continue
        evidence_positions.extend(int(pos) for pos in frame_groups[int(frame_idx)])
    if evidence_positions:
        positions["frames"] = sorted(evidence_positions)
        metadata["frames"] = {
            "positions": positions["frames"],
            "token_ids": [int(full_input_ids[pos]) for pos in positions["frames"]],
            "decoded_tokens": [
                processor.tokenizer.decode([int(full_input_ids[pos])], clean_up_tokenization_spaces=False)
                for pos in positions["frames"]
            ],
            "frame_indices": [int(frame_idx) for frame_idx in evidence_frame_indices],
        }
    else:
        warnings.append("frames:no_evidence_frame_token_positions")

    for group_name in ("character", "room"):
        token_positions = [int(pos) for pos in group_positions.get(group_name, [])]
        if not token_positions:
            warnings.append(f"{group_name}:missing_token_positions")
            continue
        positions[group_name] = token_positions
        metadata[group_name] = {
            "positions": token_positions,
            "token_ids": [int(full_input_ids[pos]) for pos in token_positions],
            "decoded_tokens": [
                processor.tokenizer.decode([int(full_input_ids[pos])], clean_up_tokenization_spaces=False)
                for pos in token_positions
            ],
        }

    prompt_len = int(inputs["input_ids"].shape[1])
    last_token_position = prompt_len - 1
    if last_token_position < 0:
        warnings.append("last_token:prompt_len_zero")
    else:
        positions["last_token"] = [int(last_token_position)]
        metadata["last_token"] = {
            "positions": [int(last_token_position)],
            "token_ids": [int(full_input_ids[last_token_position])],
            "decoded_tokens": [
                processor.tokenizer.decode([int(full_input_ids[last_token_position])], clean_up_tokenization_spaces=False)
            ],
        }

    return positions, metadata, warnings


def write_group_ld_report(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    lines: List[str] = []
    for sample in sample_metrics:
        lines.append(f"sample_id={sample['sample_id']}")
        lines.append(
            f"clean_answer_score={sample['clean_answer_score']:.8f}"
        )
        lines.append(f"groups={sample['active_groups']}")
        lines.append(f"group_token_counts={sample['group_token_counts']}")
        for layer_metrics in sample["layer_metrics"]["layers"]:
            lines.append(
                f"layer={layer_metrics['layer']} "
                f"groups={layer_metrics['groups']} "
                f"corrupted_score={[round(float(x), 8) for x in layer_metrics['corrupted_score']]} "
                f"r={[round(float(x), 8) for x in layer_metrics['r']]} "
                f"r_normalized={[round(float(x), 8) for x in layer_metrics['r_normalized']]} "
                f"total_importance={float(layer_metrics['total_importance']):.8f}"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_metrics.txt"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def build_aggregate_metrics(
    sample_metrics: List[Dict[str, Any]],
    num_layers: int,
    selected_layers: List[int],
    group_order: List[str],
) -> Dict[str, Any]:
    mean_importance_by_group: Dict[str, List[float]] = {group: [0.0 for _ in range(num_layers)] for group in group_order}
    mean_normalized_importance_by_group: Dict[str, List[float]] = {
        group: [0.0 for _ in range(num_layers)] for group in group_order
    }
    counts_by_group: Dict[str, List[int]] = {group: [0 for _ in range(num_layers)] for group in group_order}

    for sample in sample_metrics:
        for layer_metrics in sample["layer_metrics"]["layers"]:
            layer_idx = int(layer_metrics["layer"])
            groups = list(layer_metrics["groups"])
            raw_values = [float(x) for x in layer_metrics["r"]]
            norm_values = [float(x) for x in layer_metrics["r_normalized"]]
            for idx, group_name in enumerate(groups):
                mean_importance_by_group[group_name][layer_idx] += raw_values[idx]
                mean_normalized_importance_by_group[group_name][layer_idx] += norm_values[idx]
                counts_by_group[group_name][layer_idx] += 1

    for group_name in group_order:
        for layer_idx in selected_layers:
            count = counts_by_group[group_name][layer_idx]
            if count > 0:
                mean_importance_by_group[group_name][layer_idx] /= count
                mean_normalized_importance_by_group[group_name][layer_idx] /= count

    return {
        "groups": list(group_order),
        "selected_layers": list(selected_layers),
        "mean_importance_by_group": mean_importance_by_group,
        "mean_normalized_importance_by_group": mean_normalized_importance_by_group,
        "counts_by_group": counts_by_group,
    }


def plot_group_importance_lines(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    num_layers: int,
    selected_layers: List[int],
    group_order: List[str],
    seq_len_label: Optional[str] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping group-line plot: matplotlib is not available ({exc})")
        return None
    if num_layers <= 0 or not sample_metrics or not selected_layers:
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
    rng = tgi.random.Random(seed)
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
    title = "Mean Group Importance by Layer"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean importance")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    tick_step = max(1, tgi.math.ceil(len(selected_layers) / 32))
    xticks = selected_layers[::tick_step]
    if selected_layers[-1] not in xticks:
        xticks.append(selected_layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    suffix = f"_{seq_len_label}" if seq_len_label else ""
    plot_path = output_dir / f"group_importance_lines{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_aggregate_metrics_json(payload: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "aggregate_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    start_time = time.perf_counter()

    # This script compares evidence-frame patching against aligned text and
    # last-token controls on the same clean sample.
    ap = argparse.ArgumentParser(
        description=(
            "Compare layer-wise importance of evidence-frame tokens, character, room, and the last prompt "
            "token using the same input-corrupted control run and LD-based activation patching."
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
        help="Directory containing clean_scores.json for loading/updating the clean-answer cache. Defaults to --output.",
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--min_clean_correct_prob",
        type=float,
        default=0.4,
        help="Keep a sample only if the clean correct answer probability among valid numeric answers is at least this value.",
    )
    ap.add_argument("--lambda", dest="lambda_threshold", type=float, default=None)
    ap.add_argument(
        "--min_clean_ld",
        type=float,
        default=None,
        help="Backward-compatible alias for --min_clean_correct_prob.",
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
    ap.add_argument(
        "--include_groups",
        type=str,
        default="frames,character,room,last_token",
        help="Comma-separated subset of groups to patch. Supported: frames,character,room,last_token.",
    )
    ap.add_argument("--disable_plots", action="store_true")
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer")
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer")
    if args.min_clean_correct_prob < 0.0 or args.min_clean_correct_prob > 1.0:
        raise ValueError("--min_clean_correct_prob must be in [0, 1]")

    min_clean_correct_prob = float(args.min_clean_correct_prob)
    if args.lambda_threshold is not None:
        min_clean_correct_prob = float(args.lambda_threshold)
    elif args.min_clean_ld is not None:
        min_clean_correct_prob = float(args.min_clean_ld)

    data_root = Path(args.data_root)
    corrupted_data_root = Path(args.corrupted_root) if args.corrupted_root is not None else infer_corrupted_data_root(data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_ld_cache_dir = Path(args.clean_ld_cache_dir) if args.clean_ld_cache_dir is not None else output_dir
    clean_ld_cache_dir.mkdir(parents=True, exist_ok=True)
    clean_ld_cache_path = clean_ld_cache_dir / "clean_scores.json"
    clean_ld_cache = tgi.load_clean_score_cache(clean_ld_cache_path)
    cache_updates = 0

    seq_len_label = eval_utils.resolve_seq_len_label(data_root)

    lm = LanguageModel(base_model, tokenizer=processor.tokenizer)
    layers = get_layers(lm.model)
    num_layers = len(layers)
    selected_layers = tgi.parse_layer_selection(args.layers, num_layers=num_layers)
    include_groups = parse_include_groups(args.include_groups)

    sample_dirs = iter_sample_dirs(data_root)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under: {data_root}")

    processed_samples = 0
    sample_metrics: List[Dict[str, Any]] = []
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= int(args.limit):
            break

        # Start from clean samples the model already answers correctly with high confidence.
        try:
            sample_id, frames, question, states, answer = load_mmred_sample(sample_dir)
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_dir.name} skipped: load failure ({exc})")
            continue

        evidence_frame_indices = collect_evidence_frame_indices(question, states)
        if len(evidence_frame_indices) < 1:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: no evidence frames")
            continue

        try:
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, question))
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: failed to build clean inputs ({exc})")
            continue

        prompt_len = int(inputs["input_ids"].shape[1])
        a_star_text = str(answer).strip()
        try:
            a_star_ids = tgi.token_ids_of_answer(a_star_text)
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: invalid answer tokenization ({exc})")
            continue

        try:
            clean_answer_metrics, cache_was_updated = eval_utils.get_or_compute_clean_answer_metrics(
                cache=clean_ld_cache,
                sample_id=sample_id,
                num_frames=len(frames),
                answer_text=a_star_text,
                score_fn=lambda: tgi.score_valid_numeric_answers(
                    lm=lm,
                    inputs=inputs,
                    prompt_len=prompt_len,
                    num_frames=len(frames),
                ),
            )
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: clean scoring failed ({exc})")
            continue
        if cache_was_updated:
            cache_updates += 1

        clean_answer_score = float(clean_answer_metrics["clean_answer_score"])
        clean_correct_prob = float(clean_answer_metrics["clean_correct_prob"])
        clean_top1_correct = bool(clean_answer_metrics["clean_top1_correct"])
        best_answer_text = str(clean_answer_metrics["best_answer_text"])

        if not clean_top1_correct:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: clean top-1 is {best_answer_text!r}, "
                f"not correct answer {a_star_text!r}"
            )
            continue
        if clean_correct_prob < min_clean_correct_prob:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: "
                f"clean_correct_prob={clean_correct_prob:.4f} < threshold={min_clean_correct_prob:.4f}"
            )
            continue

        group_positions, group_metadata, group_warnings = locate_group_positions_and_metadata(
            inputs=inputs,
            question=question,
            num_frames=len(frames),
            evidence_frame_indices=evidence_frame_indices,
        )
        group_specific_controls, control_issues = build_group_specific_controls(
            sample_id=sample_id,
            clean_inputs=inputs,
            frames=frames,
            question=question,
            states=states,
            clean_answer_text=a_star_text,
            clean_group_positions=group_positions,
            corrupted_data_root=corrupted_data_root,
            evidence_frame_indices=evidence_frame_indices,
        )
        if group_specific_controls is None:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: failed to build aligned controls "
                f"(issues={control_issues})"
            )
            continue
        last_token_positions = [int(pos) for pos in group_positions.get("last_token", [])]
        if not last_token_positions:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: missing last_token positions "
                f"(warnings={group_warnings})"
            )
            continue
        try:
            last_token_control_source, last_token_control_scores = choose_strongest_last_token_control(
                lm=lm,
                layers=layers,
                clean_inputs=inputs,
                control_sources=group_specific_controls,
                token_position=int(last_token_positions[-1]),
            )
        except Exception as exc:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: failed to score last_token controls "
                f"({exc})"
            )
            continue

        groups_payload: List[Dict[str, Any]] = []
        group_token_counts: Dict[str, int] = {}
        skipped_groups: Dict[str, str] = {}
        for group_name in include_groups:
            token_positions = [int(pos) for pos in group_positions.get(group_name, [])]
            if not token_positions:
                skipped_groups[group_name] = "missing_token_positions"
                continue
            if group_name == "frames":
                control_source = group_specific_controls["wrong_frames"]
                control_positions = [int(pos) for pos in control_source["group_positions"].get("frames", [])]
            elif group_name == "character":
                control_source = group_specific_controls["wrong_character"]
                control_positions = [int(pos) for pos in control_source["group_positions"].get("character", [])]
            elif group_name == "room":
                control_source = group_specific_controls["wrong_room"]
                control_positions = [int(pos) for pos in control_source["group_positions"].get("room", [])]
            elif group_name == "last_token":
                control_source = group_specific_controls[last_token_control_source]
                control_positions = [int(pos) for pos in control_source["group_positions"].get("last_token", [])]
            else:
                skipped_groups[group_name] = "unsupported_group"
                continue
            if len(control_positions) != len(token_positions):
                skipped_groups[group_name] = (
                    f"token_count_mismatch(clean={len(token_positions)},control={len(control_positions)})"
                )
                continue
            groups_payload.append({
                "name": group_name,
                "clean_positions": token_positions,
                "control_positions": control_positions,
                "control_inputs": control_source["control_inputs"],
                "control_source": (
                    "wrong_frames"
                    if group_name in {"frames"}
                    else "wrong_character"
                    if group_name in {"character"}
                    else "wrong_room"
                ),
            })
            group_token_counts[group_name] = len(token_positions)

        if not groups_payload:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: no valid groups "
                f"(warnings={group_warnings}, skipped={skipped_groups})"
            )
            continue

        print(
            f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} clean_answer_score={clean_answer_score:.4f} "
            f"clean_correct_prob={clean_correct_prob:.4f} evidence_frames={evidence_frame_indices} "
            f"active_groups={[group['name'] for group in groups_payload]} batch_size={args.batch_size}"
        )
        print(f"  group token counts: {group_token_counts}")
        print(
            f"  controls: wrong_character={group_specific_controls['wrong_character']['control_value']!r} "
            f"wrong_room={group_specific_controls['wrong_room']['control_value']!r} "
            f"clean_answer={a_star_text!r} "
            f"wrong_character_answer={group_specific_controls['wrong_character']['control_answer']!r} "
            f"wrong_room_answer={group_specific_controls['wrong_room']['control_answer']!r} "
            f"last_token_source={last_token_control_source} "
            f"last_token_strengths={{{', '.join(f'{name}:{score:.4f}' for name, score in last_token_control_scores.items())}}}"
        )
        print(
            f"  alignment: character(clean_pos={group_specific_controls['wrong_character']['alignment_debug']['clean_positions']}, "
            f"control_pos={group_specific_controls['wrong_character']['alignment_debug']['control_positions']}, "
            f"clean_idx={group_specific_controls['wrong_character']['alignment_debug']['clean_prompt_index']}, "
            f"control_idx={group_specific_controls['wrong_character']['alignment_debug']['control_prompt_index']}, "
            f"passed=True) "
            f"room(clean_pos={group_specific_controls['wrong_room']['alignment_debug']['clean_positions']}, "
            f"control_pos={group_specific_controls['wrong_room']['alignment_debug']['control_positions']}, "
            f"clean_idx={group_specific_controls['wrong_room']['alignment_debug']['clean_prompt_index']}, "
            f"control_idx={group_specific_controls['wrong_room']['alignment_debug']['control_prompt_index']}, "
            f"passed=True) "
            f"last_token(clean_pos={group_specific_controls[last_token_control_source]['alignment_debug']['last_token']['clean_positions']}, "
            f"control_pos={group_specific_controls[last_token_control_source]['alignment_debug']['last_token']['control_positions']}, "
            f"clean_idx={group_specific_controls[last_token_control_source]['alignment_debug']['last_token']['clean_prompt_index']}, "
            f"control_idx={group_specific_controls[last_token_control_source]['alignment_debug']['last_token']['control_prompt_index']}, "
            f"passed=True)"
        )
        if skipped_groups:
            print(f"  skipped groups: {skipped_groups}")

        chunk_size = min(args.batch_size, len(groups_payload))
        group_chunks = [
            groups_payload[start:start + chunk_size]
            for start in range(0, len(groups_payload), chunk_size)
        ]

        # Build the batch layout once so each layer only runs the actual patching pass.
        chunk_data: List[Dict[str, Any]] = []
        try:
            for group_chunk in group_chunks:
                chunk_len = len(group_chunk)
                repeated_clean_inputs = tgi.repeat_inputs_for_batch(inputs, batch_size=chunk_len)
                clean_answer_chunk_inputs = tgi.append_answer_tokens_for_scoring(repeated_clean_inputs, a_star_ids)
                control_inputs_batch = tgi.concatenate_inputs_for_batch(
                    [group_entry["control_inputs"] for group_entry in group_chunk]
                )
                corrupted_answer_chunk_inputs = tgi.append_answer_tokens_for_scoring(control_inputs_batch, a_star_ids)
                chunk_data.append({
                    "groups": group_chunk,
                    "clean_answer_inputs": clean_answer_chunk_inputs,
                    "corrupted_answer_inputs": corrupted_answer_chunk_inputs,
                })
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: failed to build batched inputs ({exc})")
            continue

        per_layer_metrics: List[Dict[str, Any]] = []
        all_layer_corrupted_rows: List[Tuple[int, List[float]]] = []
        for layer_idx in selected_layers:
            per_group_corrupted_score: Dict[str, float] = {}
            per_group_importance: Dict[str, float] = {}
            per_group_normalized_importance: Dict[str, float] = {}

            for chunk_idx, packed in enumerate(chunk_data, start=1):
                group_chunk = packed["groups"]
                clean_positions_by_batch = [group["clean_positions"] for group in group_chunk]
                control_positions_by_batch = [group["control_positions"] for group in group_chunk]
                try:
                    corrupted_answer_scores = tgi.run_layer_multi_group_corrupted_sequence_logprob(
                        lm=lm,
                        layers=layers,
                        clean_batched_scoring_inputs=packed["clean_answer_inputs"],
                        control_batched_scoring_inputs=packed["corrupted_answer_inputs"],
                        layer_idx=layer_idx,
                        clean_token_positions_by_batch=clean_positions_by_batch,
                        control_token_positions_by_batch=control_positions_by_batch,
                        prompt_len=prompt_len,
                        answer_token_ids=a_star_ids,
                    )
                except Exception as exc:
                    print(
                        f"  layer={layer_idx} failed batched corruption forward "
                        f"(chunk {chunk_idx}/{len(chunk_data)}, {exc}); using clean score for this chunk"
                    )
                    for group in group_chunk:
                        group_name = group["name"]
                        per_group_corrupted_score[group_name] = clean_answer_score
                        per_group_importance[group_name] = 0.0
                        per_group_normalized_importance[group_name] = 0.0
                    continue

                for batch_idx, group in enumerate(group_chunk):
                    group_name = group["name"]
                    corrupted_score = float(corrupted_answer_scores[batch_idx].item())
                    importance = max(clean_answer_score - corrupted_score, 0.0)
                    token_count = max(1, len(group["clean_positions"]))
                    per_group_corrupted_score[group_name] = corrupted_score
                    per_group_importance[group_name] = importance
                    per_group_normalized_importance[group_name] = importance / float(token_count)

            layer_group_order = [group["name"] for group in groups_payload]
            corrupted_score_row = [per_group_corrupted_score.get(group_name, clean_answer_score) for group_name in layer_group_order]
            importance_row = [per_group_importance.get(group_name, 0.0) for group_name in layer_group_order]
            normalized_row = [
                per_group_normalized_importance.get(group_name, 0.0) for group_name in layer_group_order
            ]
            all_layer_corrupted_rows.append((layer_idx, list(corrupted_score_row)))

            total_importance = float(sum(importance_row))
            if total_importance > 0.0:
                probs = tgi.normalize_to_probabilities(importance_row)
                entropy_value = tgi.normalize_entropy(
                    tgi.entropy_from_probabilities(probs),
                    num_groups=len(layer_group_order),
                )
            else:
                probs = [0.0 for _ in importance_row]
                entropy_value = None

            per_layer_metrics.append({
                "layer": layer_idx,
                "groups": list(layer_group_order),
                "corrupted_score": corrupted_score_row,
                "r": importance_row,
                "r_normalized": normalized_row,
                "p": probs,
                "entropy": entropy_value,
                "total_importance": total_importance,
            })

        if all_layer_corrupted_rows:
            print("  Corrupted score table (rows=layers, columns=groups):")
            print(tgi.format_corrupted_score_table([group["name"] for group in groups_payload], all_layer_corrupted_rows))

        sample_metrics.append({
            "sample_id": sample_id,
            "question": question,
            "answer": answer,
            "clean_answer_score": clean_answer_score,
            "clean_correct_prob": clean_correct_prob,
            "clean_top1_correct": clean_top1_correct,
            "a_star_text": a_star_text,
            "a_star_ids": a_star_ids,
            "evidence_frames": [int(frame_idx) for frame_idx in evidence_frame_indices],
            "active_groups": [group["name"] for group in groups_payload],
            "group_token_counts": group_token_counts,
            "group_positions": group_positions,
            "group_metadata": group_metadata,
            "group_warnings": group_warnings,
            "control_debug": {
                "wrong_character": {
                    "control_value": group_specific_controls["wrong_character"]["control_value"],
                    "control_question": group_specific_controls["wrong_character"]["control_question"],
                    "control_answer": group_specific_controls["wrong_character"]["control_answer"],
                    "alignment": group_specific_controls["wrong_character"]["alignment_debug"],
                },
                "wrong_room": {
                    "control_value": group_specific_controls["wrong_room"]["control_value"],
                    "control_question": group_specific_controls["wrong_room"]["control_question"],
                    "control_answer": group_specific_controls["wrong_room"]["control_answer"],
                    "alignment": group_specific_controls["wrong_room"]["alignment_debug"],
                },
                "wrong_frames": {
                    "control_question": group_specific_controls["wrong_frames"]["control_question"],
                    "control_answer": group_specific_controls["wrong_frames"]["control_answer"],
                    "alignment": group_specific_controls["wrong_frames"]["alignment_debug"],
                },
                "last_token_selection": {
                    "selected_source": last_token_control_source,
                    "scores": last_token_control_scores,
                },
            },
            "skipped_groups": skipped_groups,
            "selected_layers": list(selected_layers),
            "selected_layers_spec": args.layers,
            "layer_metrics": {"layers": per_layer_metrics},
        })
        processed_samples += 1

    clean_ld_cache_path = clean_ld_cache_dir / "clean_scores.json"
    print(eval_utils.persist_clean_score_cache(clean_ld_cache_path, clean_ld_cache, cache_updates))

    text_report_path = write_group_ld_report(sample_metrics, output_dir)
    sample_json_path = tgi.write_metrics_json(sample_metrics, output_dir)
    aggregate_metrics = build_aggregate_metrics(
        sample_metrics,
        num_layers=num_layers,
        selected_layers=selected_layers,
        group_order=include_groups,
    )
    aggregate_payload = {
        "metadata": {
            "model_name": getattr(base_model.config, "_name_or_path", str(type(base_model).__name__)),
            "dataset_path": str(data_root),
            "corrupted_root": str(corrupted_data_root),
            "selected_layers": list(selected_layers),
            "selected_layers_spec": args.layers,
            "total_layer_count": int(num_layers),
            "corruption_method": _CORRUPTION_METHOD,
            "group_names": list(include_groups),
        },
        "aggregate": aggregate_metrics,
    }
    aggregate_json_path = write_aggregate_metrics_json(aggregate_payload, output_dir)
    print(f"Wrote sample metrics text report to: {text_report_path}")
    print(f"Wrote sample metrics JSON to: {sample_json_path}")
    print(f"Wrote aggregate metrics JSON to: {aggregate_json_path}")
    print(
        f"Processed {processed_samples} samples "
        f"(target limit={int(args.limit)}, min_clean_correct_prob={min_clean_correct_prob:.4f})."
    )
    tgi.print_group_summary(include_groups, sample_metrics)

    if not args.disable_plots:
        lines_path = plot_group_importance_lines(
            sample_metrics,
            output_dir,
            num_layers=num_layers,
            selected_layers=selected_layers,
            group_order=include_groups,
            seq_len_label=seq_len_label,
        )
        if lines_path is not None:
            print(f"Wrote group-importance lines plot to: {lines_path}")

    elapsed = time.perf_counter() - start_time
    print(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()

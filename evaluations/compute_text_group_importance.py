import argparse
import json
import math
import random
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

from models.model import get_layers, model as base_model, processor

try:
    from evaluations.utils import iter_sample_dirs, load_mmred_sample
except ModuleNotFoundError:
    from utils import iter_sample_dirs, load_mmred_sample

_STEPS_IN_ROOM_RE = re.compile(
    r"How many steps did\s+([A-Za-z]+)\s+spend in\s+the\s+([A-Za-z]+)",
    flags=re.IGNORECASE,
)
_QUESTION_MARKER = "Question:"
_ANSWER_MARKER = "Answer:"
_QUESTION_OPERATOR = "How many steps did"
_QUESTION_RELATION = "spend in the"
_ALL_GROUPS = [
    "character",
    "room",
    "question_operator",
    "question_relation",
    "question_marker",
    "answer_marker",
    "question_punct",
    "instruction_context",
    "instruction_output_rule",
    "assistant_prefix",
    "assistant_prefix_token_0",
    "assistant_prefix_token_1",
    "assistant_prefix_token_2",
]


def token_ids_of_answer(answer_text: str) -> List[int]:
    ids = processor.tokenizer.encode(str(answer_text).strip(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"Answer text tokenized to empty: {answer_text!r}")
    return [int(tok_id) for tok_id in ids]


def build_prompt(question: str, num_frames: int) -> str:
    return (
        f"You will be shown {num_frames} frames describing steps in a house.\n"
        f"Respond with a single integer from 0 to {num_frames} (0 is allowed). Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )


def build_inputs_from_prompt(frames: Sequence[Any], prompt: str) -> Dict[str, torch.Tensor]:
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": im} for im in frames] +
            [{"type": "text", "text": prompt}]
        ),
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return dict(inputs)


def build_inputs(frames: Sequence[Any], question: str) -> Dict[str, torch.Tensor]:
    return build_inputs_from_prompt(frames, build_prompt(question, num_frames=len(frames)))


def move_inputs_to_model_device(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    device = next(base_model.parameters()).device
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def repeat_inputs_for_batch(
    inputs: Dict[str, torch.Tensor],
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    if batch_size <= 1:
        return inputs

    repeated: Dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            repeated[key] = value
            continue
        if value.dim() == 0:
            repeated[key] = value.repeat(batch_size)
            continue
        if int(value.shape[0]) == 1:
            repeated[key] = value.repeat(batch_size, *([1] * (value.dim() - 1)))
            continue
        if key in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}:
            repeated[key] = torch.cat([value] * batch_size, dim=0)
            continue
        raise ValueError(f"Cannot batch-repeat input {key!r} with shape={tuple(value.shape)}")
    return repeated


def concatenate_inputs_for_batch(inputs_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not inputs_list:
        raise ValueError("inputs_list must be non-empty")
    if len(inputs_list) == 1:
        return inputs_list[0]

    out: Dict[str, torch.Tensor] = {}
    keys = list(inputs_list[0].keys())
    for key in keys:
        values = [inputs[key] for inputs in inputs_list]
        first_value = values[0]

        if not torch.is_tensor(first_value):
            out[key] = first_value
            continue
        if first_value.dim() == 0:
            out[key] = torch.stack(values, dim=0)
            continue
        if int(first_value.shape[0]) == 1:
            out[key] = torch.cat(values, dim=0)
            continue
        if key in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}:
            out[key] = torch.cat(values, dim=0)
            continue
        raise ValueError(f"Cannot concatenate input {key!r} with shape={tuple(first_value.shape)}")
    return out


def parse_target_character_room(question_text: str) -> Optional[Tuple[str, str]]:
    match = _STEPS_IN_ROOM_RE.search(question_text)
    if not match:
        return None
    character = match.group(1).strip()
    room = match.group(2).strip()
    normalized_room = room[:1].upper() + room[1:].lower() if room else room
    return character, normalized_room


def parse_target_character_room_with_spans(
    question_text: str,
) -> Optional[Tuple[str, str, Tuple[int, int], Tuple[int, int]]]:
    match = _STEPS_IN_ROOM_RE.search(question_text)
    if not match:
        return None
    character = match.group(1).strip()
    room = match.group(2).strip()
    normalized_room = room[:1].upper() + room[1:].lower() if room else room
    return character, normalized_room, match.span(1), match.span(2)


def rooms_to_room2chars(rooms: Dict[str, Any]) -> Dict[str, List[str]]:
    if not isinstance(rooms, dict):
        return {}
    if any(isinstance(value, list) for value in rooms.values()):
        room_to_chars: Dict[str, List[str]] = {}
        for room_name, chars in rooms.items():
            if not isinstance(room_name, str):
                continue
            normalized_room = room_name[:1].upper() + room_name[1:].lower() if room_name else room_name
            room_to_chars.setdefault(normalized_room, [])
            if isinstance(chars, list):
                room_to_chars[normalized_room].extend(str(char) for char in chars)
        return {room: sorted(set(chars)) for room, chars in room_to_chars.items()}

    room_to_chars: Dict[str, List[str]] = {}
    for char_name, room_name in rooms.items():
        if not isinstance(room_name, str):
            continue
        normalized_room = room_name[:1].upper() + room_name[1:].lower()
        room_to_chars.setdefault(normalized_room, []).append(str(char_name))
    return {room: sorted(set(chars)) for room, chars in room_to_chars.items()}


def extract_characters_from_states(states: List[Dict[str, Any]]) -> List[str]:
    chars: set[str] = set()
    for state in states:
        step_rooms = state.get("rooms", {}) if isinstance(state, dict) else {}
        room_to_chars = rooms_to_room2chars(step_rooms)
        for room_chars in room_to_chars.values():
            chars.update(str(c) for c in room_chars)
    return sorted(chars)


def extract_rooms_from_states(states: List[Dict[str, Any]]) -> List[str]:
    rooms: set[str] = set()
    for state in states:
        step_rooms = state.get("rooms", {}) if isinstance(state, dict) else {}
        room_to_chars = rooms_to_room2chars(step_rooms)
        rooms.update(str(r) for r in room_to_chars.keys())
    return sorted(rooms)


def count_steps_for_character_room(states: List[Dict[str, Any]], character: str, room: str) -> int:
    target_room = room[:1].upper() + room[1:].lower() if room else room
    count = 0
    for state in states:
        step_rooms = state.get("rooms", {}) if isinstance(state, dict) else {}
        room_to_chars = rooms_to_room2chars(step_rooms)
        if character in room_to_chars.get(target_room, []):
            count += 1
    return count


def replace_character_and_room_in_question(question: str, new_character: str, new_room: str) -> Optional[str]:
    parsed = parse_target_character_room_with_spans(question)
    if parsed is None:
        return None
    _, _, char_span, room_span = parsed
    if char_span[1] > room_span[0]:
        return None
    return (
        question[:char_span[0]] + str(new_character) +
        question[char_span[1]:room_span[0]] + str(new_room) +
        question[room_span[1]:]
    )


def _normalize_summary_text(text: str) -> str:
    compact = " ".join(str(text).split())
    return compact if compact else "<whitespace>"


def summarize_token_positions(
    input_ids: Sequence[int],
    token_positions: Sequence[int],
    max_items: int = 16,
) -> str:
    pieces: List[str] = []
    for pos in list(token_positions)[:max_items]:
        if pos < 0 or pos >= len(input_ids):
            continue
        token_id = int(input_ids[pos])
        try:
            token_text = processor.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except Exception:
            token_text = f"<id:{token_id}>"
        pieces.append(repr(token_text))
    if not pieces:
        return "<none>"
    if len(token_positions) > max_items:
        pieces.append("...")
    return " ".join(pieces)


def find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    last_start = len(haystack) - len(needle) + 1
    for start in range(last_start):
        if haystack[start:start + len(needle)] == needle:
            return start
    return None


def tokenize_with_offsets_if_available(text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
    try:
        tokenized = processor.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = [int(tok) for tok in tokenized["input_ids"]]
        offsets_raw = tokenized.get("offset_mapping")
        if offsets_raw is None:
            return token_ids, None
        offsets = [(int(pair[0]), int(pair[1])) for pair in offsets_raw]
        return token_ids, offsets
    except Exception:
        tokenized = processor.tokenizer(text, add_special_tokens=False)
        return [int(tok) for tok in tokenized["input_ids"]], None


def _token_positions_by_tokenized_substring(
    prompt_token_ids: List[int],
    prompt_text: str,
    substring_text: str,
    substring_char_start: int,
    substring_char_end: int,
) -> List[int]:
    prefix_text = prompt_text[:substring_char_start]
    prefix_ids = processor.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    substring_ids = processor.tokenizer(substring_text, add_special_tokens=False)["input_ids"]
    if not substring_ids:
        return []

    start_hint = len(prefix_ids)
    if start_hint + len(substring_ids) <= len(prompt_token_ids):
        if prompt_token_ids[start_hint:start_hint + len(substring_ids)] == substring_ids:
            return list(range(start_hint, start_hint + len(substring_ids)))

    matches: List[int] = []
    last_start = len(prompt_token_ids) - len(substring_ids) + 1
    for start in range(max(0, last_start)):
        if prompt_token_ids[start:start + len(substring_ids)] == substring_ids:
            matches.append(start)
    if not matches:
        return []
    best_start = min(matches, key=lambda s: abs(s - start_hint))
    return list(range(best_start, best_start + len(substring_ids)))


def _char_span_to_token_positions_with_offsets(
    token_offsets: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    out: List[int] = []
    for idx, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_end <= tok_start:
            continue
        if tok_end <= char_start or tok_start >= char_end:
            continue
        out.append(idx)
    return out


def _char_span_to_token_positions_by_prefix_lengths(
    prompt_text: str,
    char_start: int,
    char_end: int,
    prompt_token_count: int,
) -> List[int]:
    if char_start < 0 or char_end <= char_start or char_end > len(prompt_text):
        return []
    prefix_start_ids = processor.tokenizer(prompt_text[:char_start], add_special_tokens=False)["input_ids"]
    prefix_end_ids = processor.tokenizer(prompt_text[:char_end], add_special_tokens=False)["input_ids"]
    tok_start = int(len(prefix_start_ids))
    tok_end = int(len(prefix_end_ids))
    if tok_end <= tok_start:
        return []
    tok_start = max(0, min(tok_start, prompt_token_count))
    tok_end = max(0, min(tok_end, prompt_token_count))
    return list(range(tok_start, tok_end))


def _char_span_to_token_positions_allow_leading_space(
    prompt_text: str,
    char_start: int,
    char_end: int,
    prompt_token_count: int,
) -> List[int]:
    pos = _char_span_to_token_positions_by_prefix_lengths(
        prompt_text=prompt_text,
        char_start=char_start,
        char_end=char_end,
        prompt_token_count=prompt_token_count,
    )
    if pos:
        return pos

    # Some tokenizers merge the preceding space into the word token, so the
    # bare name/room span can map to zero tokens. In that case, retry with one
    # leading whitespace character included.
    if char_start > 0 and prompt_text[char_start - 1].isspace():
        pos = _char_span_to_token_positions_by_prefix_lengths(
            prompt_text=prompt_text,
            char_start=char_start - 1,
            char_end=char_end,
            prompt_token_count=prompt_token_count,
        )
        if pos:
            return pos
    return []


def build_prompt_group_char_spans(
    question: str,
    num_frames: int,
) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, str], List[str]]:
    skipped_reasons: List[str] = []
    parsed = parse_target_character_room_with_spans(question)
    if parsed is None:
        return {}, {}, ["question_parse_failed"]

    _, _, character_span_q, room_span_q = parsed
    prompt = build_prompt(question, num_frames=num_frames)
    line1 = f"You will be shown {num_frames} frames describing steps in a house."
    line2 = f"Respond with a single integer from 0 to {num_frames} (0 is allowed). Output only the integer."

    instruction_context_start = prompt.find(line1)
    instruction_output_start = prompt.find(line2)
    question_marker_start = prompt.find(_QUESTION_MARKER)
    answer_marker_start = prompt.rfind(_ANSWER_MARKER)

    if instruction_context_start < 0:
        skipped_reasons.append("instruction_context_not_found")
    if instruction_output_start < 0:
        skipped_reasons.append("instruction_output_rule_not_found")
    if question_marker_start < 0:
        skipped_reasons.append("question_marker_not_found")
        return {}, {}, skipped_reasons
    if answer_marker_start < 0:
        skipped_reasons.append("answer_marker_not_found")
        return {}, {}, skipped_reasons

    question_text_start = question_marker_start + len("Question: ")
    question_text_end = question_text_start + len(question)
    if question_text_end > len(prompt):
        return {}, {}, skipped_reasons + ["question_span_out_of_prompt"]

    question_operator_start_q = question.lower().find(_QUESTION_OPERATOR.lower())
    if question_operator_start_q < 0:
        skipped_reasons.append("question_operator_not_found")

    relation_start_q = character_span_q[1]
    relation_end_q = room_span_q[0]
    while relation_start_q < relation_end_q and question[relation_start_q].isspace():
        relation_start_q += 1
    while relation_end_q > relation_start_q and question[relation_end_q - 1].isspace():
        relation_end_q -= 1

    char_spans: Dict[str, List[Tuple[int, int]]] = {
        "character": [(question_text_start + character_span_q[0], question_text_start + character_span_q[1])],
        "room": [(question_text_start + room_span_q[0], question_text_start + room_span_q[1])],
        "question_marker": [(question_marker_start, question_marker_start + len(_QUESTION_MARKER))],
        "answer_marker": [(answer_marker_start, answer_marker_start + len(_ANSWER_MARKER))],
    }
    summaries: Dict[str, str] = {
        "character": _normalize_summary_text(question[character_span_q[0]:character_span_q[1]]),
        "room": _normalize_summary_text(question[room_span_q[0]:room_span_q[1]]),
        "question_marker": _QUESTION_MARKER,
        "answer_marker": _ANSWER_MARKER,
    }

    if question_operator_start_q >= 0:
        char_spans["question_operator"] = [(
            question_text_start + question_operator_start_q,
            question_text_start + question_operator_start_q + len(_QUESTION_OPERATOR),
        )]
        summaries["question_operator"] = _QUESTION_OPERATOR
    if relation_end_q > relation_start_q:
        char_spans["question_relation"] = [(
            question_text_start + relation_start_q,
            question_text_start + relation_end_q,
        )]
        summaries["question_relation"] = _normalize_summary_text(question[relation_start_q:relation_end_q])
    else:
        skipped_reasons.append("question_relation_not_found")
    if instruction_context_start >= 0:
        char_spans["instruction_context"] = [(
            instruction_context_start,
            instruction_context_start + len(line1),
        )]
        summaries["instruction_context"] = _normalize_summary_text(line1)
    if instruction_output_start >= 0:
        char_spans["instruction_output_rule"] = [(
            instruction_output_start,
            instruction_output_start + len(line2),
        )]
        summaries["instruction_output_rule"] = _normalize_summary_text(line2)
    if question.endswith("?"):
        char_spans["question_punct"] = [(
            question_text_start + len(question) - 1,
            question_text_start + len(question),
        )]
        summaries["question_punct"] = "?"
    else:
        skipped_reasons.append("question_punct_not_found")

    return char_spans, summaries, skipped_reasons


def locate_group_token_positions(
    inputs: Dict[str, torch.Tensor],
    question: str,
    num_frames: int,
) -> Tuple[Dict[str, List[int]], Dict[str, str], List[str]]:
    prompt = build_prompt(question, num_frames=num_frames)
    group_char_spans_prompt, group_summaries, skipped_reasons = build_prompt_group_char_spans(
        question=question,
        num_frames=num_frames,
    )
    if not group_char_spans_prompt:
        return {}, {}, skipped_reasons

    prompt_token_ids, prompt_offsets = tokenize_with_offsets_if_available(prompt)
    if not prompt_token_ids:
        return {}, group_summaries, ["prompt_tokenization_failed"]

    full_input_ids = [int(tok) for tok in inputs["input_ids"][0].tolist()]
    prompt_start_in_full = find_subsequence(full_input_ids, prompt_token_ids)
    if prompt_start_in_full is None:
        return {}, group_summaries, ["prompt_subsequence_not_found_in_input_ids"]

    token_positions_full: Dict[str, List[int]] = {}
    for group_name, spans in group_char_spans_prompt.items():
        all_positions: List[int] = []
        for char_start, char_end in spans:
            if group_name in {"character", "room"}:
                pos = _char_span_to_token_positions_allow_leading_space(
                    prompt_text=prompt,
                    char_start=char_start,
                    char_end=char_end,
                    prompt_token_count=len(prompt_token_ids),
                )
            else:
                pos = _char_span_to_token_positions_by_prefix_lengths(
                    prompt_text=prompt,
                    char_start=char_start,
                    char_end=char_end,
                    prompt_token_count=len(prompt_token_ids),
                )
            if not pos and prompt_offsets is not None:
                pos = _char_span_to_token_positions_with_offsets(prompt_offsets, char_start, char_end)
            if not pos:
                substring_text = prompt[char_start:char_end]
                pos = _token_positions_by_tokenized_substring(
                    prompt_token_ids=prompt_token_ids,
                    prompt_text=prompt,
                    substring_text=substring_text,
                    substring_char_start=char_start,
                    substring_char_end=char_end,
                )
            all_positions.extend(int(x) for x in pos)

        if not all_positions:
            skipped_reasons.append(f"{group_name}:token_span_not_found")
            continue
        token_positions_full[group_name] = [prompt_start_in_full + pos for pos in sorted(set(all_positions))]

    im_start_token_id = getattr(processor.tokenizer, "im_start_id", None)
    if im_start_token_id is None:
        try:
            im_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        except Exception:
            im_start_token_id = None
    prompt_end_in_full = prompt_start_in_full + len(prompt_token_ids)
    assistant_prefix_positions: List[int] = []
    if im_start_token_id is not None:
        candidate_starts = [
            idx for idx in range(prompt_end_in_full, len(full_input_ids))
            if full_input_ids[idx] == int(im_start_token_id)
        ]
        if candidate_starts:
            assistant_prefix_positions = list(range(candidate_starts[-1], len(full_input_ids)))
    if not assistant_prefix_positions:
        skipped_reasons.append("assistant_prefix:not_found")
    else:
        token_positions_full["assistant_prefix"] = assistant_prefix_positions
        group_summaries["assistant_prefix"] = summarize_token_positions(full_input_ids, assistant_prefix_positions)
        if len(assistant_prefix_positions) == 3:
            for token_idx, position in enumerate(assistant_prefix_positions):
                group_name = f"assistant_prefix_token_{token_idx}"
                token_positions_full[group_name] = [int(position)]
                group_summaries[group_name] = summarize_token_positions(full_input_ids, [int(position)])
        else:
            for token_idx in range(3):
                skipped_reasons.append(
                    f"assistant_prefix_token_{token_idx}:expected_len_3(found={len(assistant_prefix_positions)})"
                )

    return token_positions_full, group_summaries, skipped_reasons


def append_answer_tokens_for_scoring(
    inputs: Dict[str, torch.Tensor],
    answer_token_ids: List[int],
) -> Dict[str, torch.Tensor]:
    if not answer_token_ids:
        raise ValueError("answer_token_ids must be non-empty")

    input_ids = inputs["input_ids"]
    if input_ids.dim() != 2:
        raise ValueError(f"Expected input_ids to be rank-2, got shape={tuple(input_ids.shape)}")

    batch_size = int(input_ids.shape[0])
    answer_tokens = torch.tensor(
        answer_token_ids,
        dtype=input_ids.dtype,
        device=input_ids.device,
    ).unsqueeze(0).repeat(batch_size, 1)

    scored_inputs = dict(inputs)
    scored_inputs["input_ids"] = torch.cat([input_ids, answer_tokens], dim=1)

    if "attention_mask" in inputs and torch.is_tensor(inputs["attention_mask"]):
        attention_mask = inputs["attention_mask"]
        if attention_mask.dim() != 2 or int(attention_mask.shape[0]) != batch_size:
            raise ValueError(
                f"Expected attention_mask shape to be (batch, seq), got {tuple(attention_mask.shape)}"
            )
        suffix_attention = torch.ones(
            (batch_size, len(answer_token_ids)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        scored_inputs["attention_mask"] = torch.cat([attention_mask, suffix_attention], dim=1)
    return scored_inputs


def sequence_logprob_from_logits(
    logits: torch.Tensor,
    prompt_len: int,
    answer_token_ids: List[int],
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"Expected logits rank-3 [batch, seq, vocab], got {tuple(logits.shape)}")
    if prompt_len <= 0:
        raise ValueError("prompt_len must be >= 1")
    if not answer_token_ids:
        raise ValueError("answer_token_ids must be non-empty")

    batch_size = int(logits.shape[0])
    device = logits.device
    answer_len = len(answer_token_ids)
    token_positions = torch.arange(answer_len, device=device, dtype=torch.long) + (prompt_len - 1)
    target_token_ids = torch.tensor(answer_token_ids, device=device, dtype=torch.long).unsqueeze(0)
    target_token_ids = target_token_ids.repeat(batch_size, 1)

    selected_logits = logits[:, token_positions, :]
    log_probs = torch.log_softmax(selected_logits, dim=-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)
    return target_log_probs.sum(dim=1)


def _materialize_saved(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


def _to_hidden_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x:
        return _to_hidden_tensor(x[0])
    raise TypeError(f"Unsupported layer output type for corruption: {type(x)}")


def run_clean_sequence_logprob(
    lm: LanguageModel,
    scoring_inputs: Dict[str, torch.Tensor],
    prompt_len: int,
    answer_token_ids: List[int],
) -> float:
    with torch.inference_mode():
        with lm.trace(scoring_inputs):
            saved_logits = lm.output.logits.save()
    logits = _materialize_saved(saved_logits)
    scores = sequence_logprob_from_logits(logits, prompt_len=prompt_len, answer_token_ids=answer_token_ids)
    return float(scores[0].item())


def run_layer_multi_group_corrupted_sequence_logprob(
    lm: LanguageModel,
    layers: Any,
    clean_batched_scoring_inputs: Dict[str, torch.Tensor],
    control_batched_scoring_inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    clean_token_positions_by_batch: List[List[int]],
    control_token_positions_by_batch: List[List[int]],
    prompt_len: int,
    answer_token_ids: List[int],
) -> torch.Tensor:
    with torch.no_grad():
        with lm.trace(control_batched_scoring_inputs):
            control_layer_saved = _to_hidden_tensor(layers[layer_idx].output).save()

        with lm.trace(clean_batched_scoring_inputs):
            clean_layer_out = _to_hidden_tensor(layers[layer_idx].output)
            control_layer_out = _materialize_saved(control_layer_saved)

            for batch_idx, clean_positions in enumerate(clean_token_positions_by_batch):
                control_positions = control_token_positions_by_batch[batch_idx]
                if not clean_positions:
                    continue
                if len(clean_positions) != len(control_positions):
                    raise ValueError(
                        f"Token-count mismatch for patching at batch_idx={batch_idx}: "
                        f"clean={len(clean_positions)} control={len(control_positions)}"
                    )
                clean_layer_out[batch_idx, clean_positions, :] = control_layer_out[
                    batch_idx, control_positions, :
                ]

            saved_logits = lm.output.logits.save()

    logits = _materialize_saved(saved_logits)
    return sequence_logprob_from_logits(logits, prompt_len=prompt_len, answer_token_ids=answer_token_ids)


def normalize_to_probabilities(values: List[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0.0:
        return [0.0 for _ in values]
    return [float(v) / total for v in values]


def entropy_from_probabilities(probs: List[float]) -> float:
    return -sum(float(p) * math.log(float(p)) for p in probs if p > 0.0)


def normalize_entropy(entropy: float, num_groups: int) -> float:
    if num_groups <= 1:
        return 0.0
    return float(entropy / math.log(num_groups))


def format_corrupted_score_table(group_names: List[str], layer_rows: List[Tuple[int, List[float]]]) -> str:
    if not group_names or not layer_rows:
        return "<none>"
    cell_width = 12
    header = "layer".ljust(7) + " ".join(name[:cell_width].center(cell_width) for name in group_names)
    rows = [header]
    for layer_idx, row in layer_rows:
        values = " ".join(f"{value:.4f}".center(cell_width) for value in row)
        rows.append(f"{str(layer_idx).ljust(7)}{values}")
    return "\n".join(rows)


def _fixed_group_order_from_include(include_groups: List[str]) -> List[str]:
    return [name for name in _ALL_GROUPS if name in include_groups]


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
        layer_totals = {
            int(layer_metrics["layer"]): float(layer_metrics["total_importance"])
            for layer_metrics in sample["layer_metrics"]["layers"]
        }
        for layer_idx in range(num_layers):
            per_layer_values[layer_idx].append(layer_totals.get(layer_idx, 0.0))

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
    ax.fill_between(layers, mean_lo_values, mean_hi_values, color="#2ca02c", alpha=0.2, label="Mean 95% CI")
    title = "Mean Total Text-Group Importance by Layer"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Total importance")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    fig.tight_layout()

    suffix = f"_{seq_len_label}" if seq_len_label else ""
    plot_path = output_dir / f"total_importance_summary{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_layer_invalidity_rates(
    layer_sampled_counts: List[int],
    layer_invalid_counts: List[int],
    output_dir: Path,
    seq_len_label: Optional[str] = None,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping layer-invalidity plot: matplotlib is not available ({exc})")
        return None
    if not layer_sampled_counts or not any(layer_sampled_counts):
        return None

    xs = list(range(len(layer_sampled_counts)))
    ys = [
        (float(layer_invalid_counts[idx]) / float(layer_sampled_counts[idx]))
        if layer_sampled_counts[idx] > 0 else 0.0
        for idx in xs
    ]
    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    ax.plot(xs, ys, color="#ff7f0e", linewidth=2.2)
    title = "Layer Invalidity Rate"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Invalidity rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.tight_layout()

    suffix = f"_{seq_len_label}" if seq_len_label else ""
    plot_path = output_dir / f"layer_invalidity_rate{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_group_importance_heatmap(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    num_layers: int,
    include_groups: List[str],
    seq_len_label: Optional[str] = None,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping group-heatmap plot: matplotlib is not available ({exc})")
        return None
    group_order = _fixed_group_order_from_include(include_groups)
    if num_layers <= 0 or not group_order or not sample_metrics:
        return None

    sums = [[0.0 for _ in group_order] for _ in range(num_layers)]
    counts = [[0 for _ in group_order] for _ in range(num_layers)]
    for sample in sample_metrics:
        for layer_metrics in sample["layer_metrics"]["layers"]:
            layer_idx = int(layer_metrics["layer"])
            groups = list(layer_metrics["groups"])
            importance_values = [float(x) for x in layer_metrics["r"]]
            by_group = {groups[idx]: importance_values[idx] for idx in range(min(len(groups), len(importance_values)))}
            for g_idx, g_name in enumerate(group_order):
                if g_name in by_group:
                    sums[layer_idx][g_idx] += by_group[g_name]
                    counts[layer_idx][g_idx] += 1

    means = [
        [
            (sums[layer_idx][g_idx] / counts[layer_idx][g_idx]) if counts[layer_idx][g_idx] > 0 else 0.0
            for g_idx in range(len(group_order))
        ]
        for layer_idx in range(num_layers)
    ]

    fig, ax = plt.subplots(figsize=(9, max(4.5, num_layers * 0.18)), dpi=140)
    image = ax.imshow(means, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(list(range(len(group_order))))
    ax.set_xticklabels(group_order, rotation=30, ha="right")
    tick_step = max(1, math.ceil(num_layers / 32))
    y_ticks = list(range(0, num_layers, tick_step))
    if (num_layers - 1) not in y_ticks:
        y_ticks.append(num_layers - 1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks])
    ax.set_ylabel("Layer")
    ax.set_xlabel("Text group")
    title = "Mean Text-Group Importance Heatmap"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=12, pad=10)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean importance", rotation=90)
    fig.tight_layout()

    suffix = f"_{seq_len_label}" if seq_len_label else ""
    plot_path = output_dir / f"group_importance_heatmap{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_group_importance_lines(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    num_layers: int,
    include_groups: List[str],
    seq_len_label: Optional[str] = None,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping group-line plot: matplotlib is not available ({exc})")
        return None
    group_order = _fixed_group_order_from_include(include_groups)
    if num_layers <= 0 or not group_order or not sample_metrics:
        return None

    per_group_per_layer: Dict[str, List[float]] = {g: [0.0 for _ in range(num_layers)] for g in group_order}
    per_group_per_layer_counts: Dict[str, List[int]] = {g: [0 for _ in range(num_layers)] for g in group_order}
    for sample in sample_metrics:
        for layer_metrics in sample["layer_metrics"]["layers"]:
            layer_idx = int(layer_metrics["layer"])
            groups = list(layer_metrics["groups"])
            r_vals = [float(x) for x in layer_metrics["r"]]
            by_group = {groups[idx]: r_vals[idx] for idx in range(min(len(groups), len(r_vals)))}
            for g in group_order:
                if g in by_group:
                    per_group_per_layer[g][layer_idx] += by_group[g]
                    per_group_per_layer_counts[g][layer_idx] += 1

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    for g in group_order:
        mean_vals = []
        for layer_idx in range(num_layers):
            count = per_group_per_layer_counts[g][layer_idx]
            mean_vals.append((per_group_per_layer[g][layer_idx] / count) if count > 0 else 0.0)
        ax.plot(range(num_layers), mean_vals, linewidth=2.0, label=g)
    title = "Mean Text-Group Importance by Layer"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean importance")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    tick_step = max(1, math.ceil(num_layers / 32))
    xticks = list(range(0, num_layers, tick_step))
    if (num_layers - 1) not in xticks:
        xticks.append(num_layers - 1)
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    suffix = f"_{seq_len_label}" if seq_len_label else ""
    plot_path = output_dir / f"group_importance_lines{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_text_group_report(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    lines: List[str] = []
    for sample in sample_metrics:
        lines.append(f"sample_id={sample['sample_id']}")
        lines.append(f"question={sample['question']}")
        lines.append(f"answer={sample['answer']}")
        lines.append(
            f"clean_answer_score={sample['clean_answer_score']:.8f} "
            f"clean_correct_prob={sample['clean_correct_prob']:.8f} "
            f"clean_top1_correct={sample['clean_top1_correct']}"
        )
        lines.append(f"active_groups={sample['active_groups']}")
        lines.append(f"group_token_counts={sample['group_token_counts']}")
        if sample["skipped_groups"]:
            lines.append(f"skipped_groups={sample['skipped_groups']}")
        for layer_metrics in sample["layer_metrics"]["layers"]:
            lines.append(
                f"layer={layer_metrics['layer']} "
                f"groups={layer_metrics['groups']} "
                f"r={[round(float(x), 8) for x in layer_metrics['r']]} "
                f"total_importance={float(layer_metrics['total_importance']):.8f}"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_metrics.txt"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def write_metrics_json(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_metrics.json"
    output_path.write_text(json.dumps(sample_metrics, indent=2) + "\n", encoding="utf-8")
    return output_path


def print_group_summary(include_groups: List[str], sample_metrics: List[Dict[str, Any]]) -> None:
    counts_by_group = {group: 0 for group in include_groups}
    for sample in sample_metrics:
        for group in sample["active_groups"]:
            if group in counts_by_group:
                counts_by_group[group] += 1
    print("Active-group coverage:")
    for group in include_groups:
        print(f"  {group}: {counts_by_group.get(group, 0)} sample(s)")


def load_clean_score_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to parse clean-score cache at {path}: {exc}. Starting with empty cache.")
        return {}
    if not isinstance(payload, dict):
        print(f"[WARN] Invalid clean-score cache format at {path}. Expected JSON object; starting empty.")
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    for sample_id, value in payload.items():
        if isinstance(sample_id, str) and isinstance(value, dict):
            cache[sample_id] = value
    return cache


def save_clean_score_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {sample_id: cache[sample_id] for sample_id in sorted(cache.keys())}
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_include_groups(raw: str) -> List[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        return list(_ALL_GROUPS)
    invalid = [x for x in values if x not in _ALL_GROUPS]
    if invalid:
        raise ValueError(f"--include_groups has invalid entries: {invalid}. Supported groups: {_ALL_GROUPS}")
    return [x for x in _ALL_GROUPS if x in values]


def score_valid_numeric_answers(
    lm: LanguageModel,
    inputs: Dict[str, torch.Tensor],
    prompt_len: int,
    num_frames: int,
) -> Dict[str, Any]:
    scores_by_answer: Dict[str, float] = {}
    for value in range(num_frames + 1):
        answer_text = str(value)
        answer_ids = token_ids_of_answer(answer_text)
        scoring_inputs = append_answer_tokens_for_scoring(inputs, answer_ids)
        scores_by_answer[answer_text] = run_clean_sequence_logprob(
            lm=lm,
            scoring_inputs=scoring_inputs,
            prompt_len=prompt_len,
            answer_token_ids=answer_ids,
        )

    score_values = torch.tensor(list(scores_by_answer.values()), dtype=torch.float64)
    log_denom = torch.logsumexp(score_values, dim=0)
    probs_by_answer = {
        answer_text: float(torch.exp(torch.tensor(score, dtype=torch.float64) - log_denom).item())
        for answer_text, score in scores_by_answer.items()
    }
    best_answer_text = max(scores_by_answer.items(), key=lambda kv: kv[1])[0]
    return {
        "scores_by_answer": scores_by_answer,
        "probs_by_answer": probs_by_answer,
        "best_answer_text": best_answer_text,
    }


def build_control_question_candidates(
    question: str,
    states: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    parsed = parse_target_character_room(question)
    if parsed is None:
        return []
    target_character, target_room = parsed
    clean_count = count_steps_for_character_room(states, target_character, target_room)
    clean_character_token_len = len(processor.tokenizer(target_character, add_special_tokens=False)["input_ids"])
    clean_room_token_len = len(processor.tokenizer(target_room, add_special_tokens=False)["input_ids"])
    all_characters = [char for char in extract_characters_from_states(states) if char != target_character]
    all_rooms = [room for room in extract_rooms_from_states(states) if room != target_room]

    candidates: List[Tuple[Tuple[int, int, int, str, str], str, Dict[str, Any]]] = []
    for new_character in all_characters:
        for new_room in all_rooms:
            candidate_question = replace_character_and_room_in_question(question, new_character, new_room)
            if candidate_question is None:
                continue
            candidate_count = count_steps_for_character_room(states, new_character, new_room)
            if candidate_count == clean_count:
                continue
            character_token_len = len(processor.tokenizer(new_character, add_special_tokens=False)["input_ids"])
            room_token_len = len(processor.tokenizer(new_room, add_special_tokens=False)["input_ids"])
            meta = {
                "character": new_character,
                "room": new_room,
                "answer": str(candidate_count),
                "changes_answer": candidate_count != clean_count,
                "character_token_len_match": (character_token_len == clean_character_token_len),
                "room_token_len_match": (room_token_len == clean_room_token_len),
            }
            priority = (
                0,
                0 if character_token_len == clean_character_token_len else 1,
                0 if room_token_len == clean_room_token_len else 1,
                abs(len(new_character) - len(target_character)) + abs(len(new_room) - len(target_room)),
                new_character,
                new_room,
            )
            candidates.append((priority, candidate_question, meta))

    candidates.sort(key=lambda item: item[0])
    return [(question_text, meta) for _, question_text, meta in candidates]


def choose_best_control(
    frames: Sequence[Any],
    states: List[Dict[str, Any]],
    question: str,
    clean_inputs: Dict[str, torch.Tensor],
    clean_group_positions: Dict[str, List[int]],
    include_groups: List[str],
) -> Tuple[
    Optional[Dict[str, torch.Tensor]],
    Optional[str],
    Dict[str, List[int]],
    Dict[str, str],
    Dict[str, str],
    Optional[Dict[str, Any]],
]:
    clean_len = int(clean_inputs["input_ids"].shape[1])
    best_payload: Optional[
        Tuple[int, int, Dict[str, torch.Tensor], str, Dict[str, List[int]], Dict[str, str], Dict[str, str], Dict[str, Any]]
    ] = None

    for control_question, control_meta in build_control_question_candidates(question, states):
        try:
            control_inputs = move_inputs_to_model_device(build_inputs(frames, control_question))
        except Exception as exc:
            continue
        if int(control_inputs["input_ids"].shape[1]) != clean_len:
            continue

        control_group_positions, control_group_summaries, control_warnings = locate_group_token_positions(
            inputs=control_inputs,
            question=control_question,
            num_frames=len(frames),
        )
        if not control_group_positions:
            continue

        patchable_count = 0
        matched_character_room = 0
        skip_reasons: Dict[str, str] = {}
        for group_name in include_groups:
            clean_positions = clean_group_positions.get(group_name, [])
            control_positions = control_group_positions.get(group_name, [])
            if not clean_positions:
                skip_reasons[group_name] = "missing_clean_token_span"
                continue
            if not control_positions:
                skip_reasons[group_name] = "missing_control_token_span"
                continue
            if len(clean_positions) != len(control_positions):
                skip_reasons[group_name] = (
                    f"token_count_mismatch(clean={len(clean_positions)},control={len(control_positions)})"
                )
                continue
            patchable_count += 1
            if group_name in {"character", "room"}:
                matched_character_room += 1

        control_reason = {
            "control_character": control_meta["character"],
            "control_room": control_meta["room"],
            "control_answer": control_meta["answer"],
            "control_changes_answer": control_meta["changes_answer"],
            "control_warnings": control_warnings,
        }
        payload = (
            patchable_count,
            matched_character_room,
            control_inputs,
            control_question,
            control_group_positions,
            control_group_summaries,
            {**skip_reasons, **{f"warning_{idx}": warning for idx, warning in enumerate(control_warnings)}},
            control_reason,
        )
        if best_payload is None or (patchable_count, matched_character_room) > best_payload[:2]:
            best_payload = payload
        if patchable_count == len(include_groups) and matched_character_room == 2:
            break

    if best_payload is None:
        return None, None, {}, {}, {"control": "no_aligned_control_prompt_found"}, None

    _, _, control_inputs, control_question, control_group_positions, control_group_summaries, skip_info, control_reason = best_payload
    return control_inputs, control_question, control_group_positions, control_group_summaries, skip_info, control_reason


def main() -> None:
    start_time = time.perf_counter()

    ap = argparse.ArgumentParser(
        description=(
            "Measure late-layer answer-relevant information in non-image prompt groups using "
            "matched-control activation patching and full-answer log-prob scoring."
        )
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--output", type=str, default="outputs")
    ap.add_argument(
        "--clean_score_cache_dir",
        type=str,
        default=None,
        help="Directory containing clean_scores.json for loading/updating clean-answer filter cache.",
    )
    ap.add_argument(
        "--clean_ld_cache_dir",
        type=str,
        default=None,
        help="Backward-compatible alias for --clean_score_cache_dir.",
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--include_groups",
        type=str,
        default="character,room,question_operator,question_relation,question_marker,answer_marker,question_punct,instruction_context,instruction_output_rule,assistant_prefix",
        help=f"Comma-separated subset of: {','.join(_ALL_GROUPS)}",
    )
    ap.add_argument(
        "--min_clean_correct_prob",
        type=float,
        default=0.4,
        help="Keep a sample only if the clean correct answer probability among valid numeric answers is at least this value.",
    )
    ap.add_argument(
        "--min_clean_correct_logprob",
        type=float,
        default=None,
        help="Optional fallback filter: require clean correct-answer full-sequence log-prob >= this value.",
    )
    ap.add_argument("--disable_plots", action="store_true")
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer")
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer")
    if args.min_clean_correct_prob < 0.0 or args.min_clean_correct_prob > 1.0:
        raise ValueError("--min_clean_correct_prob must be in [0, 1]")

    include_groups = parse_include_groups(args.include_groups)
    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_score_cache_dir_raw = args.clean_score_cache_dir or args.clean_ld_cache_dir
    clean_score_cache_dir = Path(clean_score_cache_dir_raw) if clean_score_cache_dir_raw is not None else output_dir
    clean_score_cache_dir.mkdir(parents=True, exist_ok=True)
    clean_score_cache_path = clean_score_cache_dir / "clean_scores.json"
    clean_score_cache = load_clean_score_cache(clean_score_cache_path)
    cache_updates = 0
    if clean_score_cache:
        print(f"Loaded {len(clean_score_cache)} cached clean-score entries from: {clean_score_cache_path}")

    seq_len_match = re.search(r"(seq_len_\d+)", str(data_root))
    seq_len_label = seq_len_match.group(1) if seq_len_match else None

    lm = LanguageModel(base_model, tokenizer=processor.tokenizer)
    layers = get_layers(lm.model)
    num_layers = len(layers)

    sample_dirs = iter_sample_dirs(data_root)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under: {data_root}")

    target_processed_samples = int(args.limit)
    processed_samples = 0
    sample_metrics: List[Dict[str, Any]] = []
    layer_sampled_counts = [0 for _ in range(num_layers)]
    layer_invalid_counts = [0 for _ in range(num_layers)]

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= target_processed_samples:
            break

        try:
            sample_id, frames, question, states, answer = load_mmred_sample(sample_dir)
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_dir.name} skipped: load failure ({exc})")
            continue

        parsed = parse_target_character_room(question)
        if parsed is None:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: could not parse question")
            continue

        try:
            inputs = move_inputs_to_model_device(build_inputs(frames, question))
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: failed to build clean inputs ({exc})")
            continue

        prompt_len = int(inputs["input_ids"].shape[1])
        a_star_text = str(answer).strip()
        try:
            a_star_ids = token_ids_of_answer(a_star_text)
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: invalid answer tokenization ({exc})")
            continue

        cache_entry = clean_score_cache.get(sample_id)
        if cache_entry is not None:
            cached_num_frames = int(cache_entry.get("num_frames", -1))
            cached_answer = str(cache_entry.get("answer_text", ""))
            if cached_num_frames == len(frames) and cached_answer == a_star_text:
                clean_answer_score = float(cache_entry.get("clean_answer_score", float("-inf")))
                clean_correct_prob = float(cache_entry.get("clean_correct_prob", 0.0))
                clean_top1_correct = bool(cache_entry.get("clean_top1_correct", False))
                best_answer_text = str(cache_entry.get("best_answer_text", ""))
            else:
                cache_entry = None

        if cache_entry is None:
            try:
                candidate_scores = score_valid_numeric_answers(
                    lm=lm,
                    inputs=inputs,
                    prompt_len=prompt_len,
                    num_frames=len(frames),
                )
            except Exception as exc:
                print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: clean scoring failed ({exc})")
                continue

            clean_answer_score = float(candidate_scores["scores_by_answer"].get(a_star_text, float("-inf")))
            clean_correct_prob = float(candidate_scores["probs_by_answer"].get(a_star_text, 0.0))
            best_answer_text = str(candidate_scores["best_answer_text"])
            clean_top1_correct = (best_answer_text == a_star_text)
            clean_score_cache[sample_id] = {
                "num_frames": len(frames),
                "answer_text": a_star_text,
                "clean_answer_score": clean_answer_score,
                "clean_correct_prob": clean_correct_prob,
                "clean_top1_correct": clean_top1_correct,
                "best_answer_text": best_answer_text,
            }
            cache_updates += 1

        if not clean_top1_correct:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: clean top-1 is {best_answer_text!r}, "
                f"not correct answer {a_star_text!r}"
            )
            continue
        if clean_correct_prob < args.min_clean_correct_prob:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: "
                f"clean_correct_prob={clean_correct_prob:.4f} < threshold={args.min_clean_correct_prob:.4f}"
            )
            continue
        if args.min_clean_correct_logprob is not None and clean_answer_score < args.min_clean_correct_logprob:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: "
                f"clean_answer_score={clean_answer_score:.4f} < threshold={args.min_clean_correct_logprob:.4f}"
            )
            continue

        clean_group_positions, clean_group_summaries, clean_group_warnings = locate_group_token_positions(
            inputs=inputs,
            question=question,
            num_frames=len(frames),
        )
        if not clean_group_positions:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: no clean token groups "
                f"(warnings={clean_group_warnings})"
            )
            continue

        control_inputs, control_question, control_group_positions, control_group_summaries, control_skip_info, control_reason = choose_best_control(
            frames=frames,
            states=states,
            question=question,
            clean_inputs=inputs,
            clean_group_positions=clean_group_positions,
            include_groups=include_groups,
        )
        if control_inputs is None or control_question is None or control_reason is None:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: failed to find aligned control "
                f"(details={control_skip_info})"
            )
            continue

        groups_payload: List[Dict[str, Any]] = []
        skipped_groups: Dict[str, str] = {}
        group_summaries: Dict[str, Dict[str, str]] = {}
        group_token_counts: Dict[str, int] = {}

        for group_name in include_groups:
            clean_positions = clean_group_positions.get(group_name, [])
            control_positions = control_group_positions.get(group_name, [])
            if not clean_positions:
                skipped_groups[group_name] = "missing_clean_token_span"
                continue
            if not control_positions:
                skipped_groups[group_name] = "missing_control_token_span"
                continue
            if len(clean_positions) != len(control_positions):
                skipped_groups[group_name] = (
                    f"token_count_mismatch(clean={len(clean_positions)},control={len(control_positions)})"
                )
                continue

            group_token_counts[group_name] = len(clean_positions)
            group_summaries[group_name] = {
                "clean": clean_group_summaries.get(group_name, "<not available>"),
                "control": control_group_summaries.get(group_name, "<not available>"),
                "token_count": str(len(clean_positions)),
            }
            groups_payload.append({
                "name": group_name,
                "clean_positions": clean_positions,
                "control_positions": control_positions,
                "control_inputs": control_inputs,
            })

        if not groups_payload:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: no valid active groups "
                f"(skipped={skipped_groups}, clean_warnings={clean_group_warnings}, control_details={control_skip_info})"
            )
            continue

        active_group_names = [group["name"] for group in groups_payload]
        print(
            f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} clean_answer_score={clean_answer_score:.4f} "
            f"clean_correct_prob={clean_correct_prob:.4f} control_question={control_question!r} "
            f"control_character={control_reason['control_character']!r} "
            f"control_room={control_reason['control_room']!r} "
            f"control_answer={control_reason['control_answer']!r} "
            f"active_groups={active_group_names} batch_size={args.batch_size}"
        )
        print(f"  active group token counts: {group_token_counts}")
        if skipped_groups:
            print(f"  skipped groups: {skipped_groups}")

        chunk_size = min(args.batch_size, len(groups_payload))
        group_chunks = [
            groups_payload[start:start + chunk_size]
            for start in range(0, len(groups_payload), chunk_size)
        ]

        chunk_data: List[Dict[str, Any]] = []
        try:
            for group_chunk in group_chunks:
                chunk_len = len(group_chunk)
                repeated_clean_inputs = repeat_inputs_for_batch(inputs, batch_size=chunk_len)
                clean_scoring_inputs = append_answer_tokens_for_scoring(repeated_clean_inputs, a_star_ids)
                control_inputs_batch = concatenate_inputs_for_batch(
                    [group_entry["control_inputs"] for group_entry in group_chunk]
                )
                control_scoring_inputs = append_answer_tokens_for_scoring(control_inputs_batch, a_star_ids)
                chunk_data.append({
                    "groups": group_chunk,
                    "clean_scoring_inputs": clean_scoring_inputs,
                    "control_scoring_inputs": control_scoring_inputs,
                })
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: failed to build batched inputs ({exc})")
            continue

        per_layer_metrics: List[Dict[str, Any]] = []
        all_layer_corrupted_rows: List[Tuple[int, List[float]]] = []
        for layer_idx in range(num_layers):
            layer_sampled_counts[layer_idx] += 1
            per_group_corrupted_score: Dict[str, float] = {}
            per_group_signed_delta: Dict[str, float] = {}
            per_group_importance: Dict[str, float] = {}

            for chunk_idx, packed in enumerate(chunk_data, start=1):
                group_chunk = packed["groups"]
                clean_positions_by_batch = [group["clean_positions"] for group in group_chunk]
                control_positions_by_batch = [group["control_positions"] for group in group_chunk]
                try:
                    corrupted_scores = run_layer_multi_group_corrupted_sequence_logprob(
                        lm=lm,
                        layers=layers,
                        clean_batched_scoring_inputs=packed["clean_scoring_inputs"],
                        control_batched_scoring_inputs=packed["control_scoring_inputs"],
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
                        per_group_signed_delta[group_name] = 0.0
                        per_group_importance[group_name] = 0.0
                    continue

                for batch_idx, group in enumerate(group_chunk):
                    group_name = group["name"]
                    corrupt_score = float(corrupted_scores[batch_idx].item())
                    signed_delta = float(clean_answer_score - corrupt_score)
                    importance = max(signed_delta, 0.0)
                    per_group_corrupted_score[group_name] = corrupt_score
                    per_group_signed_delta[group_name] = signed_delta
                    per_group_importance[group_name] = importance

            layer_group_order = [group["name"] for group in groups_payload]
            corrupted_score_row = [per_group_corrupted_score.get(group_name, clean_answer_score) for group_name in layer_group_order]
            signed_delta_row = [per_group_signed_delta.get(group_name, 0.0) for group_name in layer_group_order]
            importance_row = [per_group_importance.get(group_name, 0.0) for group_name in layer_group_order]
            all_layer_corrupted_rows.append((layer_idx, list(corrupted_score_row)))

            total_importance = float(sum(importance_row))
            if total_importance > 0.0:
                probs = normalize_to_probabilities(importance_row)
                entropy_value = normalize_entropy(
                    entropy_from_probabilities(probs),
                    num_groups=len(layer_group_order),
                )
            else:
                probs = [0.0 for _ in importance_row]
                entropy_value = None
                layer_invalid_counts[layer_idx] += 1

            per_layer_metrics.append({
                "layer": layer_idx,
                "groups": list(layer_group_order),
                "corrupted_score": corrupted_score_row,
                "signed_delta": signed_delta_row,
                "r": importance_row,
                "p": probs,
                "entropy": entropy_value,
                "total_importance": total_importance,
            })

        if all_layer_corrupted_rows:
            print("  Corrupted score table (rows=layers, columns=text groups):")
            print(format_corrupted_score_table(active_group_names, all_layer_corrupted_rows))

        sample_metrics.append({
            "sample_id": sample_id,
            "answer": answer,
            "question": question,
            "control_question": control_question,
            "control_character": control_reason["control_character"],
            "control_room": control_reason["control_room"],
            "control_answer": control_reason["control_answer"],
            "clean_answer_score": clean_answer_score,
            "clean_correct_prob": clean_correct_prob,
            "clean_top1_correct": clean_top1_correct,
            "best_answer_text": best_answer_text,
            "a_star_text": a_star_text,
            "a_star_ids": a_star_ids,
            "active_groups": active_group_names,
            "group_token_counts": group_token_counts,
            "skipped_groups": skipped_groups,
            "group_summaries": group_summaries,
            "clean_group_warnings": clean_group_warnings,
            "control_group_info": control_skip_info,
            "control_reason": control_reason,
            "layer_metrics": {"layers": per_layer_metrics},
        })
        processed_samples += 1

    if cache_updates > 0:
        save_clean_score_cache(clean_score_cache_path, clean_score_cache)
        print(f"Updated clean-score cache at {clean_score_cache_path} ({cache_updates} new/changed entries).")
    elif not clean_score_cache_path.exists():
        save_clean_score_cache(clean_score_cache_path, clean_score_cache)
        print(f"Wrote empty clean-score cache to: {clean_score_cache_path}")
    else:
        print(f"No clean-score cache updates. Reused existing cache at: {clean_score_cache_path}")

    text_report_path = write_text_group_report(sample_metrics, output_dir)
    json_path = write_metrics_json(sample_metrics, output_dir)
    print(f"Wrote sample metrics text report to: {text_report_path}")
    print(f"Wrote sample metrics JSON to: {json_path}")
    print(
        f"Processed {processed_samples} samples "
        f"(target limit={target_processed_samples}, min_clean_correct_prob={args.min_clean_correct_prob:.4f})."
    )
    print_group_summary(include_groups, sample_metrics)

    if not args.disable_plots:
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

        heatmap_path = plot_group_importance_heatmap(
            sample_metrics,
            output_dir,
            num_layers=num_layers,
            include_groups=include_groups,
            seq_len_label=seq_len_label,
        )
        if heatmap_path is not None:
            print(f"Wrote group-importance heatmap to: {heatmap_path}")
        else:
            print("Skipped group-importance heatmap: insufficient data.")

        lines_path = plot_group_importance_lines(
            sample_metrics,
            output_dir,
            num_layers=num_layers,
            include_groups=include_groups,
            seq_len_label=seq_len_label,
        )
        if lines_path is not None:
            print(f"Wrote group-importance lines plot to: {lines_path}")
        else:
            print("Skipped group-importance lines plot: insufficient data.")

    elapsed = time.perf_counter() - start_time
    print(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()

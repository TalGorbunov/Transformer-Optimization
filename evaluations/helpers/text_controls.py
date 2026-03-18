from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from evaluations.helpers import utils as eval_utils
from evaluations.helpers.patching_core import (
    build_inputs,
    build_prompt,
    move_inputs_to_model_device,
    processor,
)

_QUESTION_MARKER = "Question:"
_ANSWER_MARKER = "Answer:"
_QUESTION_OPERATOR = "How many steps did"
_ANSWER_ASSISTANT_IM_START = "answer_assistant_im_start"
_ANSWER_ASSISTANT_ROLE = "answer_assistant_role"
_ANSWER_ASSISTANT_NEWLINE = "answer_assistant_newline"
ALL_GROUPS = [
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
    _ANSWER_ASSISTANT_IM_START,
    _ANSWER_ASSISTANT_ROLE,
    _ANSWER_ASSISTANT_NEWLINE,
]

parse_target_character_room = eval_utils.parse_target_character_room
parse_target_character_room_with_spans = eval_utils.parse_target_character_room_with_spans
extract_characters_from_states = eval_utils.extract_characters_from_states
extract_rooms_from_states = eval_utils.extract_rooms_from_states
count_steps_for_character_room = eval_utils.count_steps_for_character_room


def parse_include_groups(raw: str) -> List[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        return list(ALL_GROUPS)
    invalid = [value for value in values if value not in ALL_GROUPS]
    if invalid:
        raise ValueError(f"--include_groups has invalid entries: {invalid}. Supported groups: {ALL_GROUPS}")
    return [value for value in ALL_GROUPS if value in values]


def _normalize_summary_text(text: str) -> str:
    compact = " ".join(str(text).split())
    return compact if compact else "<whitespace>"


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
        tokenized = processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = [int(tok) for tok in tokenized["input_ids"]]
        offsets_raw = tokenized.get("offset_mapping")
        if offsets_raw is None:
            return token_ids, None
        return token_ids, [(int(start), int(end)) for start, end in offsets_raw]
    except Exception:
        tokenized = processor.tokenizer(text, add_special_tokens=False)
        return [int(tok) for tok in tokenized["input_ids"]], None


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
    tok_start = max(0, min(int(len(prefix_start_ids)), prompt_token_count))
    tok_end = max(0, min(int(len(prefix_end_ids)), prompt_token_count))
    if tok_end <= tok_start:
        return []
    return list(range(tok_start, tok_end))


def _char_span_to_token_positions_with_offsets(
    token_offsets: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    positions: List[int] = []
    for idx, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_end <= tok_start or tok_end <= char_start or tok_start >= char_end:
            continue
        positions.append(idx)
    return positions


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
    if prompt_token_ids[start_hint:start_hint + len(substring_ids)] == substring_ids:
        return list(range(start_hint, start_hint + len(substring_ids)))

    matches = [
        start
        for start in range(max(0, len(prompt_token_ids) - len(substring_ids) + 1))
        if prompt_token_ids[start:start + len(substring_ids)] == substring_ids
    ]
    if not matches:
        return []
    best_start = min(matches, key=lambda start: abs(start - start_hint))
    return list(range(best_start, best_start + len(substring_ids)))


def _char_span_to_token_positions_allow_leading_space(
    prompt_text: str,
    char_start: int,
    char_end: int,
    prompt_token_count: int,
) -> List[int]:
    positions = _char_span_to_token_positions_by_prefix_lengths(
        prompt_text=prompt_text,
        char_start=char_start,
        char_end=char_end,
        prompt_token_count=prompt_token_count,
    )
    if positions:
        return positions
    if char_start > 0 and prompt_text[char_start - 1].isspace():
        return _char_span_to_token_positions_by_prefix_lengths(
            prompt_text=prompt_text,
            char_start=char_start - 1,
            char_end=char_end,
            prompt_token_count=prompt_token_count,
        )
    return []


def build_prompt_group_char_spans(
    question: str,
    num_frames: int,
) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, str], List[str]]:
    warnings: List[str] = []
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
    if question_marker_start < 0 or answer_marker_start < 0:
        return {}, {}, ["prompt_markers_not_found"]

    question_text_start = question_marker_start + len("Question: ")
    question_operator_start_q = question.lower().find(_QUESTION_OPERATOR.lower())
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
    else:
        warnings.append("question_operator_not_found")
    if relation_end_q > relation_start_q:
        char_spans["question_relation"] = [(
            question_text_start + relation_start_q,
            question_text_start + relation_end_q,
        )]
        summaries["question_relation"] = _normalize_summary_text(question[relation_start_q:relation_end_q])
    else:
        warnings.append("question_relation_not_found")
    if instruction_context_start >= 0:
        char_spans["instruction_context"] = [(instruction_context_start, instruction_context_start + len(line1))]
        summaries["instruction_context"] = _normalize_summary_text(line1)
    if instruction_output_start >= 0:
        char_spans["instruction_output_rule"] = [(instruction_output_start, instruction_output_start + len(line2))]
        summaries["instruction_output_rule"] = _normalize_summary_text(line2)
    if question.endswith("?"):
        char_spans["question_punct"] = [(question_text_start + len(question) - 1, question_text_start + len(question))]
        summaries["question_punct"] = "?"
    else:
        warnings.append("question_punct_not_found")
    return char_spans, summaries, warnings


def locate_group_token_positions(
    inputs: Dict[str, torch.Tensor],
    question: str,
    num_frames: int,
) -> Tuple[Dict[str, List[int]], Dict[str, str], List[str]]:
    prompt = build_prompt(question, num_frames=num_frames)
    group_char_spans, group_summaries, warnings = build_prompt_group_char_spans(question, num_frames=num_frames)
    if not group_char_spans:
        return {}, {}, warnings

    prompt_token_ids, prompt_offsets = tokenize_with_offsets_if_available(prompt)
    if not prompt_token_ids:
        return {}, group_summaries, ["prompt_tokenization_failed"]

    full_input_ids = [int(tok) for tok in inputs["input_ids"][0].tolist()]
    prompt_start_in_full = find_subsequence(full_input_ids, prompt_token_ids)
    if prompt_start_in_full is None:
        return {}, group_summaries, ["prompt_subsequence_not_found_in_input_ids"]

    token_positions_full: Dict[str, List[int]] = {}
    for group_name, spans in group_char_spans.items():
        all_positions: List[int] = []
        for char_start, char_end in spans:
            if group_name in {"character", "room"}:
                positions = _char_span_to_token_positions_allow_leading_space(
                    prompt_text=prompt,
                    char_start=char_start,
                    char_end=char_end,
                    prompt_token_count=len(prompt_token_ids),
                )
            else:
                positions = _char_span_to_token_positions_by_prefix_lengths(
                    prompt_text=prompt,
                    char_start=char_start,
                    char_end=char_end,
                    prompt_token_count=len(prompt_token_ids),
                )
            if not positions and prompt_offsets is not None:
                positions = _char_span_to_token_positions_with_offsets(prompt_offsets, char_start, char_end)
            if not positions:
                positions = _token_positions_by_tokenized_substring(
                    prompt_token_ids=prompt_token_ids,
                    prompt_text=prompt,
                    substring_text=prompt[char_start:char_end],
                    substring_char_start=char_start,
                    substring_char_end=char_end,
                )
            all_positions.extend(int(position) for position in positions)

        if not all_positions:
            warnings.append(f"{group_name}:token_span_not_found")
            continue
        token_positions_full[group_name] = [prompt_start_in_full + position for position in sorted(set(all_positions))]

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
    if assistant_prefix_positions:
        token_positions_full["assistant_prefix"] = assistant_prefix_positions
        decoded_tail = [
            processor.tokenizer.decode([int(full_input_ids[pos])], clean_up_tokenization_spaces=False)
            for pos in assistant_prefix_positions
        ]
        group_summaries["assistant_prefix"] = _normalize_summary_text("".join(decoded_tail))
    else:
        warnings.append("assistant_prefix:not_found")
    return token_positions_full, group_summaries, warnings


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

    candidates: List[Tuple[Tuple[int, int, int, str, str], str, Dict[str, Any]]] = []
    for new_character in [char for char in extract_characters_from_states(states) if char != target_character]:
        for new_room in [room for room in extract_rooms_from_states(states) if room != target_room]:
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
            }
            priority = (
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
        except Exception:
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

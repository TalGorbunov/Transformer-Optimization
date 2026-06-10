"""AF1 sample loading, prompt layout extraction, and compatibility checks."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluations.helpers import patching_core as core
from evaluations.helpers import utils as eval_utils
from evaluations.scripts.af1.common import (
    INSTRUCTION_TRANSFER_PROMPT_SPAN,
    PER_SAMPLE_FIELDS,
    AttentionPolicy,
    PreparedSample,
    SampleLayout,
)
from models.model import find_subsequence, image_token_groups


def sanitize_token_text(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", "\\t") if text else "<empty>"


def decode_token_ids(token_ids: Sequence[int], *, processor: Any) -> List[str]:
    return [
        processor.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]


def _token_span_from_char_span(
    text: str,
    char_span: Tuple[int, int],
    *,
    processor: Any,
) -> Tuple[int, int]:
    start_char, end_char = int(char_span[0]), int(char_span[1])
    if start_char > 0 and text[start_char - 1].isspace():
        start_char -= 1
    start_token = len(processor.tokenizer(text[:start_char], add_special_tokens=False)["input_ids"])
    end_token = len(processor.tokenizer(text[:end_char], add_special_tokens=False)["input_ids"])
    return start_token, end_token


def _positions_from_token_span(base_start: int, token_span: Tuple[int, int]) -> List[int]:
    return list(range(base_start + int(token_span[0]), base_start + int(token_span[1])))


def _replace_spans(text: str, replacements: Sequence[Tuple[Tuple[int, int], str]]) -> str:
    pieces: List[str] = []
    cursor = 0
    for (start, end), replacement in sorted(replacements, key=lambda item: item[0][0]):
        pieces.append(text[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _prompt_family_key(question: str, seq_len: int) -> str:
    parsed = eval_utils.parse_target_character_room_with_spans(question)
    if parsed is None:
        raise RuntimeError(f"Could not parse character/room slots from question: {question!r}")
    _, _, character_span, room_span = parsed
    masked_question = _replace_spans(
        question,
        [
            (character_span, "{CHARACTER}"),
            (room_span, "{ROOM}"),
        ],
    )
    return core.build_prompt(masked_question, num_frames=seq_len)


def _instruction_positions_from_prompt(
    prompt_text: str,
    prompt_text_start: int,
    num_frames: int,
    *,
    processor: Any,
) -> Tuple[int, ...]:
    instruction_text = INSTRUCTION_TRANSFER_PROMPT_SPAN.format(num_frames=int(num_frames))
    instruction_start = prompt_text.find(instruction_text)
    if instruction_start < 0:
        raise RuntimeError(
            "Failed to locate the instruction span "
            f"{instruction_text!r} in the constructed prompt"
        )
    instruction_span = (instruction_start, instruction_start + len(instruction_text))
    instruction_token_span = _token_span_from_char_span(
        prompt_text,
        instruction_span,
        processor=processor,
    )
    instruction_positions = _positions_from_token_span(prompt_text_start, instruction_token_span)
    if not instruction_positions:
        raise RuntimeError(
            "Instruction span tokenized to an empty position set for "
            f"{instruction_text!r}"
        )
    return tuple(int(position) for position in instruction_positions)


def _special_token_positions(
    input_ids: Sequence[int],
    decoded_tokens: Sequence[str],
    token_text: str,
    *,
    processor: Any,
) -> Tuple[int, ...]:
    token_id = processor.tokenizer.convert_tokens_to_ids(token_text)
    positions_by_id = (
        tuple(idx for idx, input_id in enumerate(input_ids) if int(input_id) == int(token_id))
        if token_id is not None
        else tuple()
    )
    positions_by_text = tuple(
        idx
        for idx, decoded_token in enumerate(decoded_tokens)
        if str(decoded_token) == str(token_text)
    )
    if positions_by_id and positions_by_text and positions_by_id != positions_by_text:
        raise RuntimeError(
            f"Special-token position mismatch for {token_text!r}: "
            f"by_id={list(positions_by_id)} by_text={list(positions_by_text)}"
        )
    return positions_by_id or positions_by_text


def build_sample_layout(
    sample_id: str,
    frames: Sequence[Any],
    question: str,
    inputs: Dict[str, Any],
    *,
    processor: Any,
) -> SampleLayout:
    """Extract the prompt/token layout assumptions that AF1 depends on.

    The important pieces are:
    - where each frame's image token block lives in the prompt
    - which token is the prompt carrier / "last token" for next-token prediction
    - prompt length and prompt-family identity, so we can reject incompatible
      samples before donor mixing or activation patching
    """
    input_ids = [int(token_id) for token_id in inputs["input_ids"][0].detach().cpu().tolist()]
    prompt_len = len(input_ids)
    if prompt_len <= 0:
        raise RuntimeError(f"sample_id={sample_id}: empty prompt tokenization")

    prompt_text = core.build_prompt(question, num_frames=len(frames))
    prompt_text_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_text_start = find_subsequence(input_ids, [int(token_id) for token_id in prompt_text_ids])
    if prompt_text_start is None:
        raise RuntimeError(f"sample_id={sample_id}: failed to locate prompt text in multimodal prompt")

    question_fragment = f"Question: {question}\n"
    question_ids = processor.tokenizer(question_fragment, add_special_tokens=False)["input_ids"]
    question_start = find_subsequence(input_ids, [int(token_id) for token_id in question_ids])
    if question_start is None:
        raise RuntimeError(f"sample_id={sample_id}: failed to locate question span")

    parsed = eval_utils.parse_target_character_room_with_spans(question)
    if parsed is None:
        raise RuntimeError(f"sample_id={sample_id}: failed to parse character/room slots from question")
    _, room_text, character_span_in_question, room_span_in_question = parsed

    question_fragment_start_in_prompt = prompt_text.index(question_fragment)
    question_text_start_in_prompt = question_fragment_start_in_prompt + len("Question: ")

    character_span_in_prompt = (
        question_text_start_in_prompt + int(character_span_in_question[0]),
        question_text_start_in_prompt + int(character_span_in_question[1]),
    )
    room_span_in_prompt = (
        question_text_start_in_prompt + int(room_span_in_question[0]),
        question_text_start_in_prompt + int(room_span_in_question[1]),
    )

    character_token_span = _token_span_from_char_span(
        prompt_text,
        character_span_in_prompt,
        processor=processor,
    )
    character_positions = _positions_from_token_span(prompt_text_start, character_token_span)
    room_token_span = _token_span_from_char_span(
        prompt_text,
        room_span_in_prompt,
        processor=processor,
    )
    room_positions = _positions_from_token_span(prompt_text_start, room_token_span)
    if not room_positions:
        raise RuntimeError(f"sample_id={sample_id}: empty room token span")
    instruction_positions = _instruction_positions_from_prompt(
        prompt_text,
        prompt_text_start=prompt_text_start,
        num_frames=len(frames),
        processor=processor,
    )

    carrier_index = prompt_len - 1
    prompt_decoded_tokens = decode_token_ids(input_ids, processor=processor)

    frame_groups = image_token_groups(
        inputs["input_ids"][0].detach().cpu(),
        expected_num_frames=len(frames),
        processor=processor,
    )
    if len(frame_groups) != len(frames):
        raise RuntimeError(
            f"sample_id={sample_id}: expected {len(frames)} frame groups but found {len(frame_groups)}"
        )

    image_pad_positions = _special_token_positions(
        input_ids,
        prompt_decoded_tokens,
        "<|image_pad|>",
        processor=processor,
    )
    flattened_frame_positions = tuple(int(position) for group in frame_groups for position in group)
    if tuple(image_pad_positions) != flattened_frame_positions:
        raise RuntimeError(
            f"sample_id={sample_id}: image_pad positions {list(image_pad_positions)} do not match "
            f"frame groups {list(flattened_frame_positions)}"
        )
    vision_start_positions = _special_token_positions(
        input_ids,
        prompt_decoded_tokens,
        "<|vision_start|>",
        processor=processor,
    )
    vision_end_positions = _special_token_positions(
        input_ids,
        prompt_decoded_tokens,
        "<|vision_end|>",
        processor=processor,
    )

    return SampleLayout(
        sample_id=sample_id,
        seq_len=len(frames),
        prompt_len=prompt_len,
        carrier_index=carrier_index,
        carrier_token_id=int(input_ids[carrier_index]),
        carrier_token_text=sanitize_token_text(prompt_decoded_tokens[carrier_index]),
        prompt_family_key=_prompt_family_key(question, seq_len=len(frames)),
        frame_groups=tuple(tuple(int(position) for position in group) for group in frame_groups),
        image_tokens_per_frame=tuple(len(group) for group in frame_groups),
        room_text=str(room_text),
        room_positions=tuple(int(position) for position in room_positions),
        character_positions=tuple(int(position) for position in character_positions),
        instruction_positions=instruction_positions,
        image_pad_positions=tuple(int(position) for position in image_pad_positions),
        vision_start_positions=tuple(int(position) for position in vision_start_positions),
        vision_end_positions=tuple(int(position) for position in vision_end_positions),
        room_span_len=len(room_positions),
        prompt_input_ids=tuple(int(token_id) for token_id in input_ids),
        prompt_decoded_tokens=tuple(str(token) for token in prompt_decoded_tokens),
    )


def _layout_signature_payload(layout: SampleLayout) -> Dict[str, Any]:
    return {
        "seq_len": int(layout.seq_len),
        "prompt_family_key": layout.prompt_family_key,
        "prompt_len": int(layout.prompt_len),
        "carrier_index": int(layout.carrier_index),
        "carrier_token_id": int(layout.carrier_token_id),
        "carrier_token_text": layout.carrier_token_text,
        "image_tokens_per_frame": list(layout.image_tokens_per_frame),
        "frame_groups": [list(group) for group in layout.frame_groups],
        "instruction_positions": list(layout.instruction_positions),
        "image_pad_positions": list(layout.image_pad_positions),
        "vision_start_positions": list(layout.vision_start_positions),
        "vision_end_positions": list(layout.vision_end_positions),
        "room_span_len": int(layout.room_span_len),
    }


def layout_hash(layout: SampleLayout) -> str:
    payload = json.dumps(_layout_signature_payload(layout), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_and_filter_sample_dirs(data_root: Path, max_samples: int, seed: int) -> List[Path]:
    sample_dirs = list(eval_utils.iter_sample_dirs(data_root))
    rng = random.Random(seed)
    rng.shuffle(sample_dirs)
    if max_samples > 0:
        sample_dirs = sample_dirs[:max_samples]
    return sample_dirs


def inspect_and_validate_layout(
    reference_layout: SampleLayout,
    candidate_layout: SampleLayout,
    skip_hallway: bool,
) -> Dict[str, Any]:
    """Check whether a sample can safely share donors with the reference layout.

    This script intentionally uses a strict notion of compatibility because the
    AF1 conditional mean is defined over aligned frame-token groups. If prompt
    length, frame-group spans, carrier semantics, or image-token counts differ,
    we skip the sample instead of risking a silent mis-patch.
    """
    reasons: List[str] = []

    if skip_hallway and candidate_layout.room_text.lower() == "hallway":
        reasons.append("room_is_hallway")
    if int(candidate_layout.seq_len) != int(reference_layout.seq_len):
        reasons.append(
            f"seq_len_mismatch(ref={reference_layout.seq_len},cand={candidate_layout.seq_len})"
        )
    if candidate_layout.prompt_family_key != reference_layout.prompt_family_key:
        reasons.append("prompt_family_mismatch")
    if int(candidate_layout.prompt_len) != int(reference_layout.prompt_len):
        reasons.append(
            f"prompt_len_mismatch(ref={reference_layout.prompt_len},cand={candidate_layout.prompt_len})"
        )
    if int(candidate_layout.carrier_index) != int(reference_layout.carrier_index):
        reasons.append(
            f"carrier_index_mismatch(ref={reference_layout.carrier_index},cand={candidate_layout.carrier_index})"
        )
    if int(candidate_layout.carrier_token_id) != int(reference_layout.carrier_token_id):
        reasons.append(
            f"carrier_token_id_mismatch(ref={reference_layout.carrier_token_id},cand={candidate_layout.carrier_token_id})"
        )
    if tuple(candidate_layout.image_tokens_per_frame) != tuple(reference_layout.image_tokens_per_frame):
        reasons.append(
            "image_tokens_per_frame_mismatch"
            f"(ref={list(reference_layout.image_tokens_per_frame)},"
            f"cand={list(candidate_layout.image_tokens_per_frame)})"
        )
    if tuple(candidate_layout.frame_groups) != tuple(reference_layout.frame_groups):
        reasons.append("frame_group_boundaries_mismatch")
    if tuple(candidate_layout.instruction_positions) != tuple(reference_layout.instruction_positions):
        reasons.append("instruction_positions_mismatch")
    if tuple(candidate_layout.image_pad_positions) != tuple(reference_layout.image_pad_positions):
        reasons.append("image_pad_positions_mismatch")
    if tuple(candidate_layout.vision_start_positions) != tuple(reference_layout.vision_start_positions):
        reasons.append("vision_start_positions_mismatch")
    if tuple(candidate_layout.vision_end_positions) != tuple(reference_layout.vision_end_positions):
        reasons.append("vision_end_positions_mismatch")
    if int(candidate_layout.room_span_len) != int(reference_layout.room_span_len):
        reasons.append(
            f"room_span_len_mismatch(ref={reference_layout.room_span_len},cand={candidate_layout.room_span_len})"
        )

    if reasons:
        return {
            "status": "incompatible",
            "details": "; ".join(reasons),
            "reasons": reasons,
        }
    return {
        "status": "exact_match",
        "details": "exact_match",
        "reasons": [],
    }


def prepare_sample(
    sample_dir: Path,
    skip_hallway: bool,
    *,
    model_name: str,
    processor: Any,
) -> Tuple[Optional[PreparedSample], Optional[Dict[str, Any]]]:
    sample_id, frames, question, _, answer_text = eval_utils.load_mmred_sample(sample_dir)
    parsed = eval_utils.parse_target_character_room(question)
    room_text = parsed[1] if parsed is not None else ""

    if skip_hallway and room_text.lower() == "hallway":
        row = {field: "" for field in PER_SAMPLE_FIELDS}
        row.update(
            {
                "model": model_name,
                "sample_id": sample_id,
                "seq_len": int(len(frames)),
                "used": 0,
                "gold_answer": str(answer_text).strip(),
                "room_text": room_text,
                "num_frames": int(len(frames)),
                "skipped_reason": "room_is_hallway",
                "layout_match_status": "skipped",
                "layout_match_details": "room_is_hallway",
            }
        )
        return None, row

    try:
        inputs_cpu = core.build_inputs(frames, question, processor=processor)
        layout = build_sample_layout(
            sample_id=sample_id,
            frames=frames,
            question=question,
            inputs=inputs_cpu,
            processor=processor,
        )
    except Exception as exc:
        row = {field: "" for field in PER_SAMPLE_FIELDS}
        row.update(
            {
                "model": model_name,
                "sample_id": sample_id,
                "seq_len": int(len(frames)),
                "used": 0,
                "gold_answer": str(answer_text).strip(),
                "room_text": room_text,
                "num_frames": int(len(frames)),
                "skipped_reason": f"layout_build_failed({exc})",
                "layout_match_status": "skipped",
                "layout_match_details": f"layout_build_failed({exc})",
            }
        )
        return None, row

    prepared = PreparedSample(
        sample_dir=sample_dir,
        sample_id=sample_id,
        frames=frames,
        question=question,
        gold_answer=str(answer_text).strip(),
        layout=layout,
        inputs_cpu=inputs_cpu,
    )
    return prepared, None


def choose_reference_layout(samples: Sequence[PreparedSample]) -> Optional[SampleLayout]:
    if not samples:
        return None
    signature_order: List[str] = []
    signature_to_layout: Dict[str, SampleLayout] = {}
    counts: Counter[str] = Counter()
    for sample in samples:
        signature = json.dumps(_layout_signature_payload(sample.layout), sort_keys=True)
        if signature not in signature_to_layout:
            signature_order.append(signature)
            signature_to_layout[signature] = sample.layout
        counts[signature] += 1

    best_signature = max(signature_order, key=lambda signature: (counts[signature], -signature_order.index(signature)))
    return signature_to_layout[best_signature]


def _sample_seed(seed: int, *parts: str) -> int:
    raw = "::".join([str(seed)] + [str(part) for part in parts])
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16], 16)


def select_donor_pool(
    target_sample: PreparedSample,
    compatible_samples: Sequence[PreparedSample],
    k_donors: int,
    seed: int,
) -> List[PreparedSample]:
    """Choose up to `k_donors` compatible donors with deterministic shuffling.

    Donors:
    - come from the same validated seq_len pool
    - must not be the target sample itself
    - are shuffled with a target-specific seeded RNG so repeated runs are stable
    """
    candidates = [
        sample
        for sample in compatible_samples
        if sample.sample_id != target_sample.sample_id and sample.layout.seq_len == target_sample.layout.seq_len
    ]
    candidates = sorted(candidates, key=lambda sample: sample.sample_id)
    rng = random.Random(_sample_seed(seed, str(target_sample.layout.seq_len), target_sample.sample_id))
    rng.shuffle(candidates)
    return candidates[: max(0, k_donors)]


def build_hybrid_sample(
    target_sample: PreparedSample,
    donor_sample: PreparedSample,
    frame_idx: int,
) -> Dict[str, Any]:
    """Build one hybrid context for conditional-mean estimation.

    Paper-inspired semantics for the multimodal adaptation:
    - keep the target sample's text prompt fixed
    - keep frame `frame_idx` fixed from the target sample
    - replace every other frame with the corresponding frame from one donor

    The resulting prompt layout matches the target layout because only image
    pixels change, not prompt text or multimodal placeholder positions.
    """
    if target_sample.layout.seq_len != donor_sample.layout.seq_len:
        raise ValueError(
            f"Incompatible seq_len for hybrid sample: target={target_sample.layout.seq_len}, "
            f"donor={donor_sample.layout.seq_len}"
        )
    if frame_idx < 0 or frame_idx >= target_sample.layout.seq_len:
        raise IndexError(f"frame_idx={frame_idx} out of bounds for seq_len={target_sample.layout.seq_len}")

    mixed_frames = list(donor_sample.frames)
    mixed_frames[frame_idx] = target_sample.frames[frame_idx]
    return {
        "sample_id": f"{target_sample.sample_id}__frame_{frame_idx}__donor_{donor_sample.sample_id}",
        "frames": mixed_frames,
        "question": target_sample.question,
        # The text prompt is fixed to the target sample, so the prompt token
        # layout is the same as the target layout. Only the image pixels change.
        "layout": target_sample.layout,
    }


def all_non_frame_prompt_positions(layout: SampleLayout) -> Tuple[int, ...]:
    frame_positions = {int(position) for group in layout.frame_groups for position in group}
    return tuple(position for position in range(int(layout.prompt_len)) if position not in frame_positions)


def build_non_frame_hybrid_sample(
    target_sample: PreparedSample,
    donor_sample: PreparedSample,
) -> Dict[str, Any]:
    """Build one hybrid context for the all-non-frame prompt conditional mean."""
    if target_sample.layout.seq_len != donor_sample.layout.seq_len:
        raise ValueError(
            f"Incompatible seq_len for non-frame hybrid sample: target={target_sample.layout.seq_len}, "
            f"donor={donor_sample.layout.seq_len}"
        )
    return {
        "sample_id": f"{target_sample.sample_id}__non_frame__donor_{donor_sample.sample_id}",
        "frames": list(donor_sample.frames),
        "question": target_sample.question,
        # The prompt text stays fixed to the target sample; all frames come from
        # the donor so the non-frame mean is estimated from donor frame sets.
        "layout": target_sample.layout,
    }


def format_token_debug_rows(layout: SampleLayout) -> str:
    frame_lookup: Dict[int, int] = {}
    for frame_idx, group in enumerate(layout.frame_groups):
        for position in group:
            frame_lookup[int(position)] = frame_idx
    instruction_lookup = {int(position) for position in layout.instruction_positions}
    image_pad_lookup = {int(position) for position in layout.image_pad_positions}
    vision_start_lookup = {int(position) for position in layout.vision_start_positions}
    vision_end_lookup = {int(position) for position in layout.vision_end_positions}

    lines = ["idx\tid\ttoken\ttags"]
    for idx, token_id in enumerate(layout.prompt_input_ids):
        tags: List[str] = []
        if idx == layout.carrier_index:
            tags.append("CARRIER")
        if idx in frame_lookup:
            tags.append(f"frame_{frame_lookup[idx]}")
        if idx in image_pad_lookup:
            tags.append("IMAGE_PAD")
        if idx in vision_start_lookup:
            tags.append("VISION_START")
        if idx in vision_end_lookup:
            tags.append("VISION_END")
        if idx in instruction_lookup:
            tags.append("INSTRUCTION")
        lines.append(
            f"{idx}\t{token_id}\t{sanitize_token_text(layout.prompt_decoded_tokens[idx])}\t{','.join(tags) or '-'}"
        )
    return "\n".join(lines)


def format_transition_frame_debug(layout: SampleLayout, policy: AttentionPolicy) -> str:
    frame_sizes = [len(group) for group in layout.frame_groups]
    empty_frame_groups = sum(1 for group in layout.frame_groups if not group)
    dense_intra_frame = True
    cross_frame_leak = False
    for group in layout.frame_groups:
        if not group:
            continue
        group_set = {int(position) for position in group}
        transfer_keys = set(policy.frame_group_by_token.get(int(group[0]), (int(group[0]),)))
        if not group_set.issubset(transfer_keys):
            dense_intra_frame = False
        other_frame_positions = set(policy.frame_group_by_token) - group_set
        if transfer_keys & other_frame_positions:
            cross_frame_leak = True
    return (
        f"transition_frame_blocks frames={len(layout.frame_groups)} "
        f"tokens_per_frame={frame_sizes} "
        f"image_pad_tokens={len(layout.image_pad_positions)} "
        f"vision_start_tokens={len(layout.vision_start_positions)} "
        f"vision_end_tokens={len(layout.vision_end_positions)} "
        f"empty_frame_groups={empty_frame_groups} "
        f"dense_intra_frame={dense_intra_frame} "
        f"cross_frame_leak={cross_frame_leak} "
        f"instruction_mask_mode={policy.instruction_mask_mode} "
        "base_causal_preserved=True"
    )

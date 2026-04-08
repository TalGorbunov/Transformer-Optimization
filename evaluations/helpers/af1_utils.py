import csv
import json
import pickle
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from evaluations.helpers import patching_core as core
from evaluations.helpers.sdpa_attention import (
    allowed_key_positions_from_mask,
    build_prompt_allow_matrix,
    build_sdpa_layer_mask,
)
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import format_runtime, iter_sample_dirs, load_mmred_sample
from models.model import (
    DEFAULT_MODEL_ID,
    find_subsequence,
    get_default_runtime,
    get_layers,
    image_token_groups,
    prepare_attention_backend_for_forward,
)


def _runtime() -> Any:
    return get_default_runtime()


def _model() -> Any:
    return _runtime().model


def _processor() -> Any:
    return _runtime().processor


@dataclass(frozen=True)
class PatchUnit:
    name: str
    kind: str
    positions: Tuple[int, ...]
    cache_keys: Tuple[str, ...]
    cache_bucket: str


@dataclass
class TokenLayout:
    sample_id: str
    seq_len: int
    prompt_len: int
    carrier_index: int
    carrier_token_id: int
    carrier_token_text: str
    prompt_text_span: Tuple[int, int]
    question_span: Tuple[int, int]
    answer_prefix_span: Tuple[int, int]
    frame_groups: List[List[int]]
    prompt_input_ids: List[int]
    prompt_decoded_tokens: List[str]
    character_text: str
    room_text: str
    character_positions: List[int]
    room_positions: List[int]
    fixed_template_positions: List[int]
    question_fixed_positions: List[int]
    variable_question_positions: List[int]
    answer_prefix_positions: List[int]
    room_span_len: int

    def fixed_template_token_ids(self) -> List[int]:
        return [int(self.prompt_input_ids[position]) for position in self.fixed_template_positions]

    def frame_group_sizes(self) -> List[int]:
        return [len(group) for group in self.frame_groups]


def serialize_token_layout(layout: TokenLayout) -> Dict[str, Any]:
    return {
        "sample_id": layout.sample_id,
        "seq_len": int(layout.seq_len),
        "prompt_len": int(layout.prompt_len),
        "carrier_index": int(layout.carrier_index),
        "carrier_token_id": int(layout.carrier_token_id),
        "carrier_token_text": str(layout.carrier_token_text),
        "prompt_text_span": list(layout.prompt_text_span),
        "question_span": list(layout.question_span),
        "answer_prefix_span": list(layout.answer_prefix_span),
        "frame_groups": [list(group) for group in layout.frame_groups],
        "prompt_input_ids": list(layout.prompt_input_ids),
        "prompt_decoded_tokens": list(layout.prompt_decoded_tokens),
        "character_text": str(layout.character_text),
        "room_text": str(layout.room_text),
        "character_positions": list(layout.character_positions),
        "room_positions": list(layout.room_positions),
        "fixed_template_positions": list(layout.fixed_template_positions),
        "question_fixed_positions": list(layout.question_fixed_positions),
        "variable_question_positions": list(layout.variable_question_positions),
        "answer_prefix_positions": list(layout.answer_prefix_positions),
        "room_span_len": int(layout.room_span_len),
    }


def deserialize_token_layout(raw: Any) -> TokenLayout:
    if isinstance(raw, TokenLayout):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported serialized TokenLayout type: {type(raw)}")
    return TokenLayout(
        sample_id=str(raw["sample_id"]),
        seq_len=int(raw["seq_len"]),
        prompt_len=int(raw["prompt_len"]),
        carrier_index=int(raw["carrier_index"]),
        carrier_token_id=int(raw["carrier_token_id"]),
        carrier_token_text=str(raw["carrier_token_text"]),
        prompt_text_span=tuple(raw["prompt_text_span"]),
        question_span=tuple(raw["question_span"]),
        answer_prefix_span=tuple(raw["answer_prefix_span"]),
        frame_groups=[list(group) for group in raw["frame_groups"]],
        prompt_input_ids=list(raw["prompt_input_ids"]),
        prompt_decoded_tokens=list(raw["prompt_decoded_tokens"]),
        character_text=str(raw["character_text"]),
        room_text=str(raw["room_text"]),
        character_positions=list(raw["character_positions"]),
        room_positions=list(raw["room_positions"]),
        fixed_template_positions=list(raw["fixed_template_positions"]),
        question_fixed_positions=list(raw["question_fixed_positions"]),
        variable_question_positions=list(raw["variable_question_positions"]),
        answer_prefix_positions=list(raw["answer_prefix_positions"]),
        room_span_len=int(raw["room_span_len"]),
    )


def load_af1_cache_file(cache_path: Path) -> Tuple[Dict[str, Any], bool]:
    migrated_old_pickle = False
    try:
        cache = torch.load(cache_path, map_location="cpu")
    except pickle.UnpicklingError:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        migrated_old_pickle = True
    return cache, migrated_old_pickle


def batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def sanitize_token_text(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", "\\t") if text else "<empty>"


def canonical_model_slug(model_name: str) -> str:
    return model_name.lower().replace("/", "_").replace("-", "")


def seq_len_data_root(data_root_base: Path, seq_len: int, split: str) -> Path:
    return data_root_base / f"seq_len_{seq_len}" / split


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(dtype_name).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype={dtype_name!r}.")
    return mapping[key]


def model_runtime_info(requested_device: str, requested_dtype: str) -> Dict[str, str]:
    first_param = next(_model().parameters())
    actual_device = str(first_param.device)
    actual_dtype = str(first_param.dtype)
    requested_device = str(requested_device).strip()
    if requested_device and requested_device != "auto":
        requested_root = requested_device.split(":")[0]
        actual_root = actual_device.split(":")[0]
        if requested_root != actual_root:
            raise RuntimeError(
                f"Requested --device={requested_device!r}, but the current model is loaded on {actual_device!r}. "
                "The current wrapper preloads the model globally, so the requested device must match."
            )
    return {
        "model_name": DEFAULT_MODEL_ID,
        "requested_device": requested_device,
        "actual_model_device": actual_device,
        "requested_dtype": str(requested_dtype),
        "actual_model_dtype": actual_dtype,
    }


def move_inputs_to_model_device(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    device = next(_model().parameters()).device
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


def _prepare_forward_backend(
    *,
    path_name: str,
    requires_abp_mask: bool,
    output_attentions: bool = False,
) -> str:
    return prepare_attention_backend_for_forward(
        path_name=path_name,
        requires_abp_mask=requires_abp_mask,
        output_attentions=output_attentions,
        allow_sdpa_fallback=not requires_abp_mask,
        model_obj=_model(),
    )


def _backend_cache_component(attention_backend: str) -> str:
    if attention_backend == "eager":
        raise RuntimeError("Eager attention is forbidden in AF1 utility cache keys.")
    if attention_backend not in {"sdpa", "flash_attention_2"}:
        raise RuntimeError(f"Unsupported attention backend for AF1 utility cache keys: {attention_backend!r}")
    return f"backend_{attention_backend}"


def decode_token_ids(token_ids: Sequence[int]) -> List[str]:
    return [
        _processor().tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]


def _token_span_from_char_span(text: str, char_span: Tuple[int, int]) -> Tuple[int, int]:
    start_char, end_char = int(char_span[0]), int(char_span[1])
    # Tokenizers often absorb the preceding whitespace into the first token of
    # a human-readable span (for example " Michael"), so we mirror the safer
    # span-to-token logic already used elsewhere in this repo.
    if start_char > 0 and text[start_char - 1].isspace():
        start_char -= 1
    start_token = len(_processor().tokenizer(text[:start_char], add_special_tokens=False)["input_ids"])
    end_token = len(_processor().tokenizer(text[:end_char], add_special_tokens=False)["input_ids"])
    return start_token, end_token


def _positions_from_token_span(base_start: int, token_span: Tuple[int, int]) -> List[int]:
    return list(range(base_start + int(token_span[0]), base_start + int(token_span[1])))


def build_token_layout(
    sample_id: str,
    frames: Sequence[Any],
    question: str,
    inputs: Dict[str, torch.Tensor],
) -> TokenLayout:
    input_ids = [int(token_id) for token_id in inputs["input_ids"][0].detach().cpu().tolist()]
    prompt_len = len(input_ids)
    if prompt_len <= 0:
        raise RuntimeError(f"sample_id={sample_id}: prompt tokenization is empty")

    prompt_text = core.build_prompt(question, num_frames=len(frames))
    prompt_text_ids = _processor().tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_text_start = find_subsequence(input_ids, [int(token_id) for token_id in prompt_text_ids])
    if prompt_text_start is None:
        raise RuntimeError(f"sample_id={sample_id}: failed to locate prompt text in multimodal prompt")
    prompt_text_span = (prompt_text_start, prompt_text_start + len(prompt_text_ids))

    question_fragment = f"Question: {question}\n"
    question_ids = _processor().tokenizer(question_fragment, add_special_tokens=False)["input_ids"]
    question_start = find_subsequence(input_ids, [int(token_id) for token_id in question_ids])
    if question_start is None:
        raise RuntimeError(f"sample_id={sample_id}: failed to locate question span")

    answer_prefix_ids = _processor().tokenizer("Answer: ", add_special_tokens=False)["input_ids"]
    answer_prefix_start = find_subsequence(input_ids, [int(token_id) for token_id in answer_prefix_ids])
    if answer_prefix_start is None:
        raise RuntimeError(f"sample_id={sample_id}: failed to locate answer prefix")

    answer_prefix_span = (answer_prefix_start, prompt_text_span[1])
    question_span = (question_start, answer_prefix_start)

    carrier_index = prompt_len - 1
    if carrier_index < 0:
        raise RuntimeError(f"sample_id={sample_id}: invalid carrier index {carrier_index}")

    parsed = eval_utils.parse_target_character_room_with_spans(question)
    if parsed is None:
        raise RuntimeError(f"sample_id={sample_id}: failed to parse character/room slots from question")
    character_text, room_text, character_span_in_question, room_span_in_question = parsed

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

    character_token_span = _token_span_from_char_span(prompt_text, character_span_in_prompt)
    room_token_span = _token_span_from_char_span(prompt_text, room_span_in_prompt)

    character_positions = _positions_from_token_span(prompt_text_start, character_token_span)
    room_positions = _positions_from_token_span(prompt_text_start, room_token_span)
    if not character_positions:
        raise RuntimeError(
            f"sample_id={sample_id}: empty character token span "
            f"question={question!r} character_text={character_text!r} "
            f"character_span_in_prompt={character_span_in_prompt}"
        )
    if not room_positions:
        raise RuntimeError(
            f"sample_id={sample_id}: empty room token span "
            f"question={question!r} room_text={room_text!r} "
            f"room_span_in_prompt={room_span_in_prompt}"
        )

    prompt_decoded_tokens = decode_token_ids(input_ids)
    frame_groups = image_token_groups(
        inputs["input_ids"][0].detach().cpu(),
        expected_num_frames=len(frames),
        processor=_processor(),
    )
    if len(frames) > 0 and len(frame_groups) != len(frames):
        raise RuntimeError(
            f"sample_id={sample_id}: expected {len(frames)} frame token groups but found {len(frame_groups)}"
        )

    variable_question_positions = sorted(set(character_positions + room_positions))
    fixed_template_positions = [
        position
        for position in range(prompt_text_span[0], prompt_text_span[1])
        if position not in variable_question_positions
    ]
    question_fixed_positions = [
        position
        for position in range(question_span[0], question_span[1])
        if position not in variable_question_positions
    ]
    answer_prefix_positions = list(range(answer_prefix_span[0], answer_prefix_span[1]))

    return TokenLayout(
        sample_id=sample_id,
        seq_len=len(frames),
        prompt_len=prompt_len,
        carrier_index=carrier_index,
        carrier_token_id=int(input_ids[carrier_index]),
        carrier_token_text=sanitize_token_text(prompt_decoded_tokens[carrier_index]),
        prompt_text_span=prompt_text_span,
        question_span=question_span,
        answer_prefix_span=answer_prefix_span,
        frame_groups=[list(group) for group in frame_groups],
        prompt_input_ids=input_ids,
        prompt_decoded_tokens=prompt_decoded_tokens,
        character_text=str(character_text),
        room_text=str(room_text),
        character_positions=character_positions,
        room_positions=room_positions,
        fixed_template_positions=fixed_template_positions,
        question_fixed_positions=question_fixed_positions,
        variable_question_positions=variable_question_positions,
        answer_prefix_positions=answer_prefix_positions,
        room_span_len=len(room_positions),
    )


def layout_compatibility_report(
    reference: TokenLayout,
    candidate: TokenLayout,
    wait_patch_mode: str,
) -> Dict[str, Any]:
    mismatches: List[str] = []
    hard_fail_reasons: List[str] = []

    if reference.carrier_token_id != candidate.carrier_token_id:
        reason = (
            f"carrier_token_id mismatch ref={reference.carrier_token_id} cand={candidate.carrier_token_id}"
        )
        mismatches.append(reason)
        hard_fail_reasons.append(reason)
    elif reference.carrier_index != candidate.carrier_index:
        mismatches.append(
            f"carrier_index shift ref={reference.carrier_index} cand={candidate.carrier_index}"
        )

    if reference.frame_group_sizes() != candidate.frame_group_sizes():
        reason = (
            f"frame_group_sizes mismatch ref={reference.frame_group_sizes()} "
            f"cand={candidate.frame_group_sizes()}"
        )
        mismatches.append(reason)
        hard_fail_reasons.append(reason)

    if reference.fixed_template_token_ids() != candidate.fixed_template_token_ids():
        reason = "fixed template token sequence mismatch"
        mismatches.append(reason)
        hard_fail_reasons.append(reason)

    if reference.room_span_len != candidate.room_span_len:
        mismatches.append(f"room_span_len mismatch ref={reference.room_span_len} cand={candidate.room_span_len}")

    if len(reference.character_positions) != len(candidate.character_positions):
        mismatches.append(
            f"character_span_len mismatch ref={len(reference.character_positions)} cand={len(candidate.character_positions)}"
        )

    if wait_patch_mode == "template_frames_plus_room_slot" and reference.room_span_len != candidate.room_span_len:
        # Different room lengths are still allowed because the cache is bucketed by span length.
        pass

    status = "exact_match" if not mismatches else "compatible_with_variation"
    if hard_fail_reasons:
        status = "incompatible"

    return {
        "status": status,
        "mismatches": mismatches,
        "hard_fail_reasons": hard_fail_reasons,
        "summary": "; ".join(mismatches) if mismatches else "exact_match",
    }


def assert_layout_patchable(
    reference: TokenLayout,
    candidate: TokenLayout,
    wait_patch_mode: str,
) -> Dict[str, Any]:
    report = layout_compatibility_report(reference, candidate, wait_patch_mode=wait_patch_mode)
    if report["hard_fail_reasons"]:
        raise RuntimeError(
            "AF1 layout mismatch makes patching impossible.\n"
            f"reference_sample={reference.sample_id}\n"
            f"candidate_sample={candidate.sample_id}\n"
            f"status={report['status']}\n"
            f"details={json.dumps(report, sort_keys=True)}"
        )
    return report


def _make_frame_units(layout: TokenLayout) -> List[PatchUnit]:
    units: List[PatchUnit] = []
    for frame_idx, group in enumerate(layout.frame_groups):
        positions = tuple(int(position) for position in group if position != layout.carrier_index)
        if not positions:
            continue
        cache_keys = tuple(f"frame_{frame_idx}_offset_{offset}" for offset, _ in enumerate(positions))
        units.append(
            PatchUnit(
                name=f"frame_{frame_idx}",
                kind="frame_group",
                positions=positions,
                cache_keys=cache_keys,
                cache_bucket=f"frame_{frame_idx}",
            )
        )
    return units


def _make_fixed_template_units(layout: TokenLayout) -> List[PatchUnit]:
    units: List[PatchUnit] = []
    for ordinal, position in enumerate(layout.fixed_template_positions):
        if position == layout.carrier_index:
            continue
        token_id = int(layout.prompt_input_ids[position])
        cache_key = f"fixed_template_idx_{ordinal}_tok_{token_id}"
        units.append(
            PatchUnit(
                name=f"fixed_template_{ordinal}",
                kind="fixed_template_text",
                positions=(int(position),),
                cache_keys=(cache_key,),
                cache_bucket="fixed_template_text",
            )
        )
    return units


def _make_question_fixed_units(layout: TokenLayout) -> List[PatchUnit]:
    units: List[PatchUnit] = []
    for ordinal, position in enumerate(layout.question_fixed_positions):
        if position == layout.carrier_index:
            continue
        token_id = int(layout.prompt_input_ids[position])
        cache_key = f"question_fixed_idx_{ordinal}_tok_{token_id}"
        units.append(
            PatchUnit(
                name=f"question_fixed_{ordinal}",
                kind="question_fixed_text",
                positions=(int(position),),
                cache_keys=(cache_key,),
                cache_bucket="question_fixed_text",
            )
        )
    return units


def _make_character_units(layout: TokenLayout) -> List[PatchUnit]:
    units: List[PatchUnit] = []
    for offset, position in enumerate(layout.character_positions):
        if position == layout.carrier_index:
            continue
        token_id = int(layout.prompt_input_ids[position])
        cache_key = f"character_slot_offset_{offset}_tok_{token_id}"
        units.append(
            PatchUnit(
                name=f"character_slot_{offset}",
                kind="character_slot",
                positions=(int(position),),
                cache_keys=(cache_key,),
                cache_bucket=f"character_slot_tok_{token_id}",
            )
        )
    return units


def _make_room_units(layout: TokenLayout) -> List[PatchUnit]:
    units: List[PatchUnit] = []
    bucket_prefix = f"room_span_len_{layout.room_span_len}"
    for offset, position in enumerate(layout.room_positions):
        if position == layout.carrier_index:
            continue
        token_id = int(layout.prompt_input_ids[position])
        cache_key = f"{bucket_prefix}_offset_{offset}_tok_{token_id}"
        units.append(
            PatchUnit(
                name=f"room_slot_{offset}",
                kind="room_slot",
                positions=(int(position),),
                cache_keys=(cache_key,),
                cache_bucket=bucket_prefix,
            )
        )
    return units


def patch_units_for_mode(layout: TokenLayout, wait_patch_mode: str) -> List[PatchUnit]:
    frame_units = _make_frame_units(layout)
    fixed_template_units = _make_fixed_template_units(layout)
    question_fixed_units = _make_question_fixed_units(layout)
    character_units = _make_character_units(layout)
    room_units = _make_room_units(layout)

    if wait_patch_mode == "frames_only":
        candidate_units = frame_units
    elif wait_patch_mode == "text_only":
        candidate_units = fixed_template_units + character_units + room_units
    elif wait_patch_mode == "frames_plus_question_text":
        candidate_units = frame_units + question_fixed_units + character_units + room_units
    elif wait_patch_mode == "all_noncarrier":
        candidate_units = frame_units + fixed_template_units + character_units + room_units
    elif wait_patch_mode == "template_plus_frames":
        candidate_units = frame_units + fixed_template_units
    elif wait_patch_mode == "template_frames_plus_room_slot":
        candidate_units = frame_units + fixed_template_units + room_units
    else:
        raise ValueError(
            f"Unsupported wait_patch_mode={wait_patch_mode!r}. "
            "Expected one of: all_noncarrier, frames_only, text_only, frames_plus_question_text, "
            "template_plus_frames, template_frames_plus_room_slot."
        )

    filtered_units: List[PatchUnit] = []
    seen: set[Tuple[Tuple[int, ...], Tuple[str, ...]]] = set()
    for unit in candidate_units:
        positions = tuple(
            int(position)
            for position in unit.positions
            if position != layout.carrier_index
        )
        if not positions:
            continue
        key = (positions, unit.cache_keys)
        if key in seen:
            continue
        seen.add(key)
        filtered_units.append(
            PatchUnit(
                name=unit.name,
                kind=unit.kind,
                positions=positions,
                cache_keys=unit.cache_keys,
                cache_bucket=unit.cache_bucket,
            )
        )

    if not filtered_units:
        raise RuntimeError(
            f"sample_id={layout.sample_id}: wait_patch_mode={wait_patch_mode!r} produced an empty patch set"
        )
    return filtered_units


def format_token_debug_rows(layout: TokenLayout, patch_units: Sequence[PatchUnit]) -> str:
    patch_lookup: Dict[int, List[str]] = {}
    for unit in patch_units:
        for position in unit.positions:
            patch_lookup.setdefault(int(position), []).append(unit.name)

    frame_lookup: Dict[int, int] = {}
    for frame_idx, group in enumerate(layout.frame_groups):
        for position in group:
            frame_lookup[int(position)] = frame_idx

    character_set = set(layout.character_positions)
    room_set = set(layout.room_positions)
    fixed_template_set = set(layout.fixed_template_positions)
    answer_prefix_set = set(layout.answer_prefix_positions)

    lines = ["idx\tid\ttoken\ttags"]
    for idx, token_id in enumerate(layout.prompt_input_ids):
        tags: List[str] = []
        if idx == layout.carrier_index:
            tags.append("CARRIER")
        if idx in frame_lookup:
            tags.append(f"frame_{frame_lookup[idx]}")
        if idx in fixed_template_set:
            tags.append("fixed_template")
        if idx in character_set:
            tags.append("character_slot")
        if idx in room_set:
            tags.append("room_slot")
        if idx in answer_prefix_set:
            tags.append("answer_prefix")
        tags.extend(patch_lookup.get(idx, []))
        lines.append(
            f"{idx}\t{token_id}\t{sanitize_token_text(layout.prompt_decoded_tokens[idx])}\t{','.join(tags) or '-'}"
        )
    return "\n".join(lines)


def room_slot_debug_note(layout: TokenLayout, patch_units: Sequence[PatchUnit]) -> str:
    room_token_ids = [int(layout.prompt_input_ids[position]) for position in layout.room_positions]
    room_bucket = next(
        (unit.cache_bucket for unit in patch_units if unit.kind == "room_slot"),
        f"room_span_len_{layout.room_span_len}_not_patched",
    )
    return (
        f"room_text={layout.room_text!r} "
        f"room_span=({layout.room_positions[0]},{layout.room_positions[-1] + 1}) "
        f"room_span_len={layout.room_span_len} "
        f"room_token_ids={room_token_ids} "
        f"chosen_cache_bucket={room_bucket}"
    )


@contextmanager
def temporary_layer_wrappers(layers: Sequence[Any], wrapper_factory: Any) -> Iterator[None]:
    original_forwards: Dict[int, Any] = {}
    try:
        for layer_idx, layer in enumerate(layers):
            original_forwards[layer_idx] = layer.forward
            layer.forward = wrapper_factory(layer_idx, layer.forward)
        yield
    finally:
        for layer_idx, layer in enumerate(layers):
            if layer_idx in original_forwards:
                layer.forward = original_forwards[layer_idx]


def allowed_key_positions(
    query_idx: int,
    carrier_index: int,
    key_len: int,
    stage: str,
) -> List[int]:
    allowed: set[int] = {int(query_idx)}

    if query_idx == carrier_index and stage == "transfer":
        allowed.update(range(0, min(query_idx, key_len - 1) + 1))
    elif query_idx > carrier_index:
        allowed.add(query_idx - 1)

    return sorted(key for key in allowed if 0 <= key < key_len)


def build_af1_attention_allow_matrix(
    *,
    query_len: int,
    key_len: int,
    carrier_index: int,
    stage: str,
) -> torch.Tensor:
    return build_prompt_allow_matrix(
        query_len=query_len,
        key_len=key_len,
        prompt_len=query_len,
        device=torch.device("cpu"),
        allowed_keys_by_query_fn=lambda query_idx: allowed_key_positions(
            query_idx=query_idx,
            carrier_index=carrier_index,
            key_len=key_len,
            stage=stage,
        ),
    )


def build_af1_sdpa_attention_mask(
    *,
    hidden_states: torch.Tensor,
    raw_attention_mask: Optional[torch.Tensor],
    model_config: Any,
    attention_type: str,
    carrier_index: int,
    stage: str,
    cache_position: Optional[torch.Tensor] = None,
    past_key_values: Optional[Any] = None,
) -> torch.Tensor:
    key_len = int(hidden_states.shape[1]) if raw_attention_mask is None else int(raw_attention_mask.shape[-1])
    allow_matrix = build_af1_attention_allow_matrix(
        query_len=int(hidden_states.shape[1]),
        key_len=key_len,
        carrier_index=carrier_index,
        stage=stage,
    ).to(hidden_states.device)
    mask = build_sdpa_layer_mask(
        model_config=model_config,
        attention_type=attention_type,
        hidden_states=hidden_states,
        raw_attention_mask=raw_attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        allow_matrix=allow_matrix,
    )
    if mask is None:
        raise RuntimeError("AF1 SDPA masking unexpectedly produced no materialized mask.")
    return mask


def validate_attention_mask_behavior(
    layout: TokenLayout,
) -> List[str]:
    non_carrier_query = next(
        (
            position
            for position in layout.fixed_template_positions
            if position != layout.carrier_index
        ),
        max(0, layout.carrier_index - 1),
    )
    transfer_keys = allowed_key_positions(
        query_idx=layout.carrier_index,
        carrier_index=layout.carrier_index,
        key_len=layout.prompt_len,
        stage="transfer",
    )
    compute_keys = allowed_key_positions(
        query_idx=layout.carrier_index,
        carrier_index=layout.carrier_index,
        key_len=layout.prompt_len,
        stage="compute",
    )
    non_carrier_keys = allowed_key_positions(
        query_idx=non_carrier_query,
        carrier_index=layout.carrier_index,
        key_len=layout.prompt_len,
        stage="transfer",
    )

    expected_non_carrier = [non_carrier_query]
    if non_carrier_keys != expected_non_carrier:
        raise RuntimeError(
            f"AF1 mask validation failed for non-carrier token {non_carrier_query}: "
            f"expected {expected_non_carrier}, got {non_carrier_keys}"
        )

    expected_transfer = list(range(0, layout.carrier_index + 1))
    if transfer_keys != expected_transfer:
        raise RuntimeError(
            f"AF1 mask validation failed for carrier transfer stage: expected {expected_transfer}, got {transfer_keys}"
        )

    expected_compute = [layout.carrier_index]
    if compute_keys != expected_compute:
        raise RuntimeError(
            f"AF1 mask validation failed for carrier compute stage: expected {expected_compute}, got {compute_keys}"
        )

    layers = get_layers(_model())
    if not layers:
        raise RuntimeError("AF1 validation could not find any decoder layers.")
    validation_config = getattr(getattr(layers[0], "self_attn", None), "config", None)
    if validation_config is None:
        raise RuntimeError("AF1 validation could not resolve the decoder attention config.")
    prompt_len = int(layout.prompt_len)
    synthetic_hidden_states = torch.zeros((1, prompt_len, 1), dtype=torch.float32)
    synthetic_attention_mask = torch.ones((1, prompt_len), dtype=torch.bool)
    transfer_sdpa_mask = build_af1_sdpa_attention_mask(
        hidden_states=synthetic_hidden_states,
        raw_attention_mask=synthetic_attention_mask,
        model_config=validation_config,
        attention_type="full_attention",
        carrier_index=layout.carrier_index,
        stage="transfer",
    )
    compute_sdpa_mask = build_af1_sdpa_attention_mask(
        hidden_states=synthetic_hidden_states,
        raw_attention_mask=synthetic_attention_mask,
        model_config=validation_config,
        attention_type="full_attention",
        carrier_index=layout.carrier_index,
        stage="compute",
    )
    actual_transfer_keys = allowed_key_positions_from_mask(
        transfer_sdpa_mask,
        query_idx=int(layout.carrier_index),
        key_len=prompt_len,
    )
    actual_compute_keys = allowed_key_positions_from_mask(
        compute_sdpa_mask,
        query_idx=int(layout.carrier_index),
        key_len=prompt_len,
    )
    if actual_transfer_keys != expected_transfer:
        raise RuntimeError(
            f"AF1 SDPA validation failed for carrier transfer stage: "
            f"expected {expected_transfer}, got {actual_transfer_keys}"
        )
    if actual_compute_keys != expected_compute:
        raise RuntimeError(
            f"AF1 SDPA validation failed for carrier compute stage: "
            f"expected {expected_compute}, got {actual_compute_keys}"
        )

    return [
        f"non_carrier_query={non_carrier_query} allowed_keys={non_carrier_keys}",
        (
            "af1_sdpa_mask_materialization: "
            f"attention_type=full_attention transfer_mask_dtype={transfer_sdpa_mask.dtype} "
            f"compute_mask_dtype={compute_sdpa_mask.dtype}"
        ),
        f"carrier_transfer allowed_keys={actual_transfer_keys}",
        f"carrier_compute allowed_keys={actual_compute_keys}",
    ]


def stage_for_layer(layer_idx: int, wait_until_layer: int, transfer_layers: int) -> str:
    if layer_idx <= wait_until_layer:
        return "normal"
    if layer_idx <= wait_until_layer + transfer_layers:
        return "transfer"
    return "compute"


def run_clean_model(
    inputs: Dict[str, torch.Tensor],
    output_attentions: bool = False,
    output_hidden_states: bool = False,
) -> Any:
    _prepare_forward_backend(
        path_name="af1_utils_clean_forward",
        requires_abp_mask=False,
        output_attentions=output_attentions,
    )
    with torch.inference_mode():
        return _model()(
            **inputs,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


def _cache_keys_missing(cache: Dict[str, Any], patch_units: Sequence[PatchUnit]) -> List[str]:
    cache_means = cache["key_to_mean"]
    missing: List[str] = []
    for unit in patch_units:
        for cache_key in unit.cache_keys:
            if cache_key not in cache_means:
                missing.append(cache_key)
    return missing


def run_model_with_af1(
    inputs: Dict[str, torch.Tensor],
    layout: TokenLayout,
    cache: Dict[str, Any],
    wait_until_layer: int,
    transfer_layers: int,
    patch_units: Sequence[PatchUnit],
    apply_patch: bool,
    apply_mask: bool,
    output_attentions: bool = False,
) -> Any:
    layers = get_layers(_model())
    if wait_until_layer < 0 or wait_until_layer >= len(layers):
        raise ValueError(f"wait_until_layer={wait_until_layer} is outside valid range [0, {len(layers) - 1}]")

    missing_cache_keys = _cache_keys_missing(cache, patch_units) if apply_patch else []
    if missing_cache_keys:
        raise RuntimeError(
            f"Missing AF1 cache keys for sample_id={layout.sample_id}: {sorted(set(missing_cache_keys))[:10]}"
        )
    path_name = "af1_utils_clean_forward"
    if apply_patch and apply_mask:
        path_name = "af1_utils_patch_and_mask_forward"
    elif apply_patch:
        path_name = "af1_utils_patch_only_forward"
    elif apply_mask:
        path_name = "af1_utils_mask_only_forward"
    _prepare_forward_backend(
        path_name=path_name,
        requires_abp_mask=bool(apply_mask),
        output_attentions=output_attentions,
    )
    raw_attention_mask = inputs.get("attention_mask")

    def wrapper_factory(layer_idx: int, original_forward: Any) -> Any:
        layer_module = layers[layer_idx]
        attention_type = str(getattr(layer_module, "attention_type", "full_attention"))
        attention_config = getattr(getattr(layer_module, "self_attn", None), "config", None)
        if attention_config is None:
            raise RuntimeError(f"Layer {layer_idx} does not expose a self-attention config.")

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            stage = stage_for_layer(layer_idx, wait_until_layer=wait_until_layer, transfer_layers=transfer_layers)

            if apply_mask and stage != "normal":
                kwargs["attention_mask"] = build_af1_sdpa_attention_mask(
                    hidden_states=hidden_states,
                    raw_attention_mask=raw_attention_mask,
                    model_config=attention_config,
                    attention_type=attention_type,
                    carrier_index=layout.carrier_index,
                    stage=stage,
                    cache_position=kwargs.get("cache_position"),
                    past_key_values=kwargs.get("past_key_values"),
                )

            outputs = original_forward(*args, **kwargs)
            if not apply_patch or layer_idx != wait_until_layer:
                return outputs

            hidden_out = outputs[0]
            if hidden_out.dim() != 3:
                raise RuntimeError(f"Expected rank-3 hidden states at layer {layer_idx}, got {tuple(hidden_out.shape)}")
            hidden_out = hidden_out.clone()

            for unit in patch_units:
                for position, cache_key in zip(unit.positions, unit.cache_keys):
                    mean_tensor = cache["key_to_mean"][cache_key][layer_idx].to(
                        device=hidden_out.device,
                        dtype=hidden_out.dtype,
                    )
                    hidden_out[:, int(position), :] = mean_tensor.view(1, -1)

            return (hidden_out,) + tuple(outputs[1:])

        return wrapped_forward

    with temporary_layer_wrappers(layers, wrapper_factory):
        with torch.inference_mode():
            return _model()(
                **inputs,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=True,
            )


def sequence_logprob_from_outputs(
    outputs: Any,
    prompt_len: int,
    answer_token_ids: List[int],
) -> float:
    return float(
        core.sequence_logprob_from_logits(
            outputs.logits,
            prompt_len=prompt_len,
            answer_token_ids=answer_token_ids,
        )[0].item()
    )


def score_valid_numeric_answers_with_runner(
    inputs: Dict[str, torch.Tensor],
    prompt_len: int,
    num_frames: int,
    runner: Any,
) -> Dict[str, Any]:
    scores_by_answer: Dict[str, float] = {}
    for value in range(num_frames + 1):
        answer_text = str(value)
        answer_ids = core.token_ids_of_answer(answer_text)
        scoring_inputs = core.append_answer_tokens_for_scoring(inputs, answer_ids)
        outputs = runner(scoring_inputs, answer_ids)
        scores_by_answer[answer_text] = sequence_logprob_from_outputs(
            outputs,
            prompt_len=prompt_len,
            answer_token_ids=answer_ids,
        )

    ranked_scores = sorted(scores_by_answer.items(), key=lambda item: item[1], reverse=True)
    best_answer_text, best_answer_score = ranked_scores[0]
    second_best_score = ranked_scores[1][1] if len(ranked_scores) > 1 else float("-inf")
    score_values = torch.tensor(list(scores_by_answer.values()), dtype=torch.float64)
    log_denom = torch.logsumexp(score_values, dim=0)
    probs_by_answer = {
        answer_text: float(torch.exp(torch.tensor(score, dtype=torch.float64) - log_denom).item())
        for answer_text, score in scores_by_answer.items()
    }
    return {
        "scores_by_answer": scores_by_answer,
        "probs_by_answer": probs_by_answer,
        "best_answer_text": str(best_answer_text),
        "best_score": float(best_answer_score),
        "margin_over_second": float(best_answer_score - second_best_score),
    }


def _cache_path_for_mode(
    seq_len: int,
    wait_patch_mode: str,
    cache_dir: Path,
    attention_backend: str,
) -> Path:
    model_slug = canonical_model_slug(DEFAULT_MODEL_ID)
    return (
        cache_dir
        / model_slug
        / _backend_cache_component(attention_backend)
        / f"seq_len_{seq_len}"
        / f"{wait_patch_mode}.pt"
    )


def load_or_compute_mean_cache(
    seq_len: int,
    sample_dirs: Sequence[Path],
    wait_patch_mode: str,
    cache_dir: Path,
    recompute_cache: bool,
    dtype: torch.dtype,
    debug_tokenization: bool,
) -> Tuple[Dict[str, Any], TokenLayout]:
    attention_backend = _prepare_forward_backend(
        path_name="af1_utils_clean_forward",
        requires_abp_mask=False,
        output_attentions=False,
    )
    cache_path = _cache_path_for_mode(
        seq_len=seq_len,
        wait_patch_mode=wait_patch_mode,
        cache_dir=cache_dir,
        attention_backend=attention_backend,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not recompute_cache:
        cache, migrated_old_pickle = load_af1_cache_file(cache_path)
        cache["key_to_mean"] = {str(key): value for key, value in cache["key_to_mean"].items()}
        reference_layout = deserialize_token_layout(cache["reference_layout"])
        cache["reference_layout"] = serialize_token_layout(reference_layout)
        if migrated_old_pickle:
            torch.save(cache, cache_path)
            print(f"[cache] migrated legacy pickle format to safe format: {cache_path}")
        print(
            f"[cache] loaded {cache_path} "
            f"num_keys={len(cache['key_to_mean'])} "
            f"num_samples={cache['metadata']['num_samples']}"
        )
        return cache, reference_layout

    if not sample_dirs:
        raise RuntimeError(f"seq_len={seq_len}: no sample dirs available for AF1 cache generation")

    layers = get_layers(_model())
    reference_layout: Optional[TokenLayout] = None
    key_to_sum: Dict[str, torch.Tensor] = {}
    key_to_count: Dict[str, int] = {}
    key_to_meta: Dict[str, Dict[str, Any]] = {}
    sample_count = 0

    for sample_idx, sample_dir in enumerate(sample_dirs):
        sample_id, frames, question, _, _ = load_mmred_sample(sample_dir)
        inputs = move_inputs_to_model_device(core.build_inputs(frames, question))
        layout = build_token_layout(sample_id=sample_id, frames=frames, question=question, inputs=inputs)

        if reference_layout is None:
            reference_layout = layout
        else:
            assert_layout_patchable(reference_layout, layout, wait_patch_mode=wait_patch_mode)

        patch_units = patch_units_for_mode(layout, wait_patch_mode=wait_patch_mode)
        if debug_tokenization and sample_idx < 2:
            print(f"[cache] sample_id={sample_id}\n{format_token_debug_rows(layout, patch_units)}")
            print(f"[cache] {room_slot_debug_note(layout, patch_units)}")

        outputs = run_clean_model(inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) != len(layers) + 1:
            raise RuntimeError("Model did not return per-layer hidden states needed for AF1 cache generation")

        for unit in patch_units:
            for position, cache_key in zip(unit.positions, unit.cache_keys):
                position_tensor = torch.stack(
                    [hidden_states[layer_idx + 1][0, int(position), :].detach().to(dtype=torch.float32).cpu() for layer_idx in range(len(layers))],
                    dim=0,
                )
                if cache_key not in key_to_sum:
                    key_to_sum[cache_key] = torch.zeros_like(position_tensor)
                    key_to_count[cache_key] = 0
                    key_to_meta[cache_key] = {
                        "unit_name": unit.name,
                        "unit_kind": unit.kind,
                        "cache_bucket": unit.cache_bucket,
                    }
                key_to_sum[cache_key] += position_tensor
                key_to_count[cache_key] += 1

        sample_count += 1

    if reference_layout is None:
        raise RuntimeError(f"seq_len={seq_len}: AF1 cache generation produced no reference layout")

    key_to_mean = {
        cache_key: (key_to_sum[cache_key] / float(key_to_count[cache_key])).to(dtype=dtype)
        for cache_key in sorted(key_to_sum.keys())
    }
    cache = {
        "key_to_mean": key_to_mean,
        "key_to_count": key_to_count,
        "key_to_meta": key_to_meta,
        "reference_layout": serialize_token_layout(reference_layout),
        "metadata": {
            "model_name": DEFAULT_MODEL_ID,
            "seq_len": int(seq_len),
            "wait_patch_mode": wait_patch_mode,
            "num_layers": len(layers),
            "num_keys": len(key_to_mean),
            "num_samples": sample_count,
            "attention_backend": attention_backend,
            "cache_policy": (
                "position-aware means; fixed template text keyed by template order; "
                "variable text keyed by token identity; room slot additionally bucketed by room span length"
            ),
        },
    }
    torch.save(cache, cache_path)
    print(
        f"[cache] wrote {cache_path} "
        f"num_keys={len(key_to_mean)} "
        f"num_samples={sample_count}"
    )
    return cache, reference_layout


def evaluate_af1_sample(
    sample_dir: Path,
    cache: Dict[str, Any],
    reference_layout: TokenLayout,
    wait_patch_mode: str,
    wait_until_layer: int,
    transfer_layers: int,
    debug_tokenization: bool,
    sample_index: int,
    mask_only: bool,
    patch_only: bool,
) -> Dict[str, Any]:
    sample_id, frames, question, _, answer_text = load_mmred_sample(sample_dir)
    clean_inputs = move_inputs_to_model_device(core.build_inputs(frames, question))
    layout = build_token_layout(sample_id=sample_id, frames=frames, question=question, inputs=clean_inputs)
    compatibility = assert_layout_patchable(reference_layout, layout, wait_patch_mode=wait_patch_mode)
    patch_units = patch_units_for_mode(layout, wait_patch_mode=wait_patch_mode)

    if debug_tokenization and sample_index < 3:
        print(
            f"[debug] sample_id={sample_id} carrier_index={layout.carrier_index} "
            f"carrier_token={layout.carrier_token_text!r}"
        )
        print(format_token_debug_rows(layout, patch_units))
        print(f"[debug] {room_slot_debug_note(layout, patch_units)}")

    prompt_len = layout.prompt_len
    gold_answer_text = str(answer_text).strip()
    apply_patch = not mask_only
    apply_mask = not patch_only

    clean_runner = lambda scoring_inputs, answer_ids: run_clean_model(scoring_inputs, output_attentions=False)
    af1_runner = lambda scoring_inputs, answer_ids: run_model_with_af1(
        scoring_inputs,
        layout=layout,
        cache=cache,
        wait_until_layer=wait_until_layer,
        transfer_layers=transfer_layers,
        patch_units=patch_units,
        apply_patch=apply_patch,
        apply_mask=apply_mask,
        output_attentions=False,
    )

    clean_metrics = score_valid_numeric_answers_with_runner(
        clean_inputs,
        prompt_len=prompt_len,
        num_frames=len(frames),
        runner=clean_runner,
    )
    af1_metrics = score_valid_numeric_answers_with_runner(
        clean_inputs,
        prompt_len=prompt_len,
        num_frames=len(frames),
        runner=af1_runner,
    )

    clean_pred = str(clean_metrics["best_answer_text"]).strip()
    af1_pred = str(af1_metrics["best_answer_text"]).strip()
    clean_correct = clean_pred == gold_answer_text
    af1_correct = af1_pred == gold_answer_text
    num_patched_positions = sum(len(unit.positions) for unit in patch_units)

    return {
        "model": DEFAULT_MODEL_ID,
        "sample_id": sample_id,
        "seq_len": len(frames),
        "gold_answer": gold_answer_text,
        "clean_pred": clean_pred,
        "clean_correct": int(clean_correct),
        "clean_gold_prob": float(clean_metrics["probs_by_answer"].get(gold_answer_text, 0.0)),
        "clean_best_score": float(clean_metrics["best_score"]),
        "clean_margin_over_second": float(clean_metrics["margin_over_second"]),
        "af1_pred": af1_pred,
        "af1_correct": int(af1_correct),
        "af1_gold_prob": float(af1_metrics["probs_by_answer"].get(gold_answer_text, 0.0)),
        "af1_best_score": float(af1_metrics["best_score"]),
        "af1_margin_over_second": float(af1_metrics["margin_over_second"]),
        "carrier_index": layout.carrier_index,
        "carrier_token": layout.carrier_token_text,
        "layout_match_status": str(compatibility["status"]),
        "layout_match_details": str(compatibility["summary"]),
        "num_patch_units": int(len(patch_units)),
        "num_patched_positions": int(num_patched_positions),
        "wait_patch_mode": wait_patch_mode,
        "wait_until_layer": int(wait_until_layer),
        "transfer_layers": int(transfer_layers),
        "mask_only": int(bool(mask_only)),
        "patch_only": int(bool(patch_only)),
        "room_text": layout.room_text,
        "room_span_len": int(layout.room_span_len),
    }


def summarize_seq_results(
    model_name: str,
    seq_len: int,
    sample_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    n_total = len(sample_rows)
    n_clean_correct = sum(int(row["clean_correct"]) for row in sample_rows)
    n_af1_correct = sum(int(row["af1_correct"]) for row in sample_rows)
    n_both_correct = sum(
        int(bool(row["clean_correct"]) and bool(row["af1_correct"]))
        for row in sample_rows
    )
    clean_acc = (n_clean_correct / float(n_total)) if n_total else 0.0
    af1_acc = (n_af1_correct / float(n_total)) if n_total else 0.0
    af1_faith = (n_both_correct / float(n_clean_correct)) if n_clean_correct else 0.0
    return {
        "model": model_name,
        "seq_len": int(seq_len),
        "n_total": int(n_total),
        "n_clean_correct": int(n_clean_correct),
        "clean_acc": float(clean_acc),
        "af1_acc": float(af1_acc),
        "af1_faith": float(af1_faith),
    }


def format_summary_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = ["model", "seq_len", "n_total", "n_clean_correct", "clean_acc", "af1_acc", "af1_faith"]
    values = [
        [
            str(row["model"]),
            str(row["seq_len"]),
            str(row["n_total"]),
            str(row["n_clean_correct"]),
            f"{float(row['clean_acc']):.4f}",
            f"{float(row['af1_acc']):.4f}",
            f"{float(row['af1_faith']):.4f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(value[col_idx]) for value in values)) if values else len(header)
        for col_idx, header in enumerate(headers)
    ]
    header_row = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_row = "|-" + "-|-".join("-" * widths[idx] for idx in range(len(headers))) + "-|"
    data_rows = [
        "| " + " | ".join(value[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for value in values
    ]
    return "\n".join([header_row, sep_row] + data_rows)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_markdown_summary(
    path: Path,
    config: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    tokenization_notes: Sequence[str],
    cache_notes: Sequence[str],
    validation_notes: Sequence[str],
    elapsed_seconds: float,
) -> None:
    lines = [
        "# AF1 Qwen VL Faithfulness",
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
        "## Tokenization Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in tokenization_notes)
    lines.extend(["", "## Cache Notes", ""])
    lines.extend(f"- {note}" for note in cache_notes)
    lines.extend(["", "## Validation", ""])
    lines.extend(f"- {note}" for note in validation_notes)
    lines.extend(["", "## Runtime", "", f"- {format_runtime(elapsed_seconds)}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def shuffled_sample_dirs(data_root: Path, max_samples: Optional[int], seed: int) -> List[Path]:
    sample_dirs = list(iter_sample_dirs(data_root))
    rng = random.Random(seed)
    rng.shuffle(sample_dirs)
    if max_samples is not None and max_samples > 0:
        sample_dirs = sample_dirs[:max_samples]
    return sample_dirs


def inspect_room_span_examples(sample_dirs: Sequence[Path], max_to_scan: int = 64) -> List[str]:
    notes: List[str] = []
    seen_lengths: set[int] = set()
    for sample_dir in list(sample_dirs)[:max_to_scan]:
        sample_id, frames, question, _, _ = load_mmred_sample(sample_dir)
        inputs = move_inputs_to_model_device(core.build_inputs(frames, question))
        layout = build_token_layout(sample_id=sample_id, frames=frames, question=question, inputs=inputs)
        if layout.room_span_len in seen_lengths:
            continue
        seen_lengths.add(layout.room_span_len)
        room_token_ids = [int(layout.prompt_input_ids[position]) for position in layout.room_positions]
        notes.append(
            f"room-span example sample_id={sample_id} room={layout.room_text!r} "
            f"room_span_len={layout.room_span_len} room_token_ids={room_token_ids}"
        )
        if seen_lengths == {1, 2}:
            break
    return notes


def validate_single_sample(
    sample_dir: Path,
    cache: Dict[str, Any],
    reference_layout: TokenLayout,
    wait_patch_mode: str,
    wait_until_layer: int,
    transfer_layers: int,
    mask_only: bool,
    patch_only: bool,
) -> List[str]:
    sample_id, frames, question, _, answer_text = load_mmred_sample(sample_dir)
    clean_inputs = move_inputs_to_model_device(core.build_inputs(frames, question))
    layout = build_token_layout(sample_id=sample_id, frames=frames, question=question, inputs=clean_inputs)
    compatibility = assert_layout_patchable(reference_layout, layout, wait_patch_mode=wait_patch_mode)
    patch_units = patch_units_for_mode(layout, wait_patch_mode=wait_patch_mode)
    if not patch_units:
        raise RuntimeError("Validation sample produced no patch units")

    mask_notes = validate_attention_mask_behavior(layout=layout)

    clean_metrics = score_valid_numeric_answers_with_runner(
        clean_inputs,
        prompt_len=layout.prompt_len,
        num_frames=len(frames),
        runner=lambda scoring_inputs, answer_ids: run_clean_model(scoring_inputs, output_attentions=False),
    )
    af1_metrics = score_valid_numeric_answers_with_runner(
        clean_inputs,
        prompt_len=layout.prompt_len,
        num_frames=len(frames),
        runner=lambda scoring_inputs, answer_ids: run_model_with_af1(
            scoring_inputs,
            layout=layout,
            cache=cache,
            wait_until_layer=wait_until_layer,
            transfer_layers=transfer_layers,
            patch_units=patch_units,
            apply_patch=not mask_only,
            apply_mask=not patch_only,
            output_attentions=False,
        ),
    )

    notes = [
        f"validation sample_id={sample_id}",
        f"carrier_index={layout.carrier_index} carrier_token={layout.carrier_token_text!r}",
        f"patch_units={len(patch_units)} patched_positions={sum(len(unit.positions) for unit in patch_units)}",
        room_slot_debug_note(layout, patch_units),
        f"layout_match_status={compatibility['status']} details={compatibility['summary']}",
        f"clean_prediction={clean_metrics['best_answer_text']} gold={str(answer_text).strip()}",
        f"af1_prediction={af1_metrics['best_answer_text']} gold={str(answer_text).strip()}",
    ]
    notes.extend(mask_notes)
    notes.append(format_token_debug_rows(layout, patch_units))
    return notes

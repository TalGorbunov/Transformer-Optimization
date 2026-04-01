"""
AF1 wait-only sweep for MMRed with Qwen-VL.

This script minimally follows the structure of
`af1_qwen_vl_frame_cama.py`, but only supports the existing `wait_only`
intervention. It sweeps over explicit `wait_layer` values, runs the same
sample preparation / layout validation / donor selection / conditional-mean
patching pipeline for each wait layer, and plots score drop vs wait layer.

Important method notes:
- `wait_layer` follows AF1's `L_wait` semantics exactly:
  - if `wait_layer > 0`, replace `x^(L_wait)` at the output of layer
    index `wait_layer - 1`
  - if `wait_layer == 0`, replace `x^(0)` before layer 0 runs
- This script does not apply ABP masking. It is strictly the existing
  `wait_only` intervention.
- Donor hybrids keep the target prompt/question fixed while donor frames vary,
  and wait-boundary patching is applied to frame groups plus one all-non-frame
  prompt token set.
- `score_drop` is defined from the clean top-1 answer's sequence score:
  `clean_score - intervention_score`.

Example:
python evaluations/scripts/af1/wait_only_sweep.py \
  --split train \
  --seq_lens 8 \
  --max_samples 8 \
  --wait_layers 0 4 8 12 16 20 24 28 32 36 40 \
  --transfer_layers 2 \
  --k_donors 4 \
  --output_dir outputs/af1_qwen_vl_wait_only_sweep
"""

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as core
from evaluations.helpers import utils as eval_utils
from models.model import (
    MODEL_ID,
    find_subsequence,
    force_eager_attention_backend,
    get_layers,
    image_token_groups,
    model as base_model,
    processor,
)

_DONOR_POLICY = "same_seq_len_validated_layout_seeded_shuffle_exclude_target"
_MODE = "wait_only"
_SUMMARY_BY_WAIT_LAYER_FIELDS = [
    "model",
    "seq_len",
    "wait_layer",
    "n_total",
    "n_used",
    "n_clean_correct",
    "clean_acc",
    "af1_acc",
    "af1_faith",
    "mean_score_drop",
    "median_score_drop",
]
_PER_SAMPLE_BY_WAIT_LAYER_FIELDS = [
    "model",
    "mode",
    "sample_id",
    "seq_len",
    "used",
    "gold_answer",
    "clean_pred",
    "clean_correct",
    "clean_gold_prob",
    "clean_best_score",
    "clean_score",
    "clean_margin_over_second",
    "af1_pred",
    "af1_correct",
    "af1_gold_prob",
    "af1_best_score",
    "intervention_score",
    "af1_margin_over_second",
    "score_drop",
    "carrier_index",
    "carrier_token",
    "wait_layer",
    "transfer_layers",
    "transfer_layer_indices",
    "k_donors",
    "num_frames",
    "num_frame_groups",
    "prompt_len",
    "image_tokens_per_frame",
    "room_text",
    "skipped_reason",
    "donor_ids",
    "layout_match_status",
    "layout_match_details",
]


@dataclass(frozen=True)
class SampleLayout:
    sample_id: str
    seq_len: int
    prompt_len: int
    carrier_index: int
    carrier_token_id: int
    carrier_token_text: str
    prompt_family_key: str
    frame_groups: Tuple[Tuple[int, ...], ...]
    image_tokens_per_frame: Tuple[int, ...]
    room_text: str
    room_positions: Tuple[int, ...]
    character_positions: Tuple[int, ...]
    room_span_len: int
    prompt_input_ids: Tuple[int, ...]
    prompt_decoded_tokens: Tuple[str, ...]


@dataclass
class PreparedSample:
    sample_dir: Path
    sample_id: str
    frames: List[Any]
    question: str
    gold_answer: str
    layout: SampleLayout
    inputs_cpu: Dict[str, torch.Tensor]


@dataclass(frozen=True)
class WaitOnlyPolicy:
    prompt_len: int
    carrier_index: int
    wait_layer: int
    transfer_layers: int
    num_model_layers: int

    @property
    def transfer_layer_indices(self) -> Tuple[int, ...]:
        return tuple(range(self.wait_layer, self.wait_layer + self.transfer_layers))


class _WaitBoundaryCaptured(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Sweep AF1 wait-boundary frame-group plus non-frame prompt patching on MMRed/Qwen-VL "
            "without ABP masking and plot score drop vs wait layer."
        )
    )
    ap.add_argument("--model_name", type=str, default=MODEL_ID)
    ap.add_argument("--data_root_base", type=str, default="data/mmred_images")
    ap.add_argument("--split", type=str, default="all")
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--max_samples", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument(
        "--wait_layers",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Explicit AF1 wait boundaries L_wait measured in number of layers. "
            "Accepts integers and inclusive start:end:step ranges like 20:40:2. "
            "If wait_layer > 0, x^(L_wait) is patched at the output of layer wait_layer - 1."
        ),
    )
    ap.add_argument(
        "--transfer_layers",
        type=int,
        default=2,
        help="Kept for parity with the source script; no ABP masking is applied in this sweep.",
    )
    ap.add_argument(
        "--k_donors",
        type=int,
        default=4,
        help=(
            "Maximum number of compatible donors to average for each frame-group "
            "or non-frame conditional mean."
        ),
    )
    ap.add_argument("--cache_dir", type=str, default="outputs/af1_frame_cama_cache")
    ap.add_argument("--recompute_cache", action="store_true")
    ap.add_argument("--output_dir", type=str, default="outputs/af1_qwen_vl_wait_only_sweep")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_hallway", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--debug_tokenization", action="store_true")
    return ap.parse_args()


def parse_wait_layers(raw_values: Sequence[str]) -> Tuple[List[int], int]:
    wait_layers: List[int] = []
    tick_step = 1
    for raw_value in raw_values:
        token = str(raw_value).strip()
        if not token:
            continue
        if ":" not in token:
            wait_layers.append(int(token))
            continue

        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid wait-layer range {token!r}; expected start:end:step."
            )
        start, end, step = (int(part) for part in parts)
        if step == 0:
            raise ValueError(f"Invalid wait-layer range {token!r}; step must be non-zero.")
        if step > 0 and start > end:
            raise ValueError(
                f"Invalid wait-layer range {token!r}; positive step requires start <= end."
            )
        if step < 0 and start < end:
            raise ValueError(
                f"Invalid wait-layer range {token!r}; negative step requires start >= end."
            )

        tick_step = abs(int(step))
        stop = end + (1 if step > 0 else -1)
        wait_layers.extend(list(range(start, stop, step)))

    if not wait_layers:
        raise ValueError("--wait_layers must resolve to at least one layer.")
    return wait_layers, int(tick_step)


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
        raise ValueError(f"Unsupported dtype={dtype_name!r}")
    return mapping[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_model_slug(model_name: str) -> str:
    return model_name.lower().replace("/", "_").replace("-", "")


def sanitize_token_text(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", "\\t") if text else "<empty>"


def decode_token_ids(token_ids: Sequence[int]) -> List[str]:
    return [
        processor.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]


def seq_len_data_root(data_root_base: Path, seq_len: int, split: str) -> Path:
    return data_root_base / f"seq_len_{seq_len}" / split


def model_runtime_info(requested_device: str, requested_dtype: str) -> Dict[str, str]:
    first_param = next(base_model.parameters())
    actual_device = str(first_param.device)
    actual_dtype = str(first_param.dtype)
    requested_device = str(requested_device).strip()
    if requested_device and requested_device != "auto":
        if requested_device.split(":")[0] != actual_device.split(":")[0]:
            raise RuntimeError(
                f"Requested --device={requested_device!r}, but the current model is loaded on {actual_device!r}."
            )
    return {
        "model_name": MODEL_ID,
        "requested_device": requested_device,
        "actual_model_device": actual_device,
        "requested_dtype": str(requested_dtype),
        "actual_model_dtype": actual_dtype,
    }


def move_inputs_to_model_device(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    device = next(base_model.parameters()).device
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


def batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _token_span_from_char_span(text: str, char_span: Tuple[int, int]) -> Tuple[int, int]:
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


def build_sample_layout(
    sample_id: str,
    frames: Sequence[Any],
    question: str,
    inputs: Dict[str, torch.Tensor],
) -> SampleLayout:
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

    character_token_span = _token_span_from_char_span(prompt_text, character_span_in_prompt)
    character_positions = _positions_from_token_span(prompt_text_start, character_token_span)
    room_token_span = _token_span_from_char_span(prompt_text, room_span_in_prompt)
    room_positions = _positions_from_token_span(prompt_text_start, room_token_span)
    if not room_positions:
        raise RuntimeError(f"sample_id={sample_id}: empty room token span")

    carrier_index = prompt_len - 1
    prompt_decoded_tokens = decode_token_ids(input_ids)

    frame_groups = image_token_groups(inputs["input_ids"][0].detach().cpu(), expected_num_frames=len(frames))
    if len(frame_groups) != len(frames):
        raise RuntimeError(
            f"sample_id={sample_id}: expected {len(frames)} frame groups but found {len(frame_groups)}"
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
        "room_span_len": int(layout.room_span_len),
    }


def layout_hash(layout: SampleLayout) -> str:
    payload = json.dumps(_layout_signature_payload(layout), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def format_token_debug_rows(layout: SampleLayout) -> str:
    frame_lookup: Dict[int, int] = {}
    for frame_idx, group in enumerate(layout.frame_groups):
        for position in group:
            frame_lookup[int(position)] = frame_idx

    lines = ["idx\tid\ttoken\ttags"]
    for idx, token_id in enumerate(layout.prompt_input_ids):
        tags: List[str] = []
        if idx == layout.carrier_index:
            tags.append("CARRIER")
        if idx in frame_lookup:
            tags.append(f"frame_{frame_lookup[idx]}")
        lines.append(
            f"{idx}\t{token_id}\t{sanitize_token_text(layout.prompt_decoded_tokens[idx])}\t{','.join(tags) or '-'}"
        )
    return "\n".join(lines)


def load_and_filter_sample_dirs(data_root: Path, max_samples: int, seed: int) -> List[Path]:
    sample_dirs = list(eval_utils.iter_sample_dirs(data_root))
    rng = random.Random(seed)
    rng.shuffle(sample_dirs)
    if max_samples > 0:
        sample_dirs = sample_dirs[:max_samples]
    return sample_dirs


def _sample_seed(seed: int, *parts: str) -> int:
    raw = "::".join([str(seed)] + [str(part) for part in parts])
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16], 16)


def _empty_row(model_name: str, sample_id: str, seq_len: int) -> Dict[str, Any]:
    row = {field: "" for field in _PER_SAMPLE_BY_WAIT_LAYER_FIELDS}
    row["model"] = model_name
    row["sample_id"] = sample_id
    row["seq_len"] = int(seq_len)
    return row


def inspect_and_validate_layout(
    reference_layout: SampleLayout,
    candidate_layout: SampleLayout,
    skip_hallway: bool,
) -> Dict[str, Any]:
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


def _prepare_sample(sample_dir: Path, skip_hallway: bool) -> Tuple[Optional[PreparedSample], Optional[Dict[str, Any]]]:
    sample_id, frames, question, _, answer_text = eval_utils.load_mmred_sample(sample_dir)
    parsed = eval_utils.parse_target_character_room(question)
    room_text = parsed[1] if parsed is not None else ""

    if skip_hallway and room_text.lower() == "hallway":
        row = _empty_row(MODEL_ID, sample_id=sample_id, seq_len=len(frames))
        row.update(
            {
                "mode": _MODE,
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
        inputs_cpu = core.build_inputs(frames, question)
        layout = build_sample_layout(sample_id=sample_id, frames=frames, question=question, inputs=inputs_cpu)
    except Exception as exc:
        row = _empty_row(MODEL_ID, sample_id=sample_id, seq_len=len(frames))
        row.update(
            {
                "mode": _MODE,
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


def _choose_reference_layout(samples: Sequence[PreparedSample]) -> Optional[SampleLayout]:
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


def select_donor_pool(
    target_sample: PreparedSample,
    compatible_samples: Sequence[PreparedSample],
    k_donors: int,
    seed: int,
) -> List[PreparedSample]:
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
        "layout": target_sample.layout,
    }


def all_non_frame_prompt_positions(layout: SampleLayout) -> Tuple[int, ...]:
    frame_positions = {int(position) for group in layout.frame_groups for position in group}
    return tuple(position for position in range(int(layout.prompt_len)) if position not in frame_positions)


def build_non_frame_hybrid_sample(
    target_sample: PreparedSample,
    donor_sample: PreparedSample,
) -> Dict[str, Any]:
    if target_sample.layout.seq_len != donor_sample.layout.seq_len:
        raise ValueError(
            f"Incompatible seq_len for non-frame hybrid sample: target={target_sample.layout.seq_len}, "
            f"donor={donor_sample.layout.seq_len}"
        )
    return {
        "sample_id": f"{target_sample.sample_id}__non_frame__donor_{donor_sample.sample_id}",
        "frames": list(donor_sample.frames),
        "question": target_sample.question,
        "layout": target_sample.layout,
    }


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


def _to_hidden_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x:
        return _to_hidden_tensor(x[0])
    raise TypeError(f"Unsupported layer output type: {type(x)}")


def build_wait_only_policy(
    layout: SampleLayout,
    wait_layer: int,
    transfer_layers: int,
) -> WaitOnlyPolicy:
    num_layers = len(get_layers(base_model))
    if wait_layer < 0 or wait_layer > num_layers:
        raise ValueError(f"wait_layer={wait_layer} must be in [0, {num_layers}]")
    if transfer_layers < 0:
        raise ValueError("transfer_layers must be non-negative")
    if wait_layer + transfer_layers > num_layers:
        raise ValueError(
            f"wait_layer + transfer_layers must be <= {num_layers}; "
            f"received {wait_layer} + {transfer_layers}"
        )
    return WaitOnlyPolicy(
        prompt_len=int(layout.prompt_len),
        carrier_index=int(layout.carrier_index),
        wait_layer=int(wait_layer),
        transfer_layers=int(transfer_layers),
        num_model_layers=int(num_layers),
    )


def _patch_frame_groups(
    hidden_states: torch.Tensor,
    layout: SampleLayout,
    frame_group_means: Dict[int, torch.Tensor],
) -> torch.Tensor:
    patched = hidden_states.clone()
    for frame_idx, positions in enumerate(layout.frame_groups):
        replacement = frame_group_means[frame_idx].to(device=patched.device, dtype=patched.dtype)
        patched[:, list(positions), :] = replacement.unsqueeze(0)
    return patched


def _patch_non_frame_prompt_tokens(
    patched_hidden_states: torch.Tensor,
    layout: SampleLayout,
    non_frame_mean_block: torch.Tensor,
) -> torch.Tensor:
    non_frame_positions = all_non_frame_prompt_positions(layout)
    if not non_frame_positions:
        return patched_hidden_states
    replacement = non_frame_mean_block.to(
        device=patched_hidden_states.device,
        dtype=patched_hidden_states.dtype,
    )
    patched_hidden_states[:, list(non_frame_positions), :] = replacement.unsqueeze(0)
    return patched_hidden_states


def _run_model_forward(
    inputs: Dict[str, torch.Tensor],
    output_hidden_states: bool = False,
    output_attentions: bool = False,
) -> Any:
    force_eager_attention_backend()
    with torch.inference_mode():
        return base_model(
            **inputs,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


def _capture_wait_boundary_blocks(
    inputs: Dict[str, torch.Tensor],
    wait_layer: int,
    positions: Sequence[int],
) -> torch.Tensor:
    layers = get_layers(base_model)
    if wait_layer < 0 or wait_layer > len(layers):
        raise ValueError(f"wait_layer={wait_layer} must be in [0, {len(layers)}]")

    capture_positions = [int(position) for position in positions]
    captured: Dict[str, torch.Tensor] = {}
    boundary_output_layer_idx = wait_layer - 1

    def wrapper_factory(layer_idx: int, original_forward: Any) -> Any:
        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError("Layer forward received no hidden_states")

            if wait_layer == 0 and layer_idx == 0:
                captured["block"] = hidden_states[:, capture_positions, :].detach().to(dtype=torch.float32).cpu()
                raise _WaitBoundaryCaptured

            outputs = original_forward(*args, **kwargs)
            if layer_idx == boundary_output_layer_idx:
                hidden_out = _to_hidden_tensor(outputs)
                captured["block"] = hidden_out[:, capture_positions, :].detach().to(dtype=torch.float32).cpu()
                raise _WaitBoundaryCaptured
            return outputs

        return wrapped_forward

    force_eager_attention_backend()
    with temporary_layer_wrappers(layers, wrapper_factory):
        with torch.inference_mode():
            try:
                base_model(
                    **inputs,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            except _WaitBoundaryCaptured:
                pass

    if "block" not in captured:
        raise RuntimeError(
            f"Failed to capture wait-boundary block for wait_layer={wait_layer} positions={capture_positions}"
        )
    return captured["block"]


def _conditional_mean_cache_path(
    cache_dir: Path,
    model_name: str,
    seq_len: int,
    target_sample_id: str,
    frame_idx: int,
    wait_layer: int,
    k_donors_used: int,
    donor_policy: str,
    donor_ids: Sequence[str],
    layout_hash_value: str,
) -> Path:
    donor_ids_hash = hashlib.sha1(json.dumps(list(donor_ids), sort_keys=True).encode("utf-8")).hexdigest()[:16]
    donor_policy_hash = hashlib.sha1(str(donor_policy).encode("utf-8")).hexdigest()[:10]
    return (
        cache_dir
        / canonical_model_slug(model_name)
        / f"seq_len_{seq_len}"
        / f"wait_{wait_layer}"
        / f"target_{target_sample_id}"
        / (
            f"frame_{frame_idx}_k_{k_donors_used}_policy_{donor_policy_hash}_"
            f"donors_{donor_ids_hash}_layout_{layout_hash_value}.pt"
        )
    )


def _non_frame_conditional_mean_cache_path(
    cache_dir: Path,
    model_name: str,
    seq_len: int,
    target_sample_id: str,
    wait_layer: int,
    k_donors_used: int,
    donor_policy: str,
    donor_ids: Sequence[str],
    layout_hash_value: str,
) -> Path:
    donor_ids_hash = hashlib.sha1(json.dumps(list(donor_ids), sort_keys=True).encode("utf-8")).hexdigest()[:16]
    donor_policy_hash = hashlib.sha1(str(donor_policy).encode("utf-8")).hexdigest()[:10]
    return (
        cache_dir
        / canonical_model_slug(model_name)
        / f"seq_len_{seq_len}"
        / f"wait_{wait_layer}"
        / f"target_{target_sample_id}"
        / (
            f"non_frame_k_{k_donors_used}_policy_{donor_policy_hash}_"
            f"donors_{donor_ids_hash}_layout_{layout_hash_value}.pt"
        )
    )


def compute_frame_group_conditional_mean(
    target_sample: PreparedSample,
    frame_idx: int,
    donor_samples: Sequence[PreparedSample],
    wait_layer: int,
    batch_size: int,
    cache_dir: Path,
    recompute_cache: bool,
    donor_policy: str,
) -> Tuple[torch.Tensor, bool]:
    if len(donor_samples) < 2:
        raise ValueError(
            f"Need at least two donors to estimate a conditional mean, got {len(donor_samples)} "
            f"for target={target_sample.sample_id} frame={frame_idx}"
        )

    donor_ids = [sample.sample_id for sample in donor_samples]
    cache_path = _conditional_mean_cache_path(
        cache_dir=cache_dir,
        model_name=MODEL_ID,
        seq_len=target_sample.layout.seq_len,
        target_sample_id=target_sample.sample_id,
        frame_idx=frame_idx,
        wait_layer=wait_layer,
        k_donors_used=len(donor_samples),
        donor_policy=donor_policy,
        donor_ids=donor_ids,
        layout_hash_value=layout_hash(target_sample.layout),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not recompute_cache:
        payload = torch.load(cache_path, map_location="cpu")
        return payload["mean_block"].to(dtype=torch.float32), True

    frame_positions = target_sample.layout.frame_groups[frame_idx]
    blocks: List[torch.Tensor] = []
    for donor_batch in batched(list(donor_samples), batch_size=batch_size):
        hybrid_inputs_list: List[Dict[str, torch.Tensor]] = []
        for donor_sample in donor_batch:
            hybrid = build_hybrid_sample(target_sample=target_sample, donor_sample=donor_sample, frame_idx=frame_idx)
            hybrid_inputs_cpu = core.build_inputs(hybrid["frames"], hybrid["question"])
            hybrid_inputs_list.append(move_inputs_to_model_device(hybrid_inputs_cpu))

        batched_inputs = core.concatenate_inputs_for_batch(hybrid_inputs_list)
        blocks.append(_capture_wait_boundary_blocks(batched_inputs, wait_layer=wait_layer, positions=frame_positions))

    mean_block = torch.cat(blocks, dim=0).mean(dim=0).to(dtype=torch.float32).cpu()
    torch.save(
        {
            "mean_block": mean_block,
            "metadata": {
                "model_name": MODEL_ID,
                "seq_len": int(target_sample.layout.seq_len),
                "target_sample_id": target_sample.sample_id,
                "frame_idx": int(frame_idx),
                "wait_layer": int(wait_layer),
                "k_donors_requested": int(len(donor_samples)),
                "k_donors_used": int(len(donor_samples)),
                "donor_policy": donor_policy,
                "donor_ids": list(donor_ids),
                "layout_hash": layout_hash(target_sample.layout),
                "cache_semantics": (
                    "frame-group conditional mean at x^(L_wait) estimated from hybrid contexts "
                    "that keep the target frame fixed, replace the other frames with donor frames, "
                    "and keep the target text prompt fixed"
                ),
            },
        },
        cache_path,
    )
    return mean_block, False


def compute_non_frame_conditional_mean(
    target_sample: PreparedSample,
    donor_samples: Sequence[PreparedSample],
    wait_layer: int,
    batch_size: int,
    cache_dir: Path,
    recompute_cache: bool,
    donor_policy: str,
) -> Tuple[torch.Tensor, bool]:
    if len(donor_samples) < 2:
        raise ValueError(
            f"Need at least two donors to estimate a non-frame conditional mean, got {len(donor_samples)} "
            f"for target={target_sample.sample_id}"
        )

    non_frame_positions = all_non_frame_prompt_positions(target_sample.layout)
    if not non_frame_positions:
        raise RuntimeError(f"target={target_sample.sample_id}: no non-frame prompt positions found")

    donor_ids = [sample.sample_id for sample in donor_samples]
    cache_path = _non_frame_conditional_mean_cache_path(
        cache_dir=cache_dir,
        model_name=MODEL_ID,
        seq_len=target_sample.layout.seq_len,
        target_sample_id=target_sample.sample_id,
        wait_layer=wait_layer,
        k_donors_used=len(donor_samples),
        donor_policy=donor_policy,
        donor_ids=donor_ids,
        layout_hash_value=layout_hash(target_sample.layout),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not recompute_cache:
        payload = torch.load(cache_path, map_location="cpu")
        return payload["mean_block"].to(dtype=torch.float32), True

    blocks: List[torch.Tensor] = []
    for donor_batch in batched(list(donor_samples), batch_size=batch_size):
        hybrid_inputs_list: List[Dict[str, torch.Tensor]] = []
        for donor_sample in donor_batch:
            hybrid = build_non_frame_hybrid_sample(target_sample=target_sample, donor_sample=donor_sample)
            hybrid_inputs_cpu = core.build_inputs(hybrid["frames"], hybrid["question"])
            hybrid_inputs_list.append(move_inputs_to_model_device(hybrid_inputs_cpu))

        batched_inputs = core.concatenate_inputs_for_batch(hybrid_inputs_list)
        blocks.append(
            _capture_wait_boundary_blocks(
                batched_inputs,
                wait_layer=wait_layer,
                positions=non_frame_positions,
            )
        )

    mean_block = torch.cat(blocks, dim=0).mean(dim=0).to(dtype=torch.float32).cpu()
    torch.save(
        {
            "mean_block": mean_block,
            "metadata": {
                "model_name": MODEL_ID,
                "seq_len": int(target_sample.layout.seq_len),
                "target_sample_id": target_sample.sample_id,
                "wait_layer": int(wait_layer),
                "k_donors_requested": int(len(donor_samples)),
                "k_donors_used": int(len(donor_samples)),
                "donor_policy": donor_policy,
                "donor_ids": list(donor_ids),
                "layout_hash": layout_hash(target_sample.layout),
                "num_positions": int(len(non_frame_positions)),
                "cache_semantics": (
                    "all-non-frame prompt conditional mean at x^(L_wait) estimated from hybrid contexts "
                    "that keep the target text prompt fixed and replace the entire frame set with donor frames"
                ),
            },
        },
        cache_path,
    )
    return mean_block, False


def compute_all_frame_group_means_for_sample(
    target_sample: PreparedSample,
    donor_samples: Sequence[PreparedSample],
    wait_layer: int,
    batch_size: int,
    cache_dir: Path,
    recompute_cache: bool,
    donor_policy: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, int]]:
    if len(donor_samples) < 2:
        raise ValueError(
            f"Need at least two donors to estimate conditional means for target={target_sample.sample_id}"
        )

    frame_means: Dict[int, torch.Tensor] = {}
    cache_hits = 0
    cache_misses = 0
    for frame_idx in range(target_sample.layout.seq_len):
        mean_block, cache_hit = compute_frame_group_conditional_mean(
            target_sample=target_sample,
            frame_idx=frame_idx,
            donor_samples=donor_samples,
            wait_layer=wait_layer,
            batch_size=batch_size,
            cache_dir=cache_dir,
            recompute_cache=recompute_cache,
            donor_policy=donor_policy,
        )
        frame_means[frame_idx] = mean_block
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
    return frame_means, {"cache_hits": cache_hits, "cache_misses": cache_misses}


def run_clean_model(
    inputs: Dict[str, torch.Tensor],
    output_attentions: bool = False,
) -> Any:
    return _run_model_forward(
        inputs,
        output_hidden_states=False,
        output_attentions=output_attentions,
    )


def run_model_with_intervention(
    inputs: Dict[str, torch.Tensor],
    layout: SampleLayout,
    frame_group_means: Dict[int, torch.Tensor],
    non_frame_prompt_mean: torch.Tensor,
    policy: WaitOnlyPolicy,
) -> Any:
    layers = get_layers(base_model)
    if int(policy.num_model_layers) != len(layers):
        raise RuntimeError(
            f"Wait-only policy expected {policy.num_model_layers} layers but model exposes {len(layers)} layers"
        )

    boundary_output_layer_idx = policy.wait_layer - 1

    def wrapper_factory(layer_idx: int, original_forward: Any) -> Any:
        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError("Layer forward received no hidden_states")

            if policy.wait_layer == 0 and layer_idx == 0:
                patched_hidden_states = _patch_frame_groups(
                    hidden_states,
                    layout=layout,
                    frame_group_means=frame_group_means,
                )
                patched_hidden_states = _patch_non_frame_prompt_tokens(
                    patched_hidden_states,
                    layout=layout,
                    non_frame_mean_block=non_frame_prompt_mean,
                )
                if args:
                    args = (patched_hidden_states,) + tuple(args[1:])
                else:
                    kwargs["hidden_states"] = patched_hidden_states

            outputs = original_forward(*args, **kwargs)
            if policy.wait_layer > 0 and layer_idx == boundary_output_layer_idx:
                hidden_out = _to_hidden_tensor(outputs)
                patched_hidden_out = _patch_frame_groups(
                    hidden_out,
                    layout=layout,
                    frame_group_means=frame_group_means,
                )
                patched_hidden_out = _patch_non_frame_prompt_tokens(
                    patched_hidden_out,
                    layout=layout,
                    non_frame_mean_block=non_frame_prompt_mean,
                )
                return (patched_hidden_out,) + tuple(outputs[1:])
            return outputs

        return wrapped_forward

    force_eager_attention_backend()
    with temporary_layer_wrappers(layers, wrapper_factory):
        with torch.inference_mode():
            return base_model(
                **inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )


def sequence_logprob_from_outputs(outputs: Any, prompt_len: int, answer_token_ids: List[int]) -> float:
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


def run_clean_sample(sample: PreparedSample) -> Dict[str, Any]:
    clean_inputs = move_inputs_to_model_device(sample.inputs_cpu)
    return score_valid_numeric_answers_with_runner(
        clean_inputs,
        prompt_len=sample.layout.prompt_len,
        num_frames=sample.layout.seq_len,
        runner=lambda scoring_inputs, answer_ids: run_clean_model(scoring_inputs),
    )


def run_intervention_sample(
    sample: PreparedSample,
    frame_group_means: Dict[int, torch.Tensor],
    non_frame_prompt_mean: torch.Tensor,
    policy: WaitOnlyPolicy,
) -> Dict[str, Any]:
    intervention_inputs = move_inputs_to_model_device(sample.inputs_cpu)
    return score_valid_numeric_answers_with_runner(
        intervention_inputs,
        prompt_len=sample.layout.prompt_len,
        num_frames=sample.layout.seq_len,
        runner=lambda scoring_inputs, answer_ids: run_model_with_intervention(
            scoring_inputs,
            layout=sample.layout,
            frame_group_means=frame_group_means,
            non_frame_prompt_mean=non_frame_prompt_mean,
            policy=policy,
        ),
    )


def _score_of_answer_text(metrics: Dict[str, Any], answer_text: str) -> float:
    return float(metrics["scores_by_answer"].get(answer_text, float("-inf")))


def _evaluated_row(
    sample: PreparedSample,
    clean_metrics: Dict[str, Any],
    af1_metrics: Dict[str, Any],
    donor_ids: Sequence[str],
    policy: WaitOnlyPolicy,
    k_donors_requested: int,
) -> Dict[str, Any]:
    clean_pred = str(clean_metrics["best_answer_text"]).strip()
    af1_pred = str(af1_metrics["best_answer_text"]).strip()
    clean_correct = int(clean_pred == sample.gold_answer)
    af1_correct = int(af1_pred == sample.gold_answer)
    clean_score = float(clean_metrics["best_score"])
    intervention_score = _score_of_answer_text(af1_metrics, clean_pred)
    row = _empty_row(MODEL_ID, sample_id=sample.sample_id, seq_len=sample.layout.seq_len)
    row.update(
        {
            "mode": _MODE,
            "used": 1,
            "gold_answer": sample.gold_answer,
            "clean_pred": clean_pred,
            "clean_correct": clean_correct,
            "clean_gold_prob": float(clean_metrics["probs_by_answer"].get(sample.gold_answer, 0.0)),
            "clean_best_score": float(clean_metrics["best_score"]),
            "clean_score": float(clean_score),
            "clean_margin_over_second": float(clean_metrics["margin_over_second"]),
            "af1_pred": af1_pred,
            "af1_correct": af1_correct,
            "af1_gold_prob": float(af1_metrics["probs_by_answer"].get(sample.gold_answer, 0.0)),
            "af1_best_score": float(af1_metrics["best_score"]),
            "intervention_score": float(intervention_score),
            "af1_margin_over_second": float(af1_metrics["margin_over_second"]),
            "score_drop": float(clean_score - intervention_score),
            "carrier_index": int(sample.layout.carrier_index),
            "carrier_token": sample.layout.carrier_token_text,
            "wait_layer": int(policy.wait_layer),
            "transfer_layers": int(policy.transfer_layers),
            "transfer_layer_indices": json.dumps(list(policy.transfer_layer_indices)),
            "k_donors": int(k_donors_requested),
            "num_frames": int(sample.layout.seq_len),
            "num_frame_groups": int(len(sample.layout.frame_groups)),
            "prompt_len": int(sample.layout.prompt_len),
            "image_tokens_per_frame": json.dumps(list(sample.layout.image_tokens_per_frame)),
            "room_text": sample.layout.room_text,
            "skipped_reason": "",
            "donor_ids": json.dumps(list(donor_ids)),
            "layout_match_status": "exact_match",
            "layout_match_details": "exact_match",
        }
    )
    return row


def _skipped_row(
    sample_id: str,
    seq_len: int,
    gold_answer: str,
    skipped_reason: str,
    room_text: str = "",
    layout: Optional[SampleLayout] = None,
    donor_ids: Optional[Sequence[str]] = None,
    wait_layer: Optional[int] = None,
    transfer_layers: Optional[int] = None,
    k_donors: Optional[int] = None,
    layout_status: str = "skipped",
    layout_details: Optional[str] = None,
) -> Dict[str, Any]:
    row = _empty_row(MODEL_ID, sample_id=sample_id, seq_len=seq_len)
    row.update(
        {
            "mode": _MODE,
            "used": 0,
            "gold_answer": gold_answer,
            "room_text": room_text,
            "skipped_reason": skipped_reason,
            "layout_match_status": layout_status,
            "layout_match_details": layout_details or skipped_reason,
            "donor_ids": json.dumps(list(donor_ids)) if donor_ids is not None else "",
            "wait_layer": "" if wait_layer is None else int(wait_layer),
            "transfer_layers": "" if transfer_layers is None else int(transfer_layers),
            "transfer_layer_indices": (
                ""
                if wait_layer is None or transfer_layers is None
                else json.dumps(list(range(int(wait_layer), int(wait_layer) + int(transfer_layers))))
            ),
            "k_donors": "" if k_donors is None else int(k_donors),
        }
    )
    if layout is not None:
        row.update(
            {
                "carrier_index": int(layout.carrier_index),
                "carrier_token": layout.carrier_token_text,
                "num_frames": int(layout.seq_len),
                "num_frame_groups": int(len(layout.frame_groups)),
                "prompt_len": int(layout.prompt_len),
                "image_tokens_per_frame": json.dumps(list(layout.image_tokens_per_frame)),
            }
        )
    return row


def _materialize_skipped_row(
    row_template: Dict[str, Any],
    policy: WaitOnlyPolicy,
    k_donors: int,
) -> Dict[str, Any]:
    row = dict(row_template)
    row.update(
        {
            "mode": _MODE,
            "wait_layer": int(policy.wait_layer),
            "transfer_layers": int(policy.transfer_layers),
            "transfer_layer_indices": json.dumps(list(policy.transfer_layer_indices)),
            "k_donors": int(k_donors),
        }
    )
    return row


def _mean_or_zero(values: Sequence[float]) -> float:
    return 0.0 if not values else float(sum(values) / len(values))


def _median_or_zero(values: Sequence[float]) -> float:
    return 0.0 if not values else float(statistics.median(values))


def summarize_wait_layer_results(
    model_name: str,
    seq_len: int,
    wait_layer: int,
    sample_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    used_rows = [row for row in sample_rows if int(row.get("used") or 0)]
    n_total = len(sample_rows)
    n_used = len(used_rows)
    n_clean_correct = sum(int(row.get("clean_correct") or 0) for row in used_rows)
    n_af1_correct = sum(int(row.get("af1_correct") or 0) for row in used_rows)
    n_both_correct = sum(
        int(bool(int(row.get("clean_correct") or 0)) and bool(int(row.get("af1_correct") or 0)))
        for row in used_rows
    )
    score_drop_values = [float(row["score_drop"]) for row in used_rows if row.get("score_drop") != ""]
    clean_acc = (n_clean_correct / float(n_used)) if n_used else 0.0
    af1_acc = (n_af1_correct / float(n_used)) if n_used else 0.0
    af1_faith = (n_both_correct / float(n_clean_correct)) if n_clean_correct else 0.0
    return {
        "model": model_name,
        "seq_len": int(seq_len),
        "wait_layer": int(wait_layer),
        "n_total": int(n_total),
        "n_used": int(n_used),
        "n_clean_correct": int(n_clean_correct),
        "clean_acc": float(clean_acc),
        "af1_acc": float(af1_acc),
        "af1_faith": float(af1_faith),
        "mean_score_drop": _mean_or_zero(score_drop_values),
        "median_score_drop": _median_or_zero(score_drop_values),
    }


def format_summary_table(rows: Sequence[Dict[str, Any]]) -> str:
    values = [
        [
            str(row["model"]),
            str(row["seq_len"]),
            str(row["wait_layer"]),
            str(row["n_total"]),
            str(row["n_used"]),
            str(row["n_clean_correct"]),
            f"{float(row['clean_acc']):.4f}",
            f"{float(row['af1_acc']):.4f}",
            f"{float(row['af1_faith']):.4f}",
            f"{float(row['mean_score_drop']):.4f}",
            f"{float(row['median_score_drop']):.4f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(value[col_idx]) for value in values)) if values else len(header)
        for col_idx, header in enumerate(_SUMMARY_BY_WAIT_LAYER_FIELDS)
    ]
    header_row = (
        "| "
        + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(_SUMMARY_BY_WAIT_LAYER_FIELDS))
        + " |"
    )
    sep_row = "|-" + "-|-".join("-" * widths[idx] for idx in range(len(_SUMMARY_BY_WAIT_LAYER_FIELDS))) + "-|"
    data_rows = [
        "| " + " | ".join(value[idx].ljust(widths[idx]) for idx in range(len(_SUMMARY_BY_WAIT_LAYER_FIELDS))) + " |"
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


def plot_score_drop_vs_wait_layer(
    summary_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    value_key: str,
    title: str,
    ylabel: str,
    x_tick_step: int = 1,
) -> Optional[Path]:
    plotted_rows = [row for row in summary_rows if row.get("n_used")]
    if not plotted_rows:
        return None

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    for seq_len in sorted({int(row["seq_len"]) for row in plotted_rows}):
        seq_rows = sorted(
            [row for row in plotted_rows if int(row["seq_len"]) == seq_len],
            key=lambda row: int(row["wait_layer"]),
        )
        x_values = [int(row["wait_layer"]) for row in seq_rows]
        y_values = [float(row[value_key]) for row in seq_rows]
        ax.plot(x_values, y_values, marker="o", linewidth=2.0, label=f"seq_len={seq_len}")

    all_wait_layers = sorted({int(row["wait_layer"]) for row in plotted_rows})
    if all_wait_layers:
        x_tick_step = max(1, int(x_tick_step))
        x_min = min(all_wait_layers)
        x_max = max(all_wait_layers)
        ax.set_xticks(list(range(x_min, x_max + 1, x_tick_step)))

    ax.set_title(title)
    ax.set_xlabel("wait_layer")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_markdown_summary(
    path: Path,
    config: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    validation_notes: Sequence[str],
    donor_notes: Sequence[str],
    cache_notes: Sequence[str],
    output_notes: Sequence[str],
    elapsed_seconds: float,
) -> None:
    lines = [
        "# AF1 Qwen-VL Wait-Only Sweep",
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
        "## Method",
        "",
        "- Mode run: `wait_only` only.",
        "- `wait_layer` uses the same AF1 `L_wait` semantics as the source script.",
        "- Conditional-mean patching is applied to frame token groups plus one all-non-frame prompt token set.",
        "- No ABP masking is applied anywhere in this sweep.",
        "- `score_drop = clean_score - intervention_score`, where both scores are taken for the frozen clean top-1 answer.",
        "- `n_total` counts selected samples and `n_used` counts samples that passed compatibility and donor checks.",
        "",
        "## Validation",
        "",
    ]
    lines.extend(f"- {note}" for note in validation_notes)
    lines.extend(["", "## Donor Policy", ""])
    lines.extend(f"- {note}" for note in donor_notes)
    lines.extend(["", "## Cache Notes", ""])
    lines.extend(f"- {note}" for note in cache_notes)
    lines.extend(["", "## Outputs", ""])
    lines.extend(f"- {note}" for note in output_notes)
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- {eval_utils.format_runtime(elapsed_seconds)}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.wait_layers, args.wait_layer_tick_step = parse_wait_layers(args.wait_layers)
    if args.model_name != MODEL_ID:
        raise ValueError(
            f"This script is pinned to {MODEL_ID!r}; received --model_name={args.model_name!r}"
        )
    if args.k_donors < 2:
        raise ValueError("--k_donors must be at least 2 because one donor is not a conditional mean.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    parse_dtype(args.dtype)
    set_seed(args.seed)
    start_time = time.time()
    runtime_info = model_runtime_info(requested_device=args.device, requested_dtype=args.dtype)
    print(json.dumps(runtime_info, indent=2, sort_keys=True))
    print(
        f"[config] mode={_MODE} skip_hallway={bool(args.skip_hallway)} "
        f"wait_layers={list(args.wait_layers)}"
    )

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_sample_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    validation_notes: List[str] = []
    donor_notes: List[str] = []
    cache_notes: List[str] = []
    clean_metrics_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}

    validation_notes.append(
        "mode=wait_only: wait-boundary patching is applied to frame groups plus all non-frame prompt tokens; later attention remains clean."
    )
    validation_notes.append(
        "wait_layer semantics: if wait_layer > 0, patch x^(L_wait) at layer output wait_layer - 1; "
        "if wait_layer == 0, patch x^(0) before layer 0."
    )

    for seq_len in args.seq_lens:
        data_root = seq_len_data_root(Path(args.data_root_base), seq_len=seq_len, split=args.split)
        if not data_root.is_dir():
            raise FileNotFoundError(f"seq_len={seq_len}: data root not found: {data_root}")

        sample_dirs = load_and_filter_sample_dirs(
            data_root=data_root,
            max_samples=args.max_samples,
            seed=args.seed + seq_len,
        )
        if not sample_dirs:
            raise RuntimeError(f"seq_len={seq_len}: no samples found under {data_root}")

        print(
            f"[seq_len={seq_len}][mode={_MODE}] samples={len(sample_dirs)} data_root={data_root} "
            f"wait_layers={list(args.wait_layers)} transfer_layers={args.transfer_layers} k_donors={args.k_donors}"
        )

        loaded_items: List[Any] = []
        prepared_samples: List[PreparedSample] = []
        for sample_dir in sample_dirs:
            prepared_sample, skipped_row = _prepare_sample(sample_dir, skip_hallway=bool(args.skip_hallway))
            if skipped_row is not None:
                loaded_items.append(skipped_row)
                validation_notes.append(
                    f"seq_len={seq_len} sample_id={skipped_row['sample_id']} skipped: {skipped_row['skipped_reason']}"
                )
                continue
            loaded_items.append(prepared_sample)
            prepared_samples.append(prepared_sample)

        reference_layout = _choose_reference_layout(prepared_samples)
        if reference_layout is None:
            validation_notes.append(f"seq_len={seq_len}: no compatible non-hallway samples remained after filtering")
            donor_notes.append(
                f"seq_len={seq_len}: no donor selection was used because no compatible reference layout remained."
            )
            cache_notes.append(
                f"seq_len={seq_len}: no conditional-mean cache activity because no compatible reference layout remained."
            )
            for wait_layer in args.wait_layers:
                policy = build_wait_only_policy(
                    layout=SampleLayout(
                        sample_id="none",
                        seq_len=seq_len,
                        prompt_len=0,
                        carrier_index=0,
                        carrier_token_id=0,
                        carrier_token_text="",
                        prompt_family_key="",
                        frame_groups=tuple(),
                        image_tokens_per_frame=tuple(),
                        room_text="",
                        room_positions=tuple(),
                        character_positions=tuple(),
                        room_span_len=0,
                        prompt_input_ids=tuple(),
                        prompt_decoded_tokens=tuple(),
                    ),
                    wait_layer=int(wait_layer),
                    transfer_layers=args.transfer_layers,
                )
                seq_rows = [
                    _materialize_skipped_row(item, policy=policy, k_donors=args.k_donors)
                    for item in loaded_items
                    if isinstance(item, dict)
                ]
                summary_rows.append(
                    summarize_wait_layer_results(
                        MODEL_ID,
                        seq_len=seq_len,
                        wait_layer=int(wait_layer),
                        sample_rows=seq_rows,
                    )
                )
                all_sample_rows.extend(seq_rows)
            continue

        compatible_samples: List[PreparedSample] = []
        compatible_layout_hash = layout_hash(reference_layout)
        exact_match_count = 0
        ordered_items: List[Any] = []
        for item in loaded_items:
            if isinstance(item, dict):
                ordered_items.append(item)
                continue
            report = inspect_and_validate_layout(
                reference_layout=reference_layout,
                candidate_layout=item.layout,
                skip_hallway=bool(args.skip_hallway),
            )
            if report["status"] != "exact_match":
                incompatible_row = _empty_row(MODEL_ID, sample_id=item.sample_id, seq_len=item.layout.seq_len)
                incompatible_row.update(
                    {
                        "mode": _MODE,
                        "used": 0,
                        "gold_answer": item.gold_answer,
                        "room_text": item.layout.room_text,
                        "carrier_index": int(item.layout.carrier_index),
                        "carrier_token": item.layout.carrier_token_text,
                        "num_frames": int(item.layout.seq_len),
                        "num_frame_groups": int(len(item.layout.frame_groups)),
                        "prompt_len": int(item.layout.prompt_len),
                        "image_tokens_per_frame": json.dumps(list(item.layout.image_tokens_per_frame)),
                        "skipped_reason": "layout_incompatible",
                        "layout_match_status": report["status"],
                        "layout_match_details": report["details"],
                        "donor_ids": "",
                    }
                )
                ordered_items.append(incompatible_row)
                validation_notes.append(
                    f"seq_len={seq_len} sample_id={item.sample_id} incompatible: {report['details']}"
                )
                continue
            ordered_items.append(item)
            compatible_samples.append(item)
            exact_match_count += 1

        validation_notes.append(
            f"seq_len={seq_len}: reference_layout sample_id={reference_layout.sample_id} "
            f"prompt_len={reference_layout.prompt_len} carrier_index={reference_layout.carrier_index} "
            f"carrier_token={reference_layout.carrier_token_text!r} "
            f"image_tokens_per_frame={list(reference_layout.image_tokens_per_frame)} "
            f"layout_hash={compatible_layout_hash}"
        )
        validation_notes.append(
            f"seq_len={seq_len}: exact_match_samples={exact_match_count} total_selected={len(sample_dirs)}"
        )
        print(
            f"[validation][mode={_MODE}] seq_len={seq_len} reference_sample={reference_layout.sample_id} "
            f"prompt_len={reference_layout.prompt_len} carrier_index={reference_layout.carrier_index} "
            f"layout_hash={compatible_layout_hash}"
        )
        print(
            f"[validation][mode={_MODE}] seq_len={seq_len} exact_match_samples={exact_match_count} "
            f"total_selected={len(sample_dirs)}"
        )

        if args.debug_tokenization and compatible_samples:
            debug_layout = compatible_samples[0].layout
            print(f"[debug][mode={_MODE}] seq_len={seq_len} sample_id={compatible_samples[0].sample_id}")
            print(format_token_debug_rows(debug_layout))

        donor_notes.append(
            f"seq_len={seq_len}: donors come from the same seq_len pool, must pass exact layout validation, "
            f"must not equal the target sample, and are chosen with deterministic seeded shuffle "
            f"under policy={_DONOR_POLICY}"
        )

        for wait_layer in args.wait_layers:
            policy = build_wait_only_policy(
                layout=reference_layout,
                wait_layer=int(wait_layer),
                transfer_layers=args.transfer_layers,
            )
            validation_notes.append(
                f"seq_len={seq_len} wait_layer={int(wait_layer)}: ABP masking disabled; later attention remains clean."
            )
            print(
                f"[validation][mode={_MODE}] seq_len={seq_len} wait_layer={int(wait_layer)} "
                "ABP masking disabled; later attention remains clean."
            )
            print(
                f"[validation][mode={_MODE}] seq_len={seq_len} wait_layer={int(wait_layer)} "
                f"compatible_samples={len(compatible_samples)} reference_sample={reference_layout.sample_id}"
            )

            seq_cache_hits = 0
            seq_cache_misses = 0
            seq_rows: List[Dict[str, Any]] = []

            for item in ordered_items:
                if isinstance(item, dict):
                    seq_rows.append(_materialize_skipped_row(item, policy=policy, k_donors=args.k_donors))
                    print(
                        f"[seq_len={seq_len}][wait_layer={int(wait_layer)}][mode={_MODE}] "
                        f"sample_id={item['sample_id']} skipped={item['skipped_reason']}"
                    )
                    continue

                donor_pool = select_donor_pool(
                    target_sample=item,
                    compatible_samples=compatible_samples,
                    k_donors=args.k_donors,
                    seed=args.seed,
                )
                donor_ids = [sample.sample_id for sample in donor_pool]
                if len(donor_pool) < 2:
                    skipped_row = _skipped_row(
                        sample_id=item.sample_id,
                        seq_len=item.layout.seq_len,
                        gold_answer=item.gold_answer,
                        skipped_reason="insufficient_compatible_donors",
                        room_text=item.layout.room_text,
                        layout=item.layout,
                        donor_ids=donor_ids,
                        wait_layer=int(wait_layer),
                        transfer_layers=args.transfer_layers,
                        k_donors=args.k_donors,
                        layout_status="exact_match",
                        layout_details="exact_match",
                    )
                    seq_rows.append(skipped_row)
                    validation_notes.append(
                        f"seq_len={seq_len} wait_layer={int(wait_layer)} sample_id={item.sample_id} skipped: "
                        f"insufficient compatible donors (found={len(donor_pool)}, need>=2)"
                    )
                    print(
                        f"[seq_len={seq_len}][wait_layer={int(wait_layer)}][mode={_MODE}] "
                        f"sample_id={item.sample_id} skipped=insufficient_compatible_donors donors={donor_ids}"
                    )
                    continue

                frame_group_means, cache_stats = compute_all_frame_group_means_for_sample(
                    target_sample=item,
                    donor_samples=donor_pool,
                    wait_layer=int(wait_layer),
                    batch_size=args.batch_size,
                    cache_dir=cache_dir,
                    recompute_cache=bool(args.recompute_cache),
                    donor_policy=_DONOR_POLICY,
                )
                seq_cache_hits += int(cache_stats["cache_hits"])
                seq_cache_misses += int(cache_stats["cache_misses"])
                non_frame_prompt_mean, non_frame_cache_hit = compute_non_frame_conditional_mean(
                    target_sample=item,
                    donor_samples=donor_pool,
                    wait_layer=int(wait_layer),
                    batch_size=args.batch_size,
                    cache_dir=cache_dir,
                    recompute_cache=bool(args.recompute_cache),
                    donor_policy=_DONOR_POLICY,
                )
                seq_cache_hits += int(non_frame_cache_hit)
                seq_cache_misses += int(not non_frame_cache_hit)

                clean_cache_key = (int(item.layout.seq_len), item.sample_id)
                clean_metrics = clean_metrics_cache.get(clean_cache_key)
                if clean_metrics is None:
                    clean_metrics = run_clean_sample(item)
                    clean_metrics_cache[clean_cache_key] = clean_metrics
                af1_metrics = run_intervention_sample(
                    item,
                    frame_group_means=frame_group_means,
                    non_frame_prompt_mean=non_frame_prompt_mean,
                    policy=policy,
                )

                row = _evaluated_row(
                    sample=item,
                    clean_metrics=clean_metrics,
                    af1_metrics=af1_metrics,
                    donor_ids=donor_ids,
                    policy=policy,
                    k_donors_requested=args.k_donors,
                )
                seq_rows.append(row)
                print(
                    f"[seq_len={seq_len}][wait_layer={int(wait_layer)}][mode={_MODE}] "
                    f"sample_id={item.sample_id} gold={item.gold_answer} "
                    f"clean={row['clean_pred']} af1={row['af1_pred']} "
                    f"score_drop={row['score_drop']:.4f} donors={json.dumps(donor_ids)}"
                )

            cache_notes.append(
                f"seq_len={seq_len} wait_layer={int(wait_layer)}: frame-group+non-frame conditional-mean "
                f"cache_hits={seq_cache_hits} "
                f"cache_misses={seq_cache_misses} layout_hash={compatible_layout_hash}"
            )
            summary_rows.append(
                summarize_wait_layer_results(
                    MODEL_ID,
                    seq_len=seq_len,
                    wait_layer=int(wait_layer),
                    sample_rows=seq_rows,
                )
            )
            all_sample_rows.extend(seq_rows)

    summary_rows = sorted(summary_rows, key=lambda row: (int(row["seq_len"]), int(row["wait_layer"])))
    summary_table = format_summary_table(summary_rows)
    print("\nFinal Wait-Only Sweep Table")
    print(summary_table)

    summary_csv_path = output_dir / "summary_by_wait_layer.csv"
    per_sample_csv_path = output_dir / "per_sample_by_wait_layer.csv"
    mean_plot_path = output_dir / "score_drop_vs_wait_layer.png"
    median_plot_path = output_dir / "median_score_drop_vs_wait_layer.png"
    markdown_summary_path = output_dir / "summary.md"

    write_csv(summary_csv_path, summary_rows, fieldnames=_SUMMARY_BY_WAIT_LAYER_FIELDS)
    write_csv(per_sample_csv_path, all_sample_rows, fieldnames=_PER_SAMPLE_BY_WAIT_LAYER_FIELDS)
    mean_plot_written = plot_score_drop_vs_wait_layer(
        summary_rows,
        output_path=mean_plot_path,
        value_key="mean_score_drop",
        title="Mean score drop vs wait layer",
        ylabel="Mean score drop",
        x_tick_step=args.wait_layer_tick_step,
    )
    median_plot_written = plot_score_drop_vs_wait_layer(
        summary_rows,
        output_path=median_plot_path,
        value_key="median_score_drop",
        title="Median score drop vs wait layer",
        ylabel="Median score drop",
        x_tick_step=args.wait_layer_tick_step,
    )

    output_notes = [
        f"summary_by_wait_layer.csv: {summary_csv_path}",
        f"per_sample_by_wait_layer.csv: {per_sample_csv_path}",
        f"score_drop_vs_wait_layer.png: {mean_plot_path if mean_plot_written is not None else 'not_written'}",
        f"median_score_drop_vs_wait_layer.png: {median_plot_path if median_plot_written is not None else 'not_written'}",
    ]
    write_markdown_summary(
        markdown_summary_path,
        config={
            "model_name": args.model_name,
            "mode": _MODE,
            "data_root_base": args.data_root_base,
            "split": args.split,
            "seq_lens": args.seq_lens,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "wait_layers": list(args.wait_layers),
            "wait_layer_tick_step": int(args.wait_layer_tick_step),
            "transfer_layers": args.transfer_layers,
            "k_donors": args.k_donors,
            "cache_dir": str(cache_dir),
            "recompute_cache": bool(args.recompute_cache),
            "output_dir": str(output_dir),
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
            "skip_hallway": bool(args.skip_hallway),
            "donor_policy": _DONOR_POLICY,
            "wait_layer_semantics": (
                "wait_layer is AF1 L_wait measured in number of waiting layers; "
                "if wait_layer > 0 then x^(L_wait) is patched at layer output wait_layer - 1"
            ),
            "score_drop_semantics": "score_drop = clean_score - intervention_score on the frozen clean top-1 answer",
        },
        summary_rows=summary_rows,
        validation_notes=validation_notes,
        donor_notes=donor_notes,
        cache_notes=cache_notes,
        output_notes=output_notes,
        elapsed_seconds=time.time() - start_time,
    )

    print(
        json.dumps(
            {
                "summary_by_wait_layer_csv": str(summary_csv_path),
                "per_sample_by_wait_layer_csv": str(per_sample_csv_path),
                "score_drop_vs_wait_layer_png": None if mean_plot_written is None else str(mean_plot_path),
                "median_score_drop_vs_wait_layer_png": None if median_plot_written is None else str(median_plot_path),
                "summary_md": str(markdown_summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

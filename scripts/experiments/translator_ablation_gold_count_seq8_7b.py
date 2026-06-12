#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from evaluations.helpers import utils as eval_utils
from models.model import find_subsequence
from scripts.probes import run_message_memory_adapter_stage1_stage3_seq8 as prev
from scripts.probes import run_oracle_count_multilayer_injection_seq8 as oracle


EXPERIMENT_NAME = "translator_ablation_gold_count_seq8_7b"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME

BASELINE = "baseline_no_injection"
STATIC_CODEBOOK = "static_count_codebook"
LAYER_CODEBOOK = "layer_count_codebook"
LOW_RANK_TRANSLATOR = "low_rank_count_translator"
STATE_TRANSLATOR = "state_conditioned_translator"
STATE_GATED_TRANSLATOR = "state_conditioned_gated_translator"

METHOD_ALIASES = {
    "baseline": BASELINE,
    "baseline_no_injection": BASELINE,
    "static": STATIC_CODEBOOK,
    "static_count_codebook": STATIC_CODEBOOK,
    "layer": LAYER_CODEBOOK,
    "layer_codebook": LAYER_CODEBOOK,
    "layer_count_codebook": LAYER_CODEBOOK,
    "low_rank": LOW_RANK_TRANSLATOR,
    "low_rank_count_translator": LOW_RANK_TRANSLATOR,
    "state": STATE_TRANSLATOR,
    "state_conditioned": STATE_TRANSLATOR,
    "state_conditioned_translator": STATE_TRANSLATOR,
    "state_gated": STATE_GATED_TRANSLATOR,
    "state_conditioned_gated": STATE_GATED_TRANSLATOR,
    "state_conditioned_gated_translator": STATE_GATED_TRANSLATOR,
}

TOKEN_GROUP_ALIASES = {
    "none": "none",
    "room_char": "room_char",
    "all_question_tokens": "all_question_tokens",
    "question": "all_question_tokens",
    "last_token": "last_token",
    "all_question_tokens+last_token": "question_plus_last",
    "question_plus_last": "question_plus_last",
    "room_char+all_question_tokens": "room_char_plus_question",
    "room_char_plus_question": "room_char_plus_question",
    "room_char+all_question_tokens+last_token": "room_char_question_last",
    "room_char_question_last": "room_char_question_last",
}

COUNT_CONTROL_NONE = "none"
COUNT_CONTROL_SHUFFLED = "shuffled_count_control"


@dataclass
class TranslatorBatch:
    inputs: Dict[str, Any]
    target_positions: List[List[int]]
    prompt_last_indices: torch.Tensor
    gold_counts: torch.Tensor
    sample_indices: List[int]
    token_selection_ok: List[bool]
    token_selection_errors: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Oracle gold-count translator ablation for MMReD seq_len=8 Qwen2.5-VL-7B. "
            "Qwen is frozen; only tiny residual injection translators are trained."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/mmred_images_park")
    parser.add_argument("--source-run", type=Path, default=prev.DEFAULT_SOURCE_RUN)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--split", default="all_uniform")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--method", required=True, choices=sorted(METHOD_ALIASES))
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--token-group", default="room_char")
    parser.add_argument("--layer-start", type=int, default=14)
    parser.add_argument("--layer-end", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--state-bottleneck", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-samples-per-count", type=int, default=100)
    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in range(9)])
    parser.add_argument("--lambda-margin", type=float, default=0.2)
    parser.add_argument("--margin-target", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--count-control",
        default=COUNT_CONTROL_NONE,
        choices=[COUNT_CONTROL_NONE, COUNT_CONTROL_SHUFFLED],
        help="Inject the gold count or a deterministic shuffled count label.",
    )
    parser.add_argument(
        "--include-random-untrained-same-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After training, also evaluate a random untrained translator scaled to the trained parameter norm.",
    )
    parser.add_argument("--submit-mode", default="local", help="Optional bookkeeping label; the script itself runs locally.")

    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true", default=False)
    return parser.parse_args()


def canonical_method(raw: str) -> str:
    key = str(raw).strip()
    if key not in METHOD_ALIASES:
        raise ValueError(f"Unknown method={raw!r}; valid={sorted(METHOD_ALIASES)}")
    return METHOD_ALIASES[key]


def canonical_token_group(raw: str) -> str:
    key = str(raw).strip()
    if key not in TOKEN_GROUP_ALIASES:
        raise ValueError(f"Unknown token_group={raw!r}; valid={sorted(TOKEN_GROUP_ALIASES)}")
    return TOKEN_GROUP_ALIASES[key]


def safe_name(text: str) -> str:
    safe = str(text)
    for old, new in (("/", "_"), ("+", "_"), (" ", "_"), (":", "_"), (".", "p"), (",", "_")):
        safe = safe.replace(old, new)
    return safe


def layer_window_label(start: int, end: int) -> str:
    return f"{int(start)}-{int(end)}" if int(start) != int(end) else str(int(start))


def make_output_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    method = canonical_method(args.method)
    group = canonical_token_group(args.token_group)
    if method == BASELINE:
        layer_label = "no_layers"
    else:
        layer_label = f"l{int(args.layer_start)}_{int(args.layer_end)}"
    dirname = f"{stamp}_{safe_name(args.config_name)}_{safe_name(method)}_{safe_name(group)}_{layer_label}"
    return Path(args.output_root).resolve() / dirname


def default_lr_for_method(method: str) -> float:
    if method in {STATIC_CODEBOOK, LAYER_CODEBOOK, LOW_RANK_TRANSLATOR}:
        return 5e-3
    if method in {STATE_TRANSLATOR, STATE_GATED_TRANSLATOR}:
        return 1e-3
    return 0.0


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_mean(values: Sequence[Any], default: float = math.nan) -> float:
    vals = [float(x) for x in values if finite_float(x) is not None]
    return float(np.mean(vals)) if vals else float(default)


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def split_limited_indices(
    indices: Sequence[int],
    records: Sequence[prev.SampleRecord],
    limit: int,
    seed: int,
) -> List[int]:
    values = [int(x) for x in indices]
    if int(limit) <= 0 or len(values) <= int(limit):
        return values
    by_count: Dict[int, List[int]] = defaultdict(list)
    for idx in values:
        by_count[int(records[idx].gold_count)].append(int(idx))
    for count in by_count:
        by_count[count].sort(key=lambda idx: prev.stable_hash_int(f"{seed}:{idx}:{records[idx].sample_id}"))
    selected: List[int] = []
    while len(selected) < int(limit) and any(by_count.values()):
        for count in sorted(by_count):
            if len(selected) >= int(limit):
                break
            bucket = by_count[count]
            if bucket:
                selected.append(bucket.pop(0))
    random.Random(int(seed)).shuffle(selected)
    return sorted(selected)


def load_sample_index_payload(args: argparse.Namespace) -> Dict[str, Any]:
    cache_path = prev.source_cache_path(Path(args.source_run))
    if not cache_path.is_file():
        raise FileNotFoundError(f"Could not find source cache at {cache_path}")
    print(f"Loading sample ids from source cache: {cache_path}")
    payload = prev.load_torch(cache_path)
    sample_ids = [str(x) for x in payload["sample_ids"]]
    labels = payload["labels"].long()
    counts = set(int(x) for x in prev.parse_int_tokens(args.evidence_counts))
    keep = [idx for idx, label in enumerate(labels.tolist()) if int(label) in counts]
    if int(args.max_samples_per_count) > 0:
        seen: Dict[int, int] = defaultdict(int)
        limited: List[int] = []
        for idx in keep:
            count = int(labels[idx].item())
            if seen[count] < int(args.max_samples_per_count):
                limited.append(idx)
                seen[count] += 1
        keep = limited
    return {
        "cache_path": cache_path,
        "sample_ids": [sample_ids[idx] for idx in keep],
        "labels": labels[keep],
    }


def prompt_last_index(input_ids_1d: torch.Tensor, attention_mask_1d: Optional[torch.Tensor]) -> int:
    if attention_mask_1d is None:
        return int(input_ids_1d.numel()) - 1
    active = attention_mask_1d.detach().cpu().nonzero(as_tuple=True)[0]
    return int(active[-1].item()) if active.numel() else int(input_ids_1d.numel()) - 1


def prompt_text_bounds(
    record: prev.SampleRecord,
    processor: Any,
    input_ids_1d: torch.Tensor,
) -> Tuple[str, Optional[int]]:
    prompt_text = prev.core.build_prompt(record.question, num_frames=len(record.frame_paths))
    input_ids = [int(token_id) for token_id in input_ids_1d.detach().cpu().tolist()]
    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    start = find_subsequence(input_ids, [int(token_id) for token_id in prompt_ids])
    return prompt_text, None if start is None else int(start)


def fallback_question_positions(
    *,
    input_ids_1d: torch.Tensor,
    record: prev.SampleRecord,
    processor: Any,
    prompt_last: int,
) -> List[int]:
    input_ids = [int(token_id) for token_id in input_ids_1d.detach().cpu().tolist()]
    prompt_text, prompt_start = prompt_text_bounds(record, processor, input_ids_1d)
    if prompt_start is not None:
        question_fragment = f"Question: {record.question}\n"
        if question_fragment in prompt_text:
            question_start = prompt_text.index(question_fragment) + len("Question: ")
            question_end = question_start + len(record.question)
            token_span = prev._token_span_from_char_span(prompt_text, (question_start, question_end), processor)
            return [
                pos
                for pos in prev._positions_from_token_span(prompt_start, token_span)
                if 0 <= int(pos) <= int(prompt_last)
            ]
    question_ids = processor.tokenizer(record.question, add_special_tokens=False)["input_ids"]
    start = find_subsequence(input_ids, [int(token_id) for token_id in question_ids])
    if start is None:
        return []
    return [pos for pos in range(int(start), int(start) + len(question_ids)) if 0 <= pos <= int(prompt_last)]


def fallback_room_char_positions(
    *,
    input_ids_1d: torch.Tensor,
    record: prev.SampleRecord,
    processor: Any,
    prompt_last: int,
) -> List[int]:
    parsed = eval_utils.parse_target_character_room_with_spans(record.question)
    if parsed is None:
        return []
    character, room, character_span, room_span = parsed
    prompt_text, prompt_start = prompt_text_bounds(record, processor, input_ids_1d)
    positions: List[int] = []
    if prompt_start is not None:
        question_fragment = f"Question: {record.question}\n"
        if question_fragment in prompt_text:
            question_start = prompt_text.index(question_fragment) + len("Question: ")
            for span in (character_span, room_span):
                absolute_span = (question_start + int(span[0]), question_start + int(span[1]))
                token_span = prev._token_span_from_char_span(prompt_text, absolute_span, processor)
                positions.extend(prev._positions_from_token_span(prompt_start, token_span))
    if positions:
        return sorted({int(pos) for pos in positions if 0 <= int(pos) <= int(prompt_last)})

    input_ids = [int(token_id) for token_id in input_ids_1d.detach().cpu().tolist()]
    for text in (str(character), str(room)):
        token_ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        start = find_subsequence(input_ids, [int(token_id) for token_id in token_ids])
        if start is not None:
            positions.extend(range(int(start), int(start) + len(token_ids)))
    return sorted({int(pos) for pos in positions if 0 <= int(pos) <= int(prompt_last)})


def fallback_positions_for_group(
    *,
    input_ids_1d: torch.Tensor,
    attention_mask_1d: Optional[torch.Tensor],
    record: prev.SampleRecord,
    processor: Any,
    token_group: str,
) -> Tuple[List[int], int]:
    prompt_last = prompt_last_index(input_ids_1d, attention_mask_1d)
    if token_group == "none":
        return [], prompt_last
    room_char = (
        fallback_room_char_positions(
            input_ids_1d=input_ids_1d,
            record=record,
            processor=processor,
            prompt_last=prompt_last,
        )
        if token_group in {"room_char", "room_char_plus_question", "room_char_question_last"}
        else []
    )
    question = (
        fallback_question_positions(
            input_ids_1d=input_ids_1d,
            record=record,
            processor=processor,
            prompt_last=prompt_last,
        )
        if token_group in {"all_question_tokens", "room_char_plus_question", "question_plus_last", "room_char_question_last"}
        else []
    )
    last = [prompt_last] if token_group in {"last_token", "question_plus_last", "room_char_question_last"} else []
    selected = sorted({int(pos) for pos in [*room_char, *question, *last] if 0 <= int(pos) <= int(prompt_last)})
    return selected, prompt_last


def locate_positions_safe(
    *,
    input_ids_1d: torch.Tensor,
    attention_mask_1d: Optional[torch.Tensor],
    record: prev.SampleRecord,
    processor: Any,
    token_group: str,
) -> Tuple[List[int], int, bool, str]:
    if token_group == "none":
        return [], prompt_last_index(input_ids_1d, attention_mask_1d), True, ""
    try:
        positions, prompt_last, _debug = oracle.locate_positions_for_group(
            input_ids_1d=input_ids_1d,
            attention_mask_1d=attention_mask_1d,
            record=record,
            processor=processor,
            token_group=token_group,
        )
        return positions, prompt_last, bool(positions), "" if positions else "empty selection"
    except Exception as exc:
        primary_error = f"{type(exc).__name__}: {exc}"
        try:
            positions, prompt_last = fallback_positions_for_group(
                input_ids_1d=input_ids_1d,
                attention_mask_1d=attention_mask_1d,
                record=record,
                processor=processor,
                token_group=token_group,
            )
        except Exception as fallback_exc:
            return [], prompt_last_index(input_ids_1d, attention_mask_1d), False, (
                f"{primary_error}; fallback failed with {type(fallback_exc).__name__}: {fallback_exc}"
            )
        if positions:
            return positions, prompt_last, True, f"primary localization failed; used fallback: {primary_error}"
        return [], prompt_last, False, f"{primary_error}; fallback returned no positions"


def load_frames(record: prev.SampleRecord) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for frame_path in record.frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))
    return frames


def prepare_translator_batch(
    *,
    records: Sequence[prev.SampleRecord],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
    token_group: str,
) -> TranslatorBatch:
    frames_by_record = [load_frames(record) for record in records]
    conversations = [prev.build_conversation(record, frames) for record, frames in zip(records, frames_by_record)]
    try:
        if len(conversations) == 1:
            raw_inputs = processor.apply_chat_template(
                conversations[0],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            raw_inputs = processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
    finally:
        for frames in frames_by_record:
            for frame in frames:
                try:
                    frame.close()
                except Exception:
                    pass

    input_ids = raw_inputs["input_ids"]
    attention_mask = raw_inputs.get("attention_mask")
    target_positions: List[List[int]] = []
    prompt_last_indices: List[int] = []
    token_selection_ok: List[bool] = []
    token_selection_errors: List[str] = []
    for batch_idx, record in enumerate(records):
        positions, prompt_last, ok, error = locate_positions_safe(
            input_ids_1d=input_ids[batch_idx],
            attention_mask_1d=attention_mask[batch_idx] if torch.is_tensor(attention_mask) else None,
            record=record,
            processor=processor,
            token_group=token_group,
        )
        target_positions.append([int(pos) for pos in positions])
        prompt_last_indices.append(int(prompt_last))
        token_selection_ok.append(bool(ok))
        token_selection_errors.append(str(error))

    return TranslatorBatch(
        inputs=prev.move_inputs_to_device(dict(raw_inputs), device),
        target_positions=target_positions,
        prompt_last_indices=torch.tensor(prompt_last_indices, device=device, dtype=torch.long),
        gold_counts=torch.tensor([int(record.gold_count) for record in records], device=device, dtype=torch.long),
        sample_indices=[int(idx) for idx in sample_indices],
        token_selection_ok=token_selection_ok,
        token_selection_errors=token_selection_errors,
    )


class TranslatorInjectionAdapter(nn.Module):
    def __init__(
        self,
        *,
        method: str,
        count_values: Sequence[int],
        hidden_size: int,
        inject_layers: Sequence[int],
        alpha: float,
        rank: int,
        state_bottleneck: int,
    ) -> None:
        super().__init__()
        self.method = str(method)
        self.count_values = [int(x) for x in count_values]
        self.count_min = min(self.count_values)
        self.count_max = max(self.count_values)
        self.hidden_size = int(hidden_size)
        self.inject_layers = [int(x) for x in inject_layers]
        self.layer_to_pos = {int(layer): pos for pos, layer in enumerate(self.inject_layers)}
        self.alpha = float(alpha)
        self.rank = max(1, int(rank))
        self.state_bottleneck = max(1, int(state_bottleneck))
        self.enabled = True
        self._gold_counts: Optional[torch.Tensor] = None
        self._target_positions: Optional[List[List[int]]] = None
        self._handles: List[Any] = []
        self._last_energy_by_sample: Optional[torch.Tensor] = None

        num_counts = len(self.count_values)
        num_layers = len(self.inject_layers)
        if self.method == STATIC_CODEBOOK:
            self.codebook = nn.Parameter(torch.zeros(num_counts, self.hidden_size, dtype=torch.float32))
        elif self.method == LAYER_CODEBOOK:
            self.codebook = nn.Parameter(torch.zeros(num_layers, num_counts, self.hidden_size, dtype=torch.float32))
        elif self.method == LOW_RANK_TRANSLATOR:
            self.count_embeddings = nn.Parameter(torch.randn(num_counts, self.rank, dtype=torch.float32) * 0.02)
            self.layer_projection = nn.Parameter(torch.zeros(num_layers, self.hidden_size, self.rank, dtype=torch.float32))
        elif self.method in {STATE_TRANSLATOR, STATE_GATED_TRANSLATOR}:
            self.count_embeddings = nn.Parameter(torch.randn(num_counts, self.rank, dtype=torch.float32) * 0.02)
            input_dim = self.hidden_size + self.rank
            self.down = nn.ModuleList([nn.Linear(input_dim, self.state_bottleneck) for _ in range(num_layers)])
            self.up = nn.ModuleList([nn.Linear(self.state_bottleneck, self.hidden_size) for _ in range(num_layers)])
            for layer_up in self.up:
                nn.init.zeros_(layer_up.weight)
                nn.init.zeros_(layer_up.bias)
            if self.method == STATE_GATED_TRANSLATOR:
                self.gate = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_layers)])
                for gate_layer in self.gate:
                    nn.init.zeros_(gate_layer.weight)
                    nn.init.zeros_(gate_layer.bias)
        else:
            raise ValueError(f"Unknown method for trainable adapter: {self.method!r}")

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def set_context(self, gold_counts: torch.Tensor, target_positions: Sequence[Sequence[int]]) -> None:
        self._gold_counts = gold_counts.detach().long()
        self._target_positions = [[int(pos) for pos in positions] for positions in target_positions]
        self._last_energy_by_sample = None

    def clear_context(self) -> None:
        self._gold_counts = None
        self._target_positions = None
        self._last_energy_by_sample = None

    @staticmethod
    def _hidden_from_output(output: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        return None

    @staticmethod
    def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
        if torch.is_tensor(output):
            return hidden
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        if isinstance(output, list):
            return [hidden] + list(output[1:])
        return output

    def _offsets(self, device: torch.device) -> torch.Tensor:
        if self._gold_counts is None:
            raise RuntimeError("Adapter context is not set")
        offsets = self._gold_counts.to(device=device, dtype=torch.long) - int(self.count_min)
        if bool(torch.any(offsets < 0)) or bool(torch.any(offsets >= len(self.count_values))):
            raise RuntimeError("Gold count outside configured count range")
        return offsets

    def _ensure_energy(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._last_energy_by_sample is None or self._last_energy_by_sample.device != device:
            self._last_energy_by_sample = torch.zeros(int(batch_size), device=device, dtype=torch.float32)
        return self._last_energy_by_sample

    def last_energy(self) -> List[float]:
        if self._gold_counts is None:
            return []
        batch_size = int(self._gold_counts.numel())
        if self._last_energy_by_sample is None:
            return [0.0 for _ in range(batch_size)]
        return [float(x) for x in self._last_energy_by_sample.detach().float().cpu().tolist()]

    def parameter_l2(self) -> torch.Tensor:
        values = [param.float().pow(2).sum() for param in self.parameters() if param.requires_grad]
        if not values:
            return torch.tensor(0.0)
        return torch.stack(values).sum()

    def residuals_for_code_method(self, device: torch.device) -> torch.Tensor:
        offsets = self._offsets(device)
        if self.method == STATIC_CODEBOOK:
            selected = self.codebook.index_select(0, offsets)
            return selected.unsqueeze(1).expand(-1, len(self.inject_layers), -1).contiguous()
        if self.method == LAYER_CODEBOOK:
            return self.codebook[:, offsets, :].permute(1, 0, 2).contiguous()
        if self.method == LOW_RANK_TRANSLATOR:
            v = self.count_embeddings.index_select(0, offsets)
            return torch.einsum("br,lhr->blh", v, self.layer_projection).contiguous()
        raise RuntimeError(f"Not a code/residual method: {self.method}")

    def inject_code_method(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        residuals = self.residuals_for_code_method(hidden_states.device)
        layer_pos = int(self.layer_to_pos[int(layer_idx)])
        delta = residuals[:, layer_pos, :].to(dtype=hidden_states.dtype) * float(self.alpha)
        out = hidden_states.clone()
        energy = self._ensure_energy(int(hidden_states.shape[0]), hidden_states.device)
        seq_len = int(hidden_states.shape[1])
        for batch_idx, positions in enumerate(self._target_positions or []):
            valid = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len})
            if not valid:
                continue
            pos_idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
            update = delta[batch_idx].unsqueeze(0).expand(len(valid), -1)
            out[batch_idx, pos_idx, :] = out[batch_idx, pos_idx, :] + update
            with torch.no_grad():
                energy[batch_idx] += update.detach().float().pow(2).sum()
        return out

    def inject_state_method(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        offsets = self._offsets(hidden_states.device)
        layer_pos = int(self.layer_to_pos[int(layer_idx)])
        out = hidden_states.clone()
        energy = self._ensure_energy(int(hidden_states.shape[0]), hidden_states.device)
        seq_len = int(hidden_states.shape[1])
        count_embeddings = self.count_embeddings.index_select(0, offsets)
        for batch_idx, positions in enumerate(self._target_positions or []):
            valid = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len})
            if not valid:
                continue
            pos_idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
            h = hidden_states[batch_idx, pos_idx, :].float()
            h_norm = F.layer_norm(h, (self.hidden_size,))
            v = count_embeddings[batch_idx].unsqueeze(0).expand(len(valid), -1)
            x = torch.cat([h_norm, v], dim=-1)
            delta = self.up[layer_pos](F.gelu(self.down[layer_pos](x)))
            if self.method == STATE_GATED_TRANSLATOR:
                gate = torch.sigmoid(self.gate[layer_pos](x))
                delta = gate * delta
            update = (float(self.alpha) * delta).to(dtype=hidden_states.dtype)
            out[batch_idx, pos_idx, :] = out[batch_idx, pos_idx, :] + update
            with torch.no_grad():
                energy[batch_idx] += update.detach().float().pow(2).sum()
        return out

    def inject_for_layer(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not self.enabled or self._gold_counts is None or self._target_positions is None:
            return hidden_states
        if int(layer_idx) not in self.layer_to_pos:
            return hidden_states
        if self.method in {STATIC_CODEBOOK, LAYER_CODEBOOK, LOW_RANK_TRANSLATOR}:
            return self.inject_code_method(hidden_states, int(layer_idx))
        return self.inject_state_method(hidden_states, int(layer_idx))

    def register_hooks(self, model: Any) -> None:
        self.remove_hooks()
        layers = prev.get_layers(model)
        for layer_idx in self.inject_layers:
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(f"inject_layer={layer_idx} outside [0, {len(layers) - 1}]")

            def hook(_module: Any, _args: Any, output: Any, *, layer: int = int(layer_idx)) -> Any:
                hidden = self._hidden_from_output(output)
                if hidden is None:
                    return output
                return self._replace_hidden(output, self.inject_for_layer(hidden, layer))

            self._handles.append(layers[int(layer_idx)].register_forward_hook(hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def make_adapter(
    *,
    method: str,
    count_values: Sequence[int],
    hidden_size: int,
    inject_layers: Sequence[int],
    alpha: float,
    rank: int,
    state_bottleneck: int,
) -> TranslatorInjectionAdapter:
    return TranslatorInjectionAdapter(
        method=method,
        count_values=count_values,
        hidden_size=int(hidden_size),
        inject_layers=[int(x) for x in inject_layers],
        alpha=float(alpha),
        rank=int(rank),
        state_bottleneck=int(state_bottleneck),
    )


def adapter_parameter_norm(adapter: nn.Module) -> float:
    sq = 0.0
    with torch.no_grad():
        for param in adapter.parameters():
            sq += float(param.detach().float().pow(2).sum().cpu().item())
    return math.sqrt(max(0.0, sq))


def make_random_untrained_same_norm(
    trained_adapter: TranslatorInjectionAdapter,
    *,
    seed: int,
) -> TranslatorInjectionAdapter:
    random_adapter = make_adapter(
        method=trained_adapter.method,
        count_values=trained_adapter.count_values,
        hidden_size=trained_adapter.hidden_size,
        inject_layers=trained_adapter.inject_layers,
        alpha=trained_adapter.alpha,
        rank=trained_adapter.rank,
        state_bottleneck=trained_adapter.state_bottleneck,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 910031)
    with torch.no_grad():
        for param in random_adapter.parameters():
            param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.01)
        target_norm = adapter_parameter_norm(trained_adapter)
        current_norm = adapter_parameter_norm(random_adapter)
        if target_norm > 0.0 and current_norm > 0.0:
            scale = target_norm / current_norm
            for param in random_adapter.parameters():
                param.mul_(scale)
    for param in random_adapter.parameters():
        param.requires_grad_(False)
    return random_adapter


def margin_loss(count_logits: torch.Tensor, gold_offsets: torch.Tensor, margin_target: float) -> torch.Tensor:
    batch_idx = torch.arange(int(count_logits.shape[0]), device=count_logits.device, dtype=torch.long)
    gold = count_logits[batch_idx, gold_offsets]
    wrong_logits = count_logits.clone()
    wrong_logits[batch_idx, gold_offsets] = -torch.inf
    best_wrong = wrong_logits.max(dim=-1).values
    return F.relu(float(margin_target) - (gold - best_wrong)).mean()


def control_counts_for_injection(
    *,
    gold_counts: torch.Tensor,
    sample_indices: Sequence[int],
    count_values: Sequence[int],
    seed: int,
    control: str,
) -> torch.Tensor:
    if control == COUNT_CONTROL_NONE:
        return gold_counts
    if control != COUNT_CONTROL_SHUFFLED:
        raise ValueError(f"Unknown count control: {control!r}")
    values = [int(x) for x in count_values]
    shuffled: List[int] = []
    for row, idx in enumerate(sample_indices):
        gold = int(gold_counts[row].detach().cpu().item())
        pick = values[prev.stable_hash_int(f"shuffled_count:{seed}:{idx}") % len(values)]
        if len(values) > 1 and pick == gold:
            pick = values[(values.index(pick) + 1) % len(values)]
        shuffled.append(int(pick))
    return torch.tensor(shuffled, device=gold_counts.device, dtype=torch.long)


def select_gold_logits_and_margins(
    count_logits: torch.Tensor,
    gold_offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_idx = torch.arange(int(count_logits.shape[0]), device=count_logits.device, dtype=torch.long)
    gold_logits = count_logits[batch_idx, gold_offsets].float()
    wrong_logits = count_logits.float().clone()
    wrong_logits[batch_idx, gold_offsets] = -torch.inf
    best_wrong = wrong_logits.max(dim=-1).values
    margins = gold_logits - best_wrong
    return gold_logits, best_wrong, margins


def evaluate_split(
    *,
    split_name: str,
    method_label: str,
    model: Any,
    processor: Any,
    adapter: Optional[TranslatorInjectionAdapter],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    count_token_ids: Dict[int, int],
    count_values: Sequence[int],
    token_group: str,
    alpha: float,
    device: str,
    batch_size: int,
    max_eval_samples: int,
    count_control: str,
    seed: int,
) -> Dict[str, Any]:
    eval_indices = [int(x) for x in indices]
    if int(max_eval_samples) > 0:
        eval_indices = eval_indices[: int(max_eval_samples)]
    if adapter is not None:
        adapter.to(device)
        adapter.eval()
        adapter.enabled = True
        adapter.set_alpha(float(alpha))
        adapter.register_hooks(model)

    pred_by_idx: Dict[int, int] = {}
    logits_by_idx: Dict[int, List[float]] = {}
    gold_logit_by_idx: Dict[int, float] = {}
    pred_logit_by_idx: Dict[int, float] = {}
    margin_by_idx: Dict[int, float] = {}
    ce_by_idx: Dict[int, float] = {}
    injection_norm_by_idx: Dict[int, float] = {}
    token_selection_ok_by_idx: Dict[int, bool] = {}
    token_selection_error_by_idx: Dict[int, str] = {}
    num_injected_tokens_by_idx: Dict[int, int] = {}
    ce_total = 0.0
    n = 0
    count_min = min(int(x) for x in count_values)
    try:
        for batch_num, batch_indices in enumerate(prev.chunked(eval_indices, int(batch_size)), start=1):
            batch_records = [records[int(idx)] for idx in batch_indices]
            batch = prepare_translator_batch(
                records=batch_records,
                sample_indices=batch_indices,
                processor=processor,
                device=device,
                token_group=token_group if adapter is not None else "none",
            )
            if adapter is not None:
                inject_counts = control_counts_for_injection(
                    gold_counts=batch.gold_counts,
                    sample_indices=batch.sample_indices,
                    count_values=count_values,
                    seed=int(seed),
                    control=count_control,
                )
                adapter.set_context(inject_counts, batch.target_positions)
            with torch.no_grad():
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - int(count_min)
                ce_vec = F.cross_entropy(count_logits, gold_offsets, reduction="none")
                ce_total += float(ce_vec.sum().detach().cpu().item())
                n += int(batch.gold_counts.numel())
                pred_offsets = count_logits.argmax(dim=-1)
                gold_logits, _best_wrong, margins = select_gold_logits_and_margins(count_logits, gold_offsets)
                adapter_energy = adapter.last_energy() if adapter is not None else [0.0 for _ in batch_indices]
                logits_cpu = count_logits.detach().float().cpu()
                for row, idx in enumerate(batch_indices):
                    idx = int(idx)
                    pred = int(pred_offsets[row].detach().cpu().item()) + int(count_min)
                    pred_by_idx[idx] = pred
                    logits = [float(v) for v in logits_cpu[row].tolist()]
                    logits_by_idx[idx] = logits
                    gold_logit_by_idx[idx] = float(gold_logits[row].detach().cpu().item())
                    pred_offset = pred - int(count_min)
                    pred_logit_by_idx[idx] = logits[pred_offset] if 0 <= pred_offset < len(logits) else math.nan
                    margin_by_idx[idx] = float(margins[row].detach().cpu().item())
                    ce_by_idx[idx] = float(ce_vec[row].detach().cpu().item())
                    injection_norm_by_idx[idx] = float(adapter_energy[row]) if row < len(adapter_energy) else 0.0
                    token_selection_ok_by_idx[idx] = bool(batch.token_selection_ok[row]) if adapter is not None else True
                    token_selection_error_by_idx[idx] = batch.token_selection_errors[row] if adapter is not None else ""
                    num_injected_tokens_by_idx[idx] = len(batch.target_positions[row]) if adapter is not None else 0
            if adapter is not None:
                adapter.clear_context()
            if batch_num == 1 or batch_num % 50 == 0:
                print(f"  eval {method_label} {split_name}: {min(len(eval_indices), batch_num * int(batch_size))}/{len(eval_indices)}")
    finally:
        if adapter is not None:
            adapter.remove_hooks()

    y_true = [int(records[int(idx)].gold_count) for idx in eval_indices if int(idx) in pred_by_idx]
    y_pred = [pred_by_idx[int(idx)] for idx in eval_indices if int(idx) in pred_by_idx]
    return {
        "indices": eval_indices,
        "ce": ce_total / max(1, n),
        "accuracy": prev.accuracy(y_true, y_pred),
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "gold_logit_by_idx": gold_logit_by_idx,
        "pred_logit_by_idx": pred_logit_by_idx,
        "margin_by_idx": margin_by_idx,
        "ce_by_idx": ce_by_idx,
        "injection_norm_by_idx": injection_norm_by_idx,
        "token_selection_ok_by_idx": token_selection_ok_by_idx,
        "token_selection_error_by_idx": token_selection_error_by_idx,
        "num_injected_tokens_by_idx": num_injected_tokens_by_idx,
    }


def train_adapter(
    *,
    method: str,
    args: argparse.Namespace,
    output_dir: Path,
    model: Any,
    processor: Any,
    records: Sequence[prev.SampleRecord],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    count_token_ids: Dict[int, int],
    count_values: Sequence[int],
    hidden_size: int,
    inject_layers: Sequence[int],
    token_group: str,
    device: str,
    lr: float,
) -> Tuple[TranslatorInjectionAdapter, List[Dict[str, Any]], Path]:
    adapter = make_adapter(
        method=method,
        count_values=count_values,
        hidden_size=int(hidden_size),
        inject_layers=inject_layers,
        alpha=float(args.alpha),
        rank=int(args.rank),
        state_bottleneck=int(args.state_bottleneck),
    ).to(device)
    verify_trainable_parameters(model=model, adapter=adapter, method=method)
    optimizer = torch.optim.AdamW(
        [param for param in adapter.parameters() if param.requires_grad],
        lr=float(lr),
        weight_decay=0.0,
    )
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"{safe_name(args.config_name)}_{safe_name(method)}_best.pt"
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_acc = -math.inf
    best_val_ce = math.inf
    count_min = min(int(x) for x in count_values)

    for epoch in range(1, int(args.epochs) + 1):
        adapter.train()
        adapter.set_alpha(float(args.alpha))
        rng = random.Random(int(args.seed) + epoch * 997 + prev.stable_hash_int(f"{args.config_name}:{method}") % 100000)
        shuffled = [int(x) for x in train_indices]
        rng.shuffle(shuffled)
        optimizer.zero_grad(set_to_none=True)
        train_ce_total = 0.0
        train_loss_total = 0.0
        train_steps = 0
        backward_steps = 0
        skipped_no_positions = 0
        try:
            adapter.register_hooks(model)
            for step, batch_indices in enumerate(prev.chunked(shuffled, int(args.batch_size)), start=1):
                batch_records = [records[int(idx)] for idx in batch_indices]
                batch = prepare_translator_batch(
                    records=batch_records,
                    sample_indices=batch_indices,
                    processor=processor,
                    device=device,
                    token_group=token_group,
                )
                if not any(batch.target_positions):
                    skipped_no_positions += 1
                    continue
                inject_counts = control_counts_for_injection(
                    gold_counts=batch.gold_counts,
                    sample_indices=batch.sample_indices,
                    count_values=count_values,
                    seed=int(args.seed),
                    control=str(args.count_control),
                )
                adapter.set_context(inject_counts, batch.target_positions)
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - int(count_min)
                ce = F.cross_entropy(count_logits, gold_offsets)
                m_loss = margin_loss(count_logits, gold_offsets, float(args.margin_target))
                reg = adapter.parameter_l2().to(device)
                loss = ce + float(args.lambda_margin) * m_loss + float(args.weight_decay) * reg
                if not loss.requires_grad:
                    skipped_no_positions += 1
                    adapter.clear_context()
                    continue
                (loss / max(1, int(args.grad_accum))).backward()
                train_ce_total += float(ce.detach().cpu().item())
                train_loss_total += float(loss.detach().cpu().item())
                train_steps += 1
                backward_steps += 1
                adapter.clear_context()
                if backward_steps % max(1, int(args.grad_accum)) == 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 50 == 0:
                    print(
                        f"  {args.config_name}/{method} epoch={epoch} step={step} "
                        f"train_ce={train_ce_total / max(1, train_steps):.4f}"
                    )
            if backward_steps and backward_steps % max(1, int(args.grad_accum)) != 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        finally:
            adapter.remove_hooks()

        val_eval = evaluate_split(
            split_name="val",
            method_label=f"{method}__val",
            model=model,
            processor=processor,
            adapter=adapter,
            records=records,
            indices=val_indices,
            count_token_ids=count_token_ids,
            count_values=count_values,
            token_group=token_group,
            alpha=float(args.alpha),
            device=device,
            batch_size=int(args.batch_size),
            max_eval_samples=int(args.max_eval_samples),
            count_control=str(args.count_control),
            seed=int(args.seed),
        )
        row = {
            "config_name": str(args.config_name),
            "method": str(method),
            "epoch": int(epoch),
            "train_ce": train_ce_total / max(1, train_steps),
            "train_loss": train_loss_total / max(1, train_steps),
            "train_steps": int(train_steps),
            "skipped_no_positions": int(skipped_no_positions),
            "val_ce": float(val_eval["ce"]),
            "val_acc": float(val_eval["accuracy"]),
            "mean_val_injection_norm": finite_mean(list(val_eval["injection_norm_by_idx"].values()), default=0.0),
            "adapter_parameter_norm": adapter_parameter_norm(adapter),
        }
        history.append(row)
        print(
            f"  {args.config_name}/{method} epoch={epoch} train_ce={row['train_ce']:.4f} "
            f"val_ce={row['val_ce']:.4f} val_acc={row['val_acc']:.4f} "
            f"mean_inj_norm={row['mean_val_injection_norm']:.4f}"
        )
        improved = row["val_acc"] > best_val_acc + 1e-9 or (
            abs(row["val_acc"] - best_val_acc) <= 1e-9 and row["val_ce"] < best_val_ce
        )
        if improved:
            best_val_acc = float(row["val_acc"])
            best_val_ce = float(row["val_ce"])
            best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
            torch.save(
                {
                    "adapter_state_dict": best_state,
                    "config": {
                        "method": str(method),
                        "config_name": str(args.config_name),
                        "token_group": str(token_group),
                        "inject_layers": [int(x) for x in inject_layers],
                        "alpha": float(args.alpha),
                        "rank": int(args.rank),
                        "state_bottleneck": int(args.state_bottleneck),
                        "count_control": str(args.count_control),
                    },
                    "history": history,
                    "hidden_size": int(hidden_size),
                    "count_values": [int(x) for x in count_values],
                },
                checkpoint_path,
            )
    if best_state is not None:
        adapter.load_state_dict(best_state)
    return adapter.cpu(), history, checkpoint_path


def verify_trainable_parameters(*, model: Any, adapter: nn.Module, method: str) -> None:
    model_trainable = sum(int(param.requires_grad) for param in model.parameters())
    if model_trainable:
        raise RuntimeError(f"Qwen is not frozen: {model_trainable} model parameters still require grad")
    adapter_trainable = sum(int(param.requires_grad) for param in adapter.parameters())
    adapter_param_count = sum(int(param.numel()) for param in adapter.parameters() if param.requires_grad)
    if adapter_trainable <= 0 or adapter_param_count <= 0:
        raise RuntimeError(f"No trainable translator parameters for method={method}")
    print(f"Verified frozen Qwen; trainable translator tensors={adapter_trainable} params={adapter_param_count}")


def make_metrics_rows(
    *,
    eval_payload: Dict[str, Any],
    records: Sequence[prev.SampleRecord],
    count_values: Sequence[int],
    seq_len: int,
    config_name: str,
    method: str,
    token_group: str,
    requested_token_group: str,
    layer_start: int,
    layer_end: int,
    alpha: float,
    rank: int,
    lr: float,
    epoch: int,
    split_name: str,
) -> List[Dict[str, Any]]:
    count_min = min(int(x) for x in count_values)
    rows: List[Dict[str, Any]] = []
    for idx in eval_payload["indices"]:
        idx = int(idx)
        record = records[idx]
        pred = eval_payload["pred_by_idx"].get(idx, "")
        logits = eval_payload["logits_by_idx"].get(idx, [])
        gold = int(record.gold_count)
        gold_offset = gold - count_min
        pred_offset = int(pred) - count_min if pred != "" else -1
        rows.append(
            {
                "sample_id": record.sample_id,
                "sample_index": idx,
                "split": str(split_name),
                "seq_len": int(seq_len),
                "evidence_count": int(record.evidence_count),
                "gold_answer": gold,
                "pred_answer": pred,
                "correct": int(pred == gold) if pred != "" else "",
                "gold_logit": logits[gold_offset] if 0 <= gold_offset < len(logits) else "",
                "pred_logit": logits[pred_offset] if 0 <= pred_offset < len(logits) else "",
                "margin": eval_payload["margin_by_idx"].get(idx, ""),
                "injection_norm": eval_payload["injection_norm_by_idx"].get(idx, 0.0),
                "token_selection_ok": int(bool(eval_payload["token_selection_ok_by_idx"].get(idx, True))),
                "token_selection_error": eval_payload["token_selection_error_by_idx"].get(idx, ""),
                "num_injected_tokens": eval_payload["num_injected_tokens_by_idx"].get(idx, 0),
                "config_name": str(config_name),
                "method": str(method),
                "token_group": str(requested_token_group),
                "resolved_token_group": str(token_group),
                "layer_start": int(layer_start),
                "layer_end": int(layer_end),
                "layer_window": layer_window_label(int(layer_start), int(layer_end)) if method != BASELINE else "",
                "alpha": float(alpha),
                "rank": int(rank),
                "lr": float(lr),
                "epoch": int(epoch),
                "candidate_logits_json": json_dumps_compact(logits),
            }
        )
    return rows


def summarize_rows(
    *,
    metrics_rows: Sequence[Dict[str, Any]],
    train_eval: Dict[str, Any],
    eval_payload: Dict[str, Any],
    config_name: str,
    method: str,
    token_group: str,
    layer_start: int,
    layer_end: int,
    alpha: float,
    rank: int,
    lr: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    correct = [int(row["correct"]) for row in metrics_rows if row.get("correct") != ""]
    gold_logits = [value for row in metrics_rows if (value := finite_float(row.get("gold_logit"))) is not None]
    margins = [value for row in metrics_rows if (value := finite_float(row.get("margin"))) is not None]
    injection_norms = [value for row in metrics_rows if (value := finite_float(row.get("injection_norm"))) is not None]
    counts = sorted({int(row["evidence_count"]) for row in metrics_rows})
    acc_by_count: Dict[str, float] = {}
    per_count_rows: List[Dict[str, Any]] = []
    for count in counts:
        count_rows = [row for row in metrics_rows if int(row["evidence_count"]) == int(count)]
        count_correct = [int(row["correct"]) for row in count_rows if row.get("correct") != ""]
        count_margins = [value for row in count_rows if (value := finite_float(row.get("margin"))) is not None]
        count_norms = [value for row in count_rows if (value := finite_float(row.get("injection_norm"))) is not None]
        acc = float(np.mean(count_correct)) if count_correct else math.nan
        acc_by_count[str(count)] = acc
        per_count_rows.append(
            {
                "config_name": str(config_name),
                "method": str(method),
                "token_group": str(token_group),
                "layer_window": layer_window_label(int(layer_start), int(layer_end)) if method != BASELINE else "",
                "evidence_count": int(count),
                "n": len(count_correct),
                "accuracy": acc,
                "mean_margin": float(np.mean(count_margins)) if count_margins else math.nan,
                "mean_injection_norm": float(np.mean(count_norms)) if count_norms else 0.0,
            }
        )
    summary_rows = [
        {
            "config_name": str(config_name),
            "method": str(method),
            "token_group": str(token_group),
            "layer_window": layer_window_label(int(layer_start), int(layer_end)) if method != BASELINE else "",
            "overall_acc": float(np.mean(correct)) if correct else math.nan,
            "mean_gold_logit": float(np.mean(gold_logits)) if gold_logits else math.nan,
            "mean_margin": float(np.mean(margins)) if margins else math.nan,
            "mean_injection_norm": float(np.mean(injection_norms)) if injection_norms else 0.0,
            "train_acc": float(train_eval["accuracy"]),
            "eval_acc": float(eval_payload["accuracy"]),
            "acc_by_evidence_count": json_dumps_compact(acc_by_count),
            "alpha": float(alpha),
            "rank": int(rank),
            "lr": float(lr),
        }
    ]
    return summary_rows, per_count_rows


def write_dynamic_csv(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str]) -> None:
    fields = list(leading)
    for key in sorted({key for row in rows for key in row.keys()}):
        if key not in fields:
            fields.append(key)
    prev.write_csv(path, fields, rows)


def make_plots(output_dir: Path, summary_rows: Sequence[Dict[str, Any]], per_count_rows: Sequence[Dict[str, Any]]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        names = [f"{row['config_name']}\n{row['method']}" for row in summary_rows]
        accs = [float(row.get("overall_acc", math.nan)) for row in summary_rows]
        plt.figure(figsize=(max(6.0, 1.6 * len(names)), 4.5))
        plt.bar(np.arange(len(names)), accs)
        plt.xticks(np.arange(len(names)), names, rotation=20, ha="right")
        plt.ylabel("Accuracy")
        plt.title("Overall Accuracy by Method/Config")
        plt.ylim(0, max(1.0, max([x for x in accs if math.isfinite(x)] or [0.0]) * 1.1))
        plt.tight_layout()
        plt.savefig(plots_dir / "overall_accuracy_by_method_config.png", dpi=180, bbox_inches="tight")
        plt.close()

    labels = list(dict.fromkeys(f"{row['config_name']} / {row['method']}" for row in per_count_rows))
    counts = sorted({int(row["evidence_count"]) for row in per_count_rows})
    if labels and counts:
        for key, ylabel, filename in (
            ("accuracy", "Accuracy", "accuracy_vs_evidence_count.png"),
            ("mean_margin", "Mean margin", "margin_vs_evidence_count.png"),
            ("mean_injection_norm", "Mean injection norm", "injection_norm_vs_evidence_count.png"),
        ):
            plt.figure(figsize=(7.5, 4.8))
            for label in labels:
                config, method = label.split(" / ", 1)
                by_count = {
                    int(row["evidence_count"]): row
                    for row in per_count_rows
                    if row["config_name"] == config and row["method"] == method
                }
                ys = [float(by_count.get(count, {}).get(key, math.nan)) for count in counts]
                plt.plot(counts, ys, marker="o", linewidth=1.8, label=label)
            plt.xlabel("Evidence count")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} vs Evidence Count")
            plt.xticks(counts)
            plt.grid(alpha=0.25)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=180, bbox_inches="tight")
            plt.close()

        matrix = np.full((len(labels), len(counts)), np.nan, dtype=float)
        for i, label in enumerate(labels):
            config, method = label.split(" / ", 1)
            for j, count in enumerate(counts):
                matches = [
                    row
                    for row in per_count_rows
                    if row["config_name"] == config and row["method"] == method and int(row["evidence_count"]) == int(count)
                ]
                if matches:
                    matrix[i, j] = float(matches[0].get("accuracy", math.nan))
        fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(counts)), max(3.5, 0.5 * len(labels))))
        im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(counts)))
        ax.set_xticklabels(counts)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Evidence count")
        ax.set_ylabel("Method / Config")
        ax.set_title("Accuracy Heatmap")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if math.isfinite(float(matrix[i, j])):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "heatmap_method_config_x_evidence_count_accuracy.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    args = parse_args()
    method = canonical_method(args.method)
    requested_token_group = str(args.token_group)
    token_group = "none" if method == BASELINE else canonical_token_group(args.token_group)
    if int(args.seq_len) != 8:
        raise ValueError("This experiment is intentionally seq_len=8 only.")
    if int(args.layer_end) < int(args.layer_start):
        raise ValueError("--layer-end must be >= --layer-start")
    lr = float(args.lr) if args.lr is not None else default_lr_for_method(method)
    inject_layers = [] if method == BASELINE else list(range(int(args.layer_start), int(args.layer_end) + 1))
    count_values = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
    output_dir = make_output_dir(args)
    log_handle, old_stdout, old_stderr = prev.setup_logging(output_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        run_config = {
            "experiment_name": EXPERIMENT_NAME,
            "model_name": str(args.model_name),
            "dataset_root": os.fspath(args.dataset_root),
            "source_run": os.fspath(args.source_run),
            "seq_len": int(args.seq_len),
            "split": str(args.split),
            "output_root": os.fspath(Path(args.output_root).resolve()),
            "output_dir": os.fspath(output_dir),
            "config_name": str(args.config_name),
            "method": str(method),
            "requested_method": str(args.method),
            "token_group": str(requested_token_group),
            "resolved_token_group": str(token_group),
            "layer_start": int(args.layer_start),
            "layer_end": int(args.layer_end),
            "inject_layers": inject_layers,
            "alpha": float(args.alpha),
            "epochs": int(args.epochs),
            "lr": float(lr),
            "rank": int(args.rank),
            "state_bottleneck": int(args.state_bottleneck),
            "batch_size": int(args.batch_size),
            "grad_accum": int(args.grad_accum),
            "max_train_samples": int(args.max_train_samples),
            "max_eval_samples": int(args.max_eval_samples),
            "lambda_margin": float(args.lambda_margin),
            "margin_target": float(args.margin_target),
            "weight_decay": float(args.weight_decay),
            "count_control": str(args.count_control),
            "seed": int(args.seed),
            "submit_mode": str(args.submit_mode),
        }
        prev.write_json(output_dir / "run_config.json", run_config)
        print(f"Output dir: {output_dir}")
        print(f"Run config: {json.dumps(run_config, sort_keys=True)}")

        sample_payload = load_sample_index_payload(args)
        sample_ids = sample_payload["sample_ids"]
        labels = sample_payload["labels"].long()
        records = prev.load_records(args.dataset_root, args.split, args.seq_len, sample_ids)
        splits = prev.stratified_split(sample_ids, labels, int(args.seed))
        train_indices = split_limited_indices(splits["train"], records, int(args.max_train_samples), int(args.seed) + 11)
        val_indices = split_limited_indices(
            splits["val"] or splits["train"],
            records,
            int(args.max_eval_samples),
            int(args.seed) + 17,
        )
        test_indices = split_limited_indices(
            splits["test"] or splits["val"] or splits["train"],
            records,
            int(args.max_eval_samples),
            int(args.seed) + 23,
        )
        split_counts = prev.split_counts(
            {"train": train_indices, "val": val_indices, "test": test_indices},
            labels,
            count_values,
        )
        for split, row in split_counts.items():
            print(f"  {split}: " + ", ".join(f"{count}:{row.get(count, 0)}" for count in count_values))
        if not test_indices:
            raise RuntimeError("Test split is empty after limiting")
        if method != BASELINE and (not train_indices or not val_indices):
            raise RuntimeError("Train/val split is empty after limiting")

        device = prev.resolve_device(str(args.device))
        dtype = prev.resolve_dtype(str(args.dtype), device)
        model, processor = prev.load_model_and_processor(args, device=device, dtype=dtype)
        hidden_size = prev.hidden_size_from_model(model)
        candidate_format, count_token_ids = prev.candidate_token_ids(
            processor.tokenizer,
            int(args.candidate_min),
            int(args.candidate_max),
        )
        print(f"Loaded frozen Qwen hidden_size={hidden_size} candidate_format={candidate_format}")
        model_trainable = sum(int(param.requires_grad) for param in model.parameters())
        if model_trainable:
            raise RuntimeError(f"Qwen is not frozen: {model_trainable} model parameters still require grad")

        adapter: Optional[TranslatorInjectionAdapter] = None
        train_history_rows: List[Dict[str, Any]] = []
        checkpoint_path: Optional[Path] = None
        final_epoch = 0
        if method != BASELINE:
            print(f"Training {args.config_name}/{method}")
            adapter, train_history_rows, checkpoint_path = train_adapter(
                method=method,
                args=args,
                output_dir=output_dir,
                model=model,
                processor=processor,
                records=records,
                train_indices=train_indices,
                val_indices=val_indices,
                count_token_ids=count_token_ids,
                count_values=count_values,
                hidden_size=int(hidden_size),
                inject_layers=inject_layers,
                token_group=token_group,
                device=device,
                lr=float(lr),
            )
            final_epoch = int(train_history_rows[-1]["epoch"]) if train_history_rows else int(args.epochs)
            prev.write_json(
                output_dir / "checkpoint.json",
                {"trained_checkpoint": os.fspath(checkpoint_path) if checkpoint_path is not None else ""},
            )
        else:
            final_epoch = 0

        print(f"Evaluating train split for {args.config_name}/{method}")
        train_eval = evaluate_split(
            split_name="train",
            method_label=method,
            model=model,
            processor=processor,
            adapter=adapter,
            records=records,
            indices=train_indices,
            count_token_ids=count_token_ids,
            count_values=count_values,
            token_group=token_group,
            alpha=float(args.alpha) if method != BASELINE else 0.0,
            device=device,
            batch_size=int(args.batch_size),
            max_eval_samples=0,
            count_control=str(args.count_control),
            seed=int(args.seed),
        )
        print(f"Evaluating test split for {args.config_name}/{method}")
        eval_payload = evaluate_split(
            split_name="test",
            method_label=method,
            model=model,
            processor=processor,
            adapter=adapter,
            records=records,
            indices=test_indices,
            count_token_ids=count_token_ids,
            count_values=count_values,
            token_group=token_group,
            alpha=float(args.alpha) if method != BASELINE else 0.0,
            device=device,
            batch_size=int(args.batch_size),
            max_eval_samples=0,
            count_control=str(args.count_control),
            seed=int(args.seed),
        )
        metrics_rows = make_metrics_rows(
            eval_payload=eval_payload,
            records=records,
            count_values=count_values,
            seq_len=int(args.seq_len),
            config_name=str(args.config_name),
            method=str(method),
            token_group=str(token_group),
            requested_token_group=str(requested_token_group),
            layer_start=int(args.layer_start),
            layer_end=int(args.layer_end),
            alpha=float(args.alpha) if method != BASELINE else 0.0,
            rank=int(args.rank),
            lr=float(lr),
            epoch=int(final_epoch),
            split_name="test",
        )
        summary_rows, per_count_rows = summarize_rows(
            metrics_rows=metrics_rows,
            train_eval=train_eval,
            eval_payload=eval_payload,
            config_name=str(args.config_name),
            method=str(method),
            token_group=str(requested_token_group),
            layer_start=int(args.layer_start),
            layer_end=int(args.layer_end),
            alpha=float(args.alpha) if method != BASELINE else 0.0,
            rank=int(args.rank),
            lr=float(lr),
        )

        if adapter is not None and bool(args.include_random_untrained_same_norm):
            random_adapter = make_random_untrained_same_norm(adapter, seed=int(args.seed)).to(device)
            random_method = f"{method}_random_untrained_same_norm"
            print(f"Evaluating {random_method}")
            random_train_eval = evaluate_split(
                split_name="train",
                method_label=random_method,
                model=model,
                processor=processor,
                adapter=random_adapter,
                records=records,
                indices=train_indices,
                count_token_ids=count_token_ids,
                count_values=count_values,
                token_group=token_group,
                alpha=float(args.alpha),
                device=device,
                batch_size=int(args.batch_size),
                max_eval_samples=0,
                count_control=str(args.count_control),
                seed=int(args.seed),
            )
            random_eval = evaluate_split(
                split_name="test",
                method_label=random_method,
                model=model,
                processor=processor,
                adapter=random_adapter,
                records=records,
                indices=test_indices,
                count_token_ids=count_token_ids,
                count_values=count_values,
                token_group=token_group,
                alpha=float(args.alpha),
                device=device,
                batch_size=int(args.batch_size),
                max_eval_samples=0,
                count_control=str(args.count_control),
                seed=int(args.seed),
            )
            random_rows = make_metrics_rows(
                eval_payload=random_eval,
                records=records,
                count_values=count_values,
                seq_len=int(args.seq_len),
                config_name=f"{args.config_name}_random_untrained_same_norm",
                method=random_method,
                token_group=str(token_group),
                requested_token_group=str(requested_token_group),
                layer_start=int(args.layer_start),
                layer_end=int(args.layer_end),
                alpha=float(args.alpha),
                rank=int(args.rank),
                lr=float(lr),
                epoch=0,
                split_name="test",
            )
            random_summary, random_per_count = summarize_rows(
                metrics_rows=random_rows,
                train_eval=random_train_eval,
                eval_payload=random_eval,
                config_name=f"{args.config_name}_random_untrained_same_norm",
                method=random_method,
                token_group=str(requested_token_group),
                layer_start=int(args.layer_start),
                layer_end=int(args.layer_end),
                alpha=float(args.alpha),
                rank=int(args.rank),
                lr=float(lr),
            )
            metrics_rows.extend(random_rows)
            summary_rows.extend(random_summary)
            per_count_rows.extend(random_per_count)

        write_dynamic_csv(
            output_dir / "metrics.csv",
            metrics_rows,
            [
                "sample_id",
                "seq_len",
                "evidence_count",
                "gold_answer",
                "pred_answer",
                "correct",
                "gold_logit",
                "pred_logit",
                "margin",
                "injection_norm",
                "token_selection_ok",
                "config_name",
                "method",
                "token_group",
                "layer_start",
                "layer_end",
                "alpha",
                "rank",
                "lr",
                "epoch",
            ],
        )
        write_dynamic_csv(
            output_dir / "summary.csv",
            summary_rows,
            [
                "config_name",
                "method",
                "token_group",
                "layer_window",
                "overall_acc",
                "mean_gold_logit",
                "mean_margin",
                "mean_injection_norm",
                "train_acc",
                "eval_acc",
                "acc_by_evidence_count",
            ],
        )
        write_dynamic_csv(
            output_dir / "accuracy_by_evidence_count.csv",
            per_count_rows,
            ["config_name", "method", "evidence_count", "n", "accuracy", "mean_margin", "mean_injection_norm"],
        )
        if train_history_rows:
            write_dynamic_csv(
                output_dir / "train_history.csv",
                train_history_rows,
                [
                    "config_name",
                    "method",
                    "epoch",
                    "train_ce",
                    "train_loss",
                    "val_ce",
                    "val_acc",
                    "mean_val_injection_norm",
                    "adapter_parameter_norm",
                ],
            )
        if not bool(args.no_plots):
            make_plots(output_dir, summary_rows, per_count_rows)

        debug = {
            "runtime_seconds": time.time() - started,
            "source_cache": os.fspath(sample_payload["cache_path"]),
            "num_records": len(records),
            "splits": {key: len(value) for key, value in splits.items()},
            "limited_splits": {"train": len(train_indices), "val": len(val_indices), "test": len(test_indices)},
            "output_dir": os.fspath(output_dir),
        }
        prev.write_json(output_dir / "debug.json", debug)
        print(f"Finished {EXPERIMENT_NAME} config={args.config_name} method={method} in {time.time() - started:.1f}s")
        print(f"Results: {output_dir}")
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

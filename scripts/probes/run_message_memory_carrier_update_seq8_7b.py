#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
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
from torch import nn

from evaluations.helpers import utils as eval_utils
from models.model import image_token_groups
from scripts.experiments import translator_ablation_gold_count_seq8_7b as trans
from scripts.probes import run_message_memory_adapter_stage1_stage3_seq8 as prev

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
except Exception:  # pragma: no cover - version-dependent optional exact path
    apply_multimodal_rotary_pos_emb = None  # type: ignore[assignment]


EXPERIMENT_NAME = "message_memory_carrier_update_seq8_7b"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
BASELINE = "baseline_no_injection"
LAYER_LOCAL = "layer_local_message_memory"
CUMULATIVE = "cumulative_sum_message_memory"
NUM_FRAMES = 8
SIGMOID_GATE_READOUT = "sigmoid_gate"
RAW_MATRIX_READOUT = "raw_matrix"
READOUT_MODES = {SIGMOID_GATE_READOUT, RAW_MATRIX_READOUT}
ROOM_CHAR_TOKEN_GROUP = "room_char"
ALL_QUESTION_TOKEN_GROUP = "all_question"
LAST_TOKEN_GROUP = "last_token"
QUESTION_PLUS_LAST_TOKEN_GROUP = "question_plus_last"
ROOM_CHAR_PLUS_QUESTION_TOKEN_GROUP = "room_char_plus_question"
ROOM_CHAR_QUESTION_LAST_TOKEN_GROUP = "room_char_question_last"
NO_TOKEN_GROUP = "none"
TOKEN_GROUPS = {
    NO_TOKEN_GROUP,
    ROOM_CHAR_TOKEN_GROUP,
    ALL_QUESTION_TOKEN_GROUP,
    LAST_TOKEN_GROUP,
    QUESTION_PLUS_LAST_TOKEN_GROUP,
    ROOM_CHAR_PLUS_QUESTION_TOKEN_GROUP,
    ROOM_CHAR_QUESTION_LAST_TOKEN_GROUP,
}
TOKEN_GROUP_TO_LOCATOR = {
    NO_TOKEN_GROUP: "none",
    ROOM_CHAR_TOKEN_GROUP: "room_char",
    ALL_QUESTION_TOKEN_GROUP: "all_question_tokens",
    "all_question_tokens": "all_question_tokens",
    "question": "all_question_tokens",
    LAST_TOKEN_GROUP: "last_token",
    QUESTION_PLUS_LAST_TOKEN_GROUP: "question_plus_last",
    ROOM_CHAR_PLUS_QUESTION_TOKEN_GROUP: "room_char_plus_question",
    ROOM_CHAR_QUESTION_LAST_TOKEN_GROUP: "room_char_question_last",
}
TOKEN_GROUP_ALIASES = {
    "none": NO_TOKEN_GROUP,
    "room_char": ROOM_CHAR_TOKEN_GROUP,
    "all_question": ALL_QUESTION_TOKEN_GROUP,
    "all_question_tokens": ALL_QUESTION_TOKEN_GROUP,
    "question": ALL_QUESTION_TOKEN_GROUP,
    "last_token": LAST_TOKEN_GROUP,
    "all_question+last_token": QUESTION_PLUS_LAST_TOKEN_GROUP,
    "all_question_tokens+last_token": QUESTION_PLUS_LAST_TOKEN_GROUP,
    "question_plus_last": QUESTION_PLUS_LAST_TOKEN_GROUP,
    "room_char+all_question": ROOM_CHAR_PLUS_QUESTION_TOKEN_GROUP,
    "room_char+all_question_tokens": ROOM_CHAR_PLUS_QUESTION_TOKEN_GROUP,
    "room_char_plus_question": ROOM_CHAR_PLUS_QUESTION_TOKEN_GROUP,
    "room_char+all_question+last_token": ROOM_CHAR_QUESTION_LAST_TOKEN_GROUP,
    "room_char+all_question_tokens+last_token": ROOM_CHAR_QUESTION_LAST_TOKEN_GROUP,
    "room_char_question_last": ROOM_CHAR_QUESTION_LAST_TOKEN_GROUP,
}

VARIANT_ALIASES = {
    "baseline": BASELINE,
    "baseline_no_injection": BASELINE,
    "none": BASELINE,
    "layer_local": LAYER_LOCAL,
    "layer_local_message_memory": LAYER_LOCAL,
    "cumulative_sum": CUMULATIVE,
    "cumulative_sum_message_memory": CUMULATIVE,
}


@dataclass
class MemoryBatch:
    inputs: Dict[str, Any]
    target_positions: List[List[int]]
    token_group: str
    message_target_positions: List[List[int]]
    query_positions: List[List[int]]
    inject_positions: List[List[int]]
    frame_groups: List[List[List[int]]]
    prompt_last_indices: torch.Tensor
    gold_counts: torch.Tensor
    sample_indices: List[int]
    token_selection_ok: List[bool]
    token_selection_errors: List[str]
    frame_grouping_ok: List[bool]
    frame_grouping_errors: List[str]
    frame_token_counts: List[List[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learned message-memory carrier residual updates for MMReD seq_len=8 "
            "Qwen2.5-VL-7B counting. Qwen is frozen; only the memory adapter is trained."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/mmred_images_park")
    parser.add_argument("--source-run", type=Path, default=prev.DEFAULT_SOURCE_RUN)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--split", default="all_uniform")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="")

    parser.add_argument("--memory-variant", default=None, choices=sorted(VARIANT_ALIASES))
    parser.add_argument(
        "--run-baseline-and-layer-local-raw-matrix",
        action="store_true",
        default=False,
        help="Run only frozen baseline and layer-local raw-matrix adapter, then write combined comparison outputs.",
    )
    parser.add_argument("--d-mem", type=int, default=256)
    parser.add_argument("--layer-start", type=int, default=14)
    parser.add_argument("--layer-end", type=int, default=17)
    parser.add_argument("--gamma-init", type=float, default=0.05)
    parser.add_argument("--message-mode", default="auto", choices=["auto", "exact", "proxy"])
    parser.add_argument("--readout-mode", default=SIGMOID_GATE_READOUT, choices=sorted(READOUT_MODES))
    parser.add_argument("--token-group", default=ROOM_CHAR_TOKEN_GROUP, choices=sorted(TOKEN_GROUPS))

    parser.add_argument("--lambda-margin", type=float, default=0.2)
    parser.add_argument("--margin-target", type=float, default=1.0)
    parser.add_argument("--lambda-update-energy", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-samples-per-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true", default=False)

    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in range(9)])

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
    parser.add_argument("--submit-mode", default="local")
    return parser.parse_args()


def canonical_variant(raw: str) -> str:
    try:
        return VARIANT_ALIASES[str(raw)]
    except KeyError as exc:
        raise ValueError(f"Unknown memory variant {raw!r}") from exc


def canonical_token_group(raw: str) -> str:
    key = str(raw).strip()
    if key not in TOKEN_GROUP_ALIASES:
        raise ValueError(f"Unknown token_group={raw!r}; valid={sorted(TOKEN_GROUP_ALIASES)}")
    return TOKEN_GROUP_ALIASES[key]


def locator_token_group(raw: str) -> str:
    group = canonical_token_group(raw)
    return TOKEN_GROUP_TO_LOCATOR[group]


def safe_name(text: Any) -> str:
    safe = str(text)
    for old, new in (("/", "_"), ("+", "_"), (" ", "_"), (":", "_"), (".", "p"), (",", "_")):
        safe = safe.replace(old, new)
    return safe


def default_output_dir(args: argparse.Namespace, variant: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = str(args.run_name).strip() or f"{safe_name(variant)}_l{int(args.layer_start)}_{int(args.layer_end)}"
    return Path(args.output_root).resolve() / f"{stamp}_{run_name}"


def write_csv_dynamic(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str]) -> None:
    fields = list(leading)
    for key in sorted({key for row in rows for key in row.keys()}):
        if key not in fields:
            fields.append(key)
    prev.write_csv(path, fields, rows)


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_mean(values: Iterable[Any], default: float = math.nan) -> float:
    vals = [float(v) for v in values if finite_float(v) is not None]
    return float(np.mean(vals)) if vals else float(default)


def correlation(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    pairs = [(finite_float(x), finite_float(y)) for x, y in zip(xs, ys)]
    clean = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None]
    if len(clean) < 2:
        return math.nan
    x_arr = np.array([x for x, _ in clean], dtype=float)
    y_arr = np.array([y for _, y in clean], dtype=float)
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def split_limited_indices(indices: Sequence[int], records: Sequence[prev.SampleRecord], limit: int, seed: int) -> List[int]:
    return trans.split_limited_indices(indices, records, int(limit), int(seed))


def resolve_frame_groups(
    *,
    input_ids_1d: torch.Tensor,
    expected_num_frames: int,
    processor: Any,
) -> Tuple[List[List[int]], bool, str]:
    expected = int(expected_num_frames)
    groups = image_token_groups(input_ids_1d.detach().cpu(), expected, processor=processor)
    groups = [[int(pos) for pos in group] for group in groups]
    if len(groups) == expected and all(groups):
        return groups, True, ""

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    positions: List[int] = []
    if image_token_id is not None:
        positions = [int(pos.item()) for pos in (input_ids_1d.detach().cpu() == int(image_token_id)).nonzero(as_tuple=True)[0]]
    if expected > 0 and positions and len(positions) % expected == 0:
        per_frame = len(positions) // expected
        fallback = [positions[i * per_frame : (i + 1) * per_frame] for i in range(expected)]
        return fallback, False, f"contiguous image groups found {len(groups)}; used equal image-token split"

    padded = groups[:expected]
    while len(padded) < expected:
        padded.append([])
    return padded, False, f"expected {expected} frame groups, found {len(groups)}"


def evidence_frame_mask(record: prev.SampleRecord, seq_len: int) -> List[int]:
    try:
        indices = eval_utils.collect_evidence_frame_indices(record.question, record.states)
    except Exception:
        return []
    clean = {int(idx) for idx in indices if 0 <= int(idx) < int(seq_len)}
    return [1 if idx in clean else 0 for idx in range(int(seq_len))]


def prepare_memory_batch(
    *,
    records: Sequence[prev.SampleRecord],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
    token_group: str = ROOM_CHAR_TOKEN_GROUP,
    message_token_group: Optional[str] = None,
    query_token_group: Optional[str] = None,
    inject_token_group: Optional[str] = None,
) -> MemoryBatch:
    resolved_token_group = canonical_token_group(token_group)
    resolved_message_token_group = canonical_token_group(message_token_group if message_token_group is not None else resolved_token_group)
    resolved_query_token_group = canonical_token_group(query_token_group if query_token_group is not None else resolved_token_group)
    resolved_inject_token_group = canonical_token_group(inject_token_group if inject_token_group is not None else resolved_token_group)
    message_locator_group = locator_token_group(resolved_message_token_group)
    query_locator_group = locator_token_group(resolved_query_token_group)
    inject_locator_group = locator_token_group(resolved_inject_token_group)
    frames_by_record = [trans.load_frames(record) for record in records]
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
    message_target_positions: List[List[int]] = []
    query_positions: List[List[int]] = []
    inject_positions: List[List[int]] = []
    prompt_last_indices: List[int] = []
    token_selection_ok: List[bool] = []
    token_selection_errors: List[str] = []
    frame_groups: List[List[List[int]]] = []
    frame_grouping_ok: List[bool] = []
    frame_grouping_errors: List[str] = []
    frame_token_counts: List[List[int]] = []

    for batch_idx, record in enumerate(records):
        message_positions, prompt_last, message_ok, message_error = trans.locate_positions_safe(
            input_ids_1d=input_ids[batch_idx],
            attention_mask_1d=attention_mask[batch_idx] if torch.is_tensor(attention_mask) else None,
            record=record,
            processor=processor,
            token_group=message_locator_group,
        )
        query_pos, query_prompt_last, query_ok, query_error = trans.locate_positions_safe(
            input_ids_1d=input_ids[batch_idx],
            attention_mask_1d=attention_mask[batch_idx] if torch.is_tensor(attention_mask) else None,
            record=record,
            processor=processor,
            token_group=query_locator_group,
        )
        inject_pos, inject_prompt_last, inject_ok, inject_error = trans.locate_positions_safe(
            input_ids_1d=input_ids[batch_idx],
            attention_mask_1d=attention_mask[batch_idx] if torch.is_tensor(attention_mask) else None,
            record=record,
            processor=processor,
            token_group=inject_locator_group,
        )
        prompt_last = int(max(int(prompt_last), int(query_prompt_last), int(inject_prompt_last)))
        groups, groups_ok, group_error = resolve_frame_groups(
            input_ids_1d=input_ids[batch_idx],
            expected_num_frames=len(record.frame_paths),
            processor=processor,
        )
        groups = groups[:NUM_FRAMES]
        while len(groups) < NUM_FRAMES:
            groups.append([])
        target_positions.append([int(pos) for pos in inject_pos])
        message_target_positions.append([int(pos) for pos in message_positions])
        query_positions.append([int(pos) for pos in query_pos])
        inject_positions.append([int(pos) for pos in inject_pos])
        prompt_last_indices.append(int(prompt_last))
        token_selection_ok.append(bool(message_ok and query_ok and inject_ok))
        selection_errors = []
        if not message_ok:
            selection_errors.append(f"message={message_error}")
        if not query_ok:
            selection_errors.append(f"query={query_error}")
        if not inject_ok:
            selection_errors.append(f"inject={inject_error}")
        token_selection_errors.append("; ".join(selection_errors))
        frame_groups.append(groups)
        frame_grouping_ok.append(bool(groups_ok and len(groups) == NUM_FRAMES and all(groups)))
        frame_grouping_errors.append(str(group_error))
        frame_token_counts.append([len(group) for group in groups])

    return MemoryBatch(
        inputs=prev.move_inputs_to_device(dict(raw_inputs), device),
        target_positions=target_positions,
        token_group=resolved_token_group,
        message_target_positions=message_target_positions,
        query_positions=query_positions,
        inject_positions=inject_positions,
        frame_groups=frame_groups,
        prompt_last_indices=torch.tensor(prompt_last_indices, device=device, dtype=torch.long),
        gold_counts=torch.tensor([int(record.gold_count) for record in records], device=device, dtype=torch.long),
        sample_indices=[int(idx) for idx in sample_indices],
        token_selection_ok=token_selection_ok,
        token_selection_errors=token_selection_errors,
        frame_grouping_ok=frame_grouping_ok,
        frame_grouping_errors=frame_grouping_errors,
        frame_token_counts=frame_token_counts,
    )


class MessageMemoryCarrierAdapter(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        hidden_size: int,
        d_mem: int,
        inject_layers: Sequence[int],
        gamma_init: float,
        message_mode: str,
        readout_mode: str = SIGMOID_GATE_READOUT,
    ) -> None:
        super().__init__()
        self.variant = str(variant)
        self.hidden_size = int(hidden_size)
        self.d_mem = int(d_mem)
        self.inject_layers = [int(layer) for layer in inject_layers]
        self.layer_to_pos = {int(layer): pos for pos, layer in enumerate(self.inject_layers)}
        self.message_mode = str(message_mode)
        self.readout_mode = str(readout_mode)
        if self.readout_mode not in READOUT_MODES:
            raise ValueError(f"Unsupported readout_mode={self.readout_mode!r}; expected one of {sorted(READOUT_MODES)}")
        self.enabled = True

        n_layers = len(self.inject_layers)
        self.message_norm = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(n_layers)])
        self.carrier_norm = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(n_layers)])
        self.memory_norm = nn.ModuleList([nn.LayerNorm(self.d_mem) for _ in range(n_layers)])
        self.message_to_memory = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.d_mem, bias=False) for _ in range(n_layers)]
        )
        self.proxy_to_memory = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.d_mem, bias=False) for _ in range(n_layers)]
        )
        self.w_q = nn.ModuleList([nn.Linear(self.hidden_size, self.d_mem, bias=False) for _ in range(n_layers)])
        self.w_k = nn.ModuleList([nn.Linear(self.d_mem, self.d_mem, bias=False) for _ in range(n_layers)])
        self.w_v = nn.ModuleList([nn.Linear(self.d_mem, self.d_mem, bias=False) for _ in range(n_layers)])
        self.w_o = nn.ModuleList([nn.Linear(self.d_mem, self.hidden_size, bias=False) for _ in range(n_layers)])
        self.gate_bias = nn.Parameter(torch.full((n_layers,), -2.0, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.full((n_layers,), float(gamma_init), dtype=torch.float32))

        for layer in range(n_layers):
            nn.init.xavier_uniform_(self.message_to_memory[layer].weight, gain=0.5)
            nn.init.xavier_uniform_(self.proxy_to_memory[layer].weight, gain=0.5)
            nn.init.xavier_uniform_(self.w_q[layer].weight, gain=0.5)
            nn.init.xavier_uniform_(self.w_k[layer].weight, gain=0.5)
            nn.init.xavier_uniform_(self.w_v[layer].weight, gain=0.5)
            nn.init.normal_(self.w_o[layer].weight, mean=0.0, std=0.002)

        self._target_positions: Optional[List[List[int]]] = None
        self._message_target_positions: Optional[List[List[int]]] = None
        self._query_positions: Optional[List[List[int]]] = None
        self._inject_positions: Optional[List[List[int]]] = None
        self._frame_groups: Optional[List[List[List[int]]]] = None
        self._handles: List[Any] = []
        self._memory_state: Optional[torch.Tensor] = None
        self._loss_update_energies: List[torch.Tensor] = []
        self._last_stats: Dict[str, Dict[str, Any]] = {}
        self.hook_fire_counts: Dict[int, int] = defaultdict(int)
        self.message_mode_counts: Dict[str, int] = defaultdict(int)
        self.exact_failure_counts: Dict[str, int] = defaultdict(int)
        self.exact_failure_examples: List[str] = []

    def set_context(
        self,
        *,
        target_positions: Optional[Sequence[Sequence[int]]] = None,
        message_target_positions: Optional[Sequence[Sequence[int]]] = None,
        query_positions: Optional[Sequence[Sequence[int]]] = None,
        inject_positions: Optional[Sequence[Sequence[int]]] = None,
        frame_groups: Sequence[Sequence[Sequence[int]]],
    ) -> None:
        if target_positions is None and (
            message_target_positions is None or query_positions is None or inject_positions is None
        ):
            raise ValueError("target_positions or all split position groups must be provided")
        legacy_positions = target_positions if target_positions is not None else inject_positions
        assert legacy_positions is not None
        message_positions = message_target_positions if message_target_positions is not None else legacy_positions
        query_pos = query_positions if query_positions is not None else legacy_positions
        inject_pos = inject_positions if inject_positions is not None else legacy_positions
        self._target_positions = [[int(pos) for pos in positions] for positions in legacy_positions]
        self._message_target_positions = [[int(pos) for pos in positions] for positions in message_positions]
        self._query_positions = [[int(pos) for pos in positions] for positions in query_pos]
        self._inject_positions = [[int(pos) for pos in positions] for positions in inject_pos]
        self._frame_groups = [
            [[int(pos) for pos in group] for group in sample_groups]
            for sample_groups in frame_groups
        ]
        self._memory_state = None
        self._loss_update_energies = []
        self._last_stats = {
            "gate_values_by_layer": {},
            "gate_sum_by_layer": {},
            "matrix_scores_by_layer": {},
            "matrix_score_sum_by_layer": {},
            "matrix_score_abs_sum_by_layer": {},
            "matrix_score_mean_by_layer": {},
            "matrix_score_abs_mean_by_layer": {},
            "update_norm_by_layer": {},
            "memory_norm_by_layer": {},
            "message_norm_by_layer": {},
            "raw_message_norm_by_layer": {},
            "message_mode_by_layer": {},
        }

    def clear_context(self) -> None:
        self._target_positions = None
        self._message_target_positions = None
        self._query_positions = None
        self._inject_positions = None
        self._frame_groups = None
        self._memory_state = None
        self._loss_update_energies = []

    def update_energy_for_loss(self, device: torch.device) -> torch.Tensor:
        if not self._loss_update_energies:
            return torch.zeros((), device=device)
        return torch.stack(self._loss_update_energies, dim=0).sum(dim=0).mean()

    @staticmethod
    def _hidden_from_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        if args and torch.is_tensor(args[0]):
            return args[0]
        hidden = kwargs.get("hidden_states")
        return hidden if torch.is_tensor(hidden) else None

    @staticmethod
    def _replace_hidden_in_args(
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        hidden_states: torch.Tensor,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        if args and torch.is_tensor(args[0]):
            return (hidden_states,) + tuple(args[1:]), kwargs
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = hidden_states
        return args, new_kwargs

    def _record_exact_failure(self, reason: str) -> None:
        key = str(reason).split(":", 1)[0][:80]
        self.exact_failure_counts[key] += 1
        if len(self.exact_failure_examples) < 20:
            self.exact_failure_examples.append(str(reason)[:500])

    def _carrier_mean(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self._query_positions is not None
        batch, seq_len, hidden = hidden_states.shape
        rows: List[torch.Tensor] = []
        source = hidden_states.detach().float()
        for batch_idx in range(batch):
            valid = sorted({int(pos) for pos in self._query_positions[batch_idx] if 0 <= int(pos) < seq_len})
            if valid:
                idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
                rows.append(source[batch_idx, idx, :].mean(dim=0))
            else:
                rows.append(source.new_zeros((hidden,)))
        return torch.stack(rows, dim=0)

    def _proxy_messages(self, hidden_states: torch.Tensor, layer_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._frame_groups is not None
        batch, seq_len, hidden = hidden_states.shape
        source = hidden_states.detach().float()
        raw_rows: List[torch.Tensor] = []
        for batch_idx in range(batch):
            sample_rows: List[torch.Tensor] = []
            for frame_idx in range(NUM_FRAMES):
                group = self._frame_groups[batch_idx][frame_idx] if frame_idx < len(self._frame_groups[batch_idx]) else []
                valid = [int(pos) for pos in group if 0 <= int(pos) < seq_len]
                if valid:
                    idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
                    sample_rows.append(source[batch_idx, idx, :].mean(dim=0))
                else:
                    sample_rows.append(source.new_zeros((hidden,)))
            raw_rows.append(torch.stack(sample_rows, dim=0))
        raw = torch.stack(raw_rows, dim=0)
        contrib = self.proxy_to_memory[layer_pos](self.message_norm[layer_pos](raw))
        return contrib, raw

    def _repeat_kv(self, states: torch.Tensor, num_heads: int) -> torch.Tensor:
        if int(states.shape[1]) == int(num_heads):
            return states
        repeats = int(num_heads) // int(states.shape[1])
        return states.repeat_interleave(repeats, dim=1)

    def _exact_messages(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        layer_pos: int,
        kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if apply_multimodal_rotary_pos_emb is None:
            raise RuntimeError("exact unavailable: apply_multimodal_rotary_pos_emb import failed")
        if not hasattr(module, "input_layernorm") or not hasattr(module, "self_attn"):
            raise RuntimeError("exact unavailable: decoder layer does not expose input_layernorm/self_attn")
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            raise RuntimeError("exact unavailable: layer kwargs have no position_embeddings")
        assert self._message_target_positions is not None and self._frame_groups is not None
        attn = module.self_attn
        with torch.no_grad():
            hs = module.input_layernorm(hidden_states.detach())
            batch, seq_len, _hidden = hs.shape
            q = attn.q_proj(hs)
            k = attn.k_proj(hs)
            v = attn.v_proj(hs)
            head_dim = int(attn.head_dim)
            num_heads = int(attn.num_heads)
            q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, -1, head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, -1, head_dim).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.rope_scaling["mrope_section"])
            k = self._repeat_kv(k, num_heads)
            v = self._repeat_kv(v, num_heads)
            attention_mask = kwargs.get("attention_mask")
            scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
            raw_message_rows: List[torch.Tensor] = []
            projected_rows: List[torch.Tensor] = []
            arange = torch.arange(seq_len, device=hidden_states.device)

            for batch_idx in range(batch):
                carrier_positions = [
                    int(pos)
                    for pos in self._message_target_positions[batch_idx]
                    if 0 <= int(pos) < seq_len
                ]
                if not carrier_positions:
                    zero_raw = hidden_states.detach().float().new_zeros((NUM_FRAMES, self.hidden_size))
                    raw_message_rows.append(zero_raw)
                    projected_rows.append(zero_raw)
                    continue
                c_idx = torch.tensor(carrier_positions, device=hidden_states.device, dtype=torch.long)
                scores = torch.einsum(
                    "hcd,hsd->hcs",
                    q[batch_idx, :, c_idx, :].float(),
                    k[batch_idx].float(),
                ) * scaling

                causal_allowed = arange.unsqueeze(0) <= c_idx.unsqueeze(1)
                sliding_window = getattr(attn, "sliding_window", None)
                if sliding_window is not None:
                    causal_allowed &= arange.unsqueeze(0) >= (c_idx.unsqueeze(1) - int(sliding_window))
                scores = scores.masked_fill(~causal_allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
                if torch.is_tensor(attention_mask):
                    mask = attention_mask
                    if mask.dim() == 4:
                        selected_mask = mask[batch_idx : batch_idx + 1, :, c_idx, :].float()
                        scores = scores + selected_mask.squeeze(0)
                    elif mask.dim() == 2:
                        valid = mask[batch_idx].bool()
                        scores = scores.masked_fill(~valid.view(1, 1, -1), torch.finfo(scores.dtype).min)

                probs = torch.softmax(scores, dim=-1)
                sample_raw: List[torch.Tensor] = []
                sample_proj: List[torch.Tensor] = []
                for frame_idx in range(NUM_FRAMES):
                    group = self._frame_groups[batch_idx][frame_idx] if frame_idx < len(self._frame_groups[batch_idx]) else []
                    valid = [int(pos) for pos in group if 0 <= int(pos) < seq_len]
                    if not valid:
                        sample_raw.append(hidden_states.detach().float().new_zeros((self.hidden_size,)))
                        sample_proj.append(hidden_states.detach().float().new_zeros((self.hidden_size,)))
                        continue
                    f_idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
                    contrib = torch.einsum("hcf,hfd->hcd", probs[:, :, f_idx], v[batch_idx, :, f_idx, :].float())
                    contrib_flat = contrib.permute(1, 0, 2).reshape(len(carrier_positions), num_heads * head_dim)
                    projected = attn.o_proj(contrib_flat.to(dtype=hs.dtype)).detach().float()
                    pooled = projected.mean(dim=0)
                    sample_raw.append(pooled)
                    sample_proj.append(pooled)
                raw_message_rows.append(torch.stack(sample_raw, dim=0))
                projected_rows.append(torch.stack(sample_proj, dim=0))

        raw_messages = torch.stack(raw_message_rows, dim=0).to(hidden_states.device)
        contrib = self.message_to_memory[layer_pos](self.message_norm[layer_pos](raw_messages))
        return contrib, raw_messages

    def _message_contribution(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        layer_idx: int,
        layer_pos: int,
        kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.message_mode == "proxy":
            contrib, raw = self._proxy_messages(hidden_states, layer_pos)
            return contrib, raw, "proxy"
        try:
            contrib, raw = self._exact_messages(module, hidden_states, layer_pos, kwargs)
            return contrib, raw, "exact"
        except Exception as exc:
            self._record_exact_failure(f"layer {layer_idx}: {type(exc).__name__}: {exc}")
            if self.message_mode == "exact":
                raise
            contrib, raw = self._proxy_messages(hidden_states, layer_pos)
            return contrib, raw, "proxy"

    def inject_before_layer(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self._message_target_positions is None
            or self._query_positions is None
            or self._inject_positions is None
            or self._frame_groups is None
        ):
            return hidden_states
        if int(layer_idx) not in self.layer_to_pos:
            return hidden_states

        layer_pos = self.layer_to_pos[int(layer_idx)]
        self.hook_fire_counts[int(layer_idx)] += 1
        current, raw_messages, mode = self._message_contribution(module, hidden_states, int(layer_idx), layer_pos, kwargs)
        self.message_mode_counts[f"{int(layer_idx)}:{mode}"] += int(hidden_states.shape[0])

        if self.variant == CUMULATIVE:
            if self._memory_state is None:
                self._memory_state = torch.zeros(
                    int(hidden_states.shape[0]),
                    NUM_FRAMES,
                    self.d_mem,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            memory = self.memory_norm[layer_pos](self._memory_state + current.float())
            self._memory_state = memory
        else:
            memory = current.float()

        carrier = self._carrier_mean(hidden_states)
        q = self.w_q[layer_pos](self.carrier_norm[layer_pos](carrier)).float()
        k = self.w_k[layer_pos](memory).float()
        v = self.w_v[layer_pos](memory).float()
        gates: Optional[torch.Tensor] = None
        matrix_scores: Optional[torch.Tensor] = None
        if self.readout_mode == SIGMOID_GATE_READOUT:
            gates = torch.sigmoid(
                torch.einsum("bd,bfd->bf", q, k) / math.sqrt(float(self.d_mem))
                + self.gate_bias[layer_pos].float()
            )
            retrieved = torch.sum(gates.unsqueeze(-1) * v, dim=1)
        elif self.readout_mode == RAW_MATRIX_READOUT:
            matrix_scores = torch.einsum("bfd,bd->bf", k, q)
            retrieved = torch.einsum("bf,bfd->bd", matrix_scores, v)
        else:
            raise RuntimeError(f"Unsupported readout_mode={self.readout_mode!r}")
        delta = self.w_o[layer_pos](retrieved).float()
        actual_update = self.gamma[layer_pos].float() * delta

        out = hidden_states.clone()
        batch, seq_len, _hidden = hidden_states.shape
        valid_mask_values: List[float] = []
        for batch_idx, positions in enumerate(self._inject_positions):
            valid = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len})
            if not valid:
                valid_mask_values.append(0.0)
                continue
            pos_idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
            update = actual_update[batch_idx].to(dtype=hidden_states.dtype).unsqueeze(0).expand(len(valid), -1)
            out[batch_idx, pos_idx, :] = out[batch_idx, pos_idx, :] + update
            valid_mask_values.append(1.0)

        valid_mask = torch.tensor(valid_mask_values, device=hidden_states.device, dtype=actual_update.dtype)
        energy = actual_update.float().pow(2).sum(dim=-1) * valid_mask
        self._loss_update_energies.append(energy)

        layer_key = str(int(layer_idx))
        if gates is None:
            self._last_stats["gate_values_by_layer"][layer_key] = [[] for _ in range(batch)]
            self._last_stats["gate_sum_by_layer"][layer_key] = [0.0 for _ in range(batch)]
        else:
            gate_values = gates.detach().float()
            self._last_stats["gate_values_by_layer"][layer_key] = gate_values.cpu().tolist()
            self._last_stats["gate_sum_by_layer"][layer_key] = gate_values.sum(dim=-1).cpu().tolist()
        if matrix_scores is None:
            self._last_stats["matrix_scores_by_layer"][layer_key] = [[] for _ in range(batch)]
            self._last_stats["matrix_score_sum_by_layer"][layer_key] = [0.0 for _ in range(batch)]
            self._last_stats["matrix_score_abs_sum_by_layer"][layer_key] = [0.0 for _ in range(batch)]
            self._last_stats["matrix_score_mean_by_layer"][layer_key] = [0.0 for _ in range(batch)]
            self._last_stats["matrix_score_abs_mean_by_layer"][layer_key] = [0.0 for _ in range(batch)]
        else:
            score_values = matrix_scores.detach().float()
            self._last_stats["matrix_scores_by_layer"][layer_key] = score_values.cpu().tolist()
            self._last_stats["matrix_score_sum_by_layer"][layer_key] = score_values.sum(dim=-1).cpu().tolist()
            self._last_stats["matrix_score_abs_sum_by_layer"][layer_key] = score_values.abs().sum(dim=-1).cpu().tolist()
            self._last_stats["matrix_score_mean_by_layer"][layer_key] = score_values.mean(dim=-1).cpu().tolist()
            self._last_stats["matrix_score_abs_mean_by_layer"][layer_key] = score_values.abs().mean(dim=-1).cpu().tolist()
        self._last_stats["update_norm_by_layer"][layer_key] = actual_update.detach().float().norm(dim=-1).cpu().tolist()
        self._last_stats["memory_norm_by_layer"][layer_key] = memory.detach().float().norm(dim=-1).cpu().tolist()
        self._last_stats["message_norm_by_layer"][layer_key] = current.detach().float().norm(dim=-1).cpu().tolist()
        self._last_stats["raw_message_norm_by_layer"][layer_key] = raw_messages.detach().float().norm(dim=-1).cpu().tolist()
        self._last_stats["message_mode_by_layer"][layer_key] = [mode for _ in range(batch)]
        return out

    def register_hooks(self, model: Any) -> None:
        self.remove_hooks()
        layers = prev.get_layers(model)
        for layer_idx in self.inject_layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"inject_layer={layer_idx} outside [0, {len(layers) - 1}]")

            def hook(module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any], *, layer: int = int(layer_idx)) -> Any:
                hidden = self._hidden_from_args(args, kwargs)
                if hidden is None:
                    return args, kwargs
                new_hidden = self.inject_before_layer(module, hidden, layer, kwargs)
                return self._replace_hidden_in_args(args, kwargs, new_hidden)

            self._handles.append(layers[int(layer_idx)].register_forward_pre_hook(hook, with_kwargs=True))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, by_layer in self._last_stats.items():
            row_payload: Dict[str, Any] = {}
            for layer, values in by_layer.items():
                if isinstance(values, list) and row < len(values):
                    row_payload[str(layer)] = values[row]
                else:
                    row_payload[str(layer)] = values
            out[key] = row_payload
        return out


def make_adapter(args: argparse.Namespace, variant: str, hidden_size: int, inject_layers: Sequence[int]) -> MessageMemoryCarrierAdapter:
    return MessageMemoryCarrierAdapter(
        variant=str(variant),
        hidden_size=int(hidden_size),
        d_mem=int(args.d_mem),
        inject_layers=inject_layers,
        gamma_init=float(args.gamma_init),
        message_mode=str(args.message_mode),
        readout_mode=str(getattr(args, "readout_mode", SIGMOID_GATE_READOUT)),
    )


def verify_trainable_parameters(model: Any, adapter: nn.Module) -> None:
    model_trainable = sum(int(param.requires_grad) for param in model.parameters())
    if model_trainable:
        raise RuntimeError(f"Qwen is not frozen: {model_trainable} model parameter tensors still require grad")
    adapter_params = [param for param in adapter.parameters() if param.requires_grad]
    count = sum(int(param.numel()) for param in adapter_params)
    if not adapter_params or count <= 0:
        raise RuntimeError("No trainable memory adapter parameters")
    print(f"Verified frozen Qwen; trainable memory adapter tensors={len(adapter_params)} params={count}")


def select_gold_logits_and_margins(count_logits: torch.Tensor, gold_offsets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return trans.select_gold_logits_and_margins(count_logits, gold_offsets)


def margin_loss(count_logits: torch.Tensor, gold_offsets: torch.Tensor, margin_target: float) -> torch.Tensor:
    return trans.margin_loss(count_logits, gold_offsets, float(margin_target))


def chunked(values: Sequence[int], chunk_size: int) -> Iterable[List[int]]:
    yield from prev.chunked(values, int(chunk_size))


def adapter_parameter_norm(adapter: nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for param in adapter.parameters():
            total += float(param.detach().float().pow(2).sum().cpu().item())
    return math.sqrt(max(0.0, total))


def first_backward_diagnostics(model: Any, adapter: nn.Module) -> Dict[str, Any]:
    adapter_nonzero = 0
    adapter_with_grad = 0
    adapter_grad_norm_sq = 0.0
    for param in adapter.parameters():
        if param.grad is None:
            continue
        adapter_with_grad += 1
        grad_norm = float(param.grad.detach().float().norm().cpu().item())
        if grad_norm > 0:
            adapter_nonzero += 1
        adapter_grad_norm_sq += grad_norm * grad_norm
    model_grad_tensors = sum(1 for param in model.parameters() if param.grad is not None)
    return {
        "adapter_grad_tensors": int(adapter_with_grad),
        "adapter_nonzero_grad_tensors": int(adapter_nonzero),
        "adapter_grad_norm": math.sqrt(max(0.0, adapter_grad_norm_sq)),
        "model_grad_tensors": int(model_grad_tensors),
        "only_adapter_params_updated": int(model_grad_tensors == 0 and adapter_nonzero > 0),
    }


def train_adapter(
    *,
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
    variant: str,
    device: str,
) -> Tuple[MessageMemoryCarrierAdapter, List[Dict[str, Any]], Path, Dict[str, Any]]:
    adapter = make_adapter(args, variant=variant, hidden_size=int(hidden_size), inject_layers=inject_layers).to(device)
    verify_trainable_parameters(model, adapter)
    optimizer = torch.optim.AdamW(
        [param for param in adapter.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"{safe_name(variant)}_best.pt"
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_acc = -math.inf
    best_val_ce = math.inf
    count_min = min(int(x) for x in count_values)
    backward_diag: Dict[str, Any] = {}

    for epoch in range(1, int(args.epochs) + 1):
        adapter.train()
        adapter.enabled = True
        rng = random.Random(int(args.seed) + epoch * 1777)
        shuffled = [int(idx) for idx in train_indices]
        rng.shuffle(shuffled)
        optimizer.zero_grad(set_to_none=True)
        train_ce_total = 0.0
        train_loss_total = 0.0
        train_energy_total = 0.0
        train_steps = 0
        backward_steps = 0
        skipped = 0
        try:
            adapter.register_hooks(model)
            for step, batch_indices in enumerate(chunked(shuffled, int(args.batch_size)), start=1):
                batch_records = [records[int(idx)] for idx in batch_indices]
                batch = prepare_memory_batch(
                    records=batch_records,
                    sample_indices=batch_indices,
                    processor=processor,
                    device=device,
                    token_group=args.token_group,
                )
                if not any(batch.target_positions) or not any(batch.frame_grouping_ok):
                    skipped += 1
                adapter.set_context(
                    target_positions=batch.target_positions,
                    message_target_positions=batch.message_target_positions,
                    query_positions=batch.query_positions,
                    inject_positions=batch.inject_positions,
                    frame_groups=batch.frame_groups,
                )
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - int(count_min)
                ce = F.cross_entropy(count_logits, gold_offsets)
                m_loss = margin_loss(count_logits, gold_offsets, float(args.margin_target))
                update_energy = adapter.update_energy_for_loss(count_logits.device)
                loss = ce + float(args.lambda_margin) * m_loss + float(args.lambda_update_energy) * update_energy
                torch.autograd.backward(loss / max(1, int(args.grad_accum)))
                if not backward_diag:
                    backward_diag = first_backward_diagnostics(model, adapter)
                    print(f"  first backward diagnostics: {json_compact(backward_diag)}")
                train_ce_total += float(ce.detach().cpu().item())
                train_loss_total += float(loss.detach().cpu().item())
                train_energy_total += float(update_energy.detach().cpu().item())
                train_steps += 1
                backward_steps += 1
                adapter.clear_context()
                if backward_steps % max(1, int(args.grad_accum)) == 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 25 == 0:
                    print(
                        f"  {variant} epoch={epoch} step={step} "
                        f"train_ce={train_ce_total / max(1, train_steps):.4f} "
                        f"energy={train_energy_total / max(1, train_steps):.6f}"
                    )
            if backward_steps and backward_steps % max(1, int(args.grad_accum)) != 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        finally:
            adapter.remove_hooks()

        val_eval = evaluate_split(
            split_name="val",
            method_label=f"{variant}__val",
            model=model,
            processor=processor,
            adapter=adapter,
            records=records,
            indices=val_indices,
            count_token_ids=count_token_ids,
            count_values=count_values,
            device=device,
            batch_size=int(args.batch_size),
            max_eval_samples=int(args.max_eval_samples),
            token_group=args.token_group,
        )
        row = {
            "method": str(variant),
            "readout_mode": str(getattr(args, "readout_mode", SIGMOID_GATE_READOUT)),
            "epoch": int(epoch),
            "train_ce": train_ce_total / max(1, train_steps),
            "train_loss": train_loss_total / max(1, train_steps),
            "train_update_energy": train_energy_total / max(1, train_steps),
            "train_steps": int(train_steps),
            "skipped_batches_with_missing_localization": int(skipped),
            "val_ce": float(val_eval["ce"]),
            "val_accuracy": float(val_eval["accuracy"]),
            "val_mean_update_energy": finite_mean(val_eval["update_energy_total_by_idx"].values(), default=0.0),
            "adapter_parameter_norm": adapter_parameter_norm(adapter),
            "gamma_json": json_compact([float(x) for x in adapter.gamma.detach().float().cpu().tolist()]),
        }
        history.append(row)
        print(
            f"  {variant} epoch={epoch} train_ce={row['train_ce']:.4f} "
            f"val_ce={row['val_ce']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_energy={row['val_mean_update_energy']:.6f}"
        )
        improved = row["val_accuracy"] > best_val_acc + 1e-9 or (
            abs(row["val_accuracy"] - best_val_acc) <= 1e-9 and row["val_ce"] < best_val_ce
        )
        if improved:
            best_val_acc = float(row["val_accuracy"])
            best_val_ce = float(row["val_ce"])
            best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
            torch.save(
                {
                    "adapter_state_dict": best_state,
                    "history": history,
                    "backward_diagnostics": backward_diag,
                    "hidden_size": int(hidden_size),
                    "d_mem": int(args.d_mem),
                    "inject_layers": [int(x) for x in inject_layers],
                    "variant": str(variant),
                    "token_group": canonical_token_group(args.token_group),
                    "message_mode": str(args.message_mode),
                    "readout_mode": str(getattr(args, "readout_mode", SIGMOID_GATE_READOUT)),
                },
                checkpoint_path,
            )
    if best_state is not None:
        adapter.load_state_dict(best_state)
    return adapter.cpu(), history, checkpoint_path, backward_diag


def blank_row_diagnostics(layers: Sequence[int]) -> Dict[str, Any]:
    layer_map = {str(int(layer)): [] for layer in layers}
    return {
        "gate_values_by_layer": dict(layer_map),
        "gate_sum_by_layer": {str(int(layer)): 0.0 for layer in layers},
        "matrix_scores_by_layer": dict(layer_map),
        "matrix_score_sum_by_layer": {str(int(layer)): 0.0 for layer in layers},
        "matrix_score_abs_sum_by_layer": {str(int(layer)): 0.0 for layer in layers},
        "matrix_score_mean_by_layer": {str(int(layer)): 0.0 for layer in layers},
        "matrix_score_abs_mean_by_layer": {str(int(layer)): 0.0 for layer in layers},
        "update_norm_by_layer": {str(int(layer)): 0.0 for layer in layers},
        "memory_norm_by_layer": dict(layer_map),
        "message_norm_by_layer": dict(layer_map),
        "raw_message_norm_by_layer": dict(layer_map),
        "message_mode_by_layer": {str(int(layer)): "none" for layer in layers},
    }


def mean_layer_frame_value(layer_json: Dict[str, Any]) -> float:
    values: List[float] = []
    for payload in layer_json.values():
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, list):
                    values.extend(float(x) for x in item if finite_float(x) is not None)
                elif finite_float(item) is not None:
                    values.append(float(item))
        elif finite_float(payload) is not None:
            values.append(float(payload))
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def evaluate_split(
    *,
    split_name: str,
    method_label: str,
    model: Any,
    processor: Any,
    adapter: Optional[MessageMemoryCarrierAdapter],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    count_token_ids: Dict[int, int],
    count_values: Sequence[int],
    device: str,
    batch_size: int,
    max_eval_samples: int,
    token_group: str = ROOM_CHAR_TOKEN_GROUP,
) -> Dict[str, Any]:
    eval_indices = [int(idx) for idx in indices]
    if int(max_eval_samples) > 0:
        eval_indices = eval_indices[: int(max_eval_samples)]
    if adapter is not None:
        adapter.to(device)
        adapter.eval()
        adapter.enabled = True
        adapter.register_hooks(model)

    pred_by_idx: Dict[int, int] = {}
    logits_by_idx: Dict[int, List[float]] = {}
    gold_logit_by_idx: Dict[int, float] = {}
    pred_logit_by_idx: Dict[int, float] = {}
    margin_by_idx: Dict[int, float] = {}
    ce_by_idx: Dict[int, float] = {}
    update_energy_total_by_idx: Dict[int, float] = {}
    gate_sum_by_idx: Dict[int, float] = {}
    matrix_score_sum_by_idx: Dict[int, float] = {}
    matrix_score_abs_sum_by_idx: Dict[int, float] = {}
    matrix_score_mean_by_idx: Dict[int, float] = {}
    matrix_score_abs_mean_by_idx: Dict[int, float] = {}
    diagnostics_by_idx: Dict[int, Dict[str, Any]] = {}
    token_selection_ok_by_idx: Dict[int, bool] = {}
    token_selection_error_by_idx: Dict[int, str] = {}
    frame_grouping_ok_by_idx: Dict[int, bool] = {}
    frame_grouping_error_by_idx: Dict[int, str] = {}
    frame_token_counts_by_idx: Dict[int, List[int]] = {}
    room_positions_by_idx: Dict[int, List[int]] = {}
    evidence_frame_mask_by_idx: Dict[int, List[int]] = {}
    ce_total = 0.0
    n = 0
    count_min = min(int(x) for x in count_values)
    layers = adapter.inject_layers if adapter is not None else []

    try:
        for batch_num, batch_indices in enumerate(chunked(eval_indices, int(batch_size)), start=1):
            batch_records = [records[int(idx)] for idx in batch_indices]
            batch = prepare_memory_batch(
                records=batch_records,
                sample_indices=batch_indices,
                processor=processor,
                device=device,
                token_group=token_group,
            )
            if adapter is not None:
                adapter.set_context(
                    target_positions=batch.target_positions,
                    message_target_positions=batch.message_target_positions,
                    query_positions=batch.query_positions,
                    inject_positions=batch.inject_positions,
                    frame_groups=batch.frame_groups,
                )
            outputs = model(**batch.inputs, use_cache=False)
            count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
            gold_offsets = batch.gold_counts.long() - int(count_min)
            ce_vec = F.cross_entropy(count_logits, gold_offsets, reduction="none")
            ce_total += float(ce_vec.sum().detach().cpu().item())
            n += int(batch.gold_counts.numel())
            pred_offsets = count_logits.argmax(dim=-1)
            gold_logits, _best_wrong, margins = select_gold_logits_and_margins(count_logits, gold_offsets)
            logits_cpu = count_logits.detach().float().cpu()

            for row, idx in enumerate(batch_indices):
                idx = int(idx)
                record = records[idx]
                pred = int(pred_offsets[row].detach().cpu().item()) + int(count_min)
                logits = [float(v) for v in logits_cpu[row].tolist()]
                diag = adapter.stats_for_row(row) if adapter is not None else blank_row_diagnostics(layers)
                gate_sum_layer = diag.get("gate_sum_by_layer", {})
                matrix_score_sum_layer = diag.get("matrix_score_sum_by_layer", {})
                matrix_score_abs_sum_layer = diag.get("matrix_score_abs_sum_by_layer", {})
                matrix_score_mean_layer = diag.get("matrix_score_mean_by_layer", {})
                matrix_score_abs_mean_layer = diag.get("matrix_score_abs_mean_by_layer", {})
                update_norm_layer = diag.get("update_norm_by_layer", {})
                update_energy = sum(float(v) ** 2 for v in update_norm_layer.values() if finite_float(v) is not None)
                mean_gate_sum = finite_mean(gate_sum_layer.values(), default=0.0)

                pred_by_idx[idx] = pred
                logits_by_idx[idx] = logits
                gold_logit_by_idx[idx] = float(gold_logits[row].detach().cpu().item())
                pred_logit_by_idx[idx] = logits[pred - int(count_min)] if 0 <= pred - int(count_min) < len(logits) else math.nan
                margin_by_idx[idx] = float(margins[row].detach().cpu().item())
                ce_by_idx[idx] = float(ce_vec[row].detach().cpu().item())
                update_energy_total_by_idx[idx] = float(update_energy)
                gate_sum_by_idx[idx] = float(mean_gate_sum)
                matrix_score_sum_by_idx[idx] = finite_mean(matrix_score_sum_layer.values(), default=0.0)
                matrix_score_abs_sum_by_idx[idx] = finite_mean(matrix_score_abs_sum_layer.values(), default=0.0)
                matrix_score_mean_by_idx[idx] = finite_mean(matrix_score_mean_layer.values(), default=0.0)
                matrix_score_abs_mean_by_idx[idx] = finite_mean(matrix_score_abs_mean_layer.values(), default=0.0)
                diagnostics_by_idx[idx] = diag
                token_selection_ok_by_idx[idx] = bool(batch.token_selection_ok[row])
                token_selection_error_by_idx[idx] = str(batch.token_selection_errors[row])
                frame_grouping_ok_by_idx[idx] = bool(batch.frame_grouping_ok[row])
                frame_grouping_error_by_idx[idx] = str(batch.frame_grouping_errors[row])
                frame_token_counts_by_idx[idx] = [int(x) for x in batch.frame_token_counts[row]]
                room_positions_by_idx[idx] = [int(pos) for pos in batch.target_positions[row]]
                evidence_frame_mask_by_idx[idx] = evidence_frame_mask(record, NUM_FRAMES)
            if adapter is not None:
                adapter.clear_context()
            if batch_num == 1 or batch_num % 25 == 0:
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
        "update_energy_total_by_idx": update_energy_total_by_idx,
        "gate_sum_by_idx": gate_sum_by_idx,
        "matrix_score_sum_by_idx": matrix_score_sum_by_idx,
        "matrix_score_abs_sum_by_idx": matrix_score_abs_sum_by_idx,
        "matrix_score_mean_by_idx": matrix_score_mean_by_idx,
        "matrix_score_abs_mean_by_idx": matrix_score_abs_mean_by_idx,
        "diagnostics_by_idx": diagnostics_by_idx,
        "token_selection_ok_by_idx": token_selection_ok_by_idx,
        "token_selection_error_by_idx": token_selection_error_by_idx,
        "frame_grouping_ok_by_idx": frame_grouping_ok_by_idx,
        "frame_grouping_error_by_idx": frame_grouping_error_by_idx,
        "frame_token_counts_by_idx": frame_token_counts_by_idx,
        "room_positions_by_idx": room_positions_by_idx,
        "evidence_frame_mask_by_idx": evidence_frame_mask_by_idx,
    }


def make_metrics_rows(
    *,
    eval_payload: Dict[str, Any],
    records: Sequence[prev.SampleRecord],
    count_values: Sequence[int],
    variant: str,
    split_name: str,
    layer_start: int,
    layer_end: int,
    d_mem: int,
    lr: float,
    epoch: int,
    readout_mode: str,
    token_group: str = ROOM_CHAR_TOKEN_GROUP,
) -> List[Dict[str, Any]]:
    count_min = min(int(x) for x in count_values)
    resolved_token_group = canonical_token_group(token_group)
    rows: List[Dict[str, Any]] = []
    for idx in eval_payload["indices"]:
        idx = int(idx)
        if idx not in eval_payload["pred_by_idx"]:
            continue
        record = records[idx]
        pred = int(eval_payload["pred_by_idx"][idx])
        gold = int(record.gold_count)
        logits = eval_payload["logits_by_idx"].get(idx, [])
        diag = eval_payload["diagnostics_by_idx"].get(idx, {})
        message_norms = diag.get("message_norm_by_layer", {})
        memory_norms = diag.get("memory_norm_by_layer", {})
        update_norms = diag.get("update_norm_by_layer", {})
        matrix_scores = diag.get("matrix_scores_by_layer", {})
        matrix_score_sums = diag.get("matrix_score_sum_by_layer", {})
        matrix_score_abs_sums = diag.get("matrix_score_abs_sum_by_layer", {})
        matrix_score_means = diag.get("matrix_score_mean_by_layer", {})
        matrix_score_abs_means = diag.get("matrix_score_abs_mean_by_layer", {})
        row = {
            "sample_id": record.sample_id,
            "sample_index": int(idx),
            "split": str(split_name),
            "readout_mode": str(readout_mode),
            "seq_len": NUM_FRAMES,
            "evidence_count": int(record.evidence_count),
            "gold_answer": int(gold),
            "pred_answer": int(pred),
            "correct": int(pred == gold),
            "margin": eval_payload["margin_by_idx"].get(idx, math.nan),
            "gold_logit": eval_payload["gold_logit_by_idx"].get(idx, math.nan),
            "pred_logit": eval_payload["pred_logit_by_idx"].get(idx, math.nan),
            "ce": eval_payload["ce_by_idx"].get(idx, math.nan),
            "update_energy_total": eval_payload["update_energy_total_by_idx"].get(idx, 0.0),
            "approx_update_l2": math.sqrt(max(0.0, float(eval_payload["update_energy_total_by_idx"].get(idx, 0.0)))),
            "mean_gate_sum": eval_payload["gate_sum_by_idx"].get(idx, 0.0),
            "mean_matrix_score_sum": eval_payload["matrix_score_sum_by_idx"].get(idx, 0.0),
            "mean_matrix_score_abs_sum": eval_payload["matrix_score_abs_sum_by_idx"].get(idx, 0.0),
            "mean_matrix_score_mean": eval_payload["matrix_score_mean_by_idx"].get(idx, 0.0),
            "mean_matrix_score_abs_mean": eval_payload["matrix_score_abs_mean_by_idx"].get(idx, 0.0),
            "gate_sum_by_layer_json": json_compact(diag.get("gate_sum_by_layer", {})),
            "gate_values_by_layer_json": json_compact(diag.get("gate_values_by_layer", {})),
            "matrix_scores_by_layer_json": json_compact(matrix_scores),
            "matrix_score_sum_by_layer_json": json_compact(matrix_score_sums),
            "matrix_score_abs_sum_by_layer_json": json_compact(matrix_score_abs_sums),
            "matrix_score_mean_by_layer_json": json_compact(matrix_score_means),
            "matrix_score_abs_mean_by_layer_json": json_compact(matrix_score_abs_means),
            "update_norm_by_layer_json": json_compact(update_norms),
            "memory_norm_by_layer_json": json_compact(memory_norms),
            "message_norm_by_layer_json": json_compact(message_norms),
            "raw_message_norm_by_layer_json": json_compact(diag.get("raw_message_norm_by_layer", {})),
            "message_mode_by_layer_json": json_compact(diag.get("message_mode_by_layer", {})),
            "target_positions_json": json_compact(eval_payload["room_positions_by_idx"].get(idx, [])),
            "room_char_positions_json": json_compact(eval_payload["room_positions_by_idx"].get(idx, []))
            if resolved_token_group == ROOM_CHAR_TOKEN_GROUP
            else "",
            "frame_token_counts_json": json_compact(eval_payload["frame_token_counts_by_idx"].get(idx, [])),
            "evidence_frame_mask_json": json_compact(eval_payload["evidence_frame_mask_by_idx"].get(idx, [])),
            "token_selection_ok": int(bool(eval_payload["token_selection_ok_by_idx"].get(idx, False))),
            "token_selection_error": eval_payload["token_selection_error_by_idx"].get(idx, ""),
            "frame_grouping_ok": int(bool(eval_payload["frame_grouping_ok_by_idx"].get(idx, False))),
            "frame_grouping_error": eval_payload["frame_grouping_error_by_idx"].get(idx, ""),
            "method": str(variant),
            "memory_variant": str(variant),
            "token_group": resolved_token_group,
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "layer_window": f"{int(layer_start)}-{int(layer_end)}",
            "d_mem": int(d_mem),
            "lr": float(lr),
            "epoch": int(epoch),
            "candidate_logits_json": json_compact(logits),
            "mean_message_norm": mean_layer_frame_value(message_norms),
            "mean_memory_norm": mean_layer_frame_value(memory_norms),
            "mean_update_norm": finite_mean(update_norms.values(), default=0.0),
        }
        gold_offset = int(gold) - int(count_min)
        pred_offset = int(pred) - int(count_min)
        row["gold_offset"] = gold_offset
        row["pred_offset"] = pred_offset
        rows.append(row)
    return rows


def prediction_hist(rows: Sequence[Dict[str, Any]], count_values: Sequence[int]) -> Dict[str, int]:
    hist = {str(int(count)): 0 for count in count_values}
    for row in rows:
        pred = row.get("pred_answer")
        if finite_float(pred) is not None:
            key = str(int(float(pred)))
            hist[key] = hist.get(key, 0) + 1
    return hist


def summarize_rows(
    metrics_rows: Sequence[Dict[str, Any]],
    *,
    variant: str,
    count_values: Sequence[int],
    readout_mode: str,
    train_eval: Optional[Dict[str, Any]] = None,
    val_eval: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    correct = [int(row["correct"]) for row in metrics_rows]
    margins = [float(row["margin"]) for row in metrics_rows if finite_float(row.get("margin")) is not None]
    gold_logits = [float(row["gold_logit"]) for row in metrics_rows if finite_float(row.get("gold_logit")) is not None]
    update_energy = [float(row["update_energy_total"]) for row in metrics_rows if finite_float(row.get("update_energy_total")) is not None]
    update_l2 = [float(row["approx_update_l2"]) for row in metrics_rows if finite_float(row.get("approx_update_l2")) is not None]
    gate_sums = [float(row["mean_gate_sum"]) for row in metrics_rows if finite_float(row.get("mean_gate_sum")) is not None]
    matrix_score_sums = [float(row["mean_matrix_score_sum"]) for row in metrics_rows if finite_float(row.get("mean_matrix_score_sum")) is not None]
    matrix_score_abs_sums = [
        float(row["mean_matrix_score_abs_sum"])
        for row in metrics_rows
        if finite_float(row.get("mean_matrix_score_abs_sum")) is not None
    ]
    matrix_score_means = [
        float(row["mean_matrix_score_mean"])
        for row in metrics_rows
        if finite_float(row.get("mean_matrix_score_mean")) is not None
    ]
    matrix_score_abs_means = [
        float(row["mean_matrix_score_abs_mean"])
        for row in metrics_rows
        if finite_float(row.get("mean_matrix_score_abs_mean")) is not None
    ]
    evidence = [int(row["evidence_count"]) for row in metrics_rows]
    mean_gate_by_count: Dict[str, float] = {}
    per_count_rows: List[Dict[str, Any]] = []
    for count in count_values:
        count_rows = [row for row in metrics_rows if int(row["evidence_count"]) == int(count)]
        preds = [int(row["pred_answer"]) for row in count_rows]
        acc_values = [int(row["correct"]) for row in count_rows]
        count_gate = [float(row["mean_gate_sum"]) for row in count_rows if finite_float(row.get("mean_gate_sum")) is not None]
        count_matrix_sum = [float(row["mean_matrix_score_sum"]) for row in count_rows if finite_float(row.get("mean_matrix_score_sum")) is not None]
        count_matrix_abs_sum = [
            float(row["mean_matrix_score_abs_sum"])
            for row in count_rows
            if finite_float(row.get("mean_matrix_score_abs_sum")) is not None
        ]
        count_matrix_mean = [float(row["mean_matrix_score_mean"]) for row in count_rows if finite_float(row.get("mean_matrix_score_mean")) is not None]
        count_matrix_abs_mean = [
            float(row["mean_matrix_score_abs_mean"])
            for row in count_rows
            if finite_float(row.get("mean_matrix_score_abs_mean")) is not None
        ]
        count_update = [float(row["approx_update_l2"]) for row in count_rows if finite_float(row.get("approx_update_l2")) is not None]
        count_message = [float(row["mean_message_norm"]) for row in count_rows if finite_float(row.get("mean_message_norm")) is not None]
        count_memory = [float(row["mean_memory_norm"]) for row in count_rows if finite_float(row.get("mean_memory_norm")) is not None]
        mean_gate_by_count[str(int(count))] = float(np.mean(count_gate)) if count_gate else math.nan
        per_count_rows.append(
            {
                "method": str(variant),
                "memory_variant": str(variant),
                "readout_mode": str(readout_mode),
                "evidence_count": int(count),
                "n": len(count_rows),
                "accuracy": float(np.mean(acc_values)) if acc_values else math.nan,
                "mean_margin": finite_mean([row.get("margin") for row in count_rows]),
                "mean_predicted_count": float(np.mean(preds)) if preds else math.nan,
                "prediction_histogram": json_compact(prediction_hist(count_rows, count_values)),
                "mean_gate_sum": finite_mean(count_gate, default=0.0),
                "mean_matrix_score_sum": finite_mean(count_matrix_sum, default=0.0),
                "mean_matrix_score_abs_sum": finite_mean(count_matrix_abs_sum, default=0.0),
                "mean_matrix_score_mean": finite_mean(count_matrix_mean, default=0.0),
                "mean_matrix_score_abs_mean": finite_mean(count_matrix_abs_mean, default=0.0),
                "mean_update_norm": finite_mean(count_update, default=0.0),
                "mean_message_norm": finite_mean(count_message, default=0.0),
                "mean_memory_norm": finite_mean(count_memory, default=0.0),
            }
        )
    summary_rows = [
        {
            "method": str(variant),
            "memory_variant": str(variant),
            "readout_mode": str(readout_mode),
            "n": len(metrics_rows),
            "accuracy": float(np.mean(correct)) if correct else math.nan,
            "mean_margin": finite_mean(margins),
            "mean_gold_logit": finite_mean(gold_logits),
            "mean_update_energy": finite_mean(update_energy, default=0.0),
            "approx_total_update_l2": finite_mean(update_l2, default=0.0),
            "mean_gate_sum": finite_mean(gate_sums, default=0.0),
            "mean_matrix_score_sum": finite_mean(matrix_score_sums, default=0.0),
            "mean_matrix_score_abs_sum": finite_mean(matrix_score_abs_sums, default=0.0),
            "mean_matrix_score_mean": finite_mean(matrix_score_means, default=0.0),
            "mean_matrix_score_abs_mean": finite_mean(matrix_score_abs_means, default=0.0),
            "mean_gate_sum_by_evidence_count": json_compact(mean_gate_by_count),
            "corr_gate_sum_evidence_count": correlation(evidence, gate_sums),
            "corr_update_norm_evidence_count": correlation(evidence, update_l2),
            "train_accuracy": "" if train_eval is None else float(train_eval["accuracy"]),
            "val_accuracy": "" if val_eval is None else float(val_eval["accuracy"]),
            "val_ce": "" if val_eval is None else float(val_eval["ce"]),
            "token_selection_failures": sum(1 for row in metrics_rows if int(row.get("token_selection_ok", 0)) == 0),
            "frame_grouping_failures": sum(1 for row in metrics_rows if int(row.get("frame_grouping_ok", 0)) == 0),
        }
    ]
    return summary_rows, per_count_rows


def parse_json_field(row: Dict[str, Any], key: str, default: Any) -> Any:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return json.loads(str(value))
    except Exception:
        return default


def save_line_plot(path: Path, xs: Sequence[int], ys: Sequence[float], ylabel: str, title: str) -> None:
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=1.8)
    plt.xlabel("Evidence count")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(xs)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def make_plots(output_dir: Path, metrics_rows: Sequence[Dict[str, Any]], per_count_rows: Sequence[Dict[str, Any]], count_values: Sequence[int]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    counts = [int(c) for c in count_values]
    by_count = {int(row["evidence_count"]): row for row in per_count_rows}
    readout_modes = {str(row.get("readout_mode", "")).strip() for row in metrics_rows if str(row.get("readout_mode", "")).strip()}
    readout_mode = next(iter(readout_modes)) if len(readout_modes) == 1 else ""
    save_line_plot(
        plots_dir / "accuracy_vs_evidence_count.png",
        counts,
        [float(by_count.get(c, {}).get("accuracy", math.nan)) for c in counts],
        "Accuracy",
        "Accuracy vs Evidence Count",
    )
    save_line_plot(
        plots_dir / "margin_vs_evidence_count.png",
        counts,
        [float(by_count.get(c, {}).get("mean_margin", math.nan)) for c in counts],
        "Mean margin",
        "Margin vs Evidence Count",
    )
    if readout_mode == RAW_MATRIX_READOUT:
        for key, filename, ylabel, title in [
            ("mean_matrix_score_sum", "matrix_score_sum_vs_evidence_count.png", "Mean matrix score sum", "Matrix Score Sum vs Evidence Count"),
            (
                "mean_matrix_score_abs_sum",
                "matrix_score_abs_sum_vs_evidence_count.png",
                "Mean abs matrix score sum",
                "Matrix Abs Score Sum vs Evidence Count",
            ),
            ("mean_matrix_score_mean", "matrix_score_mean_vs_evidence_count.png", "Mean matrix score", "Matrix Score Mean vs Evidence Count"),
            (
                "mean_matrix_score_abs_mean",
                "matrix_score_abs_mean_vs_evidence_count.png",
                "Mean abs matrix score",
                "Matrix Abs Score Mean vs Evidence Count",
            ),
        ]:
            save_line_plot(
                plots_dir / filename,
                counts,
                [float(by_count.get(c, {}).get(key, math.nan)) for c in counts],
                ylabel,
                title,
            )
    else:
        save_line_plot(
            plots_dir / "gate_sum_vs_evidence_count.png",
            counts,
            [float(by_count.get(c, {}).get("mean_gate_sum", math.nan)) for c in counts],
            "Mean gate sum",
            "Gate Sum vs Evidence Count",
        )
    save_line_plot(
        plots_dir / "update_norm_vs_evidence_count.png",
        counts,
        [float(by_count.get(c, {}).get("mean_update_norm", math.nan)) for c in counts],
        "Mean update L2",
        "Update Norm vs Evidence Count",
    )
    save_line_plot(
        plots_dir / "message_norm_vs_evidence_count.png",
        counts,
        [float(by_count.get(c, {}).get("mean_message_norm", math.nan)) for c in counts],
        "Mean message contribution norm",
        "Message Norm vs Evidence Count",
    )
    save_line_plot(
        plots_dir / "memory_norm_vs_evidence_count.png",
        counts,
        [float(by_count.get(c, {}).get("mean_memory_norm", math.nan)) for c in counts],
        "Mean memory norm",
        "Memory Norm vs Evidence Count",
    )

    matrix = np.zeros((len(counts), len(counts)), dtype=float)
    for row in metrics_rows:
        gold = int(row["gold_answer"])
        pred = int(row["pred_answer"])
        if gold in counts and pred in counts:
            matrix[counts.index(gold), counts.index(pred)] += 1.0
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_yticks(np.arange(len(counts)))
    ax.set_xticklabels(counts)
    ax.set_yticklabels(counts)
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("Gold count")
    ax.set_title("Predicted Count Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "predicted_count_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    hist = np.zeros((len(counts), len(counts)), dtype=float)
    for i, gold in enumerate(counts):
        rows = [row for row in metrics_rows if int(row["gold_answer"]) == int(gold)]
        for row in rows:
            pred = int(row["pred_answer"])
            if pred in counts:
                hist[i, counts.index(pred)] += 1
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bottom = np.zeros(len(counts))
    for j, pred in enumerate(counts):
        ax.bar(counts, hist[:, j], bottom=bottom, label=str(pred))
        bottom += hist[:, j]
    ax.set_xlabel("Gold count")
    ax.set_ylabel("Samples")
    ax.set_title("Prediction Histogram by Gold Count")
    ax.legend(title="Pred", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "prediction_hist_by_gold_count.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if readout_mode != RAW_MATRIX_READOUT:
        xs = [int(row["evidence_count"]) for row in metrics_rows]
        ys = [float(row["mean_gate_sum"]) for row in metrics_rows]
        plt.figure(figsize=(6.2, 4.5))
        plt.scatter(xs, ys, alpha=0.65)
        plt.xlabel("Evidence count")
        plt.ylabel("Mean gate sum")
        plt.title(f"Gate Sum Correlation r={correlation(xs, ys):.3f}")
        plt.xticks(counts)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / "gate_sum_correlation_scatter.png", dpi=180, bbox_inches="tight")
        plt.close()

        frame_gate_values: Dict[int, List[float]] = defaultdict(list)
        layer_frame_values: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        evidence_gate: Dict[int, List[float]] = defaultdict(list)
        for row in metrics_rows:
            gates_by_layer = parse_json_field(row, "gate_values_by_layer_json", {})
            evidence_mask = parse_json_field(row, "evidence_frame_mask_json", [])
            for layer_text, values in gates_by_layer.items():
                try:
                    layer = int(layer_text)
                except Exception:
                    continue
                if not isinstance(values, list):
                    continue
                for frame_idx, gate in enumerate(values[:NUM_FRAMES]):
                    if finite_float(gate) is None:
                        continue
                    gate_f = float(gate)
                    frame_gate_values[int(frame_idx)].append(gate_f)
                    layer_frame_values[(layer, int(frame_idx))].append(gate_f)
                    if isinstance(evidence_mask, list) and frame_idx < len(evidence_mask):
                        evidence_gate[int(evidence_mask[frame_idx])].append(gate_f)
        plt.figure(figsize=(7.0, 4.3))
        frame_x = list(range(NUM_FRAMES))
        plt.bar(frame_x, [finite_mean(frame_gate_values[i], default=0.0) for i in frame_x])
        plt.xlabel("Frame index")
        plt.ylabel("Mean gate")
        plt.title("Mean Gate by Frame Index")
        plt.tight_layout()
        plt.savefig(plots_dir / "mean_gate_by_frame_index.png", dpi=180, bbox_inches="tight")
        plt.close()

        layers = sorted({layer for layer, _frame in layer_frame_values})
        if layers:
            heat = np.zeros((len(layers), NUM_FRAMES), dtype=float)
            for i, layer in enumerate(layers):
                for frame_idx in range(NUM_FRAMES):
                    heat[i, frame_idx] = finite_mean(layer_frame_values[(layer, frame_idx)], default=0.0)
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            im = ax.imshow(heat, aspect="auto", cmap="magma")
            ax.set_xticks(np.arange(NUM_FRAMES))
            ax.set_yticks(np.arange(len(layers)))
            ax.set_xticklabels(range(NUM_FRAMES))
            ax.set_yticklabels(layers)
            ax.set_xlabel("Frame index")
            ax.set_ylabel("Layer")
            ax.set_title("Mean Gate by Layer and Frame")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout()
            fig.savefig(plots_dir / "mean_gate_by_layer_and_frame_index_heatmap.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
        if evidence_gate.get(0) or evidence_gate.get(1):
            plt.figure(figsize=(5.4, 4.3))
            plt.boxplot([evidence_gate.get(0, []), evidence_gate.get(1, [])], labels=["nonevidence", "evidence"], showfliers=False)
            plt.ylabel("Gate")
            plt.title("Gate Values by Evidence Frame Label")
            plt.tight_layout()
            plt.savefig(plots_dir / "gate_values_evidence_vs_nonevidence_boxplot.png", dpi=180, bbox_inches="tight")
            plt.close()

    accuracy_heat = np.array([[float(by_count.get(c, {}).get("accuracy", math.nan)) for c in counts]])
    fig, ax = plt.subplots(figsize=(7.0, 1.8))
    im = ax.imshow(accuracy_heat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts)
    ax.set_yticks([0])
    ax.set_yticklabels(["accuracy"])
    ax.set_xlabel("Evidence count")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_by_evidence_count_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    margin_heat = np.array([[float(by_count.get(c, {}).get("mean_margin", math.nan)) for c in counts]])
    fig, ax = plt.subplots(figsize=(7.0, 1.8))
    im = ax.imshow(margin_heat, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts)
    ax.set_yticks([0])
    ax.set_yticklabels(["margin"])
    ax.set_xlabel("Evidence count")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "margin_by_evidence_count_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_failure_tables(output_dir: Path, metrics_rows: Sequence[Dict[str, Any]]) -> None:
    wrong = [row for row in metrics_rows if int(row.get("correct", 0)) == 0]
    top_negative = sorted(wrong, key=lambda row: float(row.get("margin", 0.0)))[:50]
    high_gate = sorted(wrong, key=lambda row: float(row.get("mean_gate_sum", 0.0)), reverse=True)[:50]
    low_gate = sorted(wrong, key=lambda row: float(row.get("mean_gate_sum", 0.0)))[:50]
    leading = [
        "sample_id",
        "sample_index",
        "evidence_count",
        "gold_answer",
        "pred_answer",
        "correct",
        "margin",
        "mean_gate_sum",
        "approx_update_l2",
        "token_selection_ok",
        "frame_grouping_ok",
    ]
    write_csv_dynamic(output_dir / "wrong_examples_top_negative_margins.csv", top_negative, leading)
    write_csv_dynamic(output_dir / "wrong_examples_high_gate_sum.csv", high_gate, leading)
    write_csv_dynamic(output_dir / "wrong_examples_low_gate_sum.csv", low_gate, leading)


def collect_sibling_summaries(parent: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not parent.exists():
        return rows
    for path in sorted(parent.rglob("summary.csv")):
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    out = dict(row)
                    out["run_dir"] = os.fspath(path.parent)
                    rows.append(out)
        except Exception:
            continue
    return rows


def method_accuracy(rows: Sequence[Dict[str, Any]], method: str) -> Optional[float]:
    vals = [finite_float(row.get("accuracy")) for row in rows if row.get("method") == method or row.get("memory_variant") == method]
    clean = [float(v) for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def load_per_count_from_siblings(parent: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(parent.rglob("accuracy_by_evidence_count.csv")):
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows.append(dict(row))
        except Exception:
            continue
    return rows


def compare_counts(per_count_rows: Sequence[Dict[str, Any]], a: str, b: str) -> Tuple[List[int], List[int]]:
    by_method_count: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in per_count_rows:
        method = str(row.get("method") or row.get("memory_variant"))
        count = finite_float(row.get("evidence_count"))
        acc = finite_float(row.get("accuracy"))
        if count is not None and acc is not None:
            by_method_count[(method, int(count))].append(float(acc))
    improved: List[int] = []
    worsened: List[int] = []
    for count in range(9):
        a_vals = by_method_count.get((a, count), [])
        b_vals = by_method_count.get((b, count), [])
        if not a_vals or not b_vals:
            continue
        delta = float(np.mean(b_vals)) - float(np.mean(a_vals))
        if delta > 1e-9:
            improved.append(count)
        elif delta < -1e-9:
            worsened.append(count)
    return improved, worsened


def write_readme(output_dir: Path, summary_rows: Sequence[Dict[str, Any]], metrics_rows: Sequence[Dict[str, Any]]) -> None:
    sibling_root = output_dir.parent
    sibling_summary = collect_sibling_summaries(sibling_root)
    sibling_per_count = load_per_count_from_siblings(sibling_root)
    baseline_acc = method_accuracy(sibling_summary, BASELINE)
    local_acc = method_accuracy(sibling_summary, LAYER_LOCAL)
    cumulative_acc = method_accuracy(sibling_summary, CUMULATIVE)
    this = summary_rows[0] if summary_rows else {}
    method = str(this.get("method", ""))
    readout_mode = str(this.get("readout_mode", "none"))
    acc = finite_float(this.get("accuracy"))
    gate_corr = finite_float(this.get("corr_gate_sum_evidence_count"))
    update_corr = finite_float(this.get("corr_update_norm_evidence_count"))
    update_l2 = finite_float(this.get("approx_total_update_l2")) or 0.0
    token_fail = int(finite_float(this.get("token_selection_failures")) or 0)
    frame_fail = int(finite_float(this.get("frame_grouping_failures")) or 0)
    under = sum(1 for row in metrics_rows if int(row["pred_answer"]) < int(row["gold_answer"]))
    over = sum(1 for row in metrics_rows if int(row["pred_answer"]) > int(row["gold_answer"]))
    wrong = sum(1 for row in metrics_rows if int(row["correct"]) == 0)
    if update_l2 < 1e-3:
        update_text = "near zero"
    elif update_l2 > 10.0:
        update_text = "huge"
    else:
        update_text = "reasonable"
    local_imp, local_worse = compare_counts(sibling_per_count, BASELINE, LAYER_LOCAL)
    cum_imp, cum_worse = compare_counts(sibling_per_count, LAYER_LOCAL, CUMULATIVE)
    matrix_values: List[float] = []
    message_modes: Counter[str] = Counter()
    for row in metrics_rows:
        matrix_values.extend(flatten_numeric(parse_json_field(row, "matrix_scores_by_layer_json", {})))
        modes = parse_json_field(row, "message_mode_by_layer_json", {})
        if isinstance(modes, dict):
            for value in modes.values():
                if isinstance(value, list):
                    message_modes.update(str(item) for item in value)
                elif value:
                    message_modes[str(value)] += 1
    matrix_finite = bool(matrix_values) and all(math.isfinite(value) for value in matrix_values)
    matrix_nonzero = any(abs(value) > 0.0 for value in matrix_values)

    lines = [
        "# Message Memory Carrier Update seq_len=8 7B",
        "",
        f"- Method: `{method}`",
        f"- Readout mode: `{readout_mode}`",
        f"- Accuracy: {acc:.4f}" if acc is not None else "- Accuracy: unavailable",
        f"- Matrix scores finite/nonzero: finite={matrix_finite}, nonzero={matrix_nonzero}",
        f"- Message mode counts from per-row diagnostics: {dict(message_modes)}",
        f"- Update norm correlation with evidence count: {update_corr:.4f}" if update_corr is not None else "- Update norm correlation with evidence count: unavailable",
        f"- Update norms look {update_text} (mean approx L2={update_l2:.6f})",
        f"- Failure shape: wrong={wrong}, undercount={under}, overcount={over}",
        f"- Localization failures: token_selection={token_fail}, frame_grouping={frame_fail}",
    ]
    if readout_mode == RAW_MATRIX_READOUT:
        lines.extend(
            [
                "",
                "## Raw Matrix Readout",
                "",
                "This run uses a raw matrix-memory readout:",
                "",
                "C = sum_f v_f k_f^T",
                "r = C q = sum_f (k_f^T q) v_f",
                "",
                "This avoids softmax normalization and does not use sigmoid slot gates. Unlike sigmoid gates, raw matrix scores are signed and unbounded, so non-evidence frames can potentially contribute negatively or be suppressed by low/negative query compatibility. This may be more task-agnostic but less stable.",
                "",
                "## Interpretation Guide",
                "",
                "- If matrix scores are finite and nonzero but accuracy is low, the memory readout is active but the residual update may not translate into the answer logits.",
                "- If update norm is near zero, gamma/init/lr/reg may be too conservative.",
                "- If update norm is huge but accuracy low, the adapter may be injecting unstable residual directions.",
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"- Did layer-local improve over baseline? {local_acc > baseline_acc} "
                    f"(baseline={baseline_acc:.4f}, layer-local={local_acc:.4f})"
                    if baseline_acc is not None and local_acc is not None
                    else "- Did layer-local improve over baseline? unavailable until both runs finish"
                ),
                (
                    f"- Did cumulative-sum improve over layer-local? {cumulative_acc > local_acc} "
                    f"(layer-local={local_acc:.4f}, cumulative={cumulative_acc:.4f})"
                    if cumulative_acc is not None and local_acc is not None
                    else "- Did cumulative-sum improve over layer-local? unavailable until both runs finish"
                ),
                f"- Evidence counts improved/worsened for layer-local vs baseline: improved={local_imp}, worsened={local_worse}",
                f"- Evidence counts improved/worsened for cumulative vs layer-local: improved={cum_imp}, worsened={cum_worse}",
                f"- Gate sum correlation with evidence count: {gate_corr:.4f}" if gate_corr is not None else "- Gate sum correlation with evidence count: unavailable",
                "",
                "## Interpretation Guide",
                "",
                "- If accuracy is low but gate sum correlates with evidence count, aggregation is working, but residual update translation into Qwen is weak.",
                "- If accuracy is low and gate sum does not correlate with evidence count, message extraction/querying is not identifying useful frame evidence.",
                "- If update norm is near zero, gamma/init/lr/reg may be too conservative.",
                "- If update norm is huge but accuracy low, the adapter is blasting residuals in wrong directions; add stronger structure or pretrain against oracle codebook direction.",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def flatten_numeric(value: Any) -> List[float]:
    if isinstance(value, dict):
        out: List[float] = []
        for item in value.values():
            out.extend(flatten_numeric(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_numeric(item))
        return out
    if finite_float(value) is not None:
        return [float(value)]
    return []


def write_diagnostics(
    *,
    output_dir: Path,
    model: Any,
    adapter: Optional[MessageMemoryCarrierAdapter],
    metrics_rows: Sequence[Dict[str, Any]],
    backward_diag: Dict[str, Any],
    train_history: Sequence[Dict[str, Any]],
    token_group: str = ROOM_CHAR_TOKEN_GROUP,
) -> None:
    model_trainable_tensors = sum(int(param.requires_grad) for param in model.parameters())
    adapter_trainable_tensors = 0 if adapter is None else sum(int(param.requires_grad) for param in adapter.parameters())
    adapter_trainable_params = 0 if adapter is None else sum(int(param.numel()) for param in adapter.parameters() if param.requires_grad)
    failed = [
        row
        for row in metrics_rows
        if int(row.get("token_selection_ok", 0)) == 0 or int(row.get("frame_grouping_ok", 0)) == 0
    ]
    matrix_values: List[float] = []
    for row in metrics_rows:
        matrix_values.extend(flatten_numeric(parse_json_field(row, "matrix_scores_by_layer_json", {})))
    update_values = [float(row.get("approx_update_l2", 0.0)) for row in metrics_rows if finite_float(row.get("approx_update_l2")) is not None]
    target_counts = [len(parse_json_field(row, "target_positions_json", [])) for row in metrics_rows]
    resolved_token_group = canonical_token_group(token_group)
    diagnostics = {
        "token_group": resolved_token_group,
        "qwen_frozen": int(model_trainable_tensors == 0),
        "model_trainable_tensors": int(model_trainable_tensors),
        "adapter_trainable_tensors": int(adapter_trainable_tensors),
        "adapter_trainable_params": int(adapter_trainable_params),
        "only_adapter_params_trainable": int(model_trainable_tensors == 0 and adapter_trainable_tensors > 0)
        if adapter is not None
        else "",
        "hooks_fire_counts": {} if adapter is None else {str(k): int(v) for k, v in sorted(adapter.hook_fire_counts.items())},
        "readout_mode": "none" if adapter is None else str(getattr(adapter, "readout_mode", "unknown")),
        "message_mode_counts": {} if adapter is None else dict(adapter.message_mode_counts),
        "exact_failure_counts": {} if adapter is None else dict(adapter.exact_failure_counts),
        "exact_failure_examples": [] if adapter is None else list(adapter.exact_failure_examples),
        "backward_diagnostics": backward_diag,
        "train_history_last": dict(train_history[-1]) if train_history else {},
        "num_failed_localization_samples": len(failed),
        "failed_localization_sample_ids": [row.get("sample_id") for row in failed[:50]],
        "hooks_ok": int(adapter is None or all(adapter.hook_fire_counts.get(layer, 0) > 0 for layer in adapter.inject_layers)),
        "frame_rows_ok": int(all(len(parse_json_field(row, "frame_token_counts_json", [])) == NUM_FRAMES for row in metrics_rows)),
        "target_positions_found": int(bool(target_counts) and all(count > 0 for count in target_counts)),
        "avg_num_target_positions": finite_mean(target_counts, default=0.0),
        "min_num_target_positions": min(target_counts) if target_counts else 0,
        "max_num_target_positions": max(target_counts) if target_counts else 0,
        "room_char_positions_found": int(
            resolved_token_group == ROOM_CHAR_TOKEN_GROUP
            and all(len(parse_json_field(row, "room_char_positions_json", [])) > 0 for row in metrics_rows)
        ),
        "nonzero_gates": int(any(float(row.get("mean_gate_sum", 0.0)) > 0.0 for row in metrics_rows)),
        "nonzero_matrix_scores": int(any(abs(value) > 0.0 for value in matrix_values)),
        "finite_matrix_scores": int(bool(matrix_values) and all(math.isfinite(value) for value in matrix_values)),
        "nonzero_updates": int(any(float(row.get("approx_update_l2", 0.0)) > 0.0 for row in metrics_rows)),
        "finite_updates": int(bool(update_values) and all(math.isfinite(value) for value in update_values)),
    }
    prev.write_json(output_dir / "diagnostics.json", diagnostics)
    write_csv_dynamic(
        output_dir / "failed_localization_samples.csv",
        failed,
        ["sample_id", "sample_index", "token_selection_ok", "token_selection_error", "frame_grouping_ok", "frame_grouping_error"],
    )


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def combined_method_label(row: Dict[str, Any], fallback: str = "") -> str:
    method = str(row.get("method") or row.get("memory_variant") or fallback)
    readout_mode = str(row.get("readout_mode", ""))
    if method == BASELINE:
        return "baseline"
    if method == LAYER_LOCAL and readout_mode == RAW_MATRIX_READOUT:
        return "layer_local_raw_matrix"
    return safe_name(method or fallback)


def save_combined_line_plot(
    path: Path,
    by_count: Dict[Tuple[str, int], Dict[str, Any]],
    methods: Sequence[str],
    count_values: Sequence[int],
    y_key: str,
    ylabel: str,
    title: str,
) -> None:
    counts = [int(count) for count in count_values]
    plt.figure(figsize=(7.2, 4.5))
    for method in methods:
        ys = [
            float(by_count.get((method, count), {}).get(y_key, math.nan))
            if finite_float(by_count.get((method, count), {}).get(y_key)) is not None
            else math.nan
            for count in counts
        ]
        plt.plot(counts, ys, marker="o", linewidth=1.8, label=method)
    plt.xlabel("Evidence count")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(counts)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def confusion_for_rows(rows: Sequence[Dict[str, Any]], count_values: Sequence[int]) -> np.ndarray:
    counts = [int(count) for count in count_values]
    matrix = np.zeros((len(counts), len(counts)), dtype=float)
    for row in rows:
        gold = finite_float(row.get("gold_answer"))
        pred = finite_float(row.get("pred_answer"))
        if gold is None or pred is None:
            continue
        gold_i = int(gold)
        pred_i = int(pred)
        if gold_i in counts and pred_i in counts:
            matrix[counts.index(gold_i), counts.index(pred_i)] += 1.0
    return matrix


def save_confusion_plot(path: Path, rows: Sequence[Dict[str, Any]], count_values: Sequence[int], title: str) -> None:
    counts = [int(count) for count in count_values]
    matrix = confusion_for_rows(rows, counts)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_yticks(np.arange(len(counts)))
    ax.set_xticklabels(counts)
    ax.set_yticklabels(counts)
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("Gold count")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_combined_outputs(
    *,
    combined_dir: Path,
    baseline_dir: Path,
    local_dir: Path,
    count_values: Sequence[int],
) -> None:
    combined_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = combined_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = read_csv_rows(baseline_dir / "metrics.csv")
    local_metrics = read_csv_rows(local_dir / "metrics.csv")
    baseline_summary = read_csv_rows(baseline_dir / "summary.csv")
    local_summary = read_csv_rows(local_dir / "summary.csv")
    baseline_counts = read_csv_rows(baseline_dir / "accuracy_by_evidence_count.csv")
    local_counts = read_csv_rows(local_dir / "accuracy_by_evidence_count.csv")

    def relabel(rows: Sequence[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["method"] = label
            item["comparison_method"] = label
            out.append(item)
        return out

    combined_metrics = relabel(baseline_metrics, "baseline") + relabel(local_metrics, "layer_local_raw_matrix")
    combined_summary = relabel(baseline_summary, "baseline") + relabel(local_summary, "layer_local_raw_matrix")
    combined_counts = relabel(baseline_counts, "baseline") + relabel(local_counts, "layer_local_raw_matrix")

    write_csv_dynamic(combined_dir / "combined_metrics.csv", combined_metrics, ["comparison_method", "method", "sample_id", "evidence_count", "gold_answer", "pred_answer", "correct"])
    write_csv_dynamic(combined_dir / "combined_summary.csv", combined_summary, ["comparison_method", "method", "readout_mode", "n", "accuracy", "mean_margin"])
    write_csv_dynamic(
        combined_dir / "combined_accuracy_by_evidence_count.csv",
        combined_counts,
        ["comparison_method", "method", "readout_mode", "evidence_count", "n", "accuracy", "mean_margin", "mean_predicted_count"],
    )

    by_count: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in combined_counts:
        count = finite_float(row.get("evidence_count"))
        if count is not None:
            by_count[(str(row["comparison_method"]), int(count))] = row
    comparison_rows: List[Dict[str, Any]] = []
    for count in count_values:
        count_i = int(count)
        base = by_count.get(("baseline", count_i), {})
        local = by_count.get(("layer_local_raw_matrix", count_i), {})
        base_acc = finite_float(base.get("accuracy"))
        local_acc = finite_float(local.get("accuracy"))
        comparison_rows.append(
            {
                "evidence_count": count_i,
                "baseline_accuracy": "" if base_acc is None else float(base_acc),
                "layer_local_raw_matrix_accuracy": "" if local_acc is None else float(local_acc),
                "delta_accuracy": "" if base_acc is None or local_acc is None else float(local_acc) - float(base_acc),
                "baseline_mean_pred": base.get("mean_predicted_count", ""),
                "layer_local_raw_matrix_mean_pred": local.get("mean_predicted_count", ""),
                "baseline_mean_margin": base.get("mean_margin", ""),
                "layer_local_raw_matrix_mean_margin": local.get("mean_margin", ""),
            }
        )
    write_csv_dynamic(
        combined_dir / "comparison_by_evidence_count.csv",
        comparison_rows,
        [
            "evidence_count",
            "baseline_accuracy",
            "layer_local_raw_matrix_accuracy",
            "delta_accuracy",
            "baseline_mean_pred",
            "layer_local_raw_matrix_mean_pred",
            "baseline_mean_margin",
            "layer_local_raw_matrix_mean_margin",
        ],
    )

    methods = ["baseline", "layer_local_raw_matrix"]
    save_combined_line_plot(plots_dir / "combined_accuracy_vs_evidence_count.png", by_count, methods, count_values, "accuracy", "Accuracy", "Accuracy vs Evidence Count")
    save_combined_line_plot(plots_dir / "combined_margin_vs_evidence_count.png", by_count, methods, count_values, "mean_margin", "Mean margin", "Margin vs Evidence Count")
    save_combined_line_plot(
        plots_dir / "mean_predicted_count_vs_gold_count.png",
        by_count,
        methods,
        count_values,
        "mean_predicted_count",
        "Mean predicted count",
        "Mean Predicted Count vs Gold Count",
    )
    save_confusion_plot(plots_dir / "predicted_count_confusion_matrix_baseline.png", combined_metrics[: len(baseline_metrics)], count_values, "Baseline Confusion Matrix")
    save_confusion_plot(
        plots_dir / "predicted_count_confusion_matrix_layer_local_raw_matrix.png",
        combined_metrics[len(baseline_metrics) :],
        count_values,
        "Layer-Local Raw Matrix Confusion Matrix",
    )

    counts = [int(count) for count in count_values]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, label, rows in zip(axes, methods, [combined_metrics[: len(baseline_metrics)], combined_metrics[len(baseline_metrics) :]]):
        matrix = confusion_for_rows(rows, counts)
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(counts)))
        ax.set_xticklabels(counts)
        ax.set_yticks(np.arange(len(counts)))
        ax.set_yticklabels(counts)
        ax.set_xlabel("Predicted count")
        ax.set_title(label)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0:
                    ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Gold count")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(plots_dir / "combined_confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(7.2, 4.3))
    xs = [int(row["evidence_count"]) for row in comparison_rows]
    ys = [float(row["delta_accuracy"]) if finite_float(row.get("delta_accuracy")) is not None else math.nan for row in comparison_rows]
    colors = ["#2ca02c" if finite_float(y) is not None and float(y) >= 0 else "#d62728" for y in ys]
    plt.bar(xs, ys, color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Evidence count")
    plt.ylabel("Layer-local raw matrix minus baseline accuracy")
    plt.title("Delta Accuracy")
    plt.xticks(counts)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "delta_accuracy_layer_local_raw_matrix_minus_baseline.png", dpi=180, bbox_inches="tight")
    plt.close()

    base_acc = finite_float((combined_summary[0] if combined_summary else {}).get("accuracy"))
    local_acc = finite_float((combined_summary[1] if len(combined_summary) > 1 else {}).get("accuracy"))
    improved = [int(row["evidence_count"]) for row in comparison_rows if finite_float(row.get("delta_accuracy")) is not None and float(row["delta_accuracy"]) > 0]
    worsened = [int(row["evidence_count"]) for row in comparison_rows if finite_float(row.get("delta_accuracy")) is not None and float(row["delta_accuracy"]) < 0]
    high_rows = [row for row in comparison_rows if int(row["evidence_count"]) >= 4 and finite_float(row.get("delta_accuracy")) is not None]
    high_delta = finite_mean([row["delta_accuracy"] for row in high_rows], default=math.nan)
    local_diag = json.loads((local_dir / "diagnostics.json").read_text(encoding="utf-8")) if (local_dir / "diagnostics.json").is_file() else {}
    baseline_diag = json.loads((baseline_dir / "diagnostics.json").read_text(encoding="utf-8")) if (baseline_dir / "diagnostics.json").is_file() else {}
    lines = [
        "# Baseline vs Layer-Local Raw Matrix",
        "",
        (
            f"- Did layer-local raw_matrix improve over baseline? {float(local_acc) > float(base_acc)} "
            f"(baseline={float(base_acc):.4f}, layer_local_raw_matrix={float(local_acc):.4f})"
            if base_acc is not None and local_acc is not None
            else "- Did layer-local raw_matrix improve over baseline? unavailable"
        ),
        f"- Evidence counts improved/worsened: improved={improved}, worsened={worsened}",
        f"- Is raw_matrix better at high evidence counts 4..8? {finite_float(high_delta) is not None and float(high_delta) > 0.0} (mean delta={high_delta:.4f})",
        f"- Matrix scores finite and nonzero? finite={bool(local_diag.get('finite_matrix_scores'))}, nonzero={bool(local_diag.get('nonzero_matrix_scores'))}",
        f"- Update norms active? nonzero_updates={bool(local_diag.get('nonzero_updates'))}, finite_updates={bool(local_diag.get('finite_updates'))}",
        f"- Exact attention message mode counts: {local_diag.get('message_mode_counts', {})}",
        f"- Qwen frozen? baseline={bool(baseline_diag.get('qwen_frozen'))}, layer_local={bool(local_diag.get('qwen_frozen'))}",
        f"- Only adapter params trainable? {bool(local_diag.get('only_adapter_params_trainable'))}",
        f"- Hooks fired on layers 14..17? {bool(local_diag.get('hooks_ok'))}",
        "",
        "Raw matrix readout:",
        "",
        "C = sum_f v_f k_f^T",
        "r = C q = sum_f (k_f^T q) v_f",
        "",
        "This avoids softmax normalization and does not use sigmoid slot gates. Raw matrix scores are signed and unbounded, so non-evidence frames can contribute negatively or be suppressed by low/negative query compatibility.",
    ]
    (combined_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def combo_command(args: argparse.Namespace, *, variant: str, run_name: str, output_dir: Path, readout_mode: str) -> List[str]:
    cmd = [
        sys.executable,
        os.fspath(Path(__file__).resolve()),
        "--memory-variant",
        variant,
        "--readout-mode",
        readout_mode,
        "--model-name",
        str(args.model_name),
        "--dataset-root",
        os.fspath(args.dataset_root),
        "--source-run",
        os.fspath(args.source_run),
        "--seq-len",
        str(args.seq_len),
        "--split",
        str(args.split),
        "--output-root",
        os.fspath(args.output_root),
        "--output-dir",
        os.fspath(output_dir),
        "--run-name",
        run_name,
        "--d-mem",
        str(args.d_mem),
        "--layer-start",
        str(args.layer_start),
        "--layer-end",
        str(args.layer_end),
        "--gamma-init",
        str(args.gamma_init),
        "--message-mode",
        str(args.message_mode),
        "--lambda-margin",
        str(args.lambda_margin),
        "--margin-target",
        str(args.margin_target),
        "--lambda-update-energy",
        str(args.lambda_update_energy),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--grad-clip",
        str(args.grad_clip),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-eval-samples",
        str(args.max_eval_samples),
        "--max-samples-per-count",
        str(args.max_samples_per_count),
        "--seed",
        str(args.seed),
        "--candidate-min",
        str(args.candidate_min),
        "--candidate-max",
        str(args.candidate_max),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--attn-implementation",
        str(args.attn_implementation),
        "--submit-mode",
        str(args.submit_mode),
    ]
    if bool(args.no_plots):
        cmd.append("--no-plots")
    if bool(args.load_in_4bit):
        cmd.append("--load-in-4bit")
    else:
        cmd.append("--no-load-in-4bit")
    if args.max_pixels is not None:
        cmd.extend(["--max-pixels", str(args.max_pixels)])
    if args.min_pixels is not None:
        cmd.extend(["--min-pixels", str(args.min_pixels)])
    if args.evidence_counts:
        cmd.append("--evidence-counts")
        cmd.extend(str(x) for x in args.evidence_counts)
    return cmd


def run_baseline_and_layer_local_raw_matrix(args: argparse.Namespace) -> int:
    if int(args.seq_len) != NUM_FRAMES:
        raise ValueError("This experiment is intentionally seq_len=8 only.")
    count_values = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
    stamp = str(args.run_name).strip() or time.strftime("%Y%m%d_%H%M%S")
    baseline_name = f"{safe_name(stamp)}_baseline"
    local_name = f"{safe_name(stamp)}_layer_local_raw_matrix"
    combined_name = f"{safe_name(stamp)}_baseline_vs_layer_local_raw_matrix"
    output_root = Path(args.output_root).resolve()
    baseline_dir = output_root / baseline_name
    local_dir = output_root / local_name

    baseline_cmd = combo_command(
        args,
        variant="baseline",
        run_name=baseline_name,
        output_dir=baseline_dir,
        readout_mode=SIGMOID_GATE_READOUT,
    )
    local_cmd = combo_command(
        args,
        variant="layer_local",
        run_name=local_name,
        output_dir=local_dir,
        readout_mode=RAW_MATRIX_READOUT,
    )
    print("Running frozen Qwen baseline")
    subprocess.run(baseline_cmd, check=True)
    print("Running layer-local raw matrix adapter")
    subprocess.run(local_cmd, check=True)

    combined_dir = output_root / combined_name
    write_combined_outputs(
        combined_dir=combined_dir,
        baseline_dir=baseline_dir,
        local_dir=local_dir,
        count_values=count_values,
    )
    print(f"Combined outputs: {combined_dir}")
    return 0


def main() -> int:
    args = parse_args()
    if bool(args.run_baseline_and_layer_local_raw_matrix):
        args.evidence_counts = prev.parse_int_tokens(args.evidence_counts)
        return run_baseline_and_layer_local_raw_matrix(args)
    if not args.memory_variant:
        raise ValueError("--memory-variant is required unless --run-baseline-and-layer-local-raw-matrix is set")
    variant = canonical_variant(args.memory_variant)
    args.token_group = canonical_token_group(args.token_group)
    if int(args.seq_len) != NUM_FRAMES:
        raise ValueError("This experiment is intentionally seq_len=8 only.")
    if int(args.layer_end) < int(args.layer_start):
        raise ValueError("--layer-end must be >= --layer-start")
    if int(args.candidate_min) != 0 or int(args.candidate_max) != 8:
        raise ValueError("This runner expects candidate counts 0-8.")
    args.evidence_counts = prev.parse_int_tokens(args.evidence_counts)
    inject_layers = list(range(int(args.layer_start), int(args.layer_end) + 1))
    count_values = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else default_output_dir(args, variant)
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
            "memory_variant": str(variant),
            "requested_memory_variant": str(args.memory_variant),
            "d_mem": int(args.d_mem),
            "layer_start": int(args.layer_start),
            "layer_end": int(args.layer_end),
            "inject_layers": inject_layers,
            "token_group": str(args.token_group),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "grad_accum": int(args.grad_accum),
            "max_train_samples": int(args.max_train_samples),
            "max_eval_samples": int(args.max_eval_samples),
            "lambda_margin": float(args.lambda_margin),
            "margin_target": float(args.margin_target),
            "lambda_update_energy": float(args.lambda_update_energy),
            "gamma_init": float(args.gamma_init),
            "message_mode": str(args.message_mode),
            "readout_mode": str(args.readout_mode),
            "seed": int(args.seed),
            "submit_mode": str(args.submit_mode),
        }
        prev.write_json(output_dir / "run_config.json", run_config)
        print(f"Output dir: {output_dir}")
        print(f"Run config: {json_compact(run_config)}")

        sample_payload = trans.load_sample_index_payload(args)
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
        if variant != BASELINE and (not train_indices or not val_indices):
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

        adapter: Optional[MessageMemoryCarrierAdapter] = None
        train_history: List[Dict[str, Any]] = []
        backward_diag: Dict[str, Any] = {}
        checkpoint_path: Optional[Path] = None
        final_epoch = 0
        if variant != BASELINE:
            adapter, train_history, checkpoint_path, backward_diag = train_adapter(
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
                variant=variant,
                device=device,
            )
            final_epoch = int(train_history[-1]["epoch"]) if train_history else int(args.epochs)
            prev.write_json(
                output_dir / "checkpoint.json",
                {
                    "trained_checkpoint": os.fspath(checkpoint_path or ""),
                    "token_group": str(args.token_group),
                    "readout_mode": str(args.readout_mode),
                    "message_mode": str(args.message_mode),
                    "inject_layers": inject_layers,
                    "d_mem": int(args.d_mem),
                },
            )

        print(f"Evaluating train split for {variant}")
        train_eval = evaluate_split(
            split_name="train",
            method_label=variant,
            model=model,
            processor=processor,
            adapter=adapter,
            records=records,
            indices=train_indices,
            count_token_ids=count_token_ids,
            count_values=count_values,
            device=device,
            batch_size=int(args.batch_size),
            max_eval_samples=int(args.max_eval_samples) if variant == BASELINE else 0,
            token_group=args.token_group,
        )
        print(f"Evaluating test split for {variant}")
        test_eval = evaluate_split(
            split_name="test",
            method_label=variant,
            model=model,
            processor=processor,
            adapter=adapter,
            records=records,
            indices=test_indices,
            count_token_ids=count_token_ids,
            count_values=count_values,
            device=device,
            batch_size=int(args.batch_size),
            max_eval_samples=0,
            token_group=args.token_group,
        )

        metrics_rows = make_metrics_rows(
            eval_payload=test_eval,
            records=records,
            count_values=count_values,
            variant=variant,
            split_name="test",
            layer_start=int(args.layer_start),
            layer_end=int(args.layer_end),
            d_mem=int(args.d_mem),
            lr=float(args.lr),
            epoch=int(final_epoch),
            readout_mode="none" if variant == BASELINE else str(args.readout_mode),
            token_group=args.token_group,
        )
        summary_rows, per_count_rows = summarize_rows(
            metrics_rows,
            variant=variant,
            count_values=count_values,
            readout_mode="none" if variant == BASELINE else str(args.readout_mode),
            train_eval=train_eval,
            val_eval=None if not train_history else {"accuracy": train_history[-1]["val_accuracy"], "ce": train_history[-1]["val_ce"]},
        )

        write_csv_dynamic(
            output_dir / "metrics.csv",
            metrics_rows,
            [
                "sample_id",
                "token_group",
                "readout_mode",
                "evidence_count",
                "gold_answer",
                "pred_answer",
                "correct",
                "margin",
                "gold_logit",
                "pred_logit",
                "update_energy_total",
                "approx_update_l2",
                "mean_gate_sum",
                "mean_matrix_score_sum",
                "mean_matrix_score_abs_sum",
                "mean_matrix_score_mean",
                "mean_matrix_score_abs_mean",
                "gate_sum_by_layer_json",
                "gate_values_by_layer_json",
                "matrix_scores_by_layer_json",
                "matrix_score_sum_by_layer_json",
                "matrix_score_abs_sum_by_layer_json",
                "matrix_score_mean_by_layer_json",
                "matrix_score_abs_mean_by_layer_json",
                "update_norm_by_layer_json",
                "memory_norm_by_layer_json",
                "message_norm_by_layer_json",
                "raw_message_norm_by_layer_json",
                "message_mode_by_layer_json",
                "target_positions_json",
                "frame_token_counts_json",
                "token_selection_ok",
                "frame_grouping_ok",
            ],
        )
        write_csv_dynamic(
            output_dir / "summary.csv",
            summary_rows,
            [
                "method",
                "memory_variant",
                "readout_mode",
                "n",
                "accuracy",
                "mean_margin",
                "mean_gold_logit",
                "mean_update_energy",
                "approx_total_update_l2",
                "mean_gate_sum",
                "mean_matrix_score_sum",
                "mean_matrix_score_abs_sum",
                "mean_matrix_score_mean",
                "mean_matrix_score_abs_mean",
                "mean_gate_sum_by_evidence_count",
                "corr_gate_sum_evidence_count",
                "corr_update_norm_evidence_count",
                "token_selection_failures",
                "frame_grouping_failures",
            ],
        )
        write_csv_dynamic(
            output_dir / "accuracy_by_evidence_count.csv",
            per_count_rows,
            [
                "method",
                "memory_variant",
                "readout_mode",
                "evidence_count",
                "n",
                "accuracy",
                "mean_margin",
                "mean_predicted_count",
                "prediction_histogram",
                "mean_gate_sum",
                "mean_matrix_score_sum",
                "mean_matrix_score_abs_sum",
                "mean_matrix_score_mean",
                "mean_matrix_score_abs_mean",
                "mean_update_norm",
                "mean_message_norm",
                "mean_memory_norm",
            ],
        )
        if train_history:
            write_csv_dynamic(
                output_dir / "train_history.csv",
                train_history,
                [
                    "method",
                    "readout_mode",
                    "epoch",
                    "train_ce",
                    "train_loss",
                    "train_update_energy",
                    "val_ce",
                    "val_accuracy",
                    "val_mean_update_energy",
                    "adapter_parameter_norm",
                ],
            )
        write_failure_tables(output_dir, metrics_rows)
        if not bool(args.no_plots):
            make_plots(output_dir, metrics_rows, per_count_rows, count_values)
        write_diagnostics(
            output_dir=output_dir,
            model=model,
            adapter=adapter,
            metrics_rows=metrics_rows,
            backward_diag=backward_diag,
            train_history=train_history,
            token_group=args.token_group,
        )
        write_readme(output_dir, summary_rows, metrics_rows)
        prev.write_json(
            output_dir / "run_done.json",
            {
                "completed": True,
                "elapsed_seconds": time.time() - started,
                "variant": str(variant),
                "output_dir": os.fspath(output_dir),
            },
        )
        print(f"Done {variant}: output_dir={output_dir}")
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

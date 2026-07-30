#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
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
import torch.nn as nn
import torch.nn.functional as F

from experiments.evidence_only import evidence_only_layer_local_seq1_8_7b as base
from experiments.evidence_only import evidence_only_sum_evidence_adapter_seq1_8_7b as sum_base
from experiments.oracle_bounds import translator_ablation_gold_count_seq8_7b as trans
from experiments.carrier_probes import run_message_memory_adapter_stage1_stage3_seq8 as prev
from experiments.carrier_probes import run_message_memory_carrier_update_seq8_7b as carrier


EXPERIMENT_NAME = "distractor_posneg_write_read_adapter_seq8_7b"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images_park"

BASELINE = "baseline"
METHOD = "oracle_posneg_write_read_adapter"  # reassigned in main() from --gate-mode/--streams
METHOD_READOUT = "oracle_posneg_write14_17_read20_27"
NUM_FRAMES = 8
COUNT_VALUES = list(range(9))
GATE_MODES = ("oracle", "learned")
STREAM_MODES = ("posneg", "pos_only")


STREAM_PARAM_PREFIXES = ("w_pos.", "w_neg.", "w_read.", "gamma")


def method_label(
    gate_mode: str,
    streams: str,
    gate_source: str = "write_layer",
    gate_hard: bool = False,
    freeze_streams: bool = False,
) -> str:
    suffix = "_lategate" if str(gate_source) == "read_layer" else ""
    if gate_hard:
        suffix += "_hardgate"
    if freeze_streams:
        suffix += "_frozenreadout"
    return f"{gate_mode}_{streams}{suffix}_write_read_adapter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distractor seq_len=8 positive/negative write/read memory adapter with a "
            "configurable frame gate: --gate-mode oracle reproduces the diagnostic "
            "upper bound (gold evidence labels at inference, not a valid method); "
            "--gate-mode learned trains a sigmoid evidence gate with auxiliary "
            "supervision from the gold mask (valid at inference). --streams pos_only "
            "ablates the negative stream."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-run", type=Path, default=prev.DEFAULT_SOURCE_RUN)
    parser.add_argument("--seq-len", type=int, default=NUM_FRAMES)
    parser.add_argument("--split", default="all_uniform")
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in COUNT_VALUES])
    parser.add_argument("--max-samples-per-count", type=int, default=100)

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="")

    parser.add_argument("--run-baseline", action="store_true", default=False)
    parser.add_argument(
        "--run-oracle-posneg-write-read",
        "--run-oracle-posneg-sum",
        dest="run_oracle_posneg_write_read",
        action="store_true",
        default=False,
    )
    parser.add_argument("--run-all", action="store_true", default=False)

    parser.add_argument("--gate-mode", choices=list(GATE_MODES), default="learned")
    parser.add_argument("--streams", choices=list(STREAM_MODES), default="posneg")
    parser.add_argument(
        "--gate-source",
        choices=["write_layer", "read_layer"],
        default="write_layer",
        help=(
            "Where the learned gate runs. 'write_layer': each write layer gates its own messages "
            "with its own states (default). 'read_layer': raw write-layer messages are cached and "
            "the gate is computed once at the first read layer, where evidence is more linearly "
            "detectable (late gate, early messages)."
        ),
    )
    parser.add_argument(
        "--gate-hard",
        action="store_true",
        default=False,
        help="Straight-through gate: hard 0/1 mask in the forward pass, sigmoid gradient in the backward pass.",
    )
    parser.add_argument(
        "--init-streams-from",
        type=Path,
        default=None,
        help="Checkpoint whose w_pos/w_neg/w_read/gamma initialize the stream/readout weights (e.g. the oracle-trained run).",
    )
    parser.add_argument(
        "--freeze-streams",
        action="store_true",
        default=False,
        help="Freeze w_pos/w_neg/w_read/gamma so only the gate (selector MLP) trains.",
    )
    parser.add_argument("--query-token-group", default="all_question", choices=sorted(carrier.TOKEN_GROUP_ALIASES))
    parser.add_argument(
        "--oracle-mask-noise",
        type=float,
        default=0.0,
        help="Training-time flip probability for the oracle mask (oracle gate mode only): teaches the readout to tolerate gate errors.",
    )
    parser.add_argument(
        "--lambda-ce", type=float, default=1.0, help="Weight on the answer cross-entropy loss."
    )
    parser.add_argument(
        "--lambda-mask", type=float, default=1.0, help="Gate BCE vs gold evidence mask (learned gate only)."
    )
    parser.add_argument(
        "--lambda-count", type=float, default=0.05, help="(sum(alpha) - gold_count)^2 penalty (learned gate only)."
    )
    parser.add_argument("--smoke", action="store_true", default=False, help="Tiny run: 1 epoch, few samples, no plots.")
    parser.add_argument("--d-mem", type=int, default=256)
    parser.add_argument("--write-layer-start", type=int, default=14)
    parser.add_argument("--write-layer-end", type=int, default=17)
    parser.add_argument("--read-layer-start", type=int, default=20)
    parser.add_argument("--read-layer-end", type=int, default=27)
    parser.add_argument("--layer-start", type=int, default=None, help="Deprecated alias for --read-layer-start.")
    parser.add_argument("--layer-end", type=int, default=None, help="Deprecated alias for --read-layer-end.")
    parser.add_argument("--gamma-init", type=float, default=0.05)
    parser.add_argument("--message-mode", default="auto", choices=["auto", "exact", "proxy"])
    parser.add_argument("--message-token-group", default="all_question", choices=sorted(carrier.TOKEN_GROUP_ALIASES))
    parser.add_argument("--inject-token-group", default="last_token", choices=sorted(carrier.TOKEN_GROUP_ALIASES))

    parser.add_argument("--lambda-margin", type=float, default=0.2)
    parser.add_argument("--margin-target", type=float, default=1.0)
    parser.add_argument("--lambda-update-energy", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=800)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true", default=False)

    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
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


def canonical_group(raw: str) -> str:
    return carrier.canonical_token_group(str(raw))


def default_output_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = str(args.run_name).strip() or f"{stamp}_{str(args.gate_mode)}_{str(args.streams)}"
    return Path(args.output_root).resolve() / base.safe_name(name)


def write_read_readout_mode(
    write_layers: Sequence[int], read_layers: Sequence[int], gate_mode: str = "oracle", streams: str = "posneg"
) -> str:
    write = [int(layer) for layer in write_layers]
    read = [int(layer) for layer in read_layers]
    return f"{gate_mode}_{streams}_write{write[0]}_{write[-1]}_read{read[0]}_{read[-1]}"


def evidence_mask_for_record(record: prev.SampleRecord, *, seq_len: int = NUM_FRAMES) -> List[int]:
    mask = carrier.evidence_frame_mask(record, int(seq_len))
    if not mask:
        raise RuntimeError(f"sample_id={record.sample_id}: oracle evidence_frame_mask is missing")
    if len(mask) != int(seq_len):
        raise RuntimeError(
            f"sample_id={record.sample_id}: oracle evidence_frame_mask length {len(mask)} != seq_len {seq_len}"
        )
    clean = [int(x) for x in mask]
    if any(value not in (0, 1) for value in clean):
        raise RuntimeError(f"sample_id={record.sample_id}: oracle mask is not binary: {clean}")
    selected = int(sum(clean))
    if selected != int(record.gold_count):
        raise RuntimeError(
            f"sample_id={record.sample_id}: oracle mask selects {selected} frames but gold_count={record.gold_count}"
        )
    if selected != int(record.evidence_count):
        raise RuntimeError(
            f"sample_id={record.sample_id}: oracle mask selects {selected} frames but evidence_count={record.evidence_count}"
        )
    return clean


def validate_oracle_masks(records: Sequence[prev.SampleRecord], seq_len: int = NUM_FRAMES) -> Dict[str, Any]:
    counts: Counter[int] = Counter()
    for record in records:
        mask = evidence_mask_for_record(record, seq_len=int(seq_len))
        counts[int(sum(mask))] += 1
    return {
        "records_checked": int(len(records)),
        "seq_len": int(seq_len),
        "mask_count_histogram": {str(k): int(v) for k, v in sorted(counts.items())},
    }


def prepare_batch(
    *,
    args: argparse.Namespace,
    records: Sequence[prev.SampleRecord],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
) -> carrier.MemoryBatch:
    if not records:
        raise ValueError("records cannot be empty")
    seq_lens = {len(record.frame_paths) for record in records}
    if seq_lens != {NUM_FRAMES}:
        raise ValueError(f"Expected seq_len={NUM_FRAMES} batch, got {sorted(seq_lens)}")
    carrier.NUM_FRAMES = NUM_FRAMES
    return carrier.prepare_memory_batch(
        records=records,
        sample_indices=sample_indices,
        processor=processor,
        device=device,
        token_group=str(args.inject_token_group),
        message_token_group=str(args.message_token_group),
        query_token_group=str(getattr(args, "query_token_group", args.inject_token_group)),
        inject_token_group=str(args.inject_token_group),
    )


def oracle_masks_for_records(records: Sequence[prev.SampleRecord], device: str) -> torch.Tensor:
    masks = [evidence_mask_for_record(record, seq_len=NUM_FRAMES) for record in records]
    return torch.tensor(masks, device=device, dtype=torch.float32)


class PosNegWriteReadAdapter(sum_base.SimpleSumEvidenceAdapter):
    """Cache pos/neg memories at write layers and inject them at later read layers.

    gate_mode='oracle' masks streams with the gold evidence labels (diagnostic upper
    bound). gate_mode='learned' replaces the gold mask with a sigmoid gate computed
    from [message, query, message*query] features, trained with auxiliary BCE/count
    supervision. streams='pos_only' drops the negative stream entirely.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        d_mem: int,
        write_layers: Sequence[int],
        read_layers: Sequence[int],
        gamma_init: float,
        message_mode: str,
        readout_mode: str,
        gate_mode: str = "oracle",
        streams: str = "posneg",
        gate_source: str = "write_layer",
        gate_hard: bool = False,
        oracle_mask_noise: float = 0.0,
    ) -> None:
        super().__init__(
            hidden_size=int(hidden_size),
            d_mem=int(d_mem),
            inject_layers=[int(layer) for layer in read_layers],
            gamma_init=float(gamma_init),
            message_mode=str(message_mode),
        )
        self.write_layers = [int(layer) for layer in write_layers]
        self.read_layers = [int(layer) for layer in read_layers]
        self.inject_layers = list(self.read_layers)
        self.layer_to_pos = {int(layer): pos for pos, layer in enumerate(self.read_layers)}
        self.write_layer_to_pos = {int(layer): pos for pos, layer in enumerate(self.write_layers)}
        self.hook_layers = sorted(set(self.write_layers + self.read_layers))

        self.message_to_memory = nn.ModuleList()
        del self.w_o
        self.w_pos = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.d_mem, bias=False) for _ in self.write_layers]
        )
        self.w_neg = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.d_mem, bias=False) for _ in self.write_layers]
        )
        self.w_read = nn.ModuleList(
            [nn.Linear(len(self.write_layers) * self.d_mem, self.hidden_size, bias=False) for _ in self.read_layers]
        )
        self.gamma = nn.Parameter(torch.full((len(self.read_layers),), float(gamma_init), dtype=torch.float32))

        for layer_pos in range(len(self.write_layers)):
            nn.init.xavier_uniform_(self.w_pos[layer_pos].weight, gain=0.5)
            nn.init.xavier_uniform_(self.w_neg[layer_pos].weight, gain=0.5)
        for layer_pos in range(len(self.read_layers)):
            nn.init.normal_(self.w_read[layer_pos].weight, mean=0.0, std=0.002)

        if str(gate_mode) not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}, got {gate_mode!r}")
        if str(streams) not in STREAM_MODES:
            raise ValueError(f"streams must be one of {STREAM_MODES}, got {streams!r}")
        if str(gate_source) not in ("write_layer", "read_layer"):
            raise ValueError(f"gate_source must be write_layer or read_layer, got {gate_source!r}")
        self.gate_mode = str(gate_mode)
        self.streams = str(streams)
        self.gate_source = str(gate_source)
        self.gate_hard = bool(gate_hard)
        self.oracle_mask_noise = float(oracle_mask_noise)
        if self.gate_source == "read_layer" and self.gate_mode != "learned":
            raise ValueError("gate_source=read_layer only makes sense with gate_mode=learned")
        self.selector_mlp: Optional[nn.ModuleList] = None
        if self.gate_mode == "learned":
            self.selector_mlp = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(3 * self.hidden_size, self.d_mem),
                        nn.GELU(),
                        nn.Linear(self.d_mem, 1),
                    )
                    for _ in self.write_layers
                ]
            )
            for mlp in self.selector_mlp:
                first = mlp[0]
                last = mlp[2]
                assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
                nn.init.xavier_uniform_(first.weight, gain=0.5)
                nn.init.zeros_(first.bias)
                nn.init.normal_(last.weight, mean=0.0, std=0.02)
                nn.init.zeros_(last.bias)

        self.readout_mode = str(readout_mode)
        self._evidence_frame_masks: Optional[torch.Tensor] = None
        self._gold_counts: Optional[torch.Tensor] = None
        self._query_positions: Optional[List[List[int]]] = None
        self._loss_gate_values: List[torch.Tensor] = []
        self._memory_cache: Dict[int, torch.Tensor] = {}
        self._pending_writes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.write_hook_fire_counts: Dict[int, int] = defaultdict(int)
        self.read_hook_fire_counts: Dict[int, int] = defaultdict(int)

    def set_context(
        self,
        *,
        message_target_positions: Sequence[Sequence[int]],
        inject_positions: Sequence[Sequence[int]],
        frame_groups: Sequence[Sequence[Sequence[int]]],
        evidence_frame_masks: torch.Tensor,
        gold_counts: torch.Tensor,
        query_positions: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        super().set_context(
            message_target_positions=message_target_positions,
            inject_positions=inject_positions,
            frame_groups=frame_groups,
        )
        if evidence_frame_masks.dim() != 2:
            raise ValueError(f"evidence_frame_masks must be [batch, frames], got {tuple(evidence_frame_masks.shape)}")
        if int(evidence_frame_masks.shape[1]) != self._num_frames():
            raise ValueError(
                f"oracle mask frame dimension {int(evidence_frame_masks.shape[1])} != frame groups {self._num_frames()}"
        )
        self._evidence_frame_masks = evidence_frame_masks.detach().float()
        self._gold_counts = gold_counts.detach().long()
        self._query_positions = (
            [[int(pos) for pos in positions] for positions in query_positions] if query_positions is not None else None
        )
        if self.gate_mode == "learned" and self._query_positions is None:
            raise RuntimeError("learned gate requires query_positions in set_context")
        self._loss_gate_values = []
        self._memory_cache = {}
        self._pending_writes = {}
        self._last_stats["gate_values_by_layer"] = {}
        self._last_stats["gate_sum_by_layer"] = {}
        self._last_stats["oracle_mask_by_layer"] = {}
        self._last_stats["oracle_mask_count_by_layer"] = {}
        self._last_stats["positive_frame_count_by_layer"] = {}
        self._last_stats["negative_frame_count_by_layer"] = {}
        self._last_stats["positive_stream_norm_by_layer"] = {}
        self._last_stats["negative_stream_norm_by_layer"] = {}
        self._last_stats["combined_stream_norm_by_layer"] = {}
        self._last_stats["write_memory_norm_by_layer"] = {}
        self._last_stats["read_memory_concat_norm_by_layer"] = {}

    def clear_context(self) -> None:
        super().clear_context()
        self._evidence_frame_masks = None
        self._gold_counts = None
        self._query_positions = None
        self._loss_gate_values = []
        self._memory_cache = {}
        self._pending_writes = {}

    def gate_supervision_loss(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._loss_gate_values:
            zero = torch.zeros((), device=device)
            return zero, zero
        if self._evidence_frame_masks is None or self._gold_counts is None:
            raise RuntimeError("Gate supervision requested without evidence masks/gold counts")
        alphas = torch.stack([alpha.float() for alpha in self._loss_gate_values], dim=0)
        target = self._evidence_frame_masks.to(device=alphas.device, dtype=alphas.dtype).unsqueeze(0)
        target = target.expand_as(alphas)
        bce = F.binary_cross_entropy(alphas.clamp(1e-6, 1.0 - 1e-6), target)
        gold_counts = self._gold_counts.to(device=alphas.device, dtype=alphas.dtype).view(1, -1)
        count_loss = (alphas.sum(dim=-1) - gold_counts).pow(2).mean()
        return bce.to(device), count_loss.to(device)

    def _query_vector(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self._query_positions is not None
        batch, seq_len, hidden = hidden_states.shape
        source = hidden_states.detach().float()
        rows: List[torch.Tensor] = []
        for batch_idx in range(batch):
            valid = sorted({int(pos) for pos in self._query_positions[batch_idx] if 0 <= int(pos) < seq_len})
            if valid:
                idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
                rows.append(source[batch_idx, idx, :].mean(dim=0))
            else:
                rows.append(source.new_zeros((hidden,)))
        return torch.stack(rows, dim=0)

    def _write_before_layer(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self._message_target_positions is None
            or self._frame_groups is None
            or self._evidence_frame_masks is None
            or int(layer_idx) not in self.write_layer_to_pos
        ):
            return hidden_states

        layer_pos = self.write_layer_to_pos[int(layer_idx)]
        self.hook_fire_counts[int(layer_idx)] += 1
        self.write_hook_fire_counts[int(layer_idx)] += 1
        raw_messages, mode = self._message_contribution(module, hidden_states, int(layer_idx), kwargs)
        self.message_mode_counts[f"{int(layer_idx)}:{mode}"] += int(hidden_states.shape[0])

        positive_messages = self.w_pos[layer_pos](raw_messages.float())
        negative_messages = self.w_neg[layer_pos](raw_messages.float())
        oracle_mask = self._evidence_frame_masks.to(device=positive_messages.device, dtype=positive_messages.dtype)
        if self.gate_mode == "learned" and self.gate_source == "read_layer":
            # Late gate, early messages: cache projected streams and raw features now,
            # resolve the gate at the first read layer where evidence is more decodable.
            self._pending_writes[int(layer_idx)] = (raw_messages.float(), positive_messages, negative_messages)
            layer_key = str(int(layer_idx))
            raw_message_norms = raw_messages.detach().float().norm(dim=-1).cpu().tolist()
            self._last_stats["message_norm_by_layer"][layer_key] = raw_message_norms
            self._last_stats["raw_message_norm_by_layer"][layer_key] = raw_message_norms
            self._last_stats["oracle_mask_by_layer"][layer_key] = oracle_mask.detach().float().cpu().tolist()
            self._last_stats["oracle_mask_count_by_layer"][layer_key] = (
                oracle_mask.detach().float().sum(dim=1).cpu().tolist()
            )
            self._last_stats["message_mode_by_layer"][layer_key] = [mode for _ in range(int(hidden_states.shape[0]))]
            return hidden_states
        if self.gate_mode == "learned":
            assert self.selector_mlp is not None
            query = self._query_vector(hidden_states)
            query_expanded = query.unsqueeze(1).expand(-1, raw_messages.shape[1], -1)
            selector_features = torch.cat(
                [raw_messages.float(), query_expanded.float(), raw_messages.float() * query_expanded.float()],
                dim=-1,
            )
            gate_logits = self.selector_mlp[layer_pos](selector_features).squeeze(-1).float()
            alpha = torch.sigmoid(gate_logits)
            self._loss_gate_values.append(alpha)
            if self.gate_hard:
                alpha = alpha + ((alpha > 0.5).float() - alpha).detach()
        else:
            alpha = oracle_mask
            if self.training and self.oracle_mask_noise > 0.0:
                flip = (torch.rand_like(alpha) < self.oracle_mask_noise).float()
                alpha = (1.0 - flip) * alpha + flip * (1.0 - alpha)
        positive_mask = alpha.unsqueeze(-1).float()
        negative_mask = (1.0 - alpha).unsqueeze(-1).float()
        positive_stream = (positive_messages.float() * positive_mask).sum(dim=1)
        negative_stream = (negative_messages.float() * negative_mask).sum(dim=1)
        if self.streams == "pos_only":
            combined_stream = positive_stream
        else:
            combined_stream = positive_stream + negative_stream
        self._memory_cache[int(layer_idx)] = combined_stream
        self._last_stats["gate_values_by_layer"][str(int(layer_idx))] = alpha.detach().float().cpu().tolist()
        self._last_stats["gate_sum_by_layer"][str(int(layer_idx))] = alpha.detach().float().sum(dim=-1).cpu().tolist()

        batch = int(hidden_states.shape[0])
        layer_key = str(int(layer_idx))
        raw_message_norms = raw_messages.detach().float().norm(dim=-1).cpu().tolist()
        self._last_stats["message_norm_by_layer"][layer_key] = raw_message_norms
        self._last_stats["raw_message_norm_by_layer"][layer_key] = raw_message_norms
        self._last_stats["summed_message_norm_by_layer"][layer_key] = (
            combined_stream.detach().float().norm(dim=-1).cpu().tolist()
        )
        self._last_stats["oracle_mask_by_layer"][layer_key] = oracle_mask.detach().float().cpu().tolist()
        self._last_stats["oracle_mask_count_by_layer"][layer_key] = oracle_mask.detach().float().sum(dim=1).cpu().tolist()
        self._last_stats["positive_frame_count_by_layer"][layer_key] = oracle_mask.detach().float().sum(dim=1).cpu().tolist()
        self._last_stats["negative_frame_count_by_layer"][layer_key] = (
            (1.0 - oracle_mask.detach().float()).sum(dim=1).cpu().tolist()
        )
        self._last_stats["positive_stream_norm_by_layer"][layer_key] = (
            positive_stream.detach().float().norm(dim=-1).cpu().tolist()
        )
        self._last_stats["negative_stream_norm_by_layer"][layer_key] = (
            negative_stream.detach().float().norm(dim=-1).cpu().tolist()
        )
        self._last_stats["combined_stream_norm_by_layer"][layer_key] = (
            combined_stream.detach().float().norm(dim=-1).cpu().tolist()
        )
        self._last_stats["write_memory_norm_by_layer"][layer_key] = (
            combined_stream.detach().float().norm(dim=-1).cpu().tolist()
        )
        self._last_stats["message_mode_by_layer"][layer_key] = [mode for _ in range(batch)]
        return hidden_states

    def _resolve_deferred_gate(self, hidden_states: torch.Tensor) -> None:
        """Compute the learned gate from read-layer states and finalize cached write streams."""
        assert self.selector_mlp is not None
        query = self._query_vector(hidden_states)
        for layer_idx in sorted(self._pending_writes):
            raw_messages, positive_messages, negative_messages = self._pending_writes[layer_idx]
            layer_pos = self.write_layer_to_pos[int(layer_idx)]
            query_expanded = query.unsqueeze(1).expand(-1, raw_messages.shape[1], -1)
            selector_features = torch.cat(
                [raw_messages.float(), query_expanded.float(), raw_messages.float() * query_expanded.float()],
                dim=-1,
            )
            gate_logits = self.selector_mlp[layer_pos](selector_features).squeeze(-1).float()
            alpha = torch.sigmoid(gate_logits)
            self._loss_gate_values.append(alpha)
            if self.gate_hard:
                alpha = alpha + ((alpha > 0.5).float() - alpha).detach()
            positive_stream = (positive_messages.float() * alpha.unsqueeze(-1).float()).sum(dim=1)
            negative_stream = (negative_messages.float() * (1.0 - alpha).unsqueeze(-1).float()).sum(dim=1)
            combined_stream = positive_stream if self.streams == "pos_only" else positive_stream + negative_stream
            self._memory_cache[int(layer_idx)] = combined_stream
            layer_key = str(int(layer_idx))
            self._last_stats["gate_values_by_layer"][layer_key] = alpha.detach().float().cpu().tolist()
            self._last_stats["gate_sum_by_layer"][layer_key] = alpha.detach().float().sum(dim=-1).cpu().tolist()
            self._last_stats["summed_message_norm_by_layer"][layer_key] = (
                combined_stream.detach().float().norm(dim=-1).cpu().tolist()
            )
            self._last_stats["positive_stream_norm_by_layer"][layer_key] = (
                positive_stream.detach().float().norm(dim=-1).cpu().tolist()
            )
            self._last_stats["negative_stream_norm_by_layer"][layer_key] = (
                negative_stream.detach().float().norm(dim=-1).cpu().tolist()
            )
            self._last_stats["combined_stream_norm_by_layer"][layer_key] = (
                combined_stream.detach().float().norm(dim=-1).cpu().tolist()
            )
            self._last_stats["write_memory_norm_by_layer"][layer_key] = (
                combined_stream.detach().float().norm(dim=-1).cpu().tolist()
            )
        self._pending_writes = {}

    def _read_before_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self._inject_positions is None
            or int(layer_idx) not in self.layer_to_pos
        ):
            return hidden_states

        if self._pending_writes:
            self._resolve_deferred_gate(hidden_states)

        missing = [int(layer) for layer in self.write_layers if int(layer) not in self._memory_cache]
        if missing:
            raise RuntimeError(
                f"read layer {int(layer_idx)} missing write-layer memories {missing}; "
                f"available={sorted(self._memory_cache)}"
            )

        layer_pos = self.layer_to_pos[int(layer_idx)]
        self.hook_fire_counts[int(layer_idx)] += 1
        self.read_hook_fire_counts[int(layer_idx)] += 1
        memory_all = torch.cat([self._memory_cache[int(layer)] for layer in self.write_layers], dim=-1)
        delta = self.w_read[layer_pos](memory_all.float()).float()
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
        self._last_stats["update_norm_by_layer"][layer_key] = actual_update.detach().float().norm(dim=-1).cpu().tolist()
        self._last_stats["read_memory_concat_norm_by_layer"][layer_key] = (
            memory_all.detach().float().norm(dim=-1).cpu().tolist()
        )
        return out

    def inject_before_layer(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        if int(layer_idx) in self.write_layer_to_pos:
            return self._write_before_layer(module, hidden_states, int(layer_idx), kwargs)
        if int(layer_idx) in self.layer_to_pos:
            return self._read_before_layer(hidden_states, int(layer_idx))
        return hidden_states

    def register_hooks(self, model: Any) -> None:
        self.remove_hooks()
        layers = prev.get_layers(model)
        for layer_idx in self.hook_layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"hook layer={layer_idx} outside [0, {len(layers) - 1}]")

            def hook(module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any], *, layer: int = int(layer_idx)) -> Any:
                hidden = self._hidden_from_args(args, kwargs)
                if hidden is None:
                    return args, kwargs
                new_hidden = self.inject_before_layer(module, hidden, layer, kwargs)
                return self._replace_hidden_in_args(args, kwargs, new_hidden)

            self._handles.append(layers[int(layer_idx)].register_forward_pre_hook(hook, with_kwargs=True))


def adapter_set_context(adapter: PosNegWriteReadAdapter, batch: carrier.MemoryBatch, records: Sequence[prev.SampleRecord]) -> None:
    adapter.set_context(
        message_target_positions=batch.message_target_positions,
        inject_positions=batch.inject_positions,
        frame_groups=batch.frame_groups,
        evidence_frame_masks=oracle_masks_for_records(records, str(batch.prompt_last_indices.device)),
        gold_counts=batch.gold_counts,
        query_positions=batch.query_positions,
    )


def blank_diagnostics(write_layers: Sequence[int], read_layers: Sequence[int]) -> Dict[str, Any]:
    payload = sum_base.blank_diagnostics(read_layers)
    payload["oracle_mask_by_layer"] = {str(int(layer)): [] for layer in write_layers}
    payload["oracle_mask_count_by_layer"] = {str(int(layer)): 0.0 for layer in write_layers}
    payload["positive_frame_count_by_layer"] = {str(int(layer)): 0.0 for layer in write_layers}
    payload["negative_frame_count_by_layer"] = {str(int(layer)): 0.0 for layer in write_layers}
    payload["positive_stream_norm_by_layer"] = {str(int(layer)): 0.0 for layer in write_layers}
    payload["negative_stream_norm_by_layer"] = {str(int(layer)): 0.0 for layer in write_layers}
    payload["combined_stream_norm_by_layer"] = {str(int(layer)): 0.0 for layer in write_layers}
    payload["write_memory_norm_by_layer"] = {str(int(layer)): 0.0 for layer in write_layers}
    payload["read_memory_concat_norm_by_layer"] = {str(int(layer)): 0.0 for layer in read_layers}
    return payload


def select_count_logits(outputs: Any, prompt_last_indices: torch.Tensor, count_token_ids: Dict[int, int]) -> torch.Tensor:
    return prev.select_count_logits(outputs.logits, prompt_last_indices, count_token_ids)


@torch.no_grad()
def evaluate_model(
    *,
    args: argparse.Namespace,
    method: str,
    split_name: str,
    model: Any,
    processor: Any,
    adapter: Optional[PosNegWriteReadAdapter],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    count_token_ids: Dict[int, int],
    device: str,
    batch_size: int,
    seed: int,
    write_layers: Sequence[int],
    read_layers: Sequence[int],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    ce_total = 0.0
    n = 0
    count_min = min(COUNT_VALUES)
    message_group = canonical_group(str(args.message_token_group))
    inject_group = canonical_group(str(args.inject_token_group))

    if adapter is not None:
        adapter.to(device)
        adapter.eval()
        adapter.enabled = True
        adapter.register_hooks(model)

    try:
        batches = base.homogeneous_batches(indices, records, int(batch_size), seed=int(seed), shuffle_batches=False)
        for batch_num, batch_indices in enumerate(batches, start=1):
            batch_records = [records[int(idx)] for idx in batch_indices]
            batch = prepare_batch(
                args=args,
                records=batch_records,
                sample_indices=batch_indices,
                processor=processor,
                device=device,
            )
            if adapter is not None:
                adapter_set_context(adapter, batch, batch_records)
            outputs = model(**batch.inputs, use_cache=False)
            count_logits = select_count_logits(outputs, batch.prompt_last_indices, count_token_ids)
            gold_offsets = batch.gold_counts.long() - int(count_min)
            ce_vec = F.cross_entropy(count_logits, gold_offsets, reduction="none")
            ce_total += float(ce_vec.sum().detach().cpu().item())
            n += int(batch.gold_counts.numel())
            pred_offsets = count_logits.argmax(dim=-1)
            gold_logits, _best_wrong, margins = carrier.select_gold_logits_and_margins(count_logits, gold_offsets)
            logits_cpu = count_logits.detach().float().cpu()

            for row_idx, sample_idx in enumerate(batch_indices):
                sample_idx = int(sample_idx)
                record = records[sample_idx]
                oracle_mask = evidence_mask_for_record(record, seq_len=NUM_FRAMES)
                gold = int(record.gold_count)
                pred = int(pred_offsets[row_idx].detach().cpu().item()) + int(count_min)
                logits_list = [float(v) for v in logits_cpu[row_idx].tolist()]
                logits_map = {str(count): logits_list[count - int(count_min)] for count in COUNT_VALUES}
                diag = (
                    adapter.stats_for_row(row_idx)
                    if adapter is not None
                    else blank_diagnostics(write_layers, read_layers)
                )
                update_norm_by_layer = diag.get("update_norm_by_layer", {})
                message_norm_by_layer = diag.get("message_norm_by_layer", {})
                oracle_mask_count_by_layer = diag.get("oracle_mask_count_by_layer", {})
                positive_frame_count_by_layer = diag.get("positive_frame_count_by_layer", {})
                negative_frame_count_by_layer = diag.get("negative_frame_count_by_layer", {})
                positive_stream_norm_by_layer = diag.get("positive_stream_norm_by_layer", {})
                negative_stream_norm_by_layer = diag.get("negative_stream_norm_by_layer", {})
                combined_stream_norm_by_layer = diag.get("combined_stream_norm_by_layer", {})
                write_memory_norm_by_layer = diag.get("write_memory_norm_by_layer", {})
                read_memory_concat_norm_by_layer = diag.get("read_memory_concat_norm_by_layer", {})
                update_norm = base.finite_mean(update_norm_by_layer.values(), default=0.0)
                message_norm = sum_base.mean_layer_frame_value(message_norm_by_layer)
                positive_stream_norm = base.finite_mean(positive_stream_norm_by_layer.values(), default=0.0)
                negative_stream_norm = base.finite_mean(negative_stream_norm_by_layer.values(), default=0.0)
                combined_stream_norm = base.finite_mean(combined_stream_norm_by_layer.values(), default=0.0)
                write_memory_norm = base.finite_mean(write_memory_norm_by_layer.values(), default=0.0)
                read_memory_concat_norm = base.finite_mean(read_memory_concat_norm_by_layer.values(), default=0.0)
                oracle_layer_count = base.finite_mean(oracle_mask_count_by_layer.values(), default=float(sum(oracle_mask)))
                positive_frame_count = int(
                    round(base.finite_mean(positive_frame_count_by_layer.values(), default=float(sum(oracle_mask))))
                )
                negative_frame_count = int(
                    round(
                        base.finite_mean(
                            negative_frame_count_by_layer.values(),
                            default=float(NUM_FRAMES - int(sum(oracle_mask))),
                        )
                    )
                )
                pred_offset = pred - int(count_min)
                rows.append(
                    {
                        "method": str(method),
                        "sample_id": record.sample_id,
                        "sample_index": int(sample_idx),
                        "seq_len": NUM_FRAMES,
                        "gold_count": int(gold),
                        "evidence_count": int(record.evidence_count),
                        "oracle_mask_count": int(sum(oracle_mask)),
                        "oracle_layer_mask_count": float(oracle_layer_count),
                        "positive_frame_count": int(positive_frame_count),
                        "negative_frame_count": int(negative_frame_count),
                        "pred_count": int(pred),
                        "correct": int(pred == gold),
                        "margin": float(margins[row_idx].detach().cpu().item()),
                        "gold_logit": float(gold_logits[row_idx].detach().cpu().item()),
                        "pred_logit": logits_list[pred_offset] if 0 <= pred_offset < len(logits_list) else math.nan,
                        "candidate_logits_json": base.json_compact(logits_map),
                        "split": str(split_name),
                        "ce": float(ce_vec[row_idx].detach().cpu().item()),
                        "readout_mode": str(getattr(adapter, "readout_mode", "none")) if adapter is not None else "none",
                        "message_token_group": message_group,
                        "inject_token_group": inject_group,
                        "message_target_positions_json": base.json_compact(batch.message_target_positions[row_idx]),
                        "inject_positions_json": base.json_compact(batch.inject_positions[row_idx]),
                        "update_norm": float(update_norm) if adapter is not None else "",
                        "message_norm": float(message_norm) if adapter is not None else "",
                        "positive_stream_norm": float(positive_stream_norm) if adapter is not None else "",
                        "negative_stream_norm": float(negative_stream_norm) if adapter is not None else "",
                        "combined_stream_norm": float(combined_stream_norm) if adapter is not None else "",
                        "write_memory_norm": float(write_memory_norm) if adapter is not None else "",
                        "read_memory_concat_norm": float(read_memory_concat_norm) if adapter is not None else "",
                        "update_norm_by_layer_json": base.json_compact(update_norm_by_layer) if adapter is not None else "",
                        "message_norm_by_layer_json": base.json_compact(message_norm_by_layer) if adapter is not None else "",
                        "raw_message_norm_by_layer_json": base.json_compact(diag.get("raw_message_norm_by_layer", {}))
                        if adapter is not None
                        else "",
                        "positive_frame_count_by_layer_json": base.json_compact(positive_frame_count_by_layer)
                        if adapter is not None
                        else "",
                        "negative_frame_count_by_layer_json": base.json_compact(negative_frame_count_by_layer)
                        if adapter is not None
                        else "",
                        "positive_stream_norm_by_layer_json": base.json_compact(positive_stream_norm_by_layer)
                        if adapter is not None
                        else "",
                        "negative_stream_norm_by_layer_json": base.json_compact(negative_stream_norm_by_layer)
                        if adapter is not None
                        else "",
                        "combined_stream_norm_by_layer_json": base.json_compact(combined_stream_norm_by_layer)
                        if adapter is not None
                        else "",
                        "write_memory_norm_by_layer_json": base.json_compact(write_memory_norm_by_layer)
                        if adapter is not None
                        else "",
                        "read_memory_concat_norm_by_layer_json": base.json_compact(read_memory_concat_norm_by_layer)
                        if adapter is not None
                        else "",
                        "oracle_mask_by_layer_json": base.json_compact(diag.get("oracle_mask_by_layer", {}))
                        if adapter is not None
                        else "",
                        "gate_values_by_layer_json": base.json_compact(diag.get("gate_values_by_layer", {}))
                        if adapter is not None
                        else "",
                        "gate_sum_by_layer_json": base.json_compact(diag.get("gate_sum_by_layer", {}))
                        if adapter is not None
                        else "",
                        "oracle_mask_count_by_layer_json": base.json_compact(oracle_mask_count_by_layer)
                        if adapter is not None
                        else "",
                        "message_mode_by_layer_json": base.json_compact(diag.get("message_mode_by_layer", {}))
                        if adapter is not None
                        else "",
                        "frame_token_counts_json": base.json_compact(batch.frame_token_counts[row_idx]),
                        "evidence_frame_mask_json": base.json_compact(oracle_mask),
                        "token_selection_ok": int(bool(batch.token_selection_ok[row_idx])),
                        "token_selection_error": str(batch.token_selection_errors[row_idx]),
                        "frame_grouping_ok": int(bool(batch.frame_grouping_ok[row_idx])),
                        "frame_grouping_error": str(batch.frame_grouping_errors[row_idx]),
                    }
                )
            if adapter is not None:
                adapter.clear_context()
            if batch_num == 1 or batch_num % 25 == 0:
                print(f"  eval {method} {split_name}: {min(batch_num * int(batch_size), len(indices))}/{len(indices)}")
    finally:
        if adapter is not None:
            adapter.remove_hooks()

    y_true = [int(row["gold_count"]) for row in rows]
    y_pred = [int(row["pred_count"]) for row in rows]
    return {
        "rows": rows,
        "ce": ce_total / max(1, n),
        "accuracy": prev.accuracy(y_true, y_pred),
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
    hidden_size: int,
    write_layers: Sequence[int],
    read_layers: Sequence[int],
    readout_mode: str,
    device: str,
) -> Tuple[PosNegWriteReadAdapter, List[Dict[str, Any]], Dict[str, Any], Path]:
    adapter = PosNegWriteReadAdapter(
        hidden_size=int(hidden_size),
        d_mem=int(args.d_mem),
        write_layers=[int(x) for x in write_layers],
        read_layers=[int(x) for x in read_layers],
        gamma_init=float(args.gamma_init),
        message_mode=str(args.message_mode),
        readout_mode=str(readout_mode),
        gate_mode=str(args.gate_mode),
        streams=str(args.streams),
        gate_source=str(args.gate_source),
        gate_hard=bool(args.gate_hard),
        oracle_mask_noise=float(args.oracle_mask_noise),
    ).to(device)
    if args.init_streams_from is not None:
        payload = torch.load(Path(args.init_streams_from), map_location="cpu")
        source_sd = payload.get("adapter_state_dict", payload)
        stream_sd = {k: v for k, v in source_sd.items() if k.startswith(STREAM_PARAM_PREFIXES)}
        if not stream_sd:
            raise RuntimeError(f"No stream parameters found in {args.init_streams_from}")
        missing, unexpected = adapter.load_state_dict(stream_sd, strict=False)
        loaded = sorted(stream_sd)
        print(f"Initialized {len(loaded)} stream tensors from {args.init_streams_from} (e.g. {loaded[:3]})")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading stream init: {unexpected}")
    if bool(args.freeze_streams):
        frozen = []
        for name, param in adapter.named_parameters():
            if name.startswith(STREAM_PARAM_PREFIXES):
                param.requires_grad = False
                frozen.append(name)
        if not frozen:
            raise RuntimeError("--freeze-streams found no stream parameters to freeze")
        print(f"Froze {len(frozen)} stream tensors; trainable now: "
              f"{[n for n, q in adapter.named_parameters() if q.requires_grad][:6]}")
    carrier.verify_trainable_parameters(model, adapter)
    optimizer = torch.optim.AdamW(
        [param for param in adapter.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "posneg_write_read_adapter_best.pt"
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_acc = -math.inf
    best_val_ce = math.inf
    backward_diag: Dict[str, Any] = {}
    count_min = min(COUNT_VALUES)

    for epoch in range(1, int(args.epochs) + 1):
        adapter.train()
        adapter.enabled = True
        train_batches = base.homogeneous_batches(
            train_indices,
            records,
            int(args.batch_size),
            seed=int(args.seed) + epoch * 9973,
            shuffle_batches=True,
        )
        optimizer.zero_grad(set_to_none=True)
        train_ce_total = 0.0
        train_loss_total = 0.0
        train_energy_total = 0.0
        train_gate_bce_total = 0.0
        train_gate_count_total = 0.0
        train_correct = 0
        train_n = 0
        train_steps = 0
        backward_steps = 0
        skipped = 0

        try:
            adapter.register_hooks(model)
            for step, batch_indices in enumerate(train_batches, start=1):
                batch_records = [records[int(idx)] for idx in batch_indices]
                batch = prepare_batch(
                    args=args,
                    records=batch_records,
                    sample_indices=batch_indices,
                    processor=processor,
                    device=device,
                )
                if (
                    not any(batch.message_target_positions)
                    or not any(batch.inject_positions)
                    or not any(batch.frame_grouping_ok)
                ):
                    skipped += 1
                adapter_set_context(adapter, batch, batch_records)
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = select_count_logits(outputs, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - int(count_min)
                ce = F.cross_entropy(count_logits, gold_offsets)
                m_loss = carrier.margin_loss(count_logits, gold_offsets, float(args.margin_target))
                update_energy = adapter.update_energy_for_loss(count_logits.device)
                gate_bce, gate_count_loss = adapter.gate_supervision_loss(count_logits.device)
                loss = (
                    float(args.lambda_ce) * ce
                    + float(args.lambda_margin) * m_loss
                    + float(args.lambda_update_energy) * update_energy
                )
                if str(args.gate_mode) == "learned":
                    loss = loss + float(args.lambda_mask) * gate_bce + float(args.lambda_count) * gate_count_loss
                torch.autograd.backward(loss / max(1, int(args.grad_accum)))
                if not backward_diag:
                    backward_diag = carrier.first_backward_diagnostics(model, adapter)
                    print(f"  first backward diagnostics: {base.json_compact(backward_diag)}")
                preds = count_logits.argmax(dim=-1) + int(count_min)
                train_correct += int((preds == batch.gold_counts.long()).sum().detach().cpu().item())
                train_n += int(batch.gold_counts.numel())
                train_ce_total += float(ce.detach().cpu().item())
                train_loss_total += float(loss.detach().cpu().item())
                train_energy_total += float(update_energy.detach().cpu().item())
                train_gate_bce_total += float(gate_bce.detach().cpu().item())
                train_gate_count_total += float(gate_count_loss.detach().cpu().item())
                train_steps += 1
                backward_steps += 1
                adapter.clear_context()
                if backward_steps % max(1, int(args.grad_accum)) == 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 25 == 0:
                    print(
                        f"  {METHOD} epoch={epoch} step={step}/{len(train_batches)} "
                        f"train_ce={train_ce_total / max(1, train_steps):.4f} "
                        f"train_acc={train_correct / max(1, train_n):.4f} "
                        f"energy={train_energy_total / max(1, train_steps):.6f}"
                    )
            if backward_steps and backward_steps % max(1, int(args.grad_accum)) != 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        finally:
            adapter.remove_hooks()

        val_eval = evaluate_model(
            args=args,
            method=METHOD,
            split_name="val",
            model=model,
            processor=processor,
            adapter=adapter,
            records=records,
            indices=val_indices,
            count_token_ids=count_token_ids,
            device=device,
            batch_size=int(args.batch_size),
            seed=int(args.seed) + 444 + epoch,
            write_layers=write_layers,
            read_layers=read_layers,
        )
        row = {
            "method": METHOD,
            "readout_mode": str(adapter.readout_mode),
            "message_token_group": canonical_group(str(args.message_token_group)),
            "inject_token_group": canonical_group(str(args.inject_token_group)),
            "epoch": int(epoch),
            "train_ce": train_ce_total / max(1, train_steps),
            "train_loss": train_loss_total / max(1, train_steps),
            "train_update_energy": train_energy_total / max(1, train_steps),
            "train_gate_bce": train_gate_bce_total / max(1, train_steps),
            "train_gate_count_loss": train_gate_count_total / max(1, train_steps),
            "gate_mode": str(args.gate_mode),
            "streams": str(args.streams),
            "train_accuracy": train_correct / max(1, train_n),
            "train_steps": int(train_steps),
            "skipped_batches_with_missing_localization": int(skipped),
            "val_ce": float(val_eval["ce"]),
            "val_accuracy": float(val_eval["accuracy"]),
            "adapter_parameter_norm": carrier.adapter_parameter_norm(adapter),
            "gamma_json": base.json_compact([float(x) for x in adapter.gamma.detach().float().cpu().tolist()]),
        }
        history.append(row)
        print(
            f"  {METHOD} epoch={epoch} train_ce={row['train_ce']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} val_ce={row['val_ce']:.4f} "
            f"val_acc={row['val_accuracy']:.4f}"
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
                    "write_layers": [int(x) for x in write_layers],
                    "read_layers": [int(x) for x in read_layers],
                    "inject_layers": [int(x) for x in read_layers],
                    "variant": METHOD,
                    "method": METHOD,
                    "message_mode": str(args.message_mode),
                    "readout_mode": str(adapter.readout_mode),
                    "message_token_group": canonical_group(str(args.message_token_group)),
                    "inject_token_group": canonical_group(str(args.inject_token_group)),
                },
                checkpoint_path,
            )
    if best_state is not None:
        adapter.load_state_dict(best_state)
    return adapter, history, backward_diag, checkpoint_path


def rank_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUC with tie-averaged ranks (numpy only)."""
    y = np.asarray([int(x) for x in labels])
    s = np.asarray([float(x) for x in scores])
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0 or len(s) != len(y):
        return float("nan")
    _unique, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank_per_value = cum - (counts - 1) / 2.0
    ranks = avg_rank_per_value[inverse]
    rank_sum_pos = float(ranks[y == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def gate_auc_by_layer(metrics_rows: Sequence[Dict[str, Any]], method: str) -> Dict[str, Any]:
    """Pooled per-write-layer evidence-detection AUC of the gate over eval rows."""
    scores_by_layer: Dict[str, List[float]] = defaultdict(list)
    labels_by_layer: Dict[str, List[int]] = defaultdict(list)
    for row in metrics_rows:
        if str(row.get("method")) != str(method):
            continue
        try:
            gates = json.loads(str(row.get("gate_values_by_layer_json") or "") or "{}")
            mask = json.loads(str(row.get("evidence_frame_mask_json") or "") or "[]")
        except (TypeError, ValueError):
            continue
        if not isinstance(gates, dict) or not mask:
            continue
        for layer, values in gates.items():
            if not isinstance(values, list) or len(values) != len(mask):
                continue
            scores_by_layer[str(layer)].extend(float(v) for v in values)
            labels_by_layer[str(layer)].extend(int(x) for x in mask)
    payload: Dict[str, Any] = {}
    for layer in sorted(scores_by_layer, key=lambda x: int(x) if str(x).isdigit() else 0):
        payload[layer] = rank_auc(scores_by_layer[layer], labels_by_layer[layer])
    finite = [v for v in payload.values() if isinstance(v, float) and math.isfinite(v)]
    payload["mean"] = float(np.mean(finite)) if finite else float("nan")
    return payload


def infer_column(rows: Sequence[Dict[str, Any]], key: str, default: str = "none") -> str:
    values = sorted({str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()})
    if not values:
        return str(default)
    if len(values) == 1:
        return values[0]
    return base.json_compact(values)


def prediction_histogram(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    hist = {str(count): 0 for count in COUNT_VALUES}
    for row in rows:
        key = str(int(row["pred_count"]))
        hist[key] = hist.get(key, 0) + 1
    return hist


def method_rows(rows: Sequence[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("method") == method]


def summarize_method(
    rows: Sequence[Dict[str, Any]],
    *,
    method: str,
    train_history: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    correct = [int(row["correct"]) for row in rows]
    train_last = dict(train_history[-1]) if train_history else {}
    return {
        "method": str(method),
        "readout_mode": infer_column(rows, "readout_mode", default="none"),
        "message_token_group": infer_column(rows, "message_token_group", default=canonical_group("all_question")),
        "inject_token_group": infer_column(rows, "inject_token_group", default=canonical_group("last_token")),
        "n": len(rows),
        "accuracy": float(np.mean(correct)) if correct else math.nan,
        "mean_margin": base.finite_mean(row.get("margin") for row in rows),
        "mean_gold_logit": base.finite_mean(row.get("gold_logit") for row in rows),
        "mean_pred_count": base.finite_mean(row.get("pred_count") for row in rows),
        "mean_update_norm": base.finite_mean((row.get("update_norm") for row in rows), default=0.0)
        if method == METHOD
        else 0.0,
        "mean_message_norm": base.finite_mean((row.get("message_norm") for row in rows), default=0.0)
        if method == METHOD
        else 0.0,
        "mean_positive_stream_norm": base.finite_mean((row.get("positive_stream_norm") for row in rows), default=0.0)
        if method == METHOD
        else 0.0,
        "mean_negative_stream_norm": base.finite_mean((row.get("negative_stream_norm") for row in rows), default=0.0)
        if method == METHOD
        else 0.0,
        "mean_combined_stream_norm": base.finite_mean((row.get("combined_stream_norm") for row in rows), default=0.0)
        if method == METHOD
        else 0.0,
        "mean_write_memory_norm": base.finite_mean((row.get("write_memory_norm") for row in rows), default=0.0)
        if method == METHOD
        else 0.0,
        "mean_read_memory_concat_norm": base.finite_mean(
            (row.get("read_memory_concat_norm") for row in rows), default=0.0
        )
        if method == METHOD
        else 0.0,
        "train_accuracy": train_last.get("train_accuracy", ""),
        "val_accuracy": train_last.get("val_accuracy", ""),
        "val_ce": train_last.get("val_ce", ""),
    }


def accuracy_by_evidence_count(rows: Sequence[Dict[str, Any]], method: str, counts: Sequence[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for count in counts:
        count_rows = [row for row in rows if row["method"] == method and int(row["gold_count"]) == int(count)]
        correct = [int(row["correct"]) for row in count_rows]
        out.append(
            {
                "method": str(method),
                "evidence_count": int(count),
                "gold_count": int(count),
                "n": len(count_rows),
                "accuracy": float(np.mean(correct)) if correct else math.nan,
                "mean_margin": base.finite_mean(row.get("margin") for row in count_rows),
                "mean_pred_count": base.finite_mean(row.get("pred_count") for row in count_rows),
                "prediction_histogram": base.json_compact(prediction_histogram(count_rows)),
                "mean_oracle_mask_count": base.finite_mean(row.get("oracle_mask_count") for row in count_rows),
                "mean_update_norm": base.finite_mean((row.get("update_norm") for row in count_rows), default=0.0)
                if method == METHOD
                else 0.0,
                "mean_message_norm": base.finite_mean((row.get("message_norm") for row in count_rows), default=0.0)
                if method == METHOD
                else 0.0,
                "mean_positive_stream_norm": base.finite_mean(
                    (row.get("positive_stream_norm") for row in count_rows), default=0.0
                )
                if method == METHOD
                else 0.0,
                "mean_negative_stream_norm": base.finite_mean(
                    (row.get("negative_stream_norm") for row in count_rows), default=0.0
                )
                if method == METHOD
                else 0.0,
                "mean_combined_stream_norm": base.finite_mean(
                    (row.get("combined_stream_norm") for row in count_rows), default=0.0
                )
                if method == METHOD
                else 0.0,
                "mean_write_memory_norm": base.finite_mean(
                    (row.get("write_memory_norm") for row in count_rows), default=0.0
                )
                if method == METHOD
                else 0.0,
                "mean_read_memory_concat_norm": base.finite_mean(
                    (row.get("read_memory_concat_norm") for row in count_rows), default=0.0
                )
                if method == METHOD
                else 0.0,
            }
        )
    return out


def comparison_by_evidence_count(accuracy_rows: Sequence[Dict[str, Any]], counts: Sequence[int]) -> List[Dict[str, Any]]:
    by_key = {(row["method"], int(row["evidence_count"])): row for row in accuracy_rows}
    out: List[Dict[str, Any]] = []
    for count in counts:
        base_row = by_key.get((BASELINE, int(count)), {})
        oracle_row = by_key.get((METHOD, int(count)), {})
        base_acc = base.finite_float(base_row.get("accuracy"))
        oracle_acc = base.finite_float(oracle_row.get("accuracy"))
        out.append(
            {
                "evidence_count": int(count),
                "gold_count": int(count),
                "baseline_accuracy": "" if base_acc is None else float(base_acc),
                "oracle_posneg_write_read_accuracy": "" if oracle_acc is None else float(oracle_acc),
                "delta_accuracy": ""
                if base_acc is None or oracle_acc is None
                else float(oracle_acc) - float(base_acc),
                "baseline_mean_pred": base_row.get("mean_pred_count", ""),
                "oracle_posneg_write_read_mean_pred": oracle_row.get("mean_pred_count", ""),
                "baseline_mean_margin": base_row.get("mean_margin", ""),
                "oracle_posneg_write_read_mean_margin": oracle_row.get("mean_margin", ""),
            }
        )
    return out


def candidate_logits(row: Dict[str, Any]) -> Dict[str, float]:
    payload = base.parse_json_field(row, "candidate_logits_json", {})
    if isinstance(payload, dict):
        return {str(k): float(v) for k, v in payload.items() if base.finite_float(v) is not None}
    if isinstance(payload, list):
        return {str(i): float(v) for i, v in enumerate(payload) if base.finite_float(v) is not None}
    return {}


def save_combined_line_plot(
    path: Path,
    accuracy_rows: Sequence[Dict[str, Any]],
    *,
    y_key: str,
    ylabel: str,
    title: str,
    counts: Sequence[int],
) -> None:
    plt.figure(figsize=(7.2, 4.5))
    for method in [BASELINE, METHOD]:
        by_count = {int(row["evidence_count"]): row for row in accuracy_rows if row["method"] == method}
        ys = [float(by_count.get(int(count), {}).get(y_key, math.nan)) for count in counts]
        plt.plot(counts, ys, marker="o", linewidth=1.8, label=method)
    plt.xlabel("Evidence count / gold count")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(counts)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def confusion_matrix(rows: Sequence[Dict[str, Any]], counts: Sequence[int]) -> np.ndarray:
    mat = np.zeros((len(counts), len(COUNT_VALUES)), dtype=float)
    count_list = [int(x) for x in counts]
    for row in rows:
        gold = int(row["gold_count"])
        pred = int(row["pred_count"])
        if gold in count_list and pred in COUNT_VALUES:
            mat[count_list.index(gold), COUNT_VALUES.index(pred)] += 1.0
    return mat


def save_confusion(path: Path, rows: Sequence[Dict[str, Any]], counts: Sequence[int], title: str) -> None:
    mat = confusion_matrix(rows, counts)
    fig, ax = plt.subplots(figsize=(7.3, 5.4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(np.arange(len(COUNT_VALUES)))
    ax.set_xticklabels(COUNT_VALUES)
    ax.set_yticks(np.arange(len(counts)))
    ax.set_yticklabels(counts)
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("Gold count / evidence count")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_combined_confusions(path: Path, rows: Sequence[Dict[str, Any]], counts: Sequence[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, method in zip(axes, [BASELINE, METHOD]):
        mat = confusion_matrix([row for row in rows if row["method"] == method], counts)
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(np.arange(len(COUNT_VALUES)))
        ax.set_xticklabels(COUNT_VALUES)
        ax.set_yticks(np.arange(len(counts)))
        ax.set_yticklabels(counts)
        ax.set_xlabel("Predicted count")
        ax.set_title(method)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] > 0:
                    ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Gold count / evidence count")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_candidate_logit_curves(path: Path, rows: Sequence[Dict[str, Any]], method: str, counts: Sequence[int]) -> None:
    plt.figure(figsize=(7.4, 4.8))
    for count in counts:
        count_rows = [row for row in rows if row["method"] == method and int(row["gold_count"]) == int(count)]
        if not count_rows:
            continue
        means: List[float] = []
        for candidate in COUNT_VALUES:
            vals = [candidate_logits(row).get(str(candidate), math.nan) for row in count_rows]
            means.append(base.finite_mean(vals))
        plt.plot(COUNT_VALUES, means, marker="o", linewidth=1.3, label=f"gold {count}")
    plt.xlabel("Candidate count")
    plt.ylabel("Mean logit")
    plt.title(f"Candidate Logit Curves: {method}")
    plt.xticks(COUNT_VALUES)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def make_plots(
    output_dir: Path,
    metrics_rows: Sequence[Dict[str, Any]],
    accuracy_rows: Sequence[Dict[str, Any]],
    comparison_rows: Sequence[Dict[str, Any]],
    counts: Sequence[int],
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_combined_line_plot(
        plots_dir / "combined_accuracy_vs_evidence_count.png",
        accuracy_rows,
        y_key="accuracy",
        ylabel="Accuracy",
        title="Accuracy vs Evidence Count",
        counts=counts,
    )
    save_combined_line_plot(
        plots_dir / "combined_margin_vs_evidence_count.png",
        accuracy_rows,
        y_key="mean_margin",
        ylabel="Mean margin",
        title="Margin vs Evidence Count",
        counts=counts,
    )

    plt.figure(figsize=(7.2, 4.8))
    plt.plot([0, max(counts)], [0, max(counts)], linestyle="--", color="black", linewidth=1.2, label="perfect y=x")
    for method in [BASELINE, METHOD]:
        by_count = {int(row["evidence_count"]): row for row in accuracy_rows if row["method"] == method}
        xs = [int(count) for count in counts]
        ys = [float(by_count.get(int(count), {}).get("mean_pred_count", math.nan)) for count in counts]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=method)
    plt.xlabel("Gold count")
    plt.ylabel("Mean predicted count")
    plt.title("Mean Predicted Count vs Gold Count")
    plt.xticks(counts)
    plt.yticks(COUNT_VALUES)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_predicted_count_vs_gold_count.png", dpi=180, bbox_inches="tight")
    plt.close()

    base_rows = [row for row in metrics_rows if row["method"] == BASELINE]
    oracle_rows = [row for row in metrics_rows if row["method"] == METHOD]
    save_confusion(plots_dir / "predicted_count_confusion_matrix_baseline.png", base_rows, counts, "Baseline Confusion Matrix")
    save_confusion(
        plots_dir / "predicted_count_confusion_matrix_oracle_posneg_write_read_adapter.png",
        oracle_rows,
        counts,
        "Oracle PosNeg Write/Read Adapter Confusion Matrix",
    )
    save_combined_confusions(plots_dir / "combined_confusion_matrices.png", metrics_rows, counts)

    plt.figure(figsize=(7.2, 4.3))
    xs = [int(row["evidence_count"]) for row in comparison_rows]
    ys = [float(row["delta_accuracy"]) if base.finite_float(row.get("delta_accuracy")) is not None else math.nan for row in comparison_rows]
    colors = ["#2ca02c" if base.finite_float(y) is not None and float(y) >= 0 else "#d62728" for y in ys]
    plt.bar(xs, ys, color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Evidence count")
    plt.ylabel("Oracle posneg write/read minus baseline accuracy")
    plt.title("Delta Accuracy")
    plt.xticks(counts)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "delta_accuracy_oracle_posneg_write_read_minus_baseline.png", dpi=180, bbox_inches="tight")
    plt.close()

    by_count = {int(row["evidence_count"]): row for row in accuracy_rows if row["method"] == METHOD}
    diagnostic_specs = [
        ("mean_update_norm", "update_norm_vs_evidence_count.png", "Mean update norm", "Oracle PosNeg Write/Read Update Norm"),
        ("mean_message_norm", "message_norm_vs_evidence_count.png", "Mean message norm", "Message Norm"),
        (
            "mean_positive_stream_norm",
            "positive_stream_norm_vs_evidence_count.png",
            "Mean positive stream norm",
            "Positive Stream Norm",
        ),
        (
            "mean_negative_stream_norm",
            "negative_stream_norm_vs_evidence_count.png",
            "Mean negative stream norm",
            "Negative Stream Norm",
        ),
        (
            "mean_combined_stream_norm",
            "combined_stream_norm_vs_evidence_count.png",
            "Mean combined stream norm",
            "Combined Stream Norm",
        ),
        (
            "mean_write_memory_norm",
            "write_memory_norm_vs_evidence_count.png",
            "Mean write memory norm",
            "Write Memory Norm",
        ),
        (
            "mean_oracle_mask_count",
            "oracle_mask_count_vs_gold_count.png",
            "Mean oracle selected frames",
            "Oracle Mask Count vs Gold Count",
        ),
    ]
    for key, filename, ylabel, title in diagnostic_specs:
        plt.figure(figsize=(7.2, 4.3))
        ys_diag = [float(by_count.get(int(count), {}).get(key, math.nan)) for count in counts]
        plt.plot(counts, ys_diag, marker="o")
        if key == "mean_oracle_mask_count":
            plt.plot([0, max(counts)], [0, max(counts)], linestyle="--", color="black", linewidth=1.1)
        plt.xlabel("Gold count")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(counts)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=180, bbox_inches="tight")
        plt.close()

    save_candidate_logit_curves(
        plots_dir / "candidate_logit_curves_by_evidence_count_baseline.png",
        metrics_rows,
        BASELINE,
        counts,
    )
    save_candidate_logit_curves(
        plots_dir / "candidate_logit_curves_by_evidence_count_oracle_posneg_write_read_adapter.png",
        metrics_rows,
        METHOD,
        counts,
    )


def flatten_numeric(value: Any) -> List[float]:
    values: List[float] = []
    if isinstance(value, dict):
        for item in value.values():
            values.extend(flatten_numeric(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(flatten_numeric(item))
    elif base.finite_float(value) is not None:
        values.append(float(value))
    return values


def numeric_values_from_json_field(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        payload = base.parse_json_field(row, key, {})
        values.extend(flatten_numeric(payload))
    return values


def message_mode_resolution(metrics_rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = Counter()
    for row in metrics_rows:
        payload = base.parse_json_field(row, "message_mode_by_layer_json", {})
        if not isinstance(payload, dict):
            continue
        for value in payload.values():
            if isinstance(value, list):
                for item in value:
                    counts[str(item)] += 1
            elif str(value):
                counts[str(value)] += 1
    return dict(counts)


def mean_pred_mae(accuracy_rows: Sequence[Dict[str, Any]], method: str) -> float:
    vals = []
    for row in accuracy_rows:
        if row.get("method") != method:
            continue
        pred = base.finite_float(row.get("mean_pred_count"))
        gold = base.finite_float(row.get("gold_count"))
        if pred is not None and gold is not None:
            vals.append(abs(float(pred) - float(gold)))
    return base.finite_mean(vals)


def write_diagnostics(
    *,
    output_dir: Path,
    model: Any,
    adapter: Optional[PosNegWriteReadAdapter],
    train_history: Sequence[Dict[str, Any]],
    backward_diag: Dict[str, Any],
    metrics_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    oracle_validation: Dict[str, Any],
) -> Dict[str, Any]:
    model_trainable_tensors = sum(int(param.requires_grad) for param in model.parameters())
    adapter_trainable_tensors = 0 if adapter is None else sum(int(param.requires_grad) for param in adapter.parameters())
    adapter_trainable_params = 0 if adapter is None else sum(int(param.numel()) for param in adapter.parameters() if param.requires_grad)
    failed_ids = sorted(
        {
            str(row.get("sample_id"))
            for row in metrics_rows
            if int(row.get("token_selection_ok", 0)) == 0 or int(row.get("frame_grouping_ok", 0)) == 0
        }
    )
    oracle_rows = method_rows(metrics_rows, METHOD)
    localization_rows = oracle_rows or metrics_rows
    message_counts = [len(base.parse_json_field(row, "message_target_positions_json", [])) for row in localization_rows]
    inject_counts = [len(base.parse_json_field(row, "inject_positions_json", [])) for row in localization_rows]
    update_values = numeric_values_from_json_field(oracle_rows, "update_norm_by_layer_json")
    positive_stream_values = numeric_values_from_json_field(oracle_rows, "positive_stream_norm_by_layer_json")
    negative_stream_values = numeric_values_from_json_field(oracle_rows, "negative_stream_norm_by_layer_json")
    combined_stream_values = numeric_values_from_json_field(oracle_rows, "combined_stream_norm_by_layer_json")
    write_memory_values = numeric_values_from_json_field(oracle_rows, "write_memory_norm_by_layer_json")
    finite_updates = bool(update_values) and all(math.isfinite(float(value)) for value in update_values)
    nonzero_updates = any(abs(float(value)) > 1e-12 for value in update_values)
    finite_positive_streams = bool(positive_stream_values) and all(
        math.isfinite(float(value)) for value in positive_stream_values
    )
    finite_negative_streams = bool(negative_stream_values) and all(
        math.isfinite(float(value)) for value in negative_stream_values
    )
    finite_combined_streams = bool(combined_stream_values) and all(
        math.isfinite(float(value)) for value in combined_stream_values
    )
    finite_write_memories = bool(write_memory_values) and all(math.isfinite(float(value)) for value in write_memory_values)
    score_fields = sorted(
        {
            key
            for row in metrics_rows
            for key in row.keys()
            if "matrix_score" in str(key) or str(key).startswith("gate_") or "query_" in str(key)
        }
    )
    param_names = [] if adapter is None else [name for name, _param in adapter.named_parameters()]
    forbidden_param_names = [
        name
        for name in param_names
        if any(part in name for part in ("w_q", "w_k", "w_v", "gate", "readout", "key", "query"))
    ]
    allowed_trainable_ok = True
    trainable_param_names = [] if adapter is None else [name for name, param in adapter.named_parameters() if param.requires_grad]
    for name in trainable_param_names:
        if not (name == "gamma" or name.startswith("w_pos.") or name.startswith("w_neg.") or name.startswith("w_read.")):
            allowed_trainable_ok = False
    mode_counts = {} if adapter is None else dict(adapter.message_mode_counts)
    exact_message_rows = sum(int(value) for key, value in mode_counts.items() if str(key).endswith(":exact"))
    proxy_message_rows = sum(int(value) for key, value in mode_counts.items() if str(key).endswith(":proxy"))
    write_layers = [] if adapter is None else [int(layer) for layer in adapter.write_layers]
    read_layers = [] if adapter is None else [int(layer) for layer in adapter.read_layers]
    hooks_ok = bool(
        adapter is None
        or (
            all(int(adapter.write_hook_fire_counts.get(int(layer), 0)) > 0 for layer in write_layers)
            and all(int(adapter.read_hook_fire_counts.get(int(layer), 0)) > 0 for layer in read_layers)
        )
    )
    message_layers_seen = {
        int(str(key).split(":", 1)[0])
        for key in mode_counts
        if str(key).split(":", 1)[0].lstrip("-").isdigit()
    }
    update_layers_seen = {
        int(layer)
        for row in oracle_rows
        for layer in base.parse_json_field(row, "update_norm_by_layer_json", {}).keys()
        if str(layer).lstrip("-").isdigit()
    }
    messages_only_from_write_layers = bool(adapter is None or message_layers_seen <= set(write_layers))
    updates_only_on_read_layers = bool(adapter is None or update_layers_seen <= set(read_layers))
    all_question_localization_ok = all(
        len(base.parse_json_field(row, "message_target_positions_json", [])) > 0
        for row in localization_rows
    )
    last_token_localization_ok = all(
        len(base.parse_json_field(row, "inject_positions_json", [])) > 0
        for row in localization_rows
    )
    mask_rows = [
        row
        for row in metrics_rows
        if isinstance(base.parse_json_field(row, "evidence_frame_mask_json", []), list)
        and len(base.parse_json_field(row, "evidence_frame_mask_json", [])) == NUM_FRAMES
    ]
    mask_count_equals_gold = all(int(row.get("oracle_mask_count", -1)) == int(row.get("gold_count", -2)) for row in mask_rows)
    positive_count_equals_gold = all(
        int(row.get("positive_frame_count", -1)) == int(row.get("gold_count", -2)) for row in mask_rows
    )
    negative_count_equals_distractors = all(
        int(row.get("negative_frame_count", -1)) == NUM_FRAMES - int(row.get("gold_count", -2)) for row in mask_rows
    )
    mean_selected = base.finite_mean((row.get("oracle_mask_count") for row in mask_rows), default=0.0)
    mean_gold = base.finite_mean((row.get("gold_count") for row in mask_rows), default=0.0)
    masks_binary = all(
        all(int(x) in (0, 1) for x in base.parse_json_field(row, "evidence_frame_mask_json", []))
        for row in mask_rows
    )
    distractor_frames_zero = all(
        len(base.parse_json_field(row, "evidence_frame_mask_json", [])) == NUM_FRAMES
        and int(sum(int(x) for x in base.parse_json_field(row, "evidence_frame_mask_json", []))) == int(row["gold_count"])
        for row in mask_rows
    )
    nonfinite_fields: List[Dict[str, Any]] = []
    for row in metrics_rows:
        for field in [
            "margin",
            "gold_logit",
            "pred_logit",
            "ce",
            "update_norm",
            "message_norm",
            "positive_stream_norm",
            "negative_stream_norm",
            "combined_stream_norm",
            "write_memory_norm",
            "read_memory_concat_norm",
        ]:
            value = row.get(field, "")
            if value == "":
                continue
            if base.finite_float(value) is None:
                nonfinite_fields.append(
                    {"sample_id": row.get("sample_id"), "method": row.get("method"), "field": field, "value": value}
                )
                if len(nonfinite_fields) >= 50:
                    break
        if len(nonfinite_fields) >= 50:
            break
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "diagnostic_upper_bound": 1,
        "oracle_mask_is_valid_inference_method": 0,
        "qwen_frozen": int(model_trainable_tensors == 0),
        "model_trainable_tensors": int(model_trainable_tensors),
        "adapter_trainable_tensors": int(adapter_trainable_tensors),
        "adapter_trainable_params": int(adapter_trainable_params),
        "adapter_trainable_parameter_names": trainable_param_names,
        "only_adapter_params_trainable": int(model_trainable_tensors == 0 and adapter_trainable_tensors > 0)
        if adapter is not None
        else "",
        "only_w_pos_w_neg_w_read_gamma_trainable": int(bool(adapter is not None and allowed_trainable_ok)),
        "write_layers": write_layers,
        "read_layers": read_layers,
        "message_token_group": canonical_group(str(args.message_token_group)),
        "inject_token_group": canonical_group(str(args.inject_token_group)),
        "readout_mode": "none" if adapter is None else str(getattr(adapter, "readout_mode", "unknown")),
        "avg_num_message_target_positions": base.finite_mean(message_counts, default=0.0),
        "avg_num_inject_positions": base.finite_mean(inject_counts, default=0.0),
        "hooks_fire_counts": {} if adapter is None else {str(k): int(v) for k, v in sorted(adapter.hook_fire_counts.items())},
        "write_hook_fire_counts": {}
        if adapter is None
        else {str(k): int(v) for k, v in sorted(adapter.write_hook_fire_counts.items())},
        "read_hook_fire_counts": {}
        if adapter is None
        else {str(k): int(v) for k, v in sorted(adapter.read_hook_fire_counts.items())},
        "hooks_ok": int(hooks_ok),
        "messages_only_from_write_layers": int(messages_only_from_write_layers),
        "updates_only_on_read_layers": int(updates_only_on_read_layers),
        "no_messages_from_read_layers": int(messages_only_from_write_layers and not (message_layers_seen & set(read_layers))),
        "no_updates_on_write_layers": int(updates_only_on_read_layers and not (update_layers_seen & set(write_layers))),
        "message_mode_counts": mode_counts,
        "message_mode_resolution_from_metrics": message_mode_resolution(metrics_rows),
        "exact_message_rows": int(exact_message_rows),
        "proxy_message_rows": int(proxy_message_rows),
        "exact_messages_used": int(exact_message_rows > 0),
        "exact_failure_counts": {} if adapter is None else dict(adapter.exact_failure_counts),
        "exact_failure_examples": [] if adapter is None else list(adapter.exact_failure_examples),
        "backward_diagnostics": backward_diag,
        "train_history_last": dict(train_history[-1]) if train_history else {},
        "oracle_mask_validation": oracle_validation,
        "oracle_masks_found_for_all_samples": int(len(mask_rows) == len(metrics_rows) and bool(mask_rows)),
        "mean_selected_evidence_frames": float(mean_selected),
        "mean_gold_count": float(mean_gold),
        "mean_selected_evidence_frames_equals_mean_gold_count": int(abs(float(mean_selected) - float(mean_gold)) <= 1e-9),
        "per_sample_oracle_mask_count_equals_gold_count": int(mask_count_equals_gold and bool(mask_rows)),
        "positive_frame_count_equals_gold_count": int(positive_count_equals_gold and bool(mask_rows)),
        "negative_frame_count_equals_8_minus_gold_count": int(negative_count_equals_distractors and bool(mask_rows)),
        "oracle_masks_binary": int(masks_binary and bool(mask_rows)),
        "distractor_frames_have_mask_zero": int(distractor_frames_zero and bool(mask_rows)),
        "finite_update_norms": int(finite_updates),
        "nonzero_updates": int(nonzero_updates),
        "finite_positive_stream_norms": int(finite_positive_streams),
        "finite_negative_stream_norms": int(finite_negative_streams),
        "finite_combined_stream_norms": int(finite_combined_streams),
        "finite_write_memory_norms": int(finite_write_memories),
        "positive_stream_norm_nonzero": int(any(abs(float(value)) > 1e-12 for value in positive_stream_values)),
        "negative_stream_norm_nonzero": int(any(abs(float(value)) > 1e-12 for value in negative_stream_values)),
        "combined_stream_norm_nonzero": int(any(abs(float(value)) > 1e-12 for value in combined_stream_values)),
        "write_memory_norm_nonzero": int(any(abs(float(value)) > 1e-12 for value in write_memory_values)),
        "non_evidence_frames_contributed_through_w_neg": int(
            finite_negative_streams and any(abs(float(value)) > 1e-12 for value in negative_stream_values)
        ),
        "query_key_readout_scores_used": 0,
        "learned_gate_used": 0,
        "query_key_readout_used": 0,
        "raw_matrix_readout_used": 0,
        "softmax_over_frames_used": 0,
        "positive_only_oracle_method_used": 0,
        "query_key_readout_score_fields": score_fields,
        "query_key_readout_parameters_present": int(bool(forbidden_param_names)),
        "query_key_readout_parameter_names": forbidden_param_names,
        "no_query_key_readout_scores_used": int(not score_fields and not forbidden_param_names),
        "no_gate_query_key_readout_used": int(not score_fields and not forbidden_param_names),
        "all_question_localization_ok": int(bool(all_question_localization_ok)),
        "last_token_localization_ok": int(bool(last_token_localization_ok)),
        "num_failed_localization_samples": len(failed_ids),
        "failed_localization_sample_ids": failed_ids[:50],
        "num_nonfinite_numeric_metrics": len(nonfinite_fields),
        "nonfinite_numeric_metric_examples": nonfinite_fields,
    }
    base.write_json(output_dir / "diagnostics.json", payload)
    return payload


def write_readme(
    output_dir: Path,
    summary_rows: Sequence[Dict[str, Any]],
    accuracy_rows: Sequence[Dict[str, Any]],
    metrics_rows: Sequence[Dict[str, Any]],
    diagnostics: Dict[str, Any],
) -> None:
    summary = {row["method"]: row for row in summary_rows}
    base_acc = base.finite_float(summary.get(BASELINE, {}).get("accuracy"))
    oracle_acc = base.finite_float(summary.get(METHOD, {}).get("accuracy"))
    improved = base_acc is not None and oracle_acc is not None and float(oracle_acc) > float(base_acc)
    base_mae = mean_pred_mae(accuracy_rows, BASELINE)
    oracle_mae = mean_pred_mae(accuracy_rows, METHOD)
    better_diagonal = base.finite_float(base_mae) is not None and base.finite_float(oracle_mae) is not None and oracle_mae < base_mae
    update_norm = base.finite_mean(
        (row.get("update_norm") for row in metrics_rows if row.get("method") == METHOD), default=0.0
    )
    update_reasonable = base.finite_float(update_norm) is not None and 0.0 < float(update_norm) < 100.0
    mode_counts = diagnostics.get("message_mode_counts", {})
    metric_mode_counts = diagnostics.get("message_mode_resolution_from_metrics", {})

    lines = [
        "# Distractor Oracle PosNeg Write/Read Adapter seq_len=8 7B",
        "",
        "This is a diagnostic upper bound, not a valid inference method.",
        "",
        "It answers:",
        "",
        "\"If evidence and non-evidence frames were perfectly labeled, can separate positive and rejection streams make frozen Qwen solve the distractor task?\"",
        "",
        "At write layers 14-17, the adapter extracts exact frame-to-all-question-token messages, applies the gold evidence-frame mask y_f, and stores one positive-negative memory per write layer.",
        "",
        "s_pos^{l_w} = sum_f y_f * W_pos^{l_w} m_f^{l_w}",
        "",
        "s_neg^{l_w} = sum_f (1 - y_f) * W_neg^{l_w} m_f^{l_w}",
        "",
        "M^{l_w} = s_pos^{l_w} + s_neg^{l_w}",
        "",
        "At read layers 20-27, it concatenates [M^14; M^15; M^16; M^17], projects with a read-layer-specific W_read, and injects into the last prompt token.",
        "",
        "h_last^{l_r} <- h_last^{l_r} + gamma_{l_r} * W_read^{l_r} [M^14; M^15; M^16; M^17]",
        "",
        "There is no learned gate, query/key readout, softmax over frames, raw-matrix readout, positive-only oracle method, learned frame selection, write-layer injection, read-layer message extraction, or averaging over write layers.",
        "",
        "## Automatic Interpretation",
        "",
        (
            f"- Did oracle posneg write/read improve over baseline? {bool(improved)} "
            f"(baseline={base_acc if base_acc is not None else math.nan:.4f}, "
            f"oracle-posneg-write-read={oracle_acc if oracle_acc is not None else math.nan:.4f})."
        ),
        (
            f"- Does mean predicted count follow y=x better than baseline? {bool(better_diagonal)} "
            f"(baseline mean-pred MAE={base_mae:.4f}, oracle-posneg-write-read={oracle_mae:.4f})."
        ),
        (
            f"- Are update norms active? {bool(update_reasonable)} "
            f"(mean update norm={update_norm:.6f}, finite={bool(diagnostics.get('finite_update_norms'))}, "
            f"nonzero={bool(diagnostics.get('nonzero_updates'))})."
        ),
        f"- Were oracle masks found for all samples? {bool(diagnostics.get('oracle_masks_found_for_all_samples'))}.",
        f"- Does positive_frame_count equal gold_count? {bool(diagnostics.get('positive_frame_count_equals_gold_count'))}.",
        f"- Does negative_frame_count equal 8 - gold_count? {bool(diagnostics.get('negative_frame_count_equals_8_minus_gold_count'))}.",
        f"- Are positive and negative stream norms finite? pos={bool(diagnostics.get('finite_positive_stream_norms'))}, neg={bool(diagnostics.get('finite_negative_stream_norms'))}.",
        f"- Did non-evidence frames contribute through W_neg? {bool(diagnostics.get('non_evidence_frames_contributed_through_w_neg'))}.",
        f"- Were messages extracted only from write layers? {bool(diagnostics.get('messages_only_from_write_layers'))}.",
        f"- Were updates injected only on read layers? {bool(diagnostics.get('updates_only_on_read_layers'))}.",
        f"- Were query/key/readout/gate components avoided? {bool(diagnostics.get('no_gate_query_key_readout_used'))}.",
        f"- Did message_mode=auto resolve to exact or proxy? adapter_counts={base.json_compact(mode_counts)}, metric_counts={base.json_compact(metric_mode_counts)}.",
        f"- Did Qwen remain frozen? {bool(diagnostics.get('qwen_frozen'))}.",
        f"- Were only adapter parameters trainable? {bool(diagnostics.get('only_adapter_params_trainable'))}.",
        "",
        "## Interpretation Rules",
        "",
        "- If oracle_posneg_write_read_adapter strongly beats the positive-only oracle mask adapter, the earlier failure was likely because ignoring distractors removed useful count information.",
        "- If oracle_posneg_write_read_adapter still fails, the issue is probably representation contamination or Qwen injection compatibility.",
        "- If low-count cases improve, that supports the idea that non-evidence frames provide necessary negative evidence.",
        "",
        "## Files",
        "",
        "- `metrics.csv`: per-sample logits, predictions, oracle masks, token positions, and positive/negative stream diagnostics.",
        "- `summary.csv`: overall frozen baseline and oracle-posneg write/read summary.",
        "- `accuracy_by_evidence_count.csv`: accuracy and prediction histograms by evidence/gold count.",
        "- `comparison_by_evidence_count.csv`: baseline vs oracle-posneg write/read deltas.",
        "- `train_history.csv`: adapter training and validation history.",
        "- `diagnostics.json`: frozen-model, trainability, hook, exact-message, oracle-mask, stream-norm, update-norm, and no-query/key/gate checks.",
        "- `plots/`: requested comparison, confusion, candidate-logit, stream-norm, update/message-norm, and oracle-mask plots.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    *,
    output_dir: Path,
    metrics_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    accuracy_rows: Sequence[Dict[str, Any]],
    comparison_rows: Sequence[Dict[str, Any]],
    train_history: Sequence[Dict[str, Any]],
) -> None:
    base.write_csv_dynamic(
        output_dir / "metrics.csv",
        metrics_rows,
        [
            "method",
            "sample_id",
            "sample_index",
            "seq_len",
            "gold_count",
            "evidence_count",
            "oracle_mask_count",
            "positive_frame_count",
            "negative_frame_count",
            "pred_count",
            "correct",
            "margin",
            "gold_logit",
            "pred_logit",
            "candidate_logits_json",
            "split",
            "readout_mode",
            "message_token_group",
            "inject_token_group",
            "message_target_positions_json",
            "inject_positions_json",
            "evidence_frame_mask_json",
            "positive_stream_norm",
            "negative_stream_norm",
            "combined_stream_norm",
            "write_memory_norm",
            "read_memory_concat_norm",
            "update_norm",
            "message_norm",
            "update_norm_by_layer_json",
            "message_norm_by_layer_json",
            "raw_message_norm_by_layer_json",
            "positive_frame_count_by_layer_json",
            "negative_frame_count_by_layer_json",
            "positive_stream_norm_by_layer_json",
            "negative_stream_norm_by_layer_json",
            "combined_stream_norm_by_layer_json",
            "write_memory_norm_by_layer_json",
            "read_memory_concat_norm_by_layer_json",
            "oracle_mask_by_layer_json",
            "oracle_mask_count_by_layer_json",
            "message_mode_by_layer_json",
            "token_selection_ok",
            "frame_grouping_ok",
        ],
    )
    base.write_csv_dynamic(
        output_dir / "summary.csv",
        summary_rows,
        [
            "method",
            "readout_mode",
            "message_token_group",
            "inject_token_group",
            "n",
            "accuracy",
            "mean_margin",
            "mean_gold_logit",
            "mean_pred_count",
            "mean_update_norm",
            "mean_message_norm",
            "mean_positive_stream_norm",
            "mean_negative_stream_norm",
            "mean_combined_stream_norm",
            "mean_write_memory_norm",
            "mean_read_memory_concat_norm",
            "train_accuracy",
            "val_accuracy",
            "val_ce",
        ],
    )
    base.write_csv_dynamic(
        output_dir / "accuracy_by_evidence_count.csv",
        accuracy_rows,
        [
            "method",
            "evidence_count",
            "gold_count",
            "n",
            "accuracy",
            "mean_margin",
            "mean_pred_count",
            "prediction_histogram",
            "mean_oracle_mask_count",
            "mean_update_norm",
            "mean_message_norm",
            "mean_positive_stream_norm",
            "mean_negative_stream_norm",
            "mean_combined_stream_norm",
            "mean_write_memory_norm",
            "mean_read_memory_concat_norm",
        ],
    )
    base.write_csv_dynamic(
        output_dir / "comparison_by_evidence_count.csv",
        comparison_rows,
        [
            "evidence_count",
            "gold_count",
            "baseline_accuracy",
            "oracle_posneg_write_read_accuracy",
            "delta_accuracy",
            "baseline_mean_pred",
            "oracle_posneg_write_read_mean_pred",
            "baseline_mean_margin",
            "oracle_posneg_write_read_mean_margin",
        ],
    )
    base.write_csv_dynamic(
        output_dir / "train_history.csv",
        train_history,
        [
            "method",
            "readout_mode",
            "message_token_group",
            "inject_token_group",
            "epoch",
            "train_ce",
            "train_loss",
            "train_update_energy",
            "train_accuracy",
            "train_steps",
            "val_ce",
            "val_accuracy",
            "adapter_parameter_norm",
            "gamma_json",
        ],
    )


def main() -> int:
    global METHOD
    args = parse_args()
    METHOD = method_label(
        str(args.gate_mode), str(args.streams), str(args.gate_source), bool(args.gate_hard), bool(args.freeze_streams)
    )
    if bool(args.smoke):
        args.epochs = min(int(args.epochs), 1)
        args.max_train_samples = min(int(args.max_train_samples), 16)
        args.max_eval_samples = min(int(args.max_eval_samples), 16)
        args.max_samples_per_count = min(int(args.max_samples_per_count), 4)
        args.no_plots = True
    if args.layer_start is not None:
        args.read_layer_start = int(args.layer_start)
    if args.layer_end is not None:
        args.read_layer_end = int(args.layer_end)
    if int(args.seq_len) != NUM_FRAMES:
        raise ValueError("This diagnostic is intentionally seq_len=8 only.")
    if int(args.write_layer_end) < int(args.write_layer_start):
        raise ValueError("--write-layer-end must be >= --write-layer-start")
    if int(args.read_layer_end) < int(args.read_layer_start):
        raise ValueError("--read-layer-end must be >= --read-layer-start")
    if int(args.candidate_min) != 0 or int(args.candidate_max) != 8:
        raise ValueError("This runner expects candidate counts 0-8.")
    args.message_token_group = canonical_group(str(args.message_token_group))
    args.inject_token_group = canonical_group(str(args.inject_token_group))
    args.evidence_counts = prev.parse_int_tokens(args.evidence_counts)
    if not args.evidence_counts:
        raise ValueError("--evidence-counts cannot be empty")
    if not (args.run_baseline or args.run_oracle_posneg_write_read or args.run_all):
        args.run_all = True

    should_run_baseline = bool(args.run_baseline or args.run_all)
    should_run_oracle = bool(args.run_oracle_posneg_write_read or args.run_all)
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else default_output_dir(args)
    log_handle, old_stdout, old_stderr = base.setup_logging(output_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        write_layers = list(range(int(args.write_layer_start), int(args.write_layer_end) + 1))
        read_layers = list(range(int(args.read_layer_start), int(args.read_layer_end) + 1))
        if not write_layers:
            raise ValueError("At least one write layer is required.")
        if not read_layers:
            raise ValueError("At least one read layer is required.")
        overlap = sorted(set(write_layers) & set(read_layers))
        if overlap:
            raise ValueError(f"Write and read layers must be disjoint; overlap={overlap}")
        readout_mode = write_read_readout_mode(write_layers, read_layers, str(args.gate_mode), str(args.streams))
        if str(args.gate_source) == "read_layer":
            readout_mode += "_lategate" 
        run_config = {
            "experiment_name": EXPERIMENT_NAME,
            "method": METHOD,
            "gate_mode": str(args.gate_mode),
            "streams": str(args.streams),
            "gate_source": str(args.gate_source),
            "gate_hard": bool(args.gate_hard),
            "oracle_mask_noise": float(args.oracle_mask_noise),
            "init_streams_from": None if args.init_streams_from is None else os.fspath(Path(args.init_streams_from).resolve()),
            "freeze_streams": bool(args.freeze_streams),
            "query_token_group": str(args.query_token_group),
            "lambda_ce": float(args.lambda_ce),
            "lambda_mask": float(args.lambda_mask),
            "lambda_count": float(args.lambda_count),
            "smoke": bool(args.smoke),
            "diagnostic_upper_bound": str(args.gate_mode) == "oracle",
            "model_name": str(args.model_name),
            "dataset_root": os.fspath(Path(args.dataset_root).resolve()),
            "source_run": os.fspath(Path(args.source_run).resolve()),
            "seq_len": NUM_FRAMES,
            "split": str(args.split),
            "evidence_counts": [int(x) for x in args.evidence_counts],
            "output_root": os.fspath(Path(args.output_root).resolve()),
            "output_dir": os.fspath(output_dir),
            "run_baseline": bool(should_run_baseline),
            "run_oracle_posneg_write_read": bool(should_run_oracle),
            "d_mem": int(args.d_mem),
            "write_layer_start": int(args.write_layer_start),
            "write_layer_end": int(args.write_layer_end),
            "read_layer_start": int(args.read_layer_start),
            "read_layer_end": int(args.read_layer_end),
            "deprecated_layer_start": None if args.layer_start is None else int(args.layer_start),
            "deprecated_layer_end": None if args.layer_end is None else int(args.layer_end),
            "write_layers": write_layers,
            "read_layers": read_layers,
            "inject_layers": read_layers,
            "message_mode": str(args.message_mode),
            "readout_mode": readout_mode,
            "message_token_group": str(args.message_token_group),
            "inject_token_group": str(args.inject_token_group),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "grad_accum": int(args.grad_accum),
            "max_train_samples": int(args.max_train_samples),
            "max_eval_samples": int(args.max_eval_samples),
            "max_samples_per_count": int(args.max_samples_per_count),
            "seed": int(args.seed),
            "candidate_counts": COUNT_VALUES,
            "submit_mode": str(args.submit_mode),
        }
        base.write_json(output_dir / "run_config.json", run_config)
        print(f"Output dir: {output_dir}")
        print(f"Run config: {base.json_compact(run_config)}")

        sample_payload = trans.load_sample_index_payload(args)
        sample_ids = sample_payload["sample_ids"]
        labels = sample_payload["labels"].long()
        records = prev.load_records(args.dataset_root, args.split, args.seq_len, sample_ids)
        if len(records) != len(sample_ids):
            raise RuntimeError(f"Loaded {len(records)} records for {len(sample_ids)} sample ids")
        label_mismatches = [
            {
                "sample_id": records[idx].sample_id,
                "label": int(labels[idx].item()),
                "gold_count": int(records[idx].gold_count),
            }
            for idx in range(len(records))
            if int(labels[idx].item()) != int(records[idx].gold_count)
        ]
        if label_mismatches:
            raise RuntimeError(f"Source labels do not match record gold_count; first={label_mismatches[:3]}")
        oracle_validation = validate_oracle_masks(records, seq_len=NUM_FRAMES)
        base.write_json(output_dir / "oracle_mask_manifest.json", oracle_validation)
        print(f"Oracle mask validation: {base.json_compact(oracle_validation)}")

        splits = prev.stratified_split(sample_ids, labels, int(args.seed))
        train_indices = carrier.split_limited_indices(
            splits["train"], records, int(args.max_train_samples), int(args.seed) + 11
        )
        val_indices = carrier.split_limited_indices(
            splits["val"] or splits["train"],
            records,
            int(args.max_eval_samples),
            int(args.seed) + 17,
        )
        test_indices = carrier.split_limited_indices(
            splits["test"] or splits["val"] or splits["train"],
            records,
            int(args.max_eval_samples),
            int(args.seed) + 23,
        )
        split_counts = prev.split_counts(
            {"train": train_indices, "val": val_indices, "test": test_indices},
            labels,
            COUNT_VALUES,
        )
        for split, row in split_counts.items():
            print(f"  {split}: " + ", ".join(f"{count}:{row.get(count, 0)}" for count in COUNT_VALUES))
        if should_run_oracle and (not train_indices or not val_indices):
            raise RuntimeError("Oracle posneg adapter training requires non-empty train and val splits")
        if not test_indices:
            raise RuntimeError("Test split is empty")

        device = prev.resolve_device(str(args.device))
        dtype = prev.resolve_dtype(str(args.dtype), device)
        model, processor = prev.load_model_and_processor(args, device=device, dtype=dtype)
        hidden_size = prev.hidden_size_from_model(model)
        candidate_format, count_token_ids = prev.candidate_token_ids(processor.tokenizer, 0, 8)
        print(f"Loaded frozen Qwen hidden_size={hidden_size} candidate_format={candidate_format}")
        model_trainable = sum(int(param.requires_grad) for param in model.parameters())
        if model_trainable:
            raise RuntimeError(f"Qwen is not frozen: {model_trainable} model parameter tensors require grad")
        print("Verified Qwen frozen before experiment dispatch.")

        metrics_rows: List[Dict[str, Any]] = []
        train_history: List[Dict[str, Any]] = []
        backward_diag: Dict[str, Any] = {}
        adapter: Optional[PosNegWriteReadAdapter] = None
        checkpoint_path: Optional[Path] = None

        if should_run_baseline:
            print("Evaluating frozen Qwen baseline")
            baseline_eval = evaluate_model(
                args=args,
                method=BASELINE,
                split_name="test",
                model=model,
                processor=processor,
                adapter=None,
                records=records,
                indices=test_indices,
                count_token_ids=count_token_ids,
                device=device,
                batch_size=int(args.batch_size),
                seed=int(args.seed) + 101,
                write_layers=write_layers,
                read_layers=read_layers,
            )
            metrics_rows.extend(baseline_eval["rows"])

        if should_run_oracle:
            print("Training oracle-posneg write/read adapter")
            adapter, train_history, backward_diag, checkpoint_path = train_adapter(
                args=args,
                output_dir=output_dir,
                model=model,
                processor=processor,
                records=records,
                train_indices=train_indices,
                val_indices=val_indices,
                count_token_ids=count_token_ids,
                hidden_size=int(hidden_size),
                write_layers=write_layers,
                read_layers=read_layers,
                readout_mode=readout_mode,
                device=device,
            )
            base.write_json(
                output_dir / "checkpoint.json",
                {
                    "oracle_posneg_write_read_best_checkpoint": os.fspath(checkpoint_path),
                    "readout_mode": readout_mode,
                    "write_layers": write_layers,
                    "read_layers": read_layers,
                    "message_token_group": str(args.message_token_group),
                    "inject_token_group": str(args.inject_token_group),
                },
            )
            print("Evaluating oracle-posneg write/read adapter on test split")
            oracle_eval = evaluate_model(
                args=args,
                method=METHOD,
                split_name="test",
                model=model,
                processor=processor,
                adapter=adapter,
                records=records,
                indices=test_indices,
                count_token_ids=count_token_ids,
                device=device,
                batch_size=int(args.batch_size),
                seed=int(args.seed) + 202,
                write_layers=write_layers,
                read_layers=read_layers,
            )
            metrics_rows.extend(oracle_eval["rows"])

        summary_rows: List[Dict[str, Any]] = []
        if should_run_baseline:
            summary_rows.append(summarize_method(method_rows(metrics_rows, BASELINE), method=BASELINE))
        if should_run_oracle:
            summary_rows.append(
                summarize_method(method_rows(metrics_rows, METHOD), method=METHOD, train_history=train_history)
            )
        accuracy_rows: List[Dict[str, Any]] = []
        for method in [BASELINE, METHOD]:
            if any(row.get("method") == method for row in metrics_rows):
                accuracy_rows.extend(accuracy_by_evidence_count(metrics_rows, method, COUNT_VALUES))
        comparison_rows = comparison_by_evidence_count(accuracy_rows, COUNT_VALUES)
        write_outputs(
            output_dir=output_dir,
            metrics_rows=metrics_rows,
            summary_rows=summary_rows,
            accuracy_rows=accuracy_rows,
            comparison_rows=comparison_rows,
            train_history=train_history,
        )
        if not bool(args.no_plots):
            make_plots(output_dir, metrics_rows, accuracy_rows, comparison_rows, COUNT_VALUES)
        diagnostics = write_diagnostics(
            output_dir=output_dir,
            model=model,
            adapter=adapter,
            train_history=train_history,
            backward_diag=backward_diag,
            metrics_rows=metrics_rows,
            args=args,
            oracle_validation=oracle_validation,
        )
        gate_auc = gate_auc_by_layer(metrics_rows, METHOD)
        diagnostics["gate_auc_by_layer"] = gate_auc
        base.write_json(output_dir / "gate_auc.json", {"method": METHOD, "gate_auc_by_layer": gate_auc})
        print(f"Gate evidence-detection AUC by layer ({METHOD}): {base.json_compact(gate_auc)}")
        write_readme(output_dir, summary_rows, accuracy_rows, metrics_rows, diagnostics)
        base.write_json(
            output_dir / "run_done.json",
            {
                "completed": True,
                "elapsed_seconds": time.time() - started,
                "output_dir": os.fspath(output_dir),
                "readout_mode": readout_mode,
                "write_layers": write_layers,
                "read_layers": read_layers,
                "message_token_group": str(args.message_token_group),
                "inject_token_group": str(args.inject_token_group),
                "methods": sorted({str(row["method"]) for row in metrics_rows}),
            },
        )
        print(f"Done: output_dir={output_dir}")
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

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
import torch.nn.functional as F
from torch import nn

from experiments.evidence_only import evidence_only_layer_local_seq1_8_7b as base
from experiments.carrier_probes import run_message_memory_adapter_stage1_stage3_seq8 as prev
from experiments.carrier_probes import run_message_memory_carrier_update_seq8_7b as carrier


EXPERIMENT_NAME = "evidence_only_sum_evidence_adapter_seq1_8_7b"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images_park_evidence_only_seq1_8"
DEFAULT_SOURCE_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images_park"

BASELINE = "baseline"
SUM_EVIDENCE = "sum_evidence_adapter"
ADDITIVE_SUM_READOUT = "additive_sum"
COUNT_VALUES = list(range(9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evidence-only seq_len 1..8 Qwen2.5-VL-7B experiment: all-question exact "
            "frame messages are projected, summed over evidence frames, and injected "
            "into the last prompt token. No query/key memory readout is used."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--seq-lens", nargs="+", default=[str(x) for x in range(1, 9)])
    parser.add_argument(
        "--train-seq-lens",
        nargs="+",
        default=[],
        help=(
            "Optional subset of --seq-lens used for the train/val splits. Seq lens outside "
            "this subset contribute all their samples to the test split (length/count OOD)."
        ),
    )
    parser.add_argument("--samples-per-seq-len", type=int, default=100)
    parser.add_argument("--force-generate", action="store_true", default=False)

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="")

    parser.add_argument("--generate-dataset", action="store_true", default=False)
    parser.add_argument("--run-baseline", action="store_true", default=False)
    parser.add_argument("--run-sum-evidence", action="store_true", default=False)
    parser.add_argument("--run-all", action="store_true", default=False)

    parser.add_argument("--d-mem", type=int, default=256)
    parser.add_argument("--layer-start", type=int, default=14)
    parser.add_argument("--layer-end", type=int, default=17)
    parser.add_argument("--gamma-init", type=float, default=0.05)
    parser.add_argument("--message-mode", default="auto", choices=["auto", "exact", "proxy"])
    parser.add_argument("--message-token-group", default="all_question", choices=sorted(carrier.TOKEN_GROUP_ALIASES))
    parser.add_argument("--inject-token-group", default="last_token", choices=sorted(carrier.TOKEN_GROUP_ALIASES))
    parser.add_argument(
        "--pool",
        default="sum",
        choices=["sum", "mean", "softmax", "pna"],
        help=(
            "Frame-message aggregator. sum=DeepSets (unnormalized). mean/softmax=normalized "
            "(over-squashing baselines). pna=PNA readout: [sum,mean,max,std] x [identity,amplify,attenuate]."
        ),
    )
    parser.add_argument(
        "--share-weights",
        action="store_true",
        default=False,
        help="Single shared phi/rho (and pool query) across all injected layers, instead of per-layer weights.",
    )

    parser.add_argument("--lambda-margin", type=float, default=0.2)
    parser.add_argument("--margin-target", type=float, default=1.0)
    parser.add_argument("--lambda-update-energy", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-samples-per-seq-len", type=int, default=100)
    parser.add_argument("--max-eval-samples-per-seq-len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true", default=False)

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
    if str(args.run_name).strip():
        name = str(args.run_name).strip()
    else:
        name = f"sum_evidence_{time.strftime('%Y%m%d_%H%M%S')}"
    return Path(args.output_root).resolve() / base.safe_name(name)


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
    if len(seq_lens) != 1:
        raise ValueError(f"Expected homogeneous batch by seq_len, got {sorted(seq_lens)}")
    carrier.NUM_FRAMES = int(next(iter(seq_lens)))
    return carrier.prepare_memory_batch(
        records=records,
        sample_indices=sample_indices,
        processor=processor,
        device=device,
        token_group=str(args.inject_token_group),
        message_token_group=str(args.message_token_group),
        query_token_group=str(args.inject_token_group),
        inject_token_group=str(args.inject_token_group),
    )


def adapter_set_context(adapter: "SimpleSumEvidenceAdapter", batch: carrier.MemoryBatch) -> None:
    adapter.set_context(
        message_target_positions=batch.message_target_positions,
        inject_positions=batch.inject_positions,
        frame_groups=batch.frame_groups,
    )


def mean_layer_frame_value(layer_json: Dict[str, Any]) -> float:
    return base.mean_layer_frame_value(layer_json)


def select_count_logits(outputs: Any, prompt_last_indices: torch.Tensor, count_token_ids: Dict[int, int]) -> torch.Tensor:
    return prev.select_count_logits(outputs.logits, prompt_last_indices, count_token_ids)


class SimpleSumEvidenceAdapter(nn.Module):
    """Project exact per-frame evidence messages, sum them, and inject into the last token."""

    def __init__(
        self,
        *,
        hidden_size: int,
        d_mem: int,
        inject_layers: Sequence[int],
        gamma_init: float,
        message_mode: str,
        pool: str = "sum",
        share_weights: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.d_mem = int(d_mem)
        self.inject_layers = [int(layer) for layer in inject_layers]
        self.layer_to_pos = {int(layer): pos for pos, layer in enumerate(self.inject_layers)}
        self.message_mode = str(message_mode)
        self.pool = str(pool)
        self.share_weights = bool(share_weights)
        self.enabled = True

        n_layers = len(self.inject_layers)
        n_modules = 1 if self.share_weights else n_layers
        self._n_modules = n_modules
        self._pna_streams = 12 if self.pool == "pna" else 1
        pool_out = self.d_mem * self._pna_streams
        self.message_to_memory = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.d_mem, bias=False) for _ in range(n_modules)]
        )
        self.w_o = nn.ModuleList([nn.Linear(pool_out, self.hidden_size, bias=False) for _ in range(n_modules)])
        self.gamma = nn.Parameter(torch.full((n_layers,), float(gamma_init), dtype=torch.float32))
        self.pool_query = (
            nn.Parameter(torch.zeros(n_modules, self.d_mem)) if self.pool == "softmax" else None
        )
        if self.pool == "softmax":
            nn.init.normal_(self.pool_query, mean=0.0, std=0.02)

        for module_idx in range(n_modules):
            nn.init.xavier_uniform_(self.message_to_memory[module_idx].weight, gain=0.5)
            nn.init.normal_(self.w_o[module_idx].weight, mean=0.0, std=0.002)

        suffix = "" if self.pool == "sum" else f"_{self.pool}"
        if self.share_weights:
            suffix += "_shared"
        self.readout_mode = ADDITIVE_SUM_READOUT + suffix

        self._message_target_positions: Optional[List[List[int]]] = None
        self._inject_positions: Optional[List[List[int]]] = None
        self._frame_groups: Optional[List[List[List[int]]]] = None
        self._handles: List[Any] = []
        self._loss_update_energies: List[torch.Tensor] = []
        self._last_stats: Dict[str, Dict[str, Any]] = {}
        self.hook_fire_counts: Dict[int, int] = defaultdict(int)
        self.message_mode_counts: Dict[str, int] = defaultdict(int)
        self.exact_failure_counts: Dict[str, int] = defaultdict(int)
        self.exact_failure_examples: List[str] = []

    def set_context(
        self,
        *,
        message_target_positions: Sequence[Sequence[int]],
        inject_positions: Sequence[Sequence[int]],
        frame_groups: Sequence[Sequence[Sequence[int]]],
    ) -> None:
        self._message_target_positions = [[int(pos) for pos in positions] for positions in message_target_positions]
        self._inject_positions = [[int(pos) for pos in positions] for positions in inject_positions]
        self._frame_groups = [
            [[int(pos) for pos in group] for group in sample_groups]
            for sample_groups in frame_groups
        ]
        self._loss_update_energies = []
        self._last_stats = {
            "update_norm_by_layer": {},
            "message_norm_by_layer": {},
            "raw_message_norm_by_layer": {},
            "summed_message_norm_by_layer": {},
            "message_mode_by_layer": {},
        }

    def clear_context(self) -> None:
        self._message_target_positions = None
        self._inject_positions = None
        self._frame_groups = None
        self._loss_update_energies = []

    def update_energy_for_loss(self, device: torch.device) -> torch.Tensor:
        if not self._loss_update_energies:
            return torch.zeros((), device=device)
        return torch.stack(self._loss_update_energies, dim=0).sum(dim=0).mean()

    def _mpos(self, layer_idx: int) -> int:
        return 0 if self.share_weights else self.layer_to_pos[int(layer_idx)]

    def _pool(self, z: torch.Tensor, mpos: int) -> torch.Tensor:
        """Aggregate per-frame projected messages z=[batch, n_frames, d_mem] over frames."""
        if self.pool == "sum":
            return z.sum(dim=1)
        if self.pool == "mean":
            return z.mean(dim=1)
        if self.pool == "softmax":
            assert self.pool_query is not None
            query = self.pool_query[mpos]
            scores = (z * query).sum(dim=-1) / math.sqrt(self.d_mem)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (z * weights).sum(dim=1)
        if self.pool == "pna":
            n_frames = max(1, int(z.shape[1]))
            aggs = [z.sum(dim=1), z.mean(dim=1), z.amax(dim=1), z.std(dim=1, unbiased=False)]
            amplify = math.log(n_frames + 1.0)
            attenuate = 1.0 / amplify if amplify > 1e-6 else 1.0
            streams = [agg * scaler for scaler in (1.0, amplify, attenuate) for agg in aggs]
            return torch.cat(streams, dim=-1)
        raise ValueError(f"unknown pool {self.pool!r}")

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

    @staticmethod
    def _repeat_kv(states: torch.Tensor, num_heads: int) -> torch.Tensor:
        if int(states.shape[1]) == int(num_heads):
            return states
        repeats = int(num_heads) // int(states.shape[1])
        return states.repeat_interleave(repeats, dim=1)

    def _num_frames(self) -> int:
        if not self._frame_groups:
            return 0
        return max((len(groups) for groups in self._frame_groups), default=0)

    def _record_exact_failure(self, reason: str) -> None:
        key = str(reason).split(":", 1)[0][:80]
        self.exact_failure_counts[key] += 1
        if len(self.exact_failure_examples) < 20:
            self.exact_failure_examples.append(str(reason)[:500])

    def _proxy_messages(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self._frame_groups is not None
        batch, seq_len, hidden = hidden_states.shape
        source = hidden_states.detach().float()
        num_frames = self._num_frames()
        raw_rows: List[torch.Tensor] = []
        for batch_idx in range(batch):
            sample_rows: List[torch.Tensor] = []
            for frame_idx in range(num_frames):
                group = self._frame_groups[batch_idx][frame_idx] if frame_idx < len(self._frame_groups[batch_idx]) else []
                valid = [int(pos) for pos in group if 0 <= int(pos) < seq_len]
                if valid:
                    idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
                    sample_rows.append(source[batch_idx, idx, :].mean(dim=0))
                else:
                    sample_rows.append(source.new_zeros((hidden,)))
            raw_rows.append(torch.stack(sample_rows, dim=0))
        return torch.stack(raw_rows, dim=0)

    def _exact_messages(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        if carrier.apply_multimodal_rotary_pos_emb is None:
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
            q, k = carrier.apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.rope_scaling["mrope_section"])
            k = self._repeat_kv(k, num_heads)
            v = self._repeat_kv(v, num_heads)
            attention_mask = kwargs.get("attention_mask")
            scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
            arange = torch.arange(seq_len, device=hidden_states.device)
            num_frames = self._num_frames()
            raw_message_rows: List[torch.Tensor] = []

            for batch_idx in range(batch):
                target_positions = [
                    int(pos)
                    for pos in self._message_target_positions[batch_idx]
                    if 0 <= int(pos) < seq_len
                ]
                if not target_positions:
                    raw_message_rows.append(hidden_states.detach().float().new_zeros((num_frames, self.hidden_size)))
                    continue

                target_idx = torch.tensor(target_positions, device=hidden_states.device, dtype=torch.long)
                scores = torch.einsum(
                    "hcd,hsd->hcs",
                    q[batch_idx, :, target_idx, :].float(),
                    k[batch_idx].float(),
                ) * scaling

                causal_allowed = arange.unsqueeze(0) <= target_idx.unsqueeze(1)
                sliding_window = getattr(attn, "sliding_window", None)
                if sliding_window is not None:
                    causal_allowed &= arange.unsqueeze(0) >= (target_idx.unsqueeze(1) - int(sliding_window))
                scores = scores.masked_fill(~causal_allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
                if torch.is_tensor(attention_mask):
                    mask = attention_mask
                    if mask.dim() == 4:
                        selected_mask = mask[batch_idx : batch_idx + 1, :, target_idx, :].float()
                        scores = scores + selected_mask.squeeze(0)
                    elif mask.dim() == 2:
                        valid_mask = mask[batch_idx].bool()
                        scores = scores.masked_fill(~valid_mask.view(1, 1, -1), torch.finfo(scores.dtype).min)

                probs = torch.softmax(scores, dim=-1)
                sample_raw: List[torch.Tensor] = []
                for frame_idx in range(num_frames):
                    group = self._frame_groups[batch_idx][frame_idx] if frame_idx < len(self._frame_groups[batch_idx]) else []
                    valid = [int(pos) for pos in group if 0 <= int(pos) < seq_len]
                    if not valid:
                        sample_raw.append(hidden_states.detach().float().new_zeros((self.hidden_size,)))
                        continue
                    frame_idx_tensor = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
                    contrib = torch.einsum(
                        "hcf,hfd->hcd",
                        probs[:, :, frame_idx_tensor],
                        v[batch_idx, :, frame_idx_tensor, :].float(),
                    )
                    contrib_flat = contrib.permute(1, 0, 2).reshape(len(target_positions), num_heads * head_dim)
                    projected = attn.o_proj(contrib_flat.to(dtype=hs.dtype)).detach().float()
                    sample_raw.append(projected.mean(dim=0))
                raw_message_rows.append(torch.stack(sample_raw, dim=0))

        return torch.stack(raw_message_rows, dim=0).to(hidden_states.device)

    def _message_contribution(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, str]:
        if self.message_mode == "proxy":
            return self._proxy_messages(hidden_states), "proxy"
        try:
            return self._exact_messages(module, hidden_states, kwargs), "exact"
        except Exception as exc:
            self._record_exact_failure(f"layer {layer_idx}: {type(exc).__name__}: {exc}")
            if self.message_mode == "exact":
                raise
            return self._proxy_messages(hidden_states), "proxy"

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
            or self._inject_positions is None
            or self._frame_groups is None
            or int(layer_idx) not in self.layer_to_pos
        ):
            return hidden_states

        layer_pos = self.layer_to_pos[int(layer_idx)]
        mpos = self._mpos(int(layer_idx))
        self.hook_fire_counts[int(layer_idx)] += 1
        raw_messages, mode = self._message_contribution(module, hidden_states, int(layer_idx), kwargs)
        self.message_mode_counts[f"{int(layer_idx)}:{mode}"] += int(hidden_states.shape[0])

        projected_messages = self.message_to_memory[mpos](raw_messages.float())
        summed = self._pool(projected_messages.float(), mpos)
        delta = self.w_o[mpos](summed).float()
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
        self._last_stats["message_norm_by_layer"][layer_key] = (
            projected_messages.detach().float().norm(dim=-1).cpu().tolist()
        )
        self._last_stats["raw_message_norm_by_layer"][layer_key] = raw_messages.detach().float().norm(dim=-1).cpu().tolist()
        self._last_stats["summed_message_norm_by_layer"][layer_key] = summed.detach().float().norm(dim=-1).cpu().tolist()
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


def blank_diagnostics(layers: Sequence[int]) -> Dict[str, Any]:
    return {
        "update_norm_by_layer": {str(int(layer)): 0.0 for layer in layers},
        "message_norm_by_layer": {str(int(layer)): [] for layer in layers},
        "raw_message_norm_by_layer": {str(int(layer)): [] for layer in layers},
        "summed_message_norm_by_layer": {str(int(layer)): 0.0 for layer in layers},
        "message_mode_by_layer": {str(int(layer)): "none" for layer in layers},
    }


@torch.no_grad()
def evaluate_model(
    *,
    args: argparse.Namespace,
    method: str,
    split_name: str,
    model: Any,
    processor: Any,
    adapter: Optional[SimpleSumEvidenceAdapter],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    count_token_ids: Dict[int, int],
    device: str,
    batch_size: int,
    seed: int,
    inject_layers: Sequence[int],
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
            seq_len = len(batch_records[0].frame_paths)
            carrier.NUM_FRAMES = int(seq_len)
            batch = prepare_batch(
                args=args,
                records=batch_records,
                sample_indices=batch_indices,
                processor=processor,
                device=device,
            )
            if adapter is not None:
                adapter_set_context(adapter, batch)
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
                gold = int(record.gold_count)
                pred = int(pred_offsets[row_idx].detach().cpu().item()) + int(count_min)
                logits_list = [float(v) for v in logits_cpu[row_idx].tolist()]
                logits_map = {str(count): logits_list[count - int(count_min)] for count in COUNT_VALUES}
                diag = adapter.stats_for_row(row_idx) if adapter is not None else blank_diagnostics(inject_layers)
                readout_mode = str(getattr(adapter, "readout_mode", "none")) if adapter is not None else "none"
                update_norm_by_layer = diag.get("update_norm_by_layer", {})
                message_norm_by_layer = diag.get("message_norm_by_layer", {})
                summed_message_norm_by_layer = diag.get("summed_message_norm_by_layer", {})
                update_norm = base.finite_mean(update_norm_by_layer.values(), default=0.0)
                message_norm = mean_layer_frame_value(message_norm_by_layer)
                summed_message_norm = base.finite_mean(summed_message_norm_by_layer.values(), default=0.0)
                pred_offset = pred - int(count_min)
                rows.append(
                    {
                        "method": str(method),
                        "sample_id": record.sample_id,
                        "sample_index": int(sample_idx),
                        "seq_len": int(seq_len),
                        "gold_count": int(gold),
                        "evidence_count": int(record.evidence_count),
                        "pred_count": int(pred),
                        "correct": int(pred == gold),
                        "margin": float(margins[row_idx].detach().cpu().item()),
                        "gold_logit": float(gold_logits[row_idx].detach().cpu().item()),
                        "pred_logit": logits_list[pred_offset] if 0 <= pred_offset < len(logits_list) else math.nan,
                        "candidate_logits_json": base.json_compact(logits_map),
                        "split": str(split_name),
                        "ce": float(ce_vec[row_idx].detach().cpu().item()),
                        "readout_mode": readout_mode,
                        "message_token_group": message_group,
                        "inject_token_group": inject_group,
                        "message_target_positions_json": base.json_compact(batch.message_target_positions[row_idx]),
                        "inject_positions_json": base.json_compact(batch.inject_positions[row_idx]),
                        "update_norm": float(update_norm) if adapter is not None else "",
                        "message_norm": float(message_norm) if adapter is not None else "",
                        "summed_message_norm": float(summed_message_norm) if adapter is not None else "",
                        "update_norm_by_layer_json": base.json_compact(update_norm_by_layer) if adapter is not None else "",
                        "message_norm_by_layer_json": base.json_compact(message_norm_by_layer) if adapter is not None else "",
                        "raw_message_norm_by_layer_json": base.json_compact(diag.get("raw_message_norm_by_layer", {}))
                        if adapter is not None
                        else "",
                        "summed_message_norm_by_layer_json": base.json_compact(summed_message_norm_by_layer)
                        if adapter is not None
                        else "",
                        "message_mode_by_layer_json": base.json_compact(diag.get("message_mode_by_layer", {}))
                        if adapter is not None
                        else "",
                        "frame_token_counts_json": base.json_compact(batch.frame_token_counts[row_idx]),
                        "evidence_frame_mask_json": base.json_compact(base.evidence_frame_mask(record)),
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
    inject_layers: Sequence[int],
    device: str,
) -> Tuple[SimpleSumEvidenceAdapter, List[Dict[str, Any]], Dict[str, Any], Path]:
    adapter = SimpleSumEvidenceAdapter(
        hidden_size=int(hidden_size),
        d_mem=int(args.d_mem),
        inject_layers=[int(x) for x in inject_layers],
        gamma_init=float(args.gamma_init),
        message_mode=str(args.message_mode),
        pool=str(args.pool),
        share_weights=bool(args.share_weights),
    ).to(device)
    carrier.verify_trainable_parameters(model, adapter)
    optimizer = torch.optim.AdamW(
        [param for param in adapter.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "sum_evidence_adapter_best.pt"
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
        train_correct = 0
        train_n = 0
        train_steps = 0
        backward_steps = 0
        skipped = 0

        try:
            adapter.register_hooks(model)
            for step, batch_indices in enumerate(train_batches, start=1):
                batch_records = [records[int(idx)] for idx in batch_indices]
                seq_len = len(batch_records[0].frame_paths)
                carrier.NUM_FRAMES = int(seq_len)
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
                adapter_set_context(adapter, batch)
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = select_count_logits(outputs, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - int(count_min)
                ce = F.cross_entropy(count_logits, gold_offsets)
                m_loss = carrier.margin_loss(count_logits, gold_offsets, float(args.margin_target))
                update_energy = adapter.update_energy_for_loss(count_logits.device)
                loss = ce + float(args.lambda_margin) * m_loss + float(args.lambda_update_energy) * update_energy
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
                train_steps += 1
                backward_steps += 1
                adapter.clear_context()
                if backward_steps % max(1, int(args.grad_accum)) == 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 25 == 0:
                    print(
                        f"  {SUM_EVIDENCE} epoch={epoch} step={step}/{len(train_batches)} "
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
            method=SUM_EVIDENCE,
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
            inject_layers=inject_layers,
        )
        row = {
            "method": SUM_EVIDENCE,
            "readout_mode": ADDITIVE_SUM_READOUT,
            "message_token_group": canonical_group(str(args.message_token_group)),
            "inject_token_group": canonical_group(str(args.inject_token_group)),
            "epoch": int(epoch),
            "train_ce": train_ce_total / max(1, train_steps),
            "train_loss": train_loss_total / max(1, train_steps),
            "train_update_energy": train_energy_total / max(1, train_steps),
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
            f"  {SUM_EVIDENCE} epoch={epoch} train_ce={row['train_ce']:.4f} "
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
                    "inject_layers": [int(x) for x in inject_layers],
                    "variant": SUM_EVIDENCE,
                    "method": SUM_EVIDENCE,
                    "message_mode": str(args.message_mode),
                    "readout_mode": ADDITIVE_SUM_READOUT,
                    "message_token_group": canonical_group(str(args.message_token_group)),
                    "inject_token_group": canonical_group(str(args.inject_token_group)),
                },
                checkpoint_path,
            )
    if best_state is not None:
        adapter.load_state_dict(best_state)
    return adapter, history, backward_diag, checkpoint_path


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
        if method == SUM_EVIDENCE
        else 0.0,
        "mean_message_norm": base.finite_mean((row.get("message_norm") for row in rows), default=0.0)
        if method == SUM_EVIDENCE
        else 0.0,
        "mean_summed_message_norm": base.finite_mean((row.get("summed_message_norm") for row in rows), default=0.0)
        if method == SUM_EVIDENCE
        else 0.0,
        "train_accuracy": train_last.get("train_accuracy", ""),
        "val_accuracy": train_last.get("val_accuracy", ""),
        "val_ce": train_last.get("val_ce", ""),
    }


def accuracy_by_seq_len(rows: Sequence[Dict[str, Any]], method: str, seq_lens: Sequence[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seq_len in seq_lens:
        seq_rows = [row for row in rows if row["method"] == method and int(row["seq_len"]) == int(seq_len)]
        correct = [int(row["correct"]) for row in seq_rows]
        out.append(
            {
                "method": str(method),
                "seq_len": int(seq_len),
                "gold_count": int(seq_len),
                "n": len(seq_rows),
                "accuracy": float(np.mean(correct)) if correct else math.nan,
                "mean_margin": base.finite_mean(row.get("margin") for row in seq_rows),
                "mean_pred_count": base.finite_mean(row.get("pred_count") for row in seq_rows),
                "prediction_histogram": base.json_compact(prediction_histogram(seq_rows)),
                "mean_update_norm": base.finite_mean((row.get("update_norm") for row in seq_rows), default=0.0)
                if method == SUM_EVIDENCE
                else 0.0,
                "mean_message_norm": base.finite_mean((row.get("message_norm") for row in seq_rows), default=0.0)
                if method == SUM_EVIDENCE
                else 0.0,
                "mean_summed_message_norm": base.finite_mean(
                    (row.get("summed_message_norm") for row in seq_rows),
                    default=0.0,
                )
                if method == SUM_EVIDENCE
                else 0.0,
            }
        )
    return out


def comparison_by_seq_len(accuracy_rows: Sequence[Dict[str, Any]], seq_lens: Sequence[int]) -> List[Dict[str, Any]]:
    by_key = {(row["method"], int(row["seq_len"])): row for row in accuracy_rows}
    out: List[Dict[str, Any]] = []
    for seq_len in seq_lens:
        base_row = by_key.get((BASELINE, int(seq_len)), {})
        sum_row = by_key.get((SUM_EVIDENCE, int(seq_len)), {})
        base_acc = base.finite_float(base_row.get("accuracy"))
        sum_acc = base.finite_float(sum_row.get("accuracy"))
        out.append(
            {
                "seq_len": int(seq_len),
                "gold_count": int(seq_len),
                "baseline_accuracy": "" if base_acc is None else float(base_acc),
                "sum_evidence_accuracy": "" if sum_acc is None else float(sum_acc),
                "delta_accuracy": ""
                if base_acc is None or sum_acc is None
                else float(sum_acc) - float(base_acc),
                "baseline_mean_pred": base_row.get("mean_pred_count", ""),
                "sum_evidence_mean_pred": sum_row.get("mean_pred_count", ""),
                "baseline_mean_margin": base_row.get("mean_margin", ""),
                "sum_evidence_mean_margin": sum_row.get("mean_margin", ""),
            }
        )
    return out


def save_combined_line_plot(
    path: Path,
    accuracy_rows: Sequence[Dict[str, Any]],
    *,
    y_key: str,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(7.2, 4.5))
    for method in [BASELINE, SUM_EVIDENCE]:
        rows = sorted([row for row in accuracy_rows if row["method"] == method], key=lambda row: int(row["seq_len"]))
        xs = [int(row["seq_len"]) for row in rows]
        ys = [float(row.get(y_key, math.nan)) for row in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=method)
    plt.xlabel("seq_len / gold_count")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted({int(row["seq_len"]) for row in accuracy_rows}))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def confusion_matrix(rows: Sequence[Dict[str, Any]], seq_lens: Sequence[int]) -> np.ndarray:
    y_counts = [int(seq_len) for seq_len in seq_lens]
    x_counts = COUNT_VALUES
    mat = np.zeros((len(y_counts), len(x_counts)), dtype=float)
    for row in rows:
        gold = int(row["gold_count"])
        pred = int(row["pred_count"])
        if gold in y_counts and pred in x_counts:
            mat[y_counts.index(gold), x_counts.index(pred)] += 1.0
    return mat


def save_confusion(path: Path, rows: Sequence[Dict[str, Any]], seq_lens: Sequence[int], title: str) -> None:
    mat = confusion_matrix(rows, seq_lens)
    fig, ax = plt.subplots(figsize=(7.3, 5.4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(np.arange(len(COUNT_VALUES)))
    ax.set_xticklabels(COUNT_VALUES)
    ax.set_yticks(np.arange(len(seq_lens)))
    ax.set_yticklabels(seq_lens)
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("Gold count / seq_len")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_combined_confusions(path: Path, rows: Sequence[Dict[str, Any]], seq_lens: Sequence[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, method in zip(axes, [BASELINE, SUM_EVIDENCE]):
        mat = confusion_matrix([row for row in rows if row["method"] == method], seq_lens)
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(np.arange(len(COUNT_VALUES)))
        ax.set_xticklabels(COUNT_VALUES)
        ax.set_yticks(np.arange(len(seq_lens)))
        ax.set_yticklabels(seq_lens)
        ax.set_xlabel("Predicted count")
        ax.set_title(method)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] > 0:
                    ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Gold count / seq_len")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def candidate_logits(row: Dict[str, Any]) -> Dict[str, float]:
    payload = base.parse_json_field(row, "candidate_logits_json", {})
    if isinstance(payload, dict):
        return {str(k): float(v) for k, v in payload.items() if base.finite_float(v) is not None}
    if isinstance(payload, list):
        return {str(i): float(v) for i, v in enumerate(payload) if base.finite_float(v) is not None}
    return {}


def save_candidate_logit_curves(path: Path, rows: Sequence[Dict[str, Any]], method: str, seq_lens: Sequence[int]) -> None:
    plt.figure(figsize=(7.4, 4.8))
    for seq_len in seq_lens:
        seq_rows = [row for row in rows if row["method"] == method and int(row["seq_len"]) == int(seq_len)]
        if not seq_rows:
            continue
        means: List[float] = []
        for count in COUNT_VALUES:
            vals = [candidate_logits(row).get(str(count), math.nan) for row in seq_rows]
            means.append(base.finite_mean(vals))
        plt.plot(COUNT_VALUES, means, marker="o", linewidth=1.3, label=f"gold {seq_len}")
    plt.xlabel("Candidate count")
    plt.ylabel("Mean logit")
    plt.title(f"Candidate Logit Curves: {method}")
    plt.xticks(COUNT_VALUES)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def make_plots(output_dir: Path, metrics_rows: Sequence[Dict[str, Any]], accuracy_rows: Sequence[Dict[str, Any]], seq_lens: Sequence[int]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_combined_line_plot(
        plots_dir / "combined_accuracy_vs_seq_len.png",
        accuracy_rows,
        y_key="accuracy",
        ylabel="Accuracy",
        title="Accuracy vs Evidence-Only Sequence Length",
    )
    save_combined_line_plot(
        plots_dir / "combined_margin_vs_seq_len.png",
        accuracy_rows,
        y_key="mean_margin",
        ylabel="Mean margin",
        title="Margin vs Evidence-Only Sequence Length",
    )

    plt.figure(figsize=(7.2, 4.8))
    max_seq = max(int(x) for x in seq_lens)
    plt.plot([0, max_seq], [0, max_seq], linestyle="--", color="black", linewidth=1.2, label="perfect y=x")
    for method in [BASELINE, SUM_EVIDENCE]:
        rows = sorted([row for row in accuracy_rows if row["method"] == method], key=lambda row: int(row["seq_len"]))
        xs = [int(row["gold_count"]) for row in rows]
        ys = [float(row.get("mean_pred_count", math.nan)) for row in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=method)
    plt.xlabel("Gold count")
    plt.ylabel("Mean predicted count")
    plt.title("Mean Predicted Count vs Gold Count")
    plt.xticks(seq_lens)
    plt.yticks(COUNT_VALUES)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_predicted_count_vs_gold_count.png", dpi=180, bbox_inches="tight")
    plt.close()

    base_rows = [row for row in metrics_rows if row["method"] == BASELINE]
    sum_rows = [row for row in metrics_rows if row["method"] == SUM_EVIDENCE]
    save_confusion(plots_dir / "predicted_count_confusion_matrix_baseline.png", base_rows, seq_lens, "Baseline Confusion Matrix")
    save_confusion(
        plots_dir / "predicted_count_confusion_matrix_sum_evidence_adapter.png",
        sum_rows,
        seq_lens,
        "Sum Evidence Adapter Confusion Matrix",
    )
    save_combined_confusions(plots_dir / "combined_confusion_matrices.png", metrics_rows, seq_lens)

    comp = comparison_by_seq_len(accuracy_rows, seq_lens)
    plt.figure(figsize=(7.2, 4.3))
    xs = [int(row["seq_len"]) for row in comp]
    ys = [float(row["delta_accuracy"]) if base.finite_float(row.get("delta_accuracy")) is not None else math.nan for row in comp]
    colors = ["#2ca02c" if base.finite_float(y) is not None and float(y) >= 0 else "#d62728" for y in ys]
    plt.bar(xs, ys, color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("seq_len / gold_count")
    plt.ylabel("Sum evidence minus baseline accuracy")
    plt.title("Delta Accuracy")
    plt.xticks(seq_lens)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "delta_accuracy_sum_evidence_minus_baseline.png", dpi=180, bbox_inches="tight")
    plt.close()

    diagnostic_specs = [
        ("mean_update_norm", "update_norm_vs_seq_len.png", "Mean update norm", "Sum Evidence Update Norm vs Seq Len"),
        ("mean_message_norm", "message_norm_vs_seq_len.png", "Mean message norm", "Projected Message Norm vs Seq Len"),
        (
            "mean_summed_message_norm",
            "summed_message_norm_vs_seq_len.png",
            "Mean summed message norm",
            "Summed Message Norm vs Seq Len",
        ),
    ]
    rows = sorted([row for row in accuracy_rows if row["method"] == SUM_EVIDENCE], key=lambda row: int(row["seq_len"]))
    for key, filename, ylabel, title in diagnostic_specs:
        plt.figure(figsize=(7.2, 4.3))
        plt.plot([int(row["seq_len"]) for row in rows], [float(row.get(key, math.nan)) for row in rows], marker="o")
        plt.xlabel("seq_len / gold_count")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(seq_lens)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=180, bbox_inches="tight")
        plt.close()

    save_candidate_logit_curves(
        plots_dir / "candidate_logit_curves_by_seq_len_baseline.png",
        metrics_rows,
        BASELINE,
        seq_lens,
    )
    save_candidate_logit_curves(
        plots_dir / "candidate_logit_curves_by_seq_len_sum_evidence_adapter.png",
        metrics_rows,
        SUM_EVIDENCE,
        seq_lens,
    )


def method_rows(rows: Sequence[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("method") == method]


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


def high_count_accuracy(accuracy_rows: Sequence[Dict[str, Any]], method: str, threshold: int = 4) -> float:
    vals = [
        base.finite_float(row.get("accuracy"))
        for row in accuracy_rows
        if row.get("method") == method and int(row.get("seq_len", 0)) >= int(threshold)
    ]
    return base.finite_mean(v for v in vals if v is not None)


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


def write_diagnostics(
    *,
    output_dir: Path,
    model: Any,
    adapter: Optional[SimpleSumEvidenceAdapter],
    train_history: Sequence[Dict[str, Any]],
    backward_diag: Dict[str, Any],
    metrics_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
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
    message_counts = [len(base.parse_json_field(row, "message_target_positions_json", [])) for row in metrics_rows]
    inject_counts = [len(base.parse_json_field(row, "inject_positions_json", [])) for row in metrics_rows]
    update_values = numeric_values_from_json_field(metrics_rows, "update_norm_by_layer_json")
    finite_updates = bool(update_values) and all(math.isfinite(float(value)) for value in update_values)
    nonzero_updates = any(abs(float(value)) > 1e-12 for value in update_values)
    score_fields = sorted(
        {
            key
            for row in metrics_rows
            for key in row.keys()
            if "matrix_score" in str(key) or str(key).startswith("gate_") or "query_" in str(key)
        }
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
            "summed_message_norm",
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
    param_names = [] if adapter is None else [name for name, _param in adapter.named_parameters()]
    query_key_param_names = [
        name
        for name in param_names
        if any(part in name for part in ("w_q", "w_k", "w_v", "gate", "readout", "key", "query"))
    ]
    mode_counts = {} if adapter is None else dict(adapter.message_mode_counts)
    exact_message_rows = sum(int(value) for key, value in mode_counts.items() if str(key).endswith(":exact"))
    proxy_message_rows = sum(int(value) for key, value in mode_counts.items() if str(key).endswith(":proxy"))
    hooks_ok = bool(
        adapter is None
        or all(int(adapter.hook_fire_counts.get(int(layer), 0)) > 0 for layer in adapter.inject_layers)
    )
    localization_rows = [
        row
        for row in metrics_rows
        if row.get("method") == SUM_EVIDENCE or not any(item.get("method") == SUM_EVIDENCE for item in metrics_rows)
    ]
    all_question_localization_ok = all(
        len(base.parse_json_field(row, "message_target_positions_json", [])) > 0
        for row in localization_rows
    )
    last_token_localization_ok = all(
        len(base.parse_json_field(row, "inject_positions_json", [])) > 0
        for row in localization_rows
    )
    payload = {
        "qwen_frozen": int(model_trainable_tensors == 0),
        "model_trainable_tensors": int(model_trainable_tensors),
        "adapter_trainable_tensors": int(adapter_trainable_tensors),
        "adapter_trainable_params": int(adapter_trainable_params),
        "only_adapter_params_trainable": int(model_trainable_tensors == 0 and adapter_trainable_tensors > 0)
        if adapter is not None
        else "",
        "message_token_group": canonical_group(str(args.message_token_group)),
        "inject_token_group": canonical_group(str(args.inject_token_group)),
        "readout_mode": "none" if adapter is None else str(getattr(adapter, "readout_mode", "unknown")),
        "avg_num_message_target_positions": base.finite_mean(message_counts, default=0.0),
        "avg_num_inject_positions": base.finite_mean(inject_counts, default=0.0),
        "hooks_fire_counts": {} if adapter is None else {str(k): int(v) for k, v in sorted(adapter.hook_fire_counts.items())},
        "hooks_ok": int(hooks_ok),
        "message_mode_counts": mode_counts,
        "message_mode_resolution_from_metrics": message_mode_resolution(metrics_rows),
        "exact_message_rows": int(exact_message_rows),
        "proxy_message_rows": int(proxy_message_rows),
        "exact_messages_used": int(exact_message_rows > 0),
        "exact_failure_counts": {} if adapter is None else dict(adapter.exact_failure_counts),
        "exact_failure_examples": [] if adapter is None else list(adapter.exact_failure_examples),
        "backward_diagnostics": backward_diag,
        "train_history_last": dict(train_history[-1]) if train_history else {},
        "finite_update_norms": int(finite_updates),
        "nonzero_updates": int(nonzero_updates),
        "query_key_readout_scores_used": 0,
        "query_key_readout_score_fields": score_fields,
        "query_key_readout_parameters_present": int(bool(query_key_param_names)),
        "query_key_readout_parameter_names": query_key_param_names,
        "no_query_key_readout_scores_used": int(not score_fields and not query_key_param_names),
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
    sum_acc = base.finite_float(summary.get(SUM_EVIDENCE, {}).get("accuracy"))
    improved = base_acc is not None and sum_acc is not None and float(sum_acc) > float(base_acc)
    base_high = high_count_accuracy(accuracy_rows, BASELINE)
    sum_high = high_count_accuracy(accuracy_rows, SUM_EVIDENCE)
    high_delta = sum_high - base_high if base.finite_float(sum_high) is not None and base.finite_float(base_high) is not None else math.nan
    base_mae = mean_pred_mae(accuracy_rows, BASELINE)
    sum_mae = mean_pred_mae(accuracy_rows, SUM_EVIDENCE)
    better_diagonal = base.finite_float(base_mae) is not None and base.finite_float(sum_mae) is not None and sum_mae < base_mae
    update_norm = base.finite_mean((row.get("update_norm") for row in metrics_rows if row.get("method") == SUM_EVIDENCE), default=0.0)
    update_reasonable = base.finite_float(update_norm) is not None and 0.0 < float(update_norm) < 100.0
    mode_counts = diagnostics.get("message_mode_counts", {})
    metric_mode_counts = diagnostics.get("message_mode_resolution_from_metrics", {})

    lines = [
        "# Evidence-Only Sum Evidence Adapter seq_len 1..8 7B",
        "",
        "This experiment tests whether the all-evidence task only needs additive evidence accumulation rather than gLSTM-style query/key addressing.",
        "",
        "all-question-token frame messages -> projected evidence vectors -> sum over frames -> last-token injection",
        "",
        "Every frame is evidence, so gold_count=evidence_count=seq_len.",
        "",
        "For each layer, frame f contributes the exact attention-value message into all question tokens:",
        "",
        "m_f^l = (1 / |Q|) sum_{q in Q} W_O [ sum_{j in I_f} A^l_{q,j} V^l_j ]",
        "",
        "The adapter projects and sums those evidence messages:",
        "",
        "s^l = sum_f W_m m_f^l",
        "",
        "and injects the result into the last prompt token:",
        "",
        "h_last^l <- h_last^l + gamma_l W_o s^l",
        "",
        "There are no query vectors, key vectors, softmax readout, raw matrix scores, or sigmoid gates in this default adapter.",
        "",
        "## Automatic Interpretation",
        "",
        (
            f"- Did additive evidence accumulation improve over baseline? {bool(improved)} "
            f"(baseline={base_acc if base_acc is not None else math.nan:.4f}, "
            f"sum-evidence={sum_acc if sum_acc is not None else math.nan:.4f})."
        ),
        (
            f"- Does it improve high counts 4..8? {base.finite_float(high_delta) is not None and high_delta > 0.0} "
            f"(baseline high={base_high:.4f}, sum-evidence high={sum_high:.4f}, delta={high_delta:.4f})."
        ),
        (
            f"- Does mean predicted count follow y=x better than baseline? {bool(better_diagonal)} "
            f"(baseline mean-pred MAE={base_mae:.4f}, sum-evidence={sum_mae:.4f})."
        ),
        (
            f"- Are update norms reasonable? {bool(update_reasonable)} "
            f"(mean update norm={update_norm:.6f}, finite={bool(diagnostics.get('finite_update_norms'))}, "
            f"nonzero={bool(diagnostics.get('nonzero_updates'))})."
        ),
        f"- Were query/key/readout scores avoided? {bool(diagnostics.get('no_query_key_readout_scores_used'))}.",
        f"- Did message_mode=auto resolve to exact or proxy? adapter_counts={base.json_compact(mode_counts)}, metric_counts={base.json_compact(metric_mode_counts)}.",
        f"- Did Qwen remain frozen? {bool(diagnostics.get('qwen_frozen'))}.",
        f"- Were only adapter parameters trainable? {bool(diagnostics.get('only_adapter_params_trainable'))}.",
        "",
        "## Interpretation Notes",
        "",
        "- If this works, the evidence-only task may mostly need additive evidence accumulation instead of gLSTM-style query/key addressing.",
        "- If this fails while query/key readout works, then last-token counting may need content-addressed mixing even when every frame is evidence.",
        "- If it fails at layers 14..17, a follow-up should try later injection layers, e.g. 18..27 or 20..27.",
        "",
        "## Files",
        "",
        "- `metrics.csv`: per-sample logits, predictions, positions, and additive-sum diagnostics.",
        "- `summary.csv`: overall baseline and sum-evidence summary.",
        "- `accuracy_by_seq_len.csv`: accuracy and prediction histograms by count.",
        "- `comparison_by_seq_len.csv`: baseline vs sum-evidence deltas.",
        "- `diagnostics.json`: frozen-model, trainability, token-position, hook, exact/proxy, update-norm, and no-query/key checks.",
        "- `plots/`: combined accuracy, margins, mean predicted counts, confusion matrices, deltas, candidate logits, and additive-sum diagnostics.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    seq_lens = base.split_int_tokens(args.seq_lens)
    if not seq_lens:
        raise ValueError("--seq-lens cannot be empty")
    if any(seq_len < 1 or seq_len > 8 for seq_len in seq_lens):
        raise ValueError("This experiment expects seq_lens within 1..8")
    if int(args.layer_end) < int(args.layer_start):
        raise ValueError("--layer-end must be >= --layer-start")
    args.message_token_group = canonical_group(str(args.message_token_group))
    args.inject_token_group = canonical_group(str(args.inject_token_group))
    if not (args.generate_dataset or args.run_baseline or args.run_sum_evidence or args.run_all):
        args.run_all = True

    should_generate = bool(args.generate_dataset or args.run_all)
    should_run_baseline = bool(args.run_baseline or args.run_all)
    should_run_sum_evidence = bool(args.run_sum_evidence or args.run_all)

    if should_generate:
        base.generate_evidence_only_dataset(
            dataset_root=Path(args.dataset_root),
            source_dataset_root=Path(args.source_dataset_root),
            seq_lens=seq_lens,
            samples_per_seq_len=int(args.samples_per_seq_len),
            force=bool(args.force_generate),
        )

    if not (should_run_baseline or should_run_sum_evidence):
        print("Dataset generation complete; no run mode requested.")
        return 0

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else default_output_dir(args)
    log_handle, old_stdout, old_stderr = base.setup_logging(output_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        inject_layers = list(range(int(args.layer_start), int(args.layer_end) + 1))
        run_config = {
            "experiment_name": EXPERIMENT_NAME,
            "model_name": str(args.model_name),
            "dataset_root": os.fspath(Path(args.dataset_root).resolve()),
            "source_dataset_root": os.fspath(Path(args.source_dataset_root).resolve()),
            "seq_lens": [int(x) for x in seq_lens],
            "train_seq_lens": base.split_int_tokens(args.train_seq_lens) if args.train_seq_lens else [int(x) for x in seq_lens],
            "output_root": os.fspath(Path(args.output_root).resolve()),
            "output_dir": os.fspath(output_dir),
            "run_baseline": bool(should_run_baseline),
            "run_sum_evidence": bool(should_run_sum_evidence),
            "d_mem": int(args.d_mem),
            "layer_start": int(args.layer_start),
            "layer_end": int(args.layer_end),
            "inject_layers": inject_layers,
            "message_mode": str(args.message_mode),
            "readout_mode": ADDITIVE_SUM_READOUT,
            "message_token_group": str(args.message_token_group),
            "inject_token_group": str(args.inject_token_group),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "grad_accum": int(args.grad_accum),
            "max_train_samples_per_seq_len": int(args.max_train_samples_per_seq_len),
            "max_eval_samples_per_seq_len": int(args.max_eval_samples_per_seq_len),
            "seed": int(args.seed),
            "candidate_counts": COUNT_VALUES,
            "submit_mode": str(args.submit_mode),
        }
        base.write_json(output_dir / "run_config.json", run_config)
        print(f"Output dir: {output_dir}")
        print(f"Run config: {base.json_compact(run_config)}")

        ok, manifest = base.validate_evidence_only_dataset(Path(args.dataset_root), seq_lens, int(args.samples_per_seq_len))
        base.write_json(output_dir / "dataset_manifest_snapshot.json", manifest)
        if not ok:
            raise RuntimeError(f"Dataset failed validation: {args.dataset_root}")

        records, by_seq = base.load_all_records(Path(args.dataset_root), seq_lens)
        splits = base.make_splits(
            records,
            by_seq,
            seed=int(args.seed),
            max_train_per_seq=int(args.max_train_samples_per_seq_len),
            max_eval_per_seq=int(args.max_eval_samples_per_seq_len),
        )
        train_seq_lens = base.split_int_tokens(args.train_seq_lens) if args.train_seq_lens else list(seq_lens)
        if any(seq_len not in seq_lens for seq_len in train_seq_lens):
            raise ValueError("--train-seq-lens must be a subset of --seq-lens")
        if sorted(train_seq_lens) != sorted(seq_lens):
            train_set = {int(x) for x in train_seq_lens}

            def record_seq_len(idx: int) -> int:
                return len(records[int(idx)].frame_paths)

            ood_extra = [idx for idx in splits["train"] + splits["val"] if record_seq_len(idx) not in train_set]
            splits["train"] = [idx for idx in splits["train"] if record_seq_len(idx) in train_set]
            splits["val"] = [idx for idx in splits["val"] if record_seq_len(idx) in train_set]
            splits["test"] = sorted(
                splits["test"] + ood_extra,
                key=lambda idx: (len(records[idx].frame_paths), records[idx].sample_id),
            )
        base.print_split_counts(records, splits, seq_lens)
        if should_run_sum_evidence and (not splits["train"] or not splits["val"]):
            raise RuntimeError("Sum-evidence training requires non-empty train and val splits")
        if not splits["test"]:
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
        adapter: Optional[SimpleSumEvidenceAdapter] = None
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
                indices=splits["test"],
                count_token_ids=count_token_ids,
                device=device,
                batch_size=int(args.batch_size),
                seed=int(args.seed) + 101,
                inject_layers=inject_layers,
            )
            metrics_rows.extend(baseline_eval["rows"])

        if should_run_sum_evidence:
            print("Training simple sum-evidence adapter")
            adapter, train_history, backward_diag, checkpoint_path = train_adapter(
                args=args,
                output_dir=output_dir,
                model=model,
                processor=processor,
                records=records,
                train_indices=splits["train"],
                val_indices=splits["val"],
                count_token_ids=count_token_ids,
                hidden_size=int(hidden_size),
                inject_layers=inject_layers,
                device=device,
            )
            base.write_json(
                output_dir / "checkpoint.json",
                {
                    "sum_evidence_best_checkpoint": os.fspath(checkpoint_path),
                    "readout_mode": ADDITIVE_SUM_READOUT,
                    "message_token_group": str(args.message_token_group),
                    "inject_token_group": str(args.inject_token_group),
                },
            )
            print("Evaluating sum-evidence adapter on test split")
            layer_eval = evaluate_model(
                args=args,
                method=SUM_EVIDENCE,
                split_name="test",
                model=model,
                processor=processor,
                adapter=adapter,
                records=records,
                indices=splits["test"],
                count_token_ids=count_token_ids,
                device=device,
                batch_size=int(args.batch_size),
                seed=int(args.seed) + 202,
                inject_layers=inject_layers,
            )
            metrics_rows.extend(layer_eval["rows"])

        summary_rows: List[Dict[str, Any]] = []
        if should_run_baseline:
            summary_rows.append(summarize_method(method_rows(metrics_rows, BASELINE), method=BASELINE))
        if should_run_sum_evidence:
            summary_rows.append(
                summarize_method(method_rows(metrics_rows, SUM_EVIDENCE), method=SUM_EVIDENCE, train_history=train_history)
            )
        accuracy_rows: List[Dict[str, Any]] = []
        for method in [BASELINE, SUM_EVIDENCE]:
            if any(row.get("method") == method for row in metrics_rows):
                accuracy_rows.extend(accuracy_by_seq_len(metrics_rows, method, seq_lens))
        comparison_rows = comparison_by_seq_len(accuracy_rows, seq_lens)

        base.write_csv_dynamic(
            output_dir / "metrics.csv",
            metrics_rows,
            [
                "method",
                "sample_id",
                "seq_len",
                "gold_count",
                "evidence_count",
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
                "update_norm",
                "message_norm",
                "summed_message_norm",
                "update_norm_by_layer_json",
                "message_norm_by_layer_json",
                "raw_message_norm_by_layer_json",
                "summed_message_norm_by_layer_json",
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
                "mean_summed_message_norm",
                "train_accuracy",
                "val_accuracy",
                "val_ce",
            ],
        )
        base.write_csv_dynamic(
            output_dir / "accuracy_by_seq_len.csv",
            accuracy_rows,
            [
                "method",
                "seq_len",
                "gold_count",
                "n",
                "accuracy",
                "mean_margin",
                "mean_pred_count",
                "prediction_histogram",
                "mean_update_norm",
                "mean_message_norm",
                "mean_summed_message_norm",
            ],
        )
        base.write_csv_dynamic(
            output_dir / "comparison_by_seq_len.csv",
            comparison_rows,
            [
                "seq_len",
                "gold_count",
                "baseline_accuracy",
                "sum_evidence_accuracy",
                "delta_accuracy",
                "baseline_mean_pred",
                "sum_evidence_mean_pred",
                "baseline_mean_margin",
                "sum_evidence_mean_margin",
            ],
        )
        if train_history:
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
                    "val_ce",
                    "val_accuracy",
                    "adapter_parameter_norm",
                ],
            )
        if not bool(args.no_plots):
            make_plots(output_dir, metrics_rows, accuracy_rows, seq_lens)
        diagnostics = write_diagnostics(
            output_dir=output_dir,
            model=model,
            adapter=adapter,
            train_history=train_history,
            backward_diag=backward_diag,
            metrics_rows=metrics_rows,
            args=args,
        )
        write_readme(output_dir, summary_rows, accuracy_rows, metrics_rows, diagnostics)
        base.write_json(
            output_dir / "run_done.json",
            {
                "completed": True,
                "elapsed_seconds": time.time() - started,
                "output_dir": os.fspath(output_dir),
                "readout_mode": ADDITIVE_SUM_READOUT,
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

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from torch.utils.data import DataLoader

from scripts.probes import run_message_memory_adapter_stage1_stage3_seq8 as prev


STAGE1_CURRENT_FREE = "stage1_current_free_value"
STAGE1_SHARED = "stage1_shared_count_direction"
STAGE1_SHARED_RESIDUAL = "stage1_shared_count_direction_plus_small_residual"
STAGE3_CURRENT_FREE = "stage3_current_free_value"
STAGE3_SHARED_RESIDUAL = "stage3_shared_count_direction_plus_small_residual"

STAGE1_VARIANTS = (STAGE1_CURRENT_FREE, STAGE1_SHARED, STAGE1_SHARED_RESIDUAL)
STAGE3_VARIANTS = (STAGE3_CURRENT_FREE, STAGE3_SHARED_RESIDUAL)
STAGE3_TO_STAGE1 = {
    STAGE3_CURRENT_FREE: STAGE1_CURRENT_FREE,
    STAGE3_SHARED_RESIDUAL: STAGE1_SHARED_RESIDUAL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared count-direction memory-write experiment for seq_len=8 message-memory adapters."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/mmred_images_park")
    parser.add_argument("--source-run", type=Path, default=prev.DEFAULT_SOURCE_RUN)
    parser.add_argument("--base-source-run", type=Path, default=prev.DEFAULT_BASE_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="all_uniform")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in range(9)])
    parser.add_argument("--layers", nargs="+", default=[str(x) for x in prev.DEFAULT_LAYERS])
    parser.add_argument("--carriers", nargs="+", default=["target_char", "target_room"])
    parser.add_argument("--max-samples-per-count", type=int, default=100)
    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--bottleneck-dim", "--d-m", type=int, default=256)
    parser.add_argument("--key-dim", "--d-k", type=int, default=64)
    parser.add_argument("--value-dim", "--d-v", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--frame-gate-bce-weight", type=float, default=0.1)
    parser.add_argument("--count-direction-mse-weight", type=float, default=0.5)
    parser.add_argument("--residual-scale", type=float, default=0.1)

    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage1-weight-decay", type=float, default=1e-2)
    parser.add_argument("--stage1-patience", type=int, default=3)
    parser.add_argument("--stage1-batch-size", type=int, default=64)

    parser.add_argument("--stage3-epochs", type=int, default=3)
    parser.add_argument("--stage3-lr", type=float, default=1e-4)
    parser.add_argument("--stage3-weight-decay", type=float, default=1e-2)
    parser.add_argument("--stage3-batch-size", type=int, default=1)
    parser.add_argument("--stage3-grad-accum", type=int, default=8)
    parser.add_argument("--stage3-patience", type=int, default=3)
    parser.add_argument("--stage3-grad-clip", type=float, default=1.0)
    parser.add_argument("--aux-mem-ce-weight", type=float, default=0.2)
    parser.add_argument("--count-direction-mse-weight-stage3", type=float, default=0.1)
    parser.add_argument("--residual-l2-weight", type=float, default=1e-5)
    parser.add_argument("--inject-layer", type=int, default=18)
    parser.add_argument("--gamma-init", type=float, default=1.0)
    parser.add_argument("--train-gamma", action=argparse.BooleanOptionalAction, default=True)

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


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / f"shared_count_direction_memory_seq8_7b_{stamp}"


def stage1_to_core_variant(variant: str) -> str:
    if variant == STAGE1_CURRENT_FREE:
        return "current_free_value"
    if variant == STAGE1_SHARED:
        return "shared_count_direction"
    if variant == STAGE1_SHARED_RESIDUAL:
        return "shared_count_direction_plus_small_residual"
    raise ValueError(f"Unknown Stage 1 variant: {variant}")


def stage3_to_core_variant(variant: str) -> str:
    if variant == STAGE3_CURRENT_FREE:
        return "current_free_value"
    if variant == STAGE3_SHARED_RESIDUAL:
        return "shared_count_direction_plus_small_residual"
    raise ValueError(f"Unknown Stage 3 variant: {variant}")


def is_shared_core_variant(variant: str) -> bool:
    return str(variant) in {"shared_count_direction", "shared_count_direction_plus_small_residual"}


class SharedCountMemoryCore(nn.Module):
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        key_dim: int,
        value_dim: int,
        dropout: float,
        residual_scale: float,
        variant: str,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.residual_scale = float(residual_scale)
        self.variant = str(variant)
        self.norm = nn.LayerNorm(self.input_dim)
        self.w_p = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.w_k = nn.Linear(self.bottleneck_dim, self.key_dim, bias=False)
        self.w_v = nn.Linear(self.bottleneck_dim, self.value_dim, bias=False)
        self.w_u = nn.Linear(self.bottleneck_dim, self.value_dim, bias=False)
        self.w_alpha = nn.Linear(self.bottleneck_dim, 1)
        self.q0 = nn.Parameter(torch.randn(self.key_dim) / math.sqrt(float(self.key_dim)))
        self.count_direction = nn.Parameter(torch.randn(self.value_dim) / math.sqrt(float(self.value_dim)))
        self.projection_direction = nn.Parameter(torch.randn(self.value_dim) / math.sqrt(float(self.value_dim)))
        self.register_buffer("x_mean", torch.zeros(self.input_dim), persistent=True)
        self.register_buffer("x_std", torch.ones(self.input_dim), persistent=True)

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.x_mean.data.copy_(mean.float().reshape(-1))
        self.x_std.data.copy_(std.float().reshape(-1).clamp_min(1e-6))

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.x_mean.to(x.device)) / self.x_std.to(x.device).clamp_min(1e-6)

    def projection_axis(self) -> torch.Tensor:
        if is_shared_core_variant(self.variant):
            return F.normalize(self.count_direction, dim=0)
        return F.normalize(self.projection_direction, dim=0)

    def forward(self, x_messages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.standardize(x_messages)
        z = self.dropout(F.gelu(self.w_p(self.norm(x))))
        alpha = torch.sigmoid(self.w_alpha(z)).squeeze(-1)
        k = F.normalize(self.w_k(z), dim=-1)
        q = F.normalize(self.q0, dim=0)
        key_query_similarity = torch.matmul(k, q)

        if self.variant == "current_free_value":
            write_value = self.w_v(z)
        elif self.variant == "shared_count_direction":
            c = self.count_direction.to(z.device).view(1, 1, -1)
            write_value = c.expand(z.shape[0], z.shape[1], -1)
        elif self.variant == "shared_count_direction_plus_small_residual":
            c = self.count_direction.to(z.device).view(1, 1, -1)
            write_value = c + float(self.residual_scale) * self.w_u(z)
        else:
            raise ValueError(f"Unknown memory core variant: {self.variant}")

        write = alpha.unsqueeze(-1) * write_value
        r = torch.sum(write * key_query_similarity.unsqueeze(-1), dim=1)
        axis = self.projection_axis().to(r.device)
        count_projection = torch.matmul(r, axis)
        diagnostics = {
            "key_query_similarity": key_query_similarity,
            "count_projection": count_projection,
            "r_norm": r.float().norm(dim=-1),
            "sum_alpha": alpha.sum(dim=-1),
        }
        return r, alpha, diagnostics


class Stage1SharedCountReadout(nn.Module):
    def __init__(self, core: SharedCountMemoryCore, variant: str, num_classes: int) -> None:
        super().__init__()
        self.core = core
        self.variant = str(variant)
        self.count_head = nn.Linear(core.value_dim, int(num_classes))
        self.scalar_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.scalar_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward_with_diagnostics(
        self, x_messages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        r, alpha, diagnostics = self.core(x_messages)
        logits = self.count_head(r)
        count_scalar = self.scalar_scale.to(r.device) * diagnostics["count_projection"] + self.scalar_bias.to(r.device)
        diagnostics = dict(diagnostics)
        diagnostics["count_scalar"] = count_scalar
        return logits, alpha, r, diagnostics

    def forward(self, x_messages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, alpha, r, _diagnostics = self.forward_with_diagnostics(x_messages)
        return logits, alpha, r


class Stage3SharedCountAdapter(nn.Module):
    def __init__(
        self,
        *,
        core: SharedCountMemoryCore,
        count_head: nn.Module,
        scalar_scale: torch.Tensor,
        scalar_bias: torch.Tensor,
        variant: str,
        hidden_size: int,
        inject_layer: int,
        gamma_init: float,
        train_gamma: bool,
    ) -> None:
        super().__init__()
        self.core = core
        self.variant = str(variant)
        self.count_head = count_head
        self.scalar_scale = nn.Parameter(scalar_scale.detach().float().clone())
        self.scalar_bias = nn.Parameter(scalar_bias.detach().float().clone())
        self.w_o = nn.Linear(core.value_dim, int(hidden_size), bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32), requires_grad=bool(train_gamma))
        self.inject_layer = int(inject_layer)
        self.enabled = True
        self._x_messages: Optional[torch.Tensor] = None
        self._target_positions: Optional[List[List[int]]] = None
        self._handles: List[Any] = []
        self.last_alpha: Optional[torch.Tensor] = None
        self.last_key_query_similarity: Optional[torch.Tensor] = None
        self.last_count_projection: Optional[torch.Tensor] = None
        self.last_count_scalar: Optional[torch.Tensor] = None
        self.last_r_norm: Optional[torch.Tensor] = None
        self.last_delta_norm: Optional[torch.Tensor] = None
        self.last_alpha_for_loss: Optional[torch.Tensor] = None
        self.last_delta_for_loss: Optional[torch.Tensor] = None
        self.last_mem_logits_for_loss: Optional[torch.Tensor] = None
        self.last_count_scalar_for_loss: Optional[torch.Tensor] = None
        nn.init.zeros_(self.w_o.weight)

    def set_context(self, x_messages: torch.Tensor, target_positions: Sequence[Sequence[int]]) -> None:
        self._x_messages = x_messages
        self._target_positions = [[int(pos) for pos in positions] for positions in target_positions]
        self.last_alpha = None
        self.last_key_query_similarity = None
        self.last_count_projection = None
        self.last_count_scalar = None
        self.last_r_norm = None
        self.last_delta_norm = None
        self.last_alpha_for_loss = None
        self.last_delta_for_loss = None
        self.last_mem_logits_for_loss = None
        self.last_count_scalar_for_loss = None

    def clear_context(self) -> None:
        self._x_messages = None
        self._target_positions = None
        self.last_alpha_for_loss = None
        self.last_delta_for_loss = None
        self.last_mem_logits_for_loss = None
        self.last_count_scalar_for_loss = None

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

    def inject(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self._x_messages is None or self._target_positions is None:
            return hidden_states
        r, alpha, diagnostics = self.core(self._x_messages.to(hidden_states.device))
        mem_logits = self.count_head(r)
        count_scalar = self.scalar_scale.to(r.device) * diagnostics["count_projection"] + self.scalar_bias.to(r.device)
        delta = self.w_o(r)
        self.last_alpha_for_loss = alpha
        self.last_delta_for_loss = delta
        self.last_mem_logits_for_loss = mem_logits
        self.last_count_scalar_for_loss = count_scalar
        self.last_alpha = alpha.detach().float().cpu()
        self.last_key_query_similarity = diagnostics["key_query_similarity"].detach().float().cpu()
        self.last_count_projection = diagnostics["count_projection"].detach().float().cpu()
        self.last_count_scalar = count_scalar.detach().float().cpu()
        self.last_r_norm = diagnostics["r_norm"].detach().float().cpu()
        self.last_delta_norm = delta.detach().float().norm(dim=-1).cpu()
        updates: List[torch.Tensor] = []
        seq_len = int(hidden_states.shape[1])
        for batch_idx, positions in enumerate(self._target_positions):
            valid = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len})
            mask = hidden_states.new_zeros((seq_len, 1))
            if valid:
                pos_idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
                mask.index_fill_(0, pos_idx, 1.0)
            updates.append(mask * (self.gamma.to(hidden_states.device) * delta[batch_idx]).to(hidden_states.dtype).unsqueeze(0))
        return hidden_states + torch.stack(updates, dim=0)

    def register_hooks(self, model: Any) -> None:
        self.remove_hooks()
        layers = prev.get_layers(model)
        if self.inject_layer < 0 or self.inject_layer >= len(layers):
            raise ValueError(f"inject_layer={self.inject_layer} outside [0, {len(layers) - 1}]")

        def hook(_module: Any, _args: Any, output: Any) -> Any:
            hidden = self._hidden_from_output(output)
            if hidden is None:
                return output
            return self._replace_hidden(output, self.inject(hidden))

        self._handles.append(layers[self.inject_layer].register_forward_hook(hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def make_stage1_model(
    args: argparse.Namespace,
    variant: str,
    input_dim: int,
    num_classes: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Stage1SharedCountReadout:
    core = SharedCountMemoryCore(
        input_dim=int(input_dim),
        bottleneck_dim=int(args.bottleneck_dim),
        key_dim=int(args.key_dim),
        value_dim=int(args.value_dim),
        dropout=float(args.dropout),
        residual_scale=float(args.residual_scale),
        variant=stage1_to_core_variant(variant),
    )
    core.set_standardizer(mean, std)
    return Stage1SharedCountReadout(core, variant=variant, num_classes=int(num_classes))


def train_stage1_variant(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    variant: str,
    x_messages: torch.Tensor,
    labels: torch.Tensor,
    frame_labels: torch.Tensor,
    splits: Dict[str, List[int]],
    candidate_min: int,
    candidate_max: int,
) -> Tuple[Stage1SharedCountReadout, Dict[str, Any], Path]:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{variant}_best.pt"
    num_classes = int(candidate_max) - int(candidate_min) + 1
    input_dim = int(x_messages.shape[-1])
    train_device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    mean, std = prev.compute_standardizer(x_messages, splits["train"])
    model = make_stage1_model(args, variant, input_dim, num_classes, mean, std).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.stage1_lr), weight_decay=float(args.stage1_weight_decay))
    train_loader = DataLoader(
        prev.MessageDataset(x_messages, labels, frame_labels, splits["train"]),
        batch_size=int(args.stage1_batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(args.seed) + 11 + stable_variant_offset(variant)),
    )
    val_loader = DataLoader(
        prev.MessageDataset(x_messages, labels, frame_labels, splits["val"]),
        batch_size=int(args.stage1_batch_size),
        shuffle=False,
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_ce = math.inf
    best_val_acc = -math.inf
    best_epoch = 0
    bad_epochs = 0
    rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.stage1_epochs) + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            _idx, x, y, y_frame = prev.batch_to_device(batch, train_device)
            y_offset = (y - int(candidate_min)).long()
            optimizer.zero_grad(set_to_none=True)
            logits, alpha, _r, diagnostics = model.forward_with_diagnostics(x)
            count_loss = F.cross_entropy(logits, y_offset)
            gate_loss = F.binary_cross_entropy(alpha, y_frame.float())
            count_direction_loss = F.mse_loss(diagnostics["count_scalar"], y.float())
            loss = count_loss + float(args.frame_gate_bce_weight) * gate_loss
            if variant in (STAGE1_SHARED, STAGE1_SHARED_RESIDUAL):
                loss = loss + float(args.count_direction_mse_weight) * count_direction_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            total_batches += 1
        val = evaluate_stage1_variant(
            model=model,
            loader=val_loader,
            labels=labels,
            frame_labels=frame_labels,
            candidate_min=candidate_min,
            device=train_device,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": total_loss / max(1, total_batches),
            "val_ce": float(val["ce"]),
            "val_accuracy": float(val["accuracy"]),
            "val_gate_bce": float(val["gate_bce"]),
            "val_sum_alpha_mse": float(val["sum_alpha_mse"]),
            "val_count_direction_mse": float(val["count_direction_mse"]),
        }
        rows.append(row)
        improved = row["val_ce"] < best_val_ce - 1e-9 or (
            abs(row["val_ce"] - best_val_ce) <= 1e-9 and row["val_accuracy"] > best_val_acc
        )
        if improved:
            best_val_ce = float(row["val_ce"])
            best_val_acc = float(row["val_accuracy"])
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"  {variant} epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_ce={row['val_ce']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_dir_mse={row['val_count_direction_mse']:.4f}"
        )
        if bad_epochs >= int(args.stage1_patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model_cpu = model.cpu()
    ckpt = {
        "model_state_dict": model_cpu.state_dict(),
        "x_mean": mean.float(),
        "x_std": std.float(),
        "input_dim": int(input_dim),
        "candidate_min": int(candidate_min),
        "candidate_max": int(candidate_max),
        "variant": str(variant),
        "history": {
            "rows": rows,
            "best_epoch": int(best_epoch),
            "best_val_accuracy": float(best_val_acc),
            "best_val_ce": float(best_val_ce),
        },
    }
    torch.save(ckpt, best_path)
    print(f"Saved {variant} checkpoint: {best_path}")
    return model_cpu, ckpt["history"], best_path


@torch.no_grad()
def evaluate_stage1_variant(
    *,
    model: Stage1SharedCountReadout,
    loader: DataLoader,
    labels: torch.Tensor,
    frame_labels: torch.Tensor,
    candidate_min: int,
    device: torch.device,
) -> Dict[str, Any]:
    model = model.to(device)
    model.eval()
    pred_by_idx: Dict[int, int] = {}
    logits_by_idx: Dict[int, List[float]] = {}
    alpha_by_idx: Dict[int, List[float]] = {}
    key_query_by_idx: Dict[int, List[float]] = {}
    count_projection_by_idx: Dict[int, float] = {}
    count_scalar_by_idx: Dict[int, float] = {}
    r_norm_by_idx: Dict[int, float] = {}
    ce_total = 0.0
    bce_total = 0.0
    sum_alpha_mse_total = 0.0
    count_direction_mse_total = 0.0
    n = 0
    for batch in loader:
        idx, x, y, y_frame = prev.batch_to_device(batch, device)
        y_offset = (y - int(candidate_min)).long()
        logits, alpha, _r, diagnostics = model.forward_with_diagnostics(x)
        ce_total += float(F.cross_entropy(logits, y_offset, reduction="sum").detach().cpu().item())
        bce_total += float(F.binary_cross_entropy(alpha, y_frame.float(), reduction="sum").detach().cpu().item())
        sum_alpha_mse_total += float(F.mse_loss(alpha.sum(dim=-1), y.float(), reduction="sum").detach().cpu().item())
        count_direction_mse_total += float(
            F.mse_loss(diagnostics["count_scalar"], y.float(), reduction="sum").detach().cpu().item()
        )
        n += int(y.numel())
        pred = logits.argmax(dim=-1) + int(candidate_min)
        for row, sample_idx in enumerate(idx.detach().cpu().tolist()):
            pred_by_idx[int(sample_idx)] = int(pred[row].detach().cpu().item())
            logits_by_idx[int(sample_idx)] = [float(v) for v in logits[row].detach().float().cpu().tolist()]
            alpha_by_idx[int(sample_idx)] = [float(v) for v in alpha[row].detach().float().cpu().tolist()]
            key_query_by_idx[int(sample_idx)] = [
                float(v) for v in diagnostics["key_query_similarity"][row].detach().float().cpu().tolist()
            ]
            count_projection_by_idx[int(sample_idx)] = float(
                diagnostics["count_projection"][row].detach().float().cpu().item()
            )
            count_scalar_by_idx[int(sample_idx)] = float(diagnostics["count_scalar"][row].detach().float().cpu().item())
            r_norm_by_idx[int(sample_idx)] = float(diagnostics["r_norm"][row].detach().float().cpu().item())
    y_true = [int(labels[int(idx)].item()) for idx in pred_by_idx]
    y_pred = [pred_by_idx[int(idx)] for idx in pred_by_idx]
    return {
        "ce": ce_total / max(1, n),
        "gate_bce": bce_total / max(1, n * int(frame_labels.shape[1])),
        "sum_alpha_mse": sum_alpha_mse_total / max(1, n),
        "count_direction_mse": count_direction_mse_total / max(1, n),
        "accuracy": prev.accuracy(y_true, y_pred),
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "alpha_by_idx": alpha_by_idx,
        "key_query_by_idx": key_query_by_idx,
        "count_projection_by_idx": count_projection_by_idx,
        "count_scalar_by_idx": count_scalar_by_idx,
        "r_norm_by_idx": r_norm_by_idx,
    }


def make_stage3_adapter(
    *,
    args: argparse.Namespace,
    stage1_model: Stage1SharedCountReadout,
    stage3_variant: str,
    hidden_size: int,
) -> Stage3SharedCountAdapter:
    core = SharedCountMemoryCore(
        input_dim=stage1_model.core.input_dim,
        bottleneck_dim=stage1_model.core.bottleneck_dim,
        key_dim=stage1_model.core.key_dim,
        value_dim=stage1_model.core.value_dim,
        dropout=float(args.dropout),
        residual_scale=float(args.residual_scale),
        variant=stage3_to_core_variant(stage3_variant),
    )
    core.load_state_dict(stage1_model.core.state_dict())
    core.variant = stage3_to_core_variant(stage3_variant)
    count_head = nn.Linear(stage1_model.core.value_dim, stage1_model.count_head.out_features)
    count_head.load_state_dict(stage1_model.count_head.state_dict())
    return Stage3SharedCountAdapter(
        core=core,
        count_head=count_head,
        scalar_scale=stage1_model.scalar_scale,
        scalar_bias=stage1_model.scalar_bias,
        variant=stage3_variant,
        hidden_size=int(hidden_size),
        inject_layer=int(args.inject_layer),
        gamma_init=float(args.gamma_init),
        train_gamma=bool(args.train_gamma),
    )


def train_stage3_variant(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    stage3_variant: str,
    model: Any,
    processor: Any,
    stage1_model: Stage1SharedCountReadout,
    records: Sequence[prev.SampleRecord],
    x_messages: torch.Tensor,
    splits: Dict[str, List[int]],
    count_token_ids: Dict[int, int],
    device: str,
    hidden_size: int,
) -> Tuple[Stage3SharedCountAdapter, Dict[str, Any], Path]:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{stage3_variant}_best.pt"
    adapter = make_stage3_adapter(
        args=args,
        stage1_model=stage1_model,
        stage3_variant=stage3_variant,
        hidden_size=hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.stage3_lr), weight_decay=float(args.stage3_weight_decay))
    train_indices = list(splits["train"])
    val_indices = list(splits["val"])
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_ce = math.inf
    best_val_acc = -math.inf
    best_epoch = 0
    bad_epochs = 0
    rows: List[Dict[str, Any]] = []
    try:
        for epoch in range(1, int(args.stage3_epochs) + 1):
            adapter.train()
            adapter.register_hooks(model)
            rng = random.Random(int(args.seed) + 1000 + epoch + stable_variant_offset(stage3_variant))
            shuffled = list(train_indices)
            rng.shuffle(shuffled)
            optimizer.zero_grad(set_to_none=True)
            train_ce = 0.0
            train_mem_ce = 0.0
            train_dir_mse = 0.0
            train_l2 = 0.0
            train_steps = 0
            for step, batch_indices in enumerate(prev.chunked(shuffled, int(args.stage3_batch_size)), start=1):
                batch_records = [records[idx] for idx in batch_indices]
                batch = prev.prepare_qwen_batch(
                    records=batch_records,
                    sample_indices=batch_indices,
                    processor=processor,
                    device=device,
                    carriers=args.carriers,
                )
                adapter.set_context(x_messages[batch_indices].to(device), batch.target_positions)
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - min(count_token_ids.keys())
                ce = F.cross_entropy(count_logits, gold_offsets)
                mem_ce = aux_mem_ce_for_adapter(adapter, gold_offsets)
                dir_mse = count_direction_mse_for_adapter(adapter, batch.gold_counts)
                residual_l2 = residual_l2_for_adapter(adapter)
                loss = (
                    ce
                    + float(args.aux_mem_ce_weight) * mem_ce
                    + float(args.count_direction_mse_weight_stage3) * dir_mse
                    + float(args.residual_l2_weight) * residual_l2
                ) / max(1, int(args.stage3_grad_accum))
                loss.backward()
                train_ce += float(ce.detach().cpu().item())
                train_mem_ce += float(mem_ce.detach().cpu().item())
                train_dir_mse += float(dir_mse.detach().cpu().item())
                train_l2 += float(residual_l2.detach().cpu().item())
                train_steps += 1
                adapter.clear_context()
                if step % max(1, int(args.stage3_grad_accum)) == 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.stage3_grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 100 == 0:
                    print(
                        f"  {stage3_variant} epoch={epoch} step={step} "
                        f"train_ce={train_ce / max(1, train_steps):.4f}"
                    )
            if train_steps % max(1, int(args.stage3_grad_accum)) != 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.stage3_grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            val_eval = evaluate_qwen_count_channel(
                method=stage3_variant,
                model=model,
                processor=processor,
                adapter=adapter,
                records=records,
                indices=val_indices,
                x_messages=x_messages,
                count_token_ids=count_token_ids,
                args=args,
                device=device,
                batch_size=int(args.stage3_batch_size),
                save_gates=False,
            )
            row = {
                "epoch": int(epoch),
                "train_ce": train_ce / max(1, train_steps),
                "train_aux_mem_ce": train_mem_ce / max(1, train_steps),
                "train_count_direction_mse": train_dir_mse / max(1, train_steps),
                "train_residual_l2": train_l2 / max(1, train_steps),
                "val_ce": float(val_eval["ce"]),
                "val_accuracy": float(val_eval["accuracy"]),
                "val_count_direction_mse": float(val_eval["count_direction_mse"]),
                "gamma": float(adapter.gamma.detach().cpu().item()),
            }
            rows.append(row)
            print(
                f"  {stage3_variant} epoch={epoch} train_ce={row['train_ce']:.4f} "
                f"val_ce={row['val_ce']:.4f} val_acc={row['val_accuracy']:.4f} "
                f"val_dir_mse={row['val_count_direction_mse']:.4f} gamma={row['gamma']:.4f}"
            )
            improved = row["val_ce"] < best_val_ce - 1e-9 or (
                abs(row["val_ce"] - best_val_ce) <= 1e-9 and row["val_accuracy"] > best_val_acc
            )
            if improved:
                best_val_ce = float(row["val_ce"])
                best_val_acc = float(row["val_accuracy"])
                best_epoch = int(epoch)
                best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if int(args.stage3_patience) > 0 and bad_epochs >= int(args.stage3_patience):
                break
    finally:
        adapter.remove_hooks()
    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter_cpu = adapter.cpu()
    ckpt = {
        "adapter_state_dict": adapter_cpu.state_dict(),
        "variant": str(stage3_variant),
        "history": {
            "rows": rows,
            "best_epoch": int(best_epoch),
            "best_val_ce": float(best_val_ce),
            "best_val_accuracy": float(best_val_acc),
        },
        "hidden_size": int(hidden_size),
        "inject_layer": int(args.inject_layer),
        "carriers": list(args.carriers),
    }
    torch.save(ckpt, best_path)
    print(f"Saved {stage3_variant} checkpoint: {best_path}")
    return adapter_cpu, ckpt["history"], best_path


def aux_mem_ce_for_adapter(adapter: Stage3SharedCountAdapter, gold_offsets: torch.Tensor) -> torch.Tensor:
    if adapter.last_mem_logits_for_loss is None:
        return adapter.gamma.new_tensor(0.0)
    return F.cross_entropy(adapter.last_mem_logits_for_loss, gold_offsets.long())


def count_direction_mse_for_adapter(adapter: Stage3SharedCountAdapter, gold_counts: torch.Tensor) -> torch.Tensor:
    if adapter.last_count_scalar_for_loss is None:
        return gold_counts.float().new_tensor(0.0)
    return F.mse_loss(adapter.last_count_scalar_for_loss, gold_counts.float())


def residual_l2_for_adapter(adapter: Stage3SharedCountAdapter) -> torch.Tensor:
    if adapter.last_delta_for_loss is None:
        return adapter.gamma.new_tensor(0.0)
    return adapter.last_delta_for_loss.float().pow(2).sum(dim=-1).mean()


def evaluate_qwen_count_channel(
    *,
    method: str,
    model: Any,
    processor: Any,
    adapter: Optional[Stage3SharedCountAdapter],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    x_messages: torch.Tensor,
    count_token_ids: Dict[int, int],
    args: argparse.Namespace,
    device: str,
    batch_size: int,
    save_gates: bool,
) -> Dict[str, Any]:
    if adapter is not None:
        adapter.to(device)
        adapter.eval()
        adapter.enabled = True
        adapter.register_hooks(model)
    pred_by_idx: Dict[int, int] = {}
    logits_by_idx: Dict[int, List[float]] = {}
    gold_score_by_idx: Dict[int, float] = {}
    gate_by_idx: Dict[int, List[float]] = {}
    sum_alpha_by_idx: Dict[int, float] = {}
    key_query_by_idx: Dict[int, List[float]] = {}
    count_projection_by_idx: Dict[int, float] = {}
    count_scalar_by_idx: Dict[int, float] = {}
    r_norm_by_idx: Dict[int, float] = {}
    delta_norm_by_idx: Dict[int, float] = {}
    ce_total = 0.0
    dir_mse_total = 0.0
    n = 0
    try:
        for batch_num, batch_indices in enumerate(prev.chunked(list(indices), int(batch_size)), start=1):
            batch_records = [records[idx] for idx in batch_indices]
            batch = prev.prepare_qwen_batch(
                records=batch_records,
                sample_indices=batch_indices,
                processor=processor,
                device=device,
                carriers=args.carriers,
            )
            if adapter is not None:
                adapter.set_context(x_messages[batch_indices].to(device), batch.target_positions)
            with torch.no_grad():
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - min(count_token_ids.keys())
                ce_total += float(F.cross_entropy(count_logits, gold_offsets, reduction="sum").detach().cpu().item())
                n += int(batch.gold_counts.numel())
                pred_offsets = count_logits.argmax(dim=-1)
                logits_cpu = count_logits.detach().float().cpu()
                for row, idx in enumerate(batch_indices):
                    pred_by_idx[int(idx)] = int(pred_offsets[row].detach().cpu().item()) + min(count_token_ids.keys())
                    values = [float(v) for v in logits_cpu[row].tolist()]
                    logits_by_idx[int(idx)] = values
                    gold_offset = int(records[int(idx)].gold_count) - min(count_token_ids.keys())
                    gold_score_by_idx[int(idx)] = float(values[gold_offset])
                if adapter is not None and adapter.last_alpha is not None:
                    sums = adapter.last_alpha.sum(dim=-1)
                    gold = batch.gold_counts.detach().float().cpu()
                    if adapter.last_count_scalar is not None:
                        dir_mse_total += float(F.mse_loss(adapter.last_count_scalar.float(), gold, reduction="sum").item())
                    for row, idx in enumerate(batch_indices):
                        sum_alpha_by_idx[int(idx)] = float(sums[row].item())
                        if save_gates:
                            gate_by_idx[int(idx)] = [float(v) for v in adapter.last_alpha[row].tolist()]
                        if adapter.last_key_query_similarity is not None:
                            key_query_by_idx[int(idx)] = [
                                float(v) for v in adapter.last_key_query_similarity[row].tolist()
                            ]
                        if adapter.last_count_projection is not None:
                            count_projection_by_idx[int(idx)] = float(adapter.last_count_projection[row].item())
                        if adapter.last_count_scalar is not None:
                            count_scalar_by_idx[int(idx)] = float(adapter.last_count_scalar[row].item())
                        if adapter.last_r_norm is not None:
                            r_norm_by_idx[int(idx)] = float(adapter.last_r_norm[row].item())
                    if adapter.last_delta_norm is not None:
                        for row, idx in enumerate(batch_indices):
                            delta_norm_by_idx[int(idx)] = float(adapter.last_delta_norm[row].item())
            if adapter is not None:
                adapter.clear_context()
            if batch_num == 1 or batch_num % 50 == 0:
                print(f"  eval {method}: {min(len(indices), batch_num * int(batch_size))}/{len(indices)}")
    finally:
        if adapter is not None:
            adapter.remove_hooks()
    y_true = [int(records[int(idx)].gold_count) for idx in indices]
    y_pred = [pred_by_idx[int(idx)] for idx in indices]
    return {
        "ce": ce_total / max(1, n),
        "accuracy": prev.accuracy(y_true, y_pred),
        "count_direction_mse": dir_mse_total / max(1, n) if adapter is not None else math.nan,
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "gold_score_by_idx": gold_score_by_idx,
        "gate_by_idx": gate_by_idx,
        "sum_alpha_by_idx": sum_alpha_by_idx,
        "key_query_by_idx": key_query_by_idx,
        "count_projection_by_idx": count_projection_by_idx,
        "count_scalar_by_idx": count_scalar_by_idx,
        "r_norm_by_idx": r_norm_by_idx,
        "delta_norm_by_idx": delta_norm_by_idx,
    }


def stable_variant_offset(name: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(name))) % 100_000


def gate_debug(
    gates_by_idx: Dict[int, List[float]],
    frame_labels: torch.Tensor,
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
) -> Dict[str, Any]:
    labels: List[int] = []
    scores: List[float] = []
    sums: List[float] = []
    golds: List[float] = []
    sums_by_count: Dict[int, List[float]] = defaultdict(list)
    alpha_by_label: Dict[int, List[float]] = defaultdict(list)
    for idx in indices:
        idx = int(idx)
        gates = gates_by_idx.get(idx)
        if gates is None:
            continue
        frame_y = [int(v) for v in frame_labels[idx].int().tolist()]
        labels.extend(frame_y)
        scores.extend([float(v) for v in gates])
        for label, gate in zip(frame_y, gates):
            alpha_by_label[int(label)].append(float(gate))
        s = float(sum(gates))
        gold = float(records[idx].gold_count)
        sums.append(s)
        golds.append(gold)
        sums_by_count[int(gold)].append(s)
    pred = [1 if score >= 0.5 else 0 for score in scores]
    corr = math.nan
    if len(sums) >= 2 and float(np.std(sums)) > 0 and float(np.std(golds)) > 0:
        corr = float(np.corrcoef(np.array(sums), np.array(golds))[0, 1])
    return {
        "frame_gate_accuracy_at_0_5": prev.accuracy(labels, pred) if labels else math.nan,
        "frame_gate_auc": prev.auroc_binary(labels, scores) if labels else math.nan,
        "mean_alpha_non_evidence": float(np.mean(alpha_by_label[0])) if alpha_by_label[0] else math.nan,
        "mean_alpha_evidence": float(np.mean(alpha_by_label[1])) if alpha_by_label[1] else math.nan,
        "count_sum_mse": float(np.mean([(s - g) ** 2 for s, g in zip(sums, golds)])) if sums else math.nan,
        "sum_alpha_gold_count_corr": corr,
        "mean_sum_alpha_by_evidence_count": {
            str(count): float(np.mean(values)) for count, values in sorted(sums_by_count.items()) if values
        },
    }


def _corr(values: Sequence[float], golds: Sequence[float]) -> float:
    if len(values) < 2 or float(np.std(values)) <= 0 or float(np.std(golds)) <= 0:
        return math.nan
    return float(np.corrcoef(np.array(values), np.array(golds))[0, 1])


def memory_debug(
    *,
    gates_by_idx: Dict[int, List[float]],
    key_query_by_idx: Dict[int, List[float]],
    count_projection_by_idx: Dict[int, float],
    count_scalar_by_idx: Dict[int, float],
    r_norm_by_idx: Dict[int, float],
    delta_norm_by_idx: Optional[Dict[int, float]],
    frame_labels: torch.Tensor,
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
) -> Dict[str, Any]:
    base = gate_debug(gates_by_idx, frame_labels, records, indices)
    key_by_label: Dict[int, List[float]] = defaultdict(list)
    projection_by_count: Dict[int, List[float]] = defaultdict(list)
    scalar_by_count: Dict[int, List[float]] = defaultdict(list)
    r_norm_by_count: Dict[int, List[float]] = defaultdict(list)
    delta_norm_by_count: Dict[int, List[float]] = defaultdict(list)
    scalars: List[float] = []
    golds: List[float] = []
    for idx in indices:
        idx = int(idx)
        count = int(records[idx].evidence_count)
        labels = [int(v) for v in frame_labels[idx].int().tolist()]
        key_scores = key_query_by_idx.get(idx, [])
        for label, score in zip(labels, key_scores):
            key_by_label[int(label)].append(float(score))
        if idx in count_projection_by_idx:
            projection_by_count[count].append(float(count_projection_by_idx[idx]))
        if idx in count_scalar_by_idx:
            value = float(count_scalar_by_idx[idx])
            scalar_by_count[count].append(value)
            scalars.append(value)
            golds.append(float(records[idx].gold_count))
        if idx in r_norm_by_idx:
            r_norm_by_count[count].append(float(r_norm_by_idx[idx]))
        if delta_norm_by_idx is not None and idx in delta_norm_by_idx:
            delta_norm_by_count[count].append(float(delta_norm_by_idx[idx]))
    base.update(
        {
            "mean_key_query_non_evidence": float(np.mean(key_by_label[0])) if key_by_label[0] else math.nan,
            "mean_key_query_evidence": float(np.mean(key_by_label[1])) if key_by_label[1] else math.nan,
            "mean_count_projection_by_evidence_count": {
                str(count): float(np.mean(values)) for count, values in sorted(projection_by_count.items()) if values
            },
            "mean_count_scalar_by_evidence_count": {
                str(count): float(np.mean(values)) for count, values in sorted(scalar_by_count.items()) if values
            },
            "count_scalar_gold_count_corr": _corr(scalars, golds),
            "mean_r_norm_by_evidence_count": {
                str(count): float(np.mean(values)) for count, values in sorted(r_norm_by_count.items()) if values
            },
            "mean_delta_norm_by_evidence_count": {
                str(count): float(np.mean(values)) for count, values in sorted(delta_norm_by_count.items()) if values
            },
        }
    )
    return base


def memory_diagnostic_rows(
    *,
    debug_by_method: Dict[str, Dict[str, Any]],
    counts: Sequence[int],
    stage_by_method: Dict[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scalar_keys = [
        "frame_gate_accuracy_at_0_5",
        "frame_gate_auc",
        "mean_alpha_non_evidence",
        "mean_alpha_evidence",
        "count_sum_mse",
        "sum_alpha_gold_count_corr",
        "mean_key_query_non_evidence",
        "mean_key_query_evidence",
        "count_scalar_gold_count_corr",
    ]
    count_keys = [
        ("mean_sum_alpha_by_evidence_count", "mean_sum_alpha"),
        ("mean_count_projection_by_evidence_count", "mean_count_projection"),
        ("mean_count_scalar_by_evidence_count", "mean_count_scalar"),
        ("mean_r_norm_by_evidence_count", "mean_r_norm"),
        ("mean_delta_norm_by_evidence_count", "mean_delta_norm"),
    ]
    for method, payload in debug_by_method.items():
        stage = stage_by_method.get(method, "")
        for key in scalar_keys:
            if key in payload:
                rows.append(
                    {
                        "method": method,
                        "stage": stage,
                        "split": "test",
                        "diagnostic": key,
                        "evidence_count": "",
                        "frame_label": "",
                        "value": float(payload.get(key, math.nan)),
                    }
                )
        for source_key, diagnostic in count_keys:
            values = payload.get(source_key, {})
            for count in counts:
                rows.append(
                    {
                        "method": method,
                        "stage": stage,
                        "split": "test",
                        "diagnostic": diagnostic,
                        "evidence_count": int(count),
                        "frame_label": "",
                        "value": float(values.get(str(count), math.nan)),
                    }
                )
    return rows


def build_gate_rows(
    *,
    method: str,
    stage: str,
    records: Sequence[prev.SampleRecord],
    splits: Dict[str, List[int]],
    gates_by_idx: Dict[int, List[float]],
    key_query_by_idx: Optional[Dict[int, List[float]]] = None,
    frame_labels: torch.Tensor,
) -> List[Dict[str, Any]]:
    split_by_idx = {idx: split for split, indices in splits.items() for idx in indices}
    rows: List[Dict[str, Any]] = []
    key_query_by_idx = key_query_by_idx or {}
    for idx, gates in sorted(gates_by_idx.items()):
        key_query = key_query_by_idx.get(int(idx), [])
        for frame_idx, gate in enumerate(gates):
            rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "split": split_by_idx.get(int(idx), ""),
                    "sample_index": int(idx),
                    "sample_id": records[int(idx)].sample_id,
                    "evidence_count": int(records[int(idx)].evidence_count),
                    "frame_idx": int(frame_idx),
                    "frame_label": int(frame_labels[int(idx), int(frame_idx)].item()),
                    "gate": float(gate),
                    "key_query_similarity": float(key_query[frame_idx]) if frame_idx < len(key_query) else math.nan,
                }
            )
    return rows


def build_gold_rows(
    *,
    records: Sequence[prev.SampleRecord],
    test_indices: Sequence[int],
    base_scores: Dict[int, float],
    stage3_scores_by_method: Dict[str, Dict[int, float]],
    counts: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, stage3_scores in stage3_scores_by_method.items():
        for count in counts:
            drops: List[float] = []
            deltas: List[float] = []
            base_values: List[float] = []
            stage3_values: List[float] = []
            for idx in test_indices:
                idx = int(idx)
                if int(records[idx].gold_count) != int(count) or idx not in base_scores or idx not in stage3_scores:
                    continue
                base = float(base_scores[idx])
                stage3 = float(stage3_scores[idx])
                base_values.append(base)
                stage3_values.append(stage3)
                drops.append(base - stage3)
                deltas.append(stage3 - base)
            rows.append(
                {
                    "method": method,
                    "split": "test",
                    "evidence_count": int(count),
                    "n": len(drops),
                    "mean_base_gold_score": float(np.mean(base_values)) if base_values else math.nan,
                    "mean_stage3_gold_score": float(np.mean(stage3_values)) if stage3_values else math.nan,
                    "mean_gold_score_drop_vs_base": float(np.mean(drops)) if drops else math.nan,
                    "median_gold_score_drop_vs_base": float(np.median(drops)) if drops else math.nan,
                    "mean_gold_score_delta_vs_base": float(np.mean(deltas)) if deltas else math.nan,
                    "median_gold_score_delta_vs_base": float(np.median(deltas)) if deltas else math.nan,
                }
            )
    return rows


def build_per_sample_rows(
    *,
    records: Sequence[prev.SampleRecord],
    splits: Dict[str, List[int]],
    method_predictions: Dict[str, Dict[int, int]],
    stage1_logits_by_method: Dict[str, Dict[int, List[float]]],
    base_scores: Dict[int, float],
    stage3_scores_by_method: Dict[str, Dict[int, float]],
    stage3_logits_by_method: Dict[str, Dict[int, List[float]]],
    candidate_min: int,
) -> List[Dict[str, Any]]:
    split_by_idx = {idx: split for split, indices in splits.items() for idx in indices}
    rows: List[Dict[str, Any]] = []
    for idx in sorted(split_by_idx):
        record = records[int(idx)]
        row: Dict[str, Any] = {
            "split": split_by_idx[int(idx)],
            "sample_index": int(idx),
            "sample_id": record.sample_id,
            "sample_dir": os.fspath(record.sample_dir),
            "evidence_count": int(record.evidence_count),
            "gold_count": int(record.gold_count),
        }
        for method, preds in method_predictions.items():
            if int(idx) in preds:
                row[f"{method}_pred_count"] = int(preds[int(idx)])
                row[f"{method}_correct"] = int(preds[int(idx)] == record.gold_count)
        if int(idx) in base_scores:
            row["base_gold_score"] = float(base_scores[int(idx)])
        for method, logits_by_idx in stage1_logits_by_method.items():
            if int(idx) in logits_by_idx:
                logits = logits_by_idx[int(idx)]
                gold_offset = int(record.gold_count) - int(candidate_min)
                row[f"{method}_gold_head_logit"] = float(logits[gold_offset])
                row[f"{method}_gold_head_rank"] = int(prev.rank_of_gold(logits, gold_offset))
                row[f"{method}_count_head_logits_json"] = json.dumps(logits)
        for method, scores in stage3_scores_by_method.items():
            if int(idx) in scores:
                row[f"{method}_gold_score"] = float(scores[int(idx)])
            if int(idx) in scores and int(idx) in base_scores:
                row[f"{method}_gold_score_delta_vs_base"] = float(scores[int(idx)] - base_scores[int(idx)])
                row[f"{method}_gold_score_drop_vs_base"] = float(base_scores[int(idx)] - scores[int(idx)])
        for method, logits_by_idx in stage3_logits_by_method.items():
            if int(idx) in logits_by_idx:
                row[f"{method}_candidate_logits_json"] = json.dumps(logits_by_idx[int(idx)])
        rows.append(row)
    return rows


def mean_delta_norm_rows(
    *,
    records: Sequence[prev.SampleRecord],
    test_indices: Sequence[int],
    delta_by_method: Dict[str, Dict[int, float]],
    counts: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, delta_by_idx in delta_by_method.items():
        for count in counts:
            values = [
                float(delta_by_idx[int(idx)])
                for idx in test_indices
                if int(records[int(idx)].evidence_count) == int(count) and int(idx) in delta_by_idx
            ]
            rows.append(
                {
                    "method": method,
                    "evidence_count": int(count),
                    "mean_delta_norm": float(np.mean(values)) if values else math.nan,
                }
            )
    return rows


def gate_sum_rows_from_debug(debug_by_method: Dict[str, Dict[str, Any]], counts: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, payload in debug_by_method.items():
        means = payload.get("mean_sum_alpha_by_evidence_count", {})
        for count in counts:
            rows.append(
                {
                    "method": method,
                    "evidence_count": int(count),
                    "mean_gate_sum": float(means.get(str(count), math.nan)),
                }
            )
    return rows


def count_debug_rows_from_debug(
    debug_by_method: Dict[str, Dict[str, Any]],
    counts: Sequence[int],
    source_key: str,
    value_key: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, payload in debug_by_method.items():
        means = payload.get(source_key, {})
        for count in counts:
            rows.append(
                {
                    "method": method,
                    "evidence_count": int(count),
                    value_key: float(means.get(str(count), math.nan)),
                }
            )
    return rows


def plot_key_query_by_label(output_dir: Path, debug_by_method: Dict[str, Dict[str, Any]]) -> None:
    methods = list(debug_by_method)
    if not methods:
        return
    xs = np.arange(len(methods))
    width = 0.36
    non = [float(debug_by_method[m].get("mean_key_query_non_evidence", math.nan)) for m in methods]
    ev = [float(debug_by_method[m].get("mean_key_query_evidence", math.nan)) for m in methods]
    plt.figure(figsize=(max(7, len(methods) * 1.4), 4.5))
    plt.bar(xs - width / 2, non, width, label="non-evidence")
    plt.bar(xs + width / 2, ev, width, label="evidence")
    plt.xticks(xs, methods, rotation=20, ha="right")
    plt.ylabel("Mean k_i^T q")
    plt.title("Key-Query Similarity by Frame Label")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig((output_dir / "key_query_similarity_by_frame_label").with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight")
    plt.close()


def make_plots(
    *,
    output_dir: Path,
    metric_rows: Sequence[Dict[str, Any]],
    mean_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    memory_debug_by_method: Dict[str, Dict[str, Any]],
    delta_rows: Sequence[Dict[str, Any]],
    counts: Sequence[int],
) -> None:
    prev.plot_line(
        output_dir / "accuracy_by_evidence_count_all_methods",
        metric_rows,
        "accuracy",
        "Accuracy",
        "Accuracy by Evidence Count",
    )
    prev.plot_line(
        output_dir / "mean_predicted_count_by_evidence_count_all_methods",
        mean_rows,
        "mean_predicted_count",
        "Mean predicted count",
        "Mean Predicted Count by Evidence Count",
    )
    prev.plot_line(
        output_dir / "gate_sum_by_evidence_count",
        gate_sum_rows_from_debug(memory_debug_by_method, counts),
        "mean_gate_sum",
        "Mean sum alpha_i",
        "Gate Sum by Evidence Count",
    )
    plot_key_query_by_label(output_dir, memory_debug_by_method)
    prev.plot_line(
        output_dir / "count_direction_projection_by_evidence_count",
        count_debug_rows_from_debug(
            memory_debug_by_method,
            counts,
            "mean_count_projection_by_evidence_count",
            "mean_count_projection",
        ),
        "mean_count_projection",
        "Mean dot(r, normalized direction)",
        "Count-Direction Projection by Evidence Count",
    )
    prev.plot_line(
        output_dir / "gold_score_delta_by_evidence_count_stage3",
        gold_rows,
        "mean_gold_score_delta_vs_base",
        "Mean gold-score delta vs base",
        "Stage 3 Gold-Score Delta vs Base",
        methods=list(STAGE3_VARIANTS),
    )
    prev.plot_line(
        output_dir / "delta_norm_by_evidence_count_stage3",
        delta_rows,
        "mean_delta_norm",
        "Mean ||delta_h||",
        "Stage 3 Delta Norm by Evidence Count",
        methods=list(STAGE3_VARIANTS),
    )


def middle_count_accuracy(metric_rows: Sequence[Dict[str, Any]], method: str) -> float:
    vals = [
        float(row["accuracy"])
        for row in metric_rows
        if row.get("method") == method and int(row.get("evidence_count", -1)) in {3, 4, 5, 6}
    ]
    return float(np.mean(vals)) if vals else math.nan


def delta_by_count(gold_rows: Sequence[Dict[str, Any]], method: str) -> Dict[int, float]:
    return {
        int(row["evidence_count"]): float(row["mean_gold_score_delta_vs_base"])
        for row in gold_rows
        if row.get("method") == method
    }


def write_summary(
    *,
    output_dir: Path,
    overall_rows: Sequence[Dict[str, Any]],
    metric_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    memory_debug_by_method: Dict[str, Dict[str, Any]],
    checkpoints: Dict[str, Path],
) -> None:
    overall = {str(row["method"]): row for row in overall_rows}
    lines = [
        "Shared count-direction memory seq_len=8",
        "",
        "Overall test accuracy:",
    ]
    for row in overall_rows:
        lines.append(f"- {row['method']}: accuracy={float(row['accuracy']):.4f} n={int(row['n'])} mae={float(row['mae']):.4f}")
    lines.append("")
    lines.append("Direct answers:")

    free = overall.get(STAGE1_CURRENT_FREE, {})
    shared = overall.get(STAGE1_SHARED, {})
    plus = overall.get(STAGE1_SHARED_RESIDUAL, {})
    shared_beats = float(shared.get("accuracy", math.nan)) > float(free.get("accuracy", math.nan))
    plus_beats = float(plus.get("accuracy", math.nan)) > float(free.get("accuracy", math.nan))
    lines.append(
        "1. Does shared_count_direction beat current_free_value in Stage 1? "
        f"{'Yes' if shared_beats else 'No'} "
        f"({float(shared.get('accuracy', math.nan)):.4f} vs {float(free.get('accuracy', math.nan)):.4f})."
    )
    lines.append(
        "2. Does shared_count_direction_plus_small_residual beat current_free_value in Stage 1? "
        f"{'Yes' if plus_beats else 'No'} "
        f"({float(plus.get('accuracy', math.nan)):.4f} vs {float(free.get('accuracy', math.nan)):.4f})."
    )

    projection_bits = []
    for method in (STAGE1_SHARED, STAGE1_SHARED_RESIDUAL, STAGE3_SHARED_RESIDUAL):
        means = memory_debug_by_method.get(method, {}).get("mean_count_projection_by_evidence_count", {})
        xs = []
        ys = []
        for count_text, value in means.items():
            try:
                xs.append(float(count_text))
                ys.append(float(value))
            except Exception:
                pass
        corr = _corr(ys, xs)
        projection_bits.append(f"{method}: corr={corr:.3f}")
    lines.append(
        "3. Does dot(r, c_hat) increase roughly linearly with evidence count? "
        + "; ".join(projection_bits)
        + "."
    )

    key_bits = []
    for method in STAGE1_VARIANTS:
        payload = memory_debug_by_method.get(method, {})
        ev = float(payload.get("mean_key_query_evidence", math.nan))
        non = float(payload.get("mean_key_query_non_evidence", math.nan))
        key_bits.append(f"{method}: evidence={ev:.3f}, non={non:.3f}, higher={ev > non}")
    lines.append(
        "4. Does k_i^T q become higher for evidence frames than non-evidence frames? "
        + "; ".join(key_bits)
        + "."
    )

    base_mid = middle_count_accuracy(metric_rows, STAGE3_CURRENT_FREE)
    shared_mid = middle_count_accuracy(metric_rows, STAGE3_SHARED_RESIDUAL)
    lines.append(
        "5. Does Stage 3 shared-count residual improve over Stage 3 current-free residual, especially on counts 3-6? "
        f"{'Yes' if shared_mid > base_mid else 'No'} "
        f"({shared_mid:.4f} vs {base_mid:.4f})."
    )

    current_delta = delta_by_count(gold_rows, STAGE3_CURRENT_FREE)
    shared_delta = delta_by_count(gold_rows, STAGE3_SHARED_RESIDUAL)
    current_mid_values = [current_delta.get(count, math.nan) for count in (3, 4, 5, 6)]
    shared_mid_values = [shared_delta.get(count, math.nan) for count in (3, 4, 5, 6)]
    current_mid_delta = float(np.nanmean(current_mid_values)) if any(math.isfinite(float(v)) for v in current_mid_values) else math.nan
    shared_mid_delta = float(np.nanmean(shared_mid_values)) if any(math.isfinite(float(v)) for v in shared_mid_values) else math.nan
    lines.append(
        "6. Does the shared-count Stage 3 reduce negative gold-score deltas in middle counts? "
        f"{'Yes' if shared_mid_delta > current_mid_delta else 'No'} "
        f"(shared={shared_mid_delta:.4f}, current_free={current_mid_delta:.4f})."
    )

    lines.append("")
    lines.append("Checkpoints:")
    for method, path in checkpoints.items():
        lines.append(f"- {method}: {path}")
    (output_dir / "results_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    prev.write_csv(path, fieldnames, rows)


def main() -> int:
    args = parse_args()
    args.layers = prev.parse_int_tokens(args.layers)
    args.evidence_counts = prev.parse_int_tokens(args.evidence_counts)
    args.carriers = prev.split_tokens(args.carriers)
    output_dir = (args.output_dir if args.output_dir is not None else default_output_dir()).resolve()
    log_handle, old_stdout, old_stderr = prev.setup_logging(output_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        config = {
            "model_name": str(args.model_name),
            "dataset_root": os.fspath(args.dataset_root),
            "source_run": os.fspath(args.source_run),
            "base_source_run": os.fspath(args.base_source_run),
            "output_dir": os.fspath(output_dir),
            "split": str(args.split),
            "seq_len": int(args.seq_len),
            "evidence_counts": list(args.evidence_counts),
            "layers": list(args.layers),
            "carriers": list(args.carriers),
            "max_samples_per_count": int(args.max_samples_per_count),
            "candidate_min": int(args.candidate_min),
            "candidate_max": int(args.candidate_max),
            "stage1_variants": list(STAGE1_VARIANTS),
            "stage3_variants": list(STAGE3_VARIANTS),
            "bottleneck_dim": int(args.bottleneck_dim),
            "key_dim": int(args.key_dim),
            "value_dim": int(args.value_dim),
            "dropout": float(args.dropout),
            "frame_gate_bce_weight": float(args.frame_gate_bce_weight),
            "count_direction_mse_weight": float(args.count_direction_mse_weight),
            "count_direction_mse_weight_stage3": float(args.count_direction_mse_weight_stage3),
            "aux_mem_ce_weight": float(args.aux_mem_ce_weight),
            "residual_scale": float(args.residual_scale),
            "residual_l2_weight": float(args.residual_l2_weight),
            "stage1_epochs": int(args.stage1_epochs),
            "stage1_patience": int(args.stage1_patience),
            "stage3_epochs": int(args.stage3_epochs),
            "stage3_patience": int(args.stage3_patience),
            "inject_layer": int(args.inject_layer),
            "gamma_init": float(args.gamma_init),
            "train_gamma": bool(args.train_gamma),
            "dtype": str(args.dtype),
            "attn_implementation": str(args.attn_implementation),
            "load_in_4bit": bool(args.load_in_4bit),
            "seed": int(args.seed),
        }
        prev.write_json(output_dir / "run_config.json", config)
        print(f"Output dir: {output_dir}")
        print(f"Config: {json.dumps(config, sort_keys=True)}")

        feature_data = prev.load_message_features(args, args.layers, args.evidence_counts)
        sample_ids = feature_data["sample_ids"]
        labels = feature_data["labels"]
        frame_labels = feature_data["frame_labels"]
        x_messages = feature_data["x_messages"]
        records = prev.load_records(args.dataset_root, args.split, args.seq_len, sample_ids)
        counts = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
        splits = prev.stratified_split(sample_ids, labels, int(args.seed))
        counts_by_split = prev.split_counts(splits, labels, counts)
        print(f"x_messages shape={tuple(x_messages.shape)} D_msg={int(x_messages.shape[-1])}")
        for split, row in counts_by_split.items():
            print(f"  {split}: " + ", ".join(f"{count}:{row.get(count, 0)}" for count in counts))

        stage1_models: Dict[str, Stage1SharedCountReadout] = {}
        stage1_histories: Dict[str, Any] = {}
        stage1_eval_all: Dict[str, Dict[str, Any]] = {}
        stage1_checkpoints: Dict[str, Path] = {}
        eval_device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
        for variant in STAGE1_VARIANTS:
            print(f"Training {variant}")
            model_stage1, history, checkpoint = train_stage1_variant(
                args=args,
                output_dir=output_dir,
                variant=variant,
                x_messages=x_messages,
                labels=labels,
                frame_labels=frame_labels,
                splits=splits,
                candidate_min=int(args.candidate_min),
                candidate_max=int(args.candidate_max),
            )
            stage1_models[variant] = model_stage1
            stage1_histories[variant] = history
            stage1_checkpoints[variant] = checkpoint
            stage1_eval_all[variant] = {}
            for split_name, split_indices in splits.items():
                loader = DataLoader(
                    prev.MessageDataset(x_messages, labels, frame_labels, split_indices),
                    batch_size=int(args.stage1_batch_size),
                    shuffle=False,
                )
                stage1_eval_all[variant][split_name] = evaluate_stage1_variant(
                    model=model_stage1,
                    loader=loader,
                    labels=labels,
                    frame_labels=frame_labels,
                    candidate_min=int(args.candidate_min),
                    device=eval_device,
                )
            model_stage1.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        method_predictions: Dict[str, Dict[int, int]] = {}
        stage1_logits_by_method: Dict[str, Dict[int, List[float]]] = {}
        stage1_gates_by_method: Dict[str, Dict[int, List[float]]] = {}
        stage1_key_query_by_method: Dict[str, Dict[int, List[float]]] = {}
        stage1_projection_by_method: Dict[str, Dict[int, float]] = {}
        stage1_scalar_by_method: Dict[str, Dict[int, float]] = {}
        stage1_r_norm_by_method: Dict[str, Dict[int, float]] = {}
        for variant, evals in stage1_eval_all.items():
            method_predictions[variant] = {
                idx: pred for split_eval in evals.values() for idx, pred in split_eval["pred_by_idx"].items()
            }
            stage1_logits_by_method[variant] = {
                idx: value for split_eval in evals.values() for idx, value in split_eval["logits_by_idx"].items()
            }
            stage1_gates_by_method[variant] = {
                idx: value for split_eval in evals.values() for idx, value in split_eval["alpha_by_idx"].items()
            }
            stage1_key_query_by_method[variant] = {
                idx: value for split_eval in evals.values() for idx, value in split_eval["key_query_by_idx"].items()
            }
            stage1_projection_by_method[variant] = {
                idx: value for split_eval in evals.values() for idx, value in split_eval["count_projection_by_idx"].items()
            }
            stage1_scalar_by_method[variant] = {
                idx: value for split_eval in evals.values() for idx, value in split_eval["count_scalar_by_idx"].items()
            }
            stage1_r_norm_by_method[variant] = {
                idx: value for split_eval in evals.values() for idx, value in split_eval["r_norm_by_idx"].items()
            }

        device = prev.resolve_device(str(args.device))
        dtype = prev.resolve_dtype(str(args.dtype), device)
        model, processor = prev.load_model_and_processor(args, device=device, dtype=dtype)
        candidate_format, count_ids = prev.candidate_token_ids(processor.tokenizer, int(args.candidate_min), int(args.candidate_max))
        hidden_size = prev.hidden_size_from_model(model)
        print(f"Loaded Qwen hidden_size={hidden_size} candidate_format={candidate_format} count_ids={count_ids}")
        print("Evaluating base frozen Qwen on test split")
        base_eval = evaluate_qwen_count_channel(
            method="base_frozen_qwen",
            model=model,
            processor=processor,
            adapter=None,
            records=records,
            indices=splits["test"],
            x_messages=x_messages,
            count_token_ids=count_ids,
            args=args,
            device=device,
            batch_size=int(args.stage3_batch_size),
            save_gates=False,
        )
        method_predictions["base_frozen_qwen"] = base_eval["pred_by_idx"]

        stage3_eval_by_method: Dict[str, Dict[str, Any]] = {}
        stage3_histories: Dict[str, Any] = {}
        stage3_checkpoints: Dict[str, Path] = {}
        for variant in STAGE3_VARIANTS:
            print(f"Training {variant}")
            source_stage1 = STAGE3_TO_STAGE1[variant]
            adapter, history, checkpoint = train_stage3_variant(
                args=args,
                output_dir=output_dir,
                stage3_variant=variant,
                model=model,
                processor=processor,
                stage1_model=stage1_models[source_stage1],
                records=records,
                x_messages=x_messages,
                splits=splits,
                count_token_ids=count_ids,
                device=device,
                hidden_size=hidden_size,
            )
            stage3_histories[variant] = history
            stage3_checkpoints[variant] = checkpoint
            print(f"Evaluating {variant} on test split")
            stage3_eval = evaluate_qwen_count_channel(
                method=variant,
                model=model,
                processor=processor,
                adapter=adapter,
                records=records,
                indices=splits["test"],
                x_messages=x_messages,
                count_token_ids=count_ids,
                args=args,
                device=device,
                batch_size=int(args.stage3_batch_size),
                save_gates=True,
            )
            stage3_eval_by_method[variant] = stage3_eval
            method_predictions[variant] = stage3_eval["pred_by_idx"]
            adapter.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metric_rows, overall_rows, mean_rows = prev.metric_tables(
            records=records,
            test_indices=splits["test"],
            method_predictions=method_predictions,
            counts=counts,
        )
        stage3_scores_by_method = {
            method: payload.get("gold_score_by_idx", {}) for method, payload in stage3_eval_by_method.items()
        }
        stage3_logits_by_method = {
            method: payload.get("logits_by_idx", {}) for method, payload in stage3_eval_by_method.items()
        }
        gold_rows = build_gold_rows(
            records=records,
            test_indices=splits["test"],
            base_scores=base_eval.get("gold_score_by_idx", {}),
            stage3_scores_by_method=stage3_scores_by_method,
            counts=counts,
        )
        per_sample_rows = build_per_sample_rows(
            records=records,
            splits=splits,
            method_predictions=method_predictions,
            stage1_logits_by_method=stage1_logits_by_method,
            base_scores=base_eval.get("gold_score_by_idx", {}),
            stage3_scores_by_method=stage3_scores_by_method,
            stage3_logits_by_method=stage3_logits_by_method,
            candidate_min=int(args.candidate_min),
        )
        gate_rows: List[Dict[str, Any]] = []
        for method, gates in stage1_gates_by_method.items():
            gate_rows.extend(
                build_gate_rows(
                    method=method,
                    stage="stage1",
                    records=records,
                    splits=splits,
                    gates_by_idx=gates,
                    key_query_by_idx=stage1_key_query_by_method.get(method, {}),
                    frame_labels=frame_labels,
                )
            )
        for method, payload in stage3_eval_by_method.items():
            gate_rows.extend(
                build_gate_rows(
                    method=method,
                    stage="stage3",
                    records=records,
                    splits={"test": splits["test"]},
                    gates_by_idx=payload.get("gate_by_idx", {}),
                    key_query_by_idx=payload.get("key_query_by_idx", {}),
                    frame_labels=frame_labels,
                )
            )
        stage1_memory_debug = {
            method: memory_debug(
                gates_by_idx=gates,
                key_query_by_idx=stage1_key_query_by_method.get(method, {}),
                count_projection_by_idx=stage1_projection_by_method.get(method, {}),
                count_scalar_by_idx=stage1_scalar_by_method.get(method, {}),
                r_norm_by_idx=stage1_r_norm_by_method.get(method, {}),
                delta_norm_by_idx=None,
                frame_labels=frame_labels,
                records=records,
                indices=splits["test"],
            )
            for method, gates in stage1_gates_by_method.items()
        }
        stage3_memory_debug = {
            method: memory_debug(
                gates_by_idx=payload.get("gate_by_idx", {}),
                key_query_by_idx=payload.get("key_query_by_idx", {}),
                count_projection_by_idx=payload.get("count_projection_by_idx", {}),
                count_scalar_by_idx=payload.get("count_scalar_by_idx", {}),
                r_norm_by_idx=payload.get("r_norm_by_idx", {}),
                delta_norm_by_idx=payload.get("delta_norm_by_idx", {}),
                frame_labels=frame_labels,
                records=records,
                indices=splits["test"],
            )
            for method, payload in stage3_eval_by_method.items()
        }
        memory_debug_by_method = {**stage1_memory_debug, **stage3_memory_debug}
        memory_rows = memory_diagnostic_rows(
            debug_by_method=memory_debug_by_method,
            counts=counts,
            stage_by_method={**{method: "stage1" for method in STAGE1_VARIANTS}, **{method: "stage3" for method in STAGE3_VARIANTS}},
        )
        delta_by_method = {
            method: payload.get("delta_norm_by_idx", {}) for method, payload in stage3_eval_by_method.items()
        }
        delta_rows = mean_delta_norm_rows(
            records=records,
            test_indices=splits["test"],
            delta_by_method=delta_by_method,
            counts=counts,
        )
        debug = {
            "x_messages_shape": list(x_messages.shape),
            "D_msg": int(x_messages.shape[-1]),
            "d_m": int(args.bottleneck_dim),
            "d_k": int(args.key_dim),
            "d_v": int(args.value_dim),
            "split_counts": {split: {str(k): int(v) for k, v in row.items()} for split, row in counts_by_split.items()},
            "stage1": {
                method: {
                    "history": stage1_histories[method],
                    "checkpoint": os.fspath(stage1_checkpoints[method]),
                    "memory_debug_test": stage1_memory_debug[method],
                }
                for method in STAGE1_VARIANTS
            },
            "stage3": {
                method: {
                    "history": stage3_histories[method],
                    "checkpoint": os.fspath(stage3_checkpoints[method]),
                    "memory_debug_test": stage3_memory_debug[method],
                }
                for method in STAGE3_VARIANTS
            },
            "inject_layer": int(args.inject_layer),
            "injection_target": "target_char + target_room tokens",
            "source_cache": os.fspath(feature_data["cache_path"]),
            "runtime_seconds": time.time() - started,
        }

        write_csv(output_dir / "metrics.csv", ["method", "split", "evidence_count", "n", "accuracy", "mae"], metric_rows)
        write_csv(output_dir / "overall_metrics.csv", ["method", "split", "n", "accuracy", "mae", "mean_predicted_count"], overall_rows)
        write_csv(
            output_dir / "mean_predicted_count_by_evidence_count.csv",
            ["method", "split", "evidence_count", "n", "mean_predicted_count"],
            mean_rows,
        )
        write_csv(
            output_dir / "gold_score_drop_by_evidence_count.csv",
            [
                "method",
                "split",
                "evidence_count",
                "n",
                "mean_base_gold_score",
                "mean_stage3_gold_score",
                "mean_gold_score_drop_vs_base",
                "median_gold_score_drop_vs_base",
                "mean_gold_score_delta_vs_base",
                "median_gold_score_delta_vs_base",
            ],
            gold_rows,
        )
        per_sample_fields = sorted({key for row in per_sample_rows for key in row.keys()})
        leading = ["split", "sample_index", "sample_id", "sample_dir", "evidence_count", "gold_count"]
        per_sample_fields = leading + [field for field in per_sample_fields if field not in leading]
        write_csv(output_dir / "per_sample_predictions.csv", per_sample_fields, per_sample_rows)
        write_csv(
            output_dir / "per_frame_gates.csv",
            [
                "method",
                "stage",
                "split",
                "sample_index",
                "sample_id",
                "evidence_count",
                "frame_idx",
                "frame_label",
                "gate",
                "key_query_similarity",
            ],
            gate_rows,
        )
        write_csv(
            output_dir / "memory_diagnostics.csv",
            ["method", "stage", "split", "diagnostic", "evidence_count", "frame_label", "value"],
            memory_rows,
        )
        prev.write_json(output_dir / "adapter_debug.json", debug)
        checkpoints = {**stage1_checkpoints, **stage3_checkpoints}
        write_summary(
            output_dir=output_dir,
            overall_rows=overall_rows,
            metric_rows=metric_rows,
            gold_rows=gold_rows,
            memory_debug_by_method=memory_debug_by_method,
            checkpoints=checkpoints,
        )
        if not bool(args.no_plots):
            make_plots(
                output_dir=output_dir,
                metric_rows=metric_rows,
                mean_rows=mean_rows,
                gold_rows=gold_rows,
                memory_debug_by_method=memory_debug_by_method,
                delta_rows=delta_rows,
                counts=counts,
            )
        print(f"Wrote outputs to {output_dir}")
        print(f"Total runtime seconds: {time.time() - started:.2f}")
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

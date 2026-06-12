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
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from scripts.probes import run_message_memory_adapter_stage1_stage3_seq8 as prev


STAGE1_VARIANTS = ("stage1_memory_only", "stage1_gate_sum_only", "stage1_memory_plus_gate_sum")
STAGE3_VARIANTS = ("stage3_memory_only", "stage3_gate_sum_only", "stage3_memory_plus_gate_sum")
STAGE3_TO_STAGE1 = {
    "stage3_memory_only": "stage1_memory_only",
    "stage3_gate_sum_only": "stage1_gate_sum_only",
    "stage3_memory_plus_gate_sum": "stage1_memory_plus_gate_sum",
}
SCALAR_FEATURE_DIM = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count-channel ablation for seq_len=8 message-memory adapters."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/mmred_images_park")
    parser.add_argument("--source-run", type=Path, default=prev.DEFAULT_SOURCE_RUN)
    parser.add_argument("--base-source-run", type=Path, default=prev.DEFAULT_BASE_SOURCE_RUN)
    parser.add_argument(
        "--previous-run",
        type=Path,
        default=PROJECT_ROOT / "outputs/message_memory_adapter_stage1_stage3_seq8_7b_20260526_212606",
    )
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
    parser.add_argument("--count-sum-mse-weight", type=float, default=0.5)

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
    parser.add_argument("--count-sum-mse-weight-stage3", type=float, default=0.1)
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
    return PROJECT_ROOT / "outputs" / f"message_memory_count_channel_ablation_seq8_7b_{stamp}"


def scalar_count_features(sum_alpha: torch.Tensor, seq_len: int) -> torch.Tensor:
    s = sum_alpha.float().unsqueeze(-1)
    u = s / max(1.0, float(seq_len))
    return torch.cat(
        [
            s,
            u,
            u * u,
            u * u * u,
            torch.sin(math.pi * u),
            torch.cos(math.pi * u),
            torch.sin(2.0 * math.pi * u),
            torch.cos(2.0 * math.pi * u),
        ],
        dim=-1,
    )


def feature_dim_for_variant(variant: str, value_dim: int) -> int:
    if variant.endswith("memory_only"):
        return int(value_dim)
    if variant.endswith("gate_sum_only"):
        return SCALAR_FEATURE_DIM
    if variant.endswith("memory_plus_gate_sum"):
        return int(value_dim) + SCALAR_FEATURE_DIM
    raise ValueError(f"Unknown variant: {variant}")


class Stage1CountChannelReadout(nn.Module):
    def __init__(self, core: prev.MessageMemoryCore, variant: str, num_classes: int, seq_len: int) -> None:
        super().__init__()
        self.core = core
        self.variant = str(variant)
        self.seq_len = int(seq_len)
        feat_dim = feature_dim_for_variant(self.variant, core.value_dim)
        if self.variant.endswith("memory_only"):
            self.count_head = nn.Linear(feat_dim, int(num_classes))
        else:
            self.count_head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, 64),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(64, int(num_classes)),
            )

    def count_features(self, r: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        scalar = scalar_count_features(alpha.sum(dim=-1), self.seq_len)
        if self.variant.endswith("memory_only"):
            return r
        if self.variant.endswith("gate_sum_only"):
            return scalar
        if self.variant.endswith("memory_plus_gate_sum"):
            return torch.cat([r, scalar.to(r.device)], dim=-1)
        raise ValueError(f"Unknown variant: {self.variant}")

    def forward(self, x_messages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r, alpha = self.core(x_messages)
        logits = self.count_head(self.count_features(r, alpha))
        return logits, alpha, r


class Stage3CountChannelAdapter(nn.Module):
    def __init__(
        self,
        *,
        core: prev.MessageMemoryCore,
        variant: str,
        hidden_size: int,
        inject_layer: int,
        gamma_init: float,
        train_gamma: bool,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.core = core
        self.variant = str(variant)
        self.seq_len = int(seq_len)
        self.w_o = nn.Linear(feature_dim_for_variant(self.variant, core.value_dim), int(hidden_size), bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32), requires_grad=bool(train_gamma))
        self.inject_layer = int(inject_layer)
        self.enabled = True
        self._x_messages: Optional[torch.Tensor] = None
        self._target_positions: Optional[List[List[int]]] = None
        self._handles: List[Any] = []
        self.last_alpha: Optional[torch.Tensor] = None
        self.last_delta_norm: Optional[torch.Tensor] = None
        self.last_alpha_for_loss: Optional[torch.Tensor] = None
        self.last_delta_for_loss: Optional[torch.Tensor] = None
        nn.init.zeros_(self.w_o.weight)

    def set_context(self, x_messages: torch.Tensor, target_positions: Sequence[Sequence[int]]) -> None:
        self._x_messages = x_messages
        self._target_positions = [[int(pos) for pos in positions] for positions in target_positions]
        self.last_alpha = None
        self.last_delta_norm = None
        self.last_alpha_for_loss = None
        self.last_delta_for_loss = None

    def clear_context(self) -> None:
        self._x_messages = None
        self._target_positions = None
        self.last_alpha_for_loss = None
        self.last_delta_for_loss = None

    def count_features(self, r: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        scalar = scalar_count_features(alpha.sum(dim=-1), self.seq_len).to(r.device)
        if self.variant.endswith("memory_only"):
            return r
        if self.variant.endswith("gate_sum_only"):
            return scalar
        if self.variant.endswith("memory_plus_gate_sum"):
            return torch.cat([r, scalar], dim=-1)
        raise ValueError(f"Unknown variant: {self.variant}")

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
        r, alpha = self.core(self._x_messages.to(hidden_states.device))
        delta = self.w_o(self.count_features(r, alpha))
        self.last_alpha_for_loss = alpha
        self.last_delta_for_loss = delta
        self.last_alpha = alpha.detach().float().cpu()
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
) -> Stage1CountChannelReadout:
    core = prev.MessageMemoryCore(
        input_dim=int(input_dim),
        bottleneck_dim=int(args.bottleneck_dim),
        key_dim=int(args.key_dim),
        value_dim=int(args.value_dim),
        dropout=float(args.dropout),
    )
    core.set_standardizer(mean, std)
    return Stage1CountChannelReadout(core, variant=variant, num_classes=int(num_classes), seq_len=int(args.seq_len))


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
) -> Tuple[Stage1CountChannelReadout, Dict[str, Any], Path]:
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
    best_metric = -math.inf
    best_val_ce = math.inf
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
            logits, alpha, _r = model(x)
            count_loss = F.cross_entropy(logits, y_offset)
            gate_loss = F.binary_cross_entropy(alpha, y_frame.float())
            sum_loss = F.mse_loss(alpha.sum(dim=-1), y.float())
            loss = (
                count_loss
                + float(args.frame_gate_bce_weight) * gate_loss
                + float(args.count_sum_mse_weight) * sum_loss
            )
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
            "val_count_sum_mse": float(val["count_sum_mse"]),
        }
        rows.append(row)
        improved = row["val_accuracy"] > best_metric + 1e-9 or (
            abs(row["val_accuracy"] - best_metric) <= 1e-9 and row["val_ce"] < best_val_ce
        )
        if improved:
            best_metric = float(row["val_accuracy"])
            best_val_ce = float(row["val_ce"])
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"  {variant} epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_ce={row['val_ce']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_sum_mse={row['val_count_sum_mse']:.4f}"
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
            "best_val_accuracy": float(best_metric),
            "best_val_ce": float(best_val_ce),
        },
    }
    torch.save(ckpt, best_path)
    print(f"Saved {variant} checkpoint: {best_path}")
    return model_cpu, ckpt["history"], best_path


@torch.no_grad()
def evaluate_stage1_variant(
    *,
    model: Stage1CountChannelReadout,
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
    ce_total = 0.0
    bce_total = 0.0
    mse_total = 0.0
    n = 0
    for batch in loader:
        idx, x, y, y_frame = prev.batch_to_device(batch, device)
        y_offset = (y - int(candidate_min)).long()
        logits, alpha, _r = model(x)
        ce_total += float(F.cross_entropy(logits, y_offset, reduction="sum").detach().cpu().item())
        bce_total += float(F.binary_cross_entropy(alpha, y_frame.float(), reduction="sum").detach().cpu().item())
        mse_total += float(F.mse_loss(alpha.sum(dim=-1), y.float(), reduction="sum").detach().cpu().item())
        n += int(y.numel())
        pred = logits.argmax(dim=-1) + int(candidate_min)
        for row, sample_idx in enumerate(idx.detach().cpu().tolist()):
            pred_by_idx[int(sample_idx)] = int(pred[row].detach().cpu().item())
            logits_by_idx[int(sample_idx)] = [float(v) for v in logits[row].detach().float().cpu().tolist()]
            alpha_by_idx[int(sample_idx)] = [float(v) for v in alpha[row].detach().float().cpu().tolist()]
    y_true = [int(labels[int(idx)].item()) for idx in pred_by_idx]
    y_pred = [pred_by_idx[int(idx)] for idx in pred_by_idx]
    return {
        "ce": ce_total / max(1, n),
        "gate_bce": bce_total / max(1, n * int(frame_labels.shape[1])),
        "count_sum_mse": mse_total / max(1, n),
        "accuracy": prev.accuracy(y_true, y_pred),
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "alpha_by_idx": alpha_by_idx,
    }


def make_stage3_adapter(
    *,
    args: argparse.Namespace,
    stage1_model: Stage1CountChannelReadout,
    stage3_variant: str,
    hidden_size: int,
) -> Stage3CountChannelAdapter:
    core = prev.MessageMemoryCore(
        input_dim=stage1_model.core.input_dim,
        bottleneck_dim=stage1_model.core.bottleneck_dim,
        key_dim=stage1_model.core.key_dim,
        value_dim=stage1_model.core.value_dim,
        dropout=float(args.dropout),
    )
    core.load_state_dict(stage1_model.core.state_dict())
    return Stage3CountChannelAdapter(
        core=core,
        variant=stage3_variant,
        hidden_size=int(hidden_size),
        inject_layer=int(args.inject_layer),
        gamma_init=float(args.gamma_init),
        train_gamma=bool(args.train_gamma),
        seq_len=int(args.seq_len),
    )


def train_stage3_variant(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    stage3_variant: str,
    model: Any,
    processor: Any,
    stage1_model: Stage1CountChannelReadout,
    records: Sequence[prev.SampleRecord],
    x_messages: torch.Tensor,
    splits: Dict[str, List[int]],
    count_token_ids: Dict[int, int],
    device: str,
    hidden_size: int,
) -> Tuple[Stage3CountChannelAdapter, Dict[str, Any], Path]:
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
            train_sum_mse = 0.0
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
                sum_mse = count_sum_mse_for_adapter(adapter, batch.gold_counts)
                residual_l2 = residual_l2_for_adapter(adapter)
                loss = (
                    ce
                    + float(args.count_sum_mse_weight_stage3) * sum_mse
                    + float(args.residual_l2_weight) * residual_l2
                ) / max(1, int(args.stage3_grad_accum))
                loss.backward()
                train_ce += float(ce.detach().cpu().item())
                train_sum_mse += float(sum_mse.detach().cpu().item())
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
                "train_count_sum_mse": train_sum_mse / max(1, train_steps),
                "train_residual_l2": train_l2 / max(1, train_steps),
                "val_ce": float(val_eval["ce"]),
                "val_accuracy": float(val_eval["accuracy"]),
                "val_count_sum_mse": float(val_eval["count_sum_mse"]),
                "gamma": float(adapter.gamma.detach().cpu().item()),
            }
            rows.append(row)
            print(
                f"  {stage3_variant} epoch={epoch} train_ce={row['train_ce']:.4f} "
                f"val_ce={row['val_ce']:.4f} val_acc={row['val_accuracy']:.4f} "
                f"val_sum_mse={row['val_count_sum_mse']:.4f} gamma={row['gamma']:.4f}"
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


def count_sum_mse_for_adapter(adapter: Stage3CountChannelAdapter, gold_counts: torch.Tensor) -> torch.Tensor:
    if adapter.last_alpha_for_loss is None:
        return gold_counts.float().new_tensor(0.0)
    return F.mse_loss(adapter.last_alpha_for_loss.sum(dim=-1), gold_counts.float())


def residual_l2_for_adapter(adapter: Stage3CountChannelAdapter) -> torch.Tensor:
    if adapter.last_delta_for_loss is None:
        return adapter.gamma.new_tensor(0.0)
    return adapter.last_delta_for_loss.float().pow(2).sum(dim=-1).mean()


def evaluate_qwen_count_channel(
    *,
    method: str,
    model: Any,
    processor: Any,
    adapter: Optional[Stage3CountChannelAdapter],
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
    delta_norm_by_idx: Dict[int, float] = {}
    ce_total = 0.0
    mse_total = 0.0
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
                    mse_total += float(F.mse_loss(sums.float(), gold, reduction="sum").item())
                    for row, idx in enumerate(batch_indices):
                        sum_alpha_by_idx[int(idx)] = float(sums[row].item())
                        if save_gates:
                            gate_by_idx[int(idx)] = [float(v) for v in adapter.last_alpha[row].tolist()]
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
        "count_sum_mse": mse_total / max(1, n) if adapter is not None else math.nan,
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "gold_score_by_idx": gold_score_by_idx,
        "gate_by_idx": gate_by_idx,
        "sum_alpha_by_idx": sum_alpha_by_idx,
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


def build_gate_rows(
    *,
    method: str,
    stage: str,
    records: Sequence[prev.SampleRecord],
    splits: Dict[str, List[int]],
    gates_by_idx: Dict[int, List[float]],
    frame_labels: torch.Tensor,
) -> List[Dict[str, Any]]:
    split_by_idx = {idx: split for split, indices in splits.items() for idx in indices}
    rows: List[Dict[str, Any]] = []
    for idx, gates in sorted(gates_by_idx.items()):
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


def make_plots(
    *,
    output_dir: Path,
    metric_rows: Sequence[Dict[str, Any]],
    mean_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    stage1_gate_debug: Dict[str, Dict[str, Any]],
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
        output_dir / "gate_sum_by_evidence_count_stage1_variants",
        gate_sum_rows_from_debug(stage1_gate_debug, counts),
        "mean_gate_sum",
        "Mean sum alpha_i",
        "Stage 1 Gate Sum by Evidence Count",
        methods=list(STAGE1_VARIANTS),
    )
    prev.plot_line(
        output_dir / "gold_score_delta_by_evidence_count_stage3_variants",
        gold_rows,
        "mean_gold_score_delta_vs_base",
        "Mean gold-score delta vs base",
        "Stage 3 Gold-Score Delta vs Base",
        methods=list(STAGE3_VARIANTS),
    )
    prev.plot_line(
        output_dir / "delta_norm_by_evidence_count_stage3_variants",
        delta_rows,
        "mean_delta_norm",
        "Mean ||delta_h||",
        "Stage 3 Delta Norm by Evidence Count",
        methods=list(STAGE3_VARIANTS),
    )


def load_previous_deltas(previous_run: Path) -> Dict[int, float]:
    path = Path(previous_run) / "gold_score_drop_by_evidence_count.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    if "mean_gold_score_delta_vs_base" not in df.columns:
        return {}
    return {
        int(row.evidence_count): float(row.mean_gold_score_delta_vs_base)
        for row in df.itertuples()
        if str(getattr(row, "method", "")) == "stage3_room_char_residual"
    }


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
    stage1_gate_debug: Dict[str, Dict[str, Any]],
    previous_deltas: Dict[int, float],
    checkpoints: Dict[str, Path],
) -> None:
    overall = {str(row["method"]): row for row in overall_rows}
    lines = [
        "Message-memory count-channel ablation seq_len=8",
        "",
        "Overall test accuracy:",
    ]
    for row in overall_rows:
        lines.append(f"- {row['method']}: accuracy={float(row['accuracy']):.4f} n={int(row['n'])} mae={float(row['mae']):.4f}")
    lines.append("")
    lines.append("Direct answers:")

    mem = overall.get("stage1_memory_only", {})
    gate = overall.get("stage1_gate_sum_only", {})
    plus = overall.get("stage1_memory_plus_gate_sum", {})
    gate_beats = float(gate.get("accuracy", math.nan)) > float(mem.get("accuracy", math.nan))
    plus_beats = float(plus.get("accuracy", math.nan)) > float(mem.get("accuracy", math.nan))
    lines.append(
        "1. Does gate_sum_only beat memory_only in Stage 1? "
        f"{'Yes' if gate_beats else 'No'} "
        f"({float(gate.get('accuracy', math.nan)):.4f} vs {float(mem.get('accuracy', math.nan)):.4f})."
    )
    lines.append(
        "2. Does memory_plus_gate_sum beat memory_only in Stage 1? "
        f"{'Yes' if plus_beats else 'No'} "
        f"({float(plus.get('accuracy', math.nan)):.4f} vs {float(mem.get('accuracy', math.nan)):.4f})."
    )

    base_mid = middle_count_accuracy(metric_rows, "stage3_memory_only")
    gate_mid = middle_count_accuracy(metric_rows, "stage3_gate_sum_only")
    plus_mid = middle_count_accuracy(metric_rows, "stage3_memory_plus_gate_sum")
    improved = gate_mid > base_mid or plus_mid > base_mid
    lines.append(
        "3. Does gate_sum or memory_plus_gate_sum improve Stage 3 middle counts 3-6? "
        f"{'Yes' if improved else 'No'} "
        f"(memory_only={base_mid:.4f}, gate_sum_only={gate_mid:.4f}, memory_plus_gate_sum={plus_mid:.4f})."
    )

    comparison_bits: List[str] = []
    for method in STAGE3_VARIANTS:
        new = delta_by_count(gold_rows, method)
        if not previous_deltas:
            comparison_bits.append(f"{method}: previous deltas unavailable")
            continue
        count_bits = []
        for count in (4, 5, 6):
            old = previous_deltas.get(count, math.nan)
            value = new.get(count, math.nan)
            if math.isfinite(old) and math.isfinite(value):
                count_bits.append(f"{count}:{value:.3f}>{old:.3f}={value > old}")
        comparison_bits.append(f"{method} " + ", ".join(count_bits))
    lines.append(
        "4. Are gold-score deltas for counts 4-6 less negative than the previous Stage 3 memory-only run? "
        + "; ".join(comparison_bits)
        + "."
    )

    track_bits = []
    for method in STAGE1_VARIANTS:
        payload = stage1_gate_debug.get(method, {})
        corr = float(payload.get("sum_alpha_gold_count_corr", math.nan))
        mse = float(payload.get("count_sum_mse", math.nan))
        means = payload.get("mean_sum_alpha_by_evidence_count", {})
        edge = f"0->{float(means.get('0', math.nan)):.2f}, 8->{float(means.get('8', math.nan)):.2f}"
        track_bits.append(f"{method}: corr={corr:.3f}, mse={mse:.3f}, {edge}")
    lines.append("5. Does sum(alpha) track gold count? " + "; ".join(track_bits) + ".")

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
            "previous_run": os.fspath(args.previous_run),
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
            "count_sum_mse_weight": float(args.count_sum_mse_weight),
            "count_sum_mse_weight_stage3": float(args.count_sum_mse_weight_stage3),
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

        stage1_models: Dict[str, Stage1CountChannelReadout] = {}
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
                    frame_labels=frame_labels,
                )
            )
        stage1_gate_debug = {
            method: gate_debug(gates, frame_labels, records, splits["test"])
            for method, gates in stage1_gates_by_method.items()
        }
        stage3_gate_debug = {
            method: gate_debug(payload.get("gate_by_idx", {}), frame_labels, records, splits["test"])
            for method, payload in stage3_eval_by_method.items()
        }
        delta_by_method = {
            method: payload.get("delta_norm_by_idx", {}) for method, payload in stage3_eval_by_method.items()
        }
        delta_rows = mean_delta_norm_rows(
            records=records,
            test_indices=splits["test"],
            delta_by_method=delta_by_method,
            counts=counts,
        )
        previous_deltas = load_previous_deltas(args.previous_run)
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
                    "gate_debug_test": stage1_gate_debug[method],
                }
                for method in STAGE1_VARIANTS
            },
            "stage3": {
                method: {
                    "history": stage3_histories[method],
                    "checkpoint": os.fspath(stage3_checkpoints[method]),
                    "gate_debug_test": stage3_gate_debug[method],
                    "mean_delta_norm_by_evidence_count": {
                        str(row["evidence_count"]): float(row["mean_delta_norm"])
                        for row in delta_rows
                        if row["method"] == method
                    },
                }
                for method in STAGE3_VARIANTS
            },
            "previous_stage3_memory_only_gold_score_delta_by_evidence_count": {
                str(k): float(v) for k, v in sorted(previous_deltas.items())
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
            ["method", "stage", "split", "sample_index", "sample_id", "evidence_count", "frame_idx", "frame_label", "gate"],
            gate_rows,
        )
        prev.write_json(output_dir / "adapter_debug.json", debug)
        checkpoints = {**stage1_checkpoints, **stage3_checkpoints}
        write_summary(
            output_dir=output_dir,
            overall_rows=overall_rows,
            metric_rows=metric_rows,
            gold_rows=gold_rows,
            stage1_gate_debug=stage1_gate_debug,
            previous_deltas=previous_deltas,
            checkpoints=checkpoints,
        )
        if not bool(args.no_plots):
            make_plots(
                output_dir=output_dir,
                metric_rows=metric_rows,
                mean_rows=mean_rows,
                gold_rows=gold_rows,
                stage1_gate_debug=stage1_gate_debug,
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

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
from scripts.probes import run_shared_count_direction_memory_seq8 as shared


STAGE1_CURRENT_FREE = shared.STAGE1_CURRENT_FREE
STAGE1_SHARED_RESIDUAL = shared.STAGE1_SHARED_RESIDUAL
STAGE3_CURRENT_FREE = shared.STAGE3_CURRENT_FREE
STAGE3_SHARED_RESIDUAL = shared.STAGE3_SHARED_RESIDUAL
STAGE3_CALIBRATED = "stage3_shared_count_direction_plus_small_residual_calibrated"

STAGE1_VARIANTS = (STAGE1_CURRENT_FREE, STAGE1_SHARED_RESIDUAL)
BASE_STAGE3_VARIANTS = (STAGE3_CURRENT_FREE, STAGE3_SHARED_RESIDUAL)
MIDDLE_COUNTS = (3, 4, 5, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrated shared count-direction memory adapter experiment for MMReD seq_len=8."
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
    parser.add_argument("--lambda-counts", nargs="+", default=["0.05", "0.1", "0.2"])
    parser.add_argument("--lambda-res", nargs="+", default=["0.001", "0.01"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
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
    return PROJECT_ROOT / "outputs" / f"shared_count_direction_calibrated_seq8_7b_{stamp}"


def float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def calibrated_method_name(lambda_count: float, lambda_res: float) -> str:
    return f"{STAGE3_CALIBRATED}_lc{float_tag(lambda_count)}_lres{float_tag(lambda_res)}"


def safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(".", "p")


def method_config_id(method: str, method_configs: Dict[str, Dict[str, Any]]) -> str:
    config = method_configs.get(str(method), {})
    return str(config.get("config_id", "baseline"))


def method_variant(method: str, method_configs: Dict[str, Dict[str, Any]]) -> str:
    config = method_configs.get(str(method), {})
    if config:
        return str(config["variant"])
    if method == STAGE3_CURRENT_FREE or method == STAGE1_CURRENT_FREE:
        return "current_free_value"
    if method == STAGE3_SHARED_RESIDUAL or method == STAGE1_SHARED_RESIDUAL:
        return "shared_count_direction_plus_small_residual"
    if method == "base_frozen_qwen":
        return "base_frozen_qwen"
    return str(method)


def method_stage(method: str) -> str:
    if str(method).startswith("stage1_"):
        return "stage1"
    if str(method).startswith("stage3_"):
        return "stage3"
    if method == "base_frozen_qwen":
        return "base"
    return ""


def short_label(method: str, method_configs: Dict[str, Dict[str, Any]]) -> str:
    if method == STAGE3_CURRENT_FREE:
        return "current_free"
    if method == STAGE3_SHARED_RESIDUAL:
        return "shared_residual"
    config = method_configs.get(str(method))
    if config:
        return f"cal {config['lambda_count']:g}/{config['lambda_res']:g}"
    if method == "base_frozen_qwen":
        return "base_qwen"
    return str(method).replace("stage3_", "")


def add_method_metadata(rows: Sequence[Dict[str, Any]], method_configs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method", ""))
        config = method_configs.get(method, {})
        new_row = dict(row)
        new_row["stage"] = method_stage(method)
        new_row["variant"] = method_variant(method, method_configs)
        new_row["config_id"] = method_config_id(method, method_configs)
        new_row["lambda_count"] = config.get("lambda_count", "")
        new_row["lambda_res"] = config.get("lambda_res", "")
        out.append(new_row)
    return out


def stage3_to_stage1(method: str) -> str:
    if method == STAGE3_CURRENT_FREE:
        return STAGE1_CURRENT_FREE
    if method == STAGE3_SHARED_RESIDUAL or method.startswith(STAGE3_CALIBRATED):
        return STAGE1_SHARED_RESIDUAL
    raise ValueError(f"Unknown Stage 3 method: {method}")


def stage3_core_variant(method: str) -> str:
    if method == STAGE3_CURRENT_FREE:
        return "current_free_value"
    if method == STAGE3_SHARED_RESIDUAL or method.startswith(STAGE3_CALIBRATED):
        return "shared_count_direction_plus_small_residual"
    raise ValueError(f"Unknown Stage 3 method: {method}")


def is_calibrated_method(method: str) -> bool:
    return str(method).startswith(STAGE3_CALIBRATED)


class CalibratedStage3Adapter(shared.Stage3SharedCountAdapter):
    def set_context(self, x_messages: torch.Tensor, target_positions: Sequence[Sequence[int]]) -> None:
        super().set_context(x_messages, target_positions)
        self.last_r_for_loss: Optional[torch.Tensor] = None
        self.last_count_projection_for_loss: Optional[torch.Tensor] = None
        self.last_r_perp_for_loss: Optional[torch.Tensor] = None
        self.last_r_perp_norm: Optional[torch.Tensor] = None

    def clear_context(self) -> None:
        super().clear_context()
        self.last_r_for_loss = None
        self.last_count_projection_for_loss = None
        self.last_r_perp_for_loss = None

    def inject(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self._x_messages is None or self._target_positions is None:
            return hidden_states
        r, alpha, diagnostics = self.core(self._x_messages.to(hidden_states.device))
        axis = self.core.projection_axis().to(r.device).to(r.dtype)
        projection = diagnostics["count_projection"].to(r.dtype)
        r_perp = r - projection.unsqueeze(-1) * axis.view(1, -1)
        mem_logits = self.count_head(r)
        count_scalar = self.scalar_scale.to(r.device) * diagnostics["count_projection"] + self.scalar_bias.to(r.device)
        delta = self.w_o(r)
        self.last_alpha_for_loss = alpha
        self.last_delta_for_loss = delta
        self.last_mem_logits_for_loss = mem_logits
        self.last_count_scalar_for_loss = count_scalar
        self.last_r_for_loss = r
        self.last_count_projection_for_loss = diagnostics["count_projection"]
        self.last_r_perp_for_loss = r_perp
        self.last_alpha = alpha.detach().float().cpu()
        self.last_key_query_similarity = diagnostics["key_query_similarity"].detach().float().cpu()
        self.last_count_projection = diagnostics["count_projection"].detach().float().cpu()
        self.last_count_scalar = count_scalar.detach().float().cpu()
        self.last_r_norm = diagnostics["r_norm"].detach().float().cpu()
        self.last_r_perp_norm = r_perp.detach().float().norm(dim=-1).cpu()
        self.last_delta_norm = delta.detach().float().norm(dim=-1).cpu()
        updates: List[torch.Tensor] = []
        seq_len = int(hidden_states.shape[1])
        for batch_idx, positions in enumerate(self._target_positions):
            valid = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len})
            mask = hidden_states.new_zeros((seq_len, 1))
            if valid:
                pos_idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
                mask.index_fill_(0, pos_idx, 1.0)
            update = (self.gamma.to(hidden_states.device) * delta[batch_idx]).to(hidden_states.dtype).unsqueeze(0)
            updates.append(mask * update)
        return hidden_states + torch.stack(updates, dim=0)


def make_stage3_adapter(
    *,
    args: argparse.Namespace,
    stage1_model: shared.Stage1SharedCountReadout,
    stage3_method: str,
    hidden_size: int,
) -> CalibratedStage3Adapter:
    core = shared.SharedCountMemoryCore(
        input_dim=stage1_model.core.input_dim,
        bottleneck_dim=stage1_model.core.bottleneck_dim,
        key_dim=stage1_model.core.key_dim,
        value_dim=stage1_model.core.value_dim,
        dropout=float(args.dropout),
        residual_scale=float(args.residual_scale),
        variant=stage3_core_variant(stage3_method),
    )
    core.load_state_dict(stage1_model.core.state_dict())
    core.variant = stage3_core_variant(stage3_method)
    count_head = nn.Linear(stage1_model.core.value_dim, stage1_model.count_head.out_features)
    count_head.load_state_dict(stage1_model.count_head.state_dict())
    return CalibratedStage3Adapter(
        core=core,
        count_head=count_head,
        scalar_scale=stage1_model.scalar_scale,
        scalar_bias=stage1_model.scalar_bias,
        variant=stage3_method,
        hidden_size=int(hidden_size),
        inject_layer=int(args.inject_layer),
        gamma_init=float(args.gamma_init),
        train_gamma=bool(args.train_gamma),
    )


def count_projection_huber_for_adapter(
    adapter: CalibratedStage3Adapter, gold_counts: torch.Tensor, huber_delta: float
) -> torch.Tensor:
    if adapter.last_count_projection_for_loss is None:
        return gold_counts.float().new_tensor(0.0)
    return F.huber_loss(
        adapter.last_count_projection_for_loss.float(),
        gold_counts.float(),
        reduction="mean",
        delta=float(huber_delta),
    )


def r_perp_l2_for_adapter(adapter: CalibratedStage3Adapter) -> torch.Tensor:
    if adapter.last_r_perp_for_loss is None:
        return adapter.gamma.new_tensor(0.0)
    return adapter.last_r_perp_for_loss.float().pow(2).sum(dim=-1).mean()


def train_stage3_method(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    stage3_method: str,
    method_configs: Dict[str, Dict[str, Any]],
    model: Any,
    processor: Any,
    stage1_model: shared.Stage1SharedCountReadout,
    records: Sequence[prev.SampleRecord],
    x_messages: torch.Tensor,
    splits: Dict[str, List[int]],
    count_token_ids: Dict[int, int],
    device: str,
    hidden_size: int,
) -> Tuple[CalibratedStage3Adapter, Dict[str, Any], Path]:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{safe_name(stage3_method)}_best.pt"
    adapter = make_stage3_adapter(
        args=args,
        stage1_model=stage1_model,
        stage3_method=stage3_method,
        hidden_size=hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.stage3_lr), weight_decay=float(args.stage3_weight_decay))
    train_indices = list(splits["train"])
    val_indices = list(splits["val"])
    config = method_configs.get(stage3_method, {})
    calibrated = bool(config)
    lambda_count = float(config.get("lambda_count", 0.0))
    lambda_res = float(config.get("lambda_res", 0.0))
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
            rng = random.Random(int(args.seed) + 1000 + epoch + shared.stable_variant_offset(stage3_method))
            shuffled = list(train_indices)
            rng.shuffle(shuffled)
            optimizer.zero_grad(set_to_none=True)
            train_ce = 0.0
            train_mem_ce = 0.0
            train_dir_mse = 0.0
            train_huber = 0.0
            train_delta_l2 = 0.0
            train_r_perp_l2 = 0.0
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
                mem_ce = shared.aux_mem_ce_for_adapter(adapter, gold_offsets)
                dir_mse = shared.count_direction_mse_for_adapter(adapter, batch.gold_counts)
                delta_l2 = shared.residual_l2_for_adapter(adapter)
                huber = count_projection_huber_for_adapter(adapter, batch.gold_counts, float(args.huber_delta))
                r_perp_l2 = r_perp_l2_for_adapter(adapter)
                if calibrated:
                    loss_raw = ce + lambda_count * huber + lambda_res * r_perp_l2
                else:
                    loss_raw = (
                        ce
                        + float(args.aux_mem_ce_weight) * mem_ce
                        + float(args.count_direction_mse_weight_stage3) * dir_mse
                        + float(args.residual_l2_weight) * delta_l2
                    )
                loss = loss_raw / max(1, int(args.stage3_grad_accum))
                loss.backward()
                train_ce += float(ce.detach().cpu().item())
                train_mem_ce += float(mem_ce.detach().cpu().item())
                train_dir_mse += float(dir_mse.detach().cpu().item())
                train_huber += float(huber.detach().cpu().item())
                train_delta_l2 += float(delta_l2.detach().cpu().item())
                train_r_perp_l2 += float(r_perp_l2.detach().cpu().item())
                train_steps += 1
                adapter.clear_context()
                if step % max(1, int(args.stage3_grad_accum)) == 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.stage3_grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 100 == 0:
                    suffix = f" huber={train_huber / max(1, train_steps):.4f}" if calibrated else ""
                    print(
                        f"  {stage3_method} epoch={epoch} step={step} "
                        f"train_ce={train_ce / max(1, train_steps):.4f}{suffix}"
                    )
            if train_steps % max(1, int(args.stage3_grad_accum)) != 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.stage3_grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            val_eval = evaluate_qwen_count_channel(
                method=stage3_method,
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
                "train_count_projection_huber": train_huber / max(1, train_steps),
                "train_delta_l2": train_delta_l2 / max(1, train_steps),
                "train_r_perp_l2": train_r_perp_l2 / max(1, train_steps),
                "val_ce": float(val_eval["ce"]),
                "val_accuracy": float(val_eval["accuracy"]),
                "val_count_projection_mae": float(val_eval["count_projection_mae"]),
                "val_mean_r_perp_norm": float(val_eval["mean_r_perp_norm"]),
                "gamma": float(adapter.gamma.detach().cpu().item()),
                "lambda_count": lambda_count if calibrated else "",
                "lambda_res": lambda_res if calibrated else "",
            }
            rows.append(row)
            print(
                f"  {stage3_method} epoch={epoch} train_ce={row['train_ce']:.4f} "
                f"val_ce={row['val_ce']:.4f} val_acc={row['val_accuracy']:.4f} "
                f"val_cal_mae={row['val_count_projection_mae']:.4f} "
                f"r_perp={row['val_mean_r_perp_norm']:.4f} gamma={row['gamma']:.4f}"
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
        "variant": str(stage3_method),
        "config": dict(config),
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
    print(f"Saved {stage3_method} checkpoint: {best_path}")
    return adapter_cpu, ckpt["history"], best_path


def evaluate_qwen_count_channel(
    *,
    method: str,
    model: Any,
    processor: Any,
    adapter: Optional[CalibratedStage3Adapter],
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
    r_perp_norm_by_idx: Dict[int, float] = {}
    delta_norm_by_idx: Dict[int, float] = {}
    ce_total = 0.0
    cal_abs_total = 0.0
    r_perp_total = 0.0
    cal_n = 0
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
                    if adapter.last_count_projection is not None:
                        z = adapter.last_count_projection.float()
                        cal_abs_total += float((z - gold).abs().sum().item())
                        cal_n += int(gold.numel())
                    if adapter.last_r_perp_norm is not None:
                        r_perp_total += float(adapter.last_r_perp_norm.float().sum().item())
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
                        if adapter.last_r_perp_norm is not None:
                            r_perp_norm_by_idx[int(idx)] = float(adapter.last_r_perp_norm[row].item())
                        if adapter.last_delta_norm is not None:
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
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "gold_score_by_idx": gold_score_by_idx,
        "gate_by_idx": gate_by_idx,
        "sum_alpha_by_idx": sum_alpha_by_idx,
        "key_query_by_idx": key_query_by_idx,
        "count_projection_by_idx": count_projection_by_idx,
        "count_scalar_by_idx": count_scalar_by_idx,
        "r_norm_by_idx": r_norm_by_idx,
        "r_perp_norm_by_idx": r_perp_norm_by_idx,
        "delta_norm_by_idx": delta_norm_by_idx,
        "count_projection_mae": cal_abs_total / max(1, cal_n) if adapter is not None else math.nan,
        "mean_r_perp_norm": r_perp_total / max(1, cal_n) if adapter is not None else math.nan,
    }


def ci95(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if len(clean) <= 1:
        return 0.0 if len(clean) == 1 else math.nan
    return float(1.96 * np.std(np.array(clean), ddof=1) / math.sqrt(len(clean)))


def corr_or_nan(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or float(np.std(xs)) <= 0 or float(np.std(ys)) <= 0:
        return math.nan
    return float(np.corrcoef(np.array(xs), np.array(ys))[0, 1])


def build_gold_score_rows(
    *,
    records: Sequence[prev.SampleRecord],
    test_indices: Sequence[int],
    base_scores: Dict[int, float],
    stage3_scores_by_method: Dict[str, Dict[int, float]],
    counts: Sequence[int],
    method_configs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, stage3_scores in stage3_scores_by_method.items():
        config = method_configs.get(method, {})
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
                    "stage": "stage3",
                    "variant": method_variant(method, method_configs),
                    "config_id": method_config_id(method, method_configs),
                    "lambda_count": config.get("lambda_count", ""),
                    "lambda_res": config.get("lambda_res", ""),
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


def build_calibration_rows(
    *,
    records: Sequence[prev.SampleRecord],
    test_indices: Sequence[int],
    stage3_eval_by_method: Dict[str, Dict[str, Any]],
    counts: Sequence[int],
    method_configs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, payload in stage3_eval_by_method.items():
        config = method_configs.get(method, {})
        projections = payload.get("count_projection_by_idx", {})
        xs_all: List[float] = []
        ys_all: List[float] = []
        for idx in test_indices:
            idx = int(idx)
            if idx in projections:
                xs_all.append(float(projections[idx]))
                ys_all.append(float(records[idx].gold_count))
        corr = corr_or_nan(xs_all, ys_all)
        for count in counts:
            values = [
                float(projections[int(idx)])
                for idx in test_indices
                if int(records[int(idx)].gold_count) == int(count) and int(idx) in projections
            ]
            abs_errors = [abs(value - float(count)) for value in values]
            rows.append(
                {
                    "method": method,
                    "stage": "stage3",
                    "variant": method_variant(method, method_configs),
                    "config_id": method_config_id(method, method_configs),
                    "lambda_count": config.get("lambda_count", ""),
                    "lambda_res": config.get("lambda_res", ""),
                    "split": "test",
                    "evidence_count": int(count),
                    "n": len(values),
                    "mean_count_projection": float(np.mean(values)) if values else math.nan,
                    "count_projection_ci95": ci95(values),
                    "mean_abs_calibration_error": float(np.mean(abs_errors)) if abs_errors else math.nan,
                    "calibration_error_ci95": ci95(abs_errors),
                    "corr_dot_true_count": corr,
                }
            )
    return rows


def build_residual_norm_rows(
    *,
    records: Sequence[prev.SampleRecord],
    test_indices: Sequence[int],
    stage3_eval_by_method: Dict[str, Dict[str, Any]],
    counts: Sequence[int],
    method_configs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, payload in stage3_eval_by_method.items():
        config = method_configs.get(method, {})
        r_perp = payload.get("r_perp_norm_by_idx", {})
        r_norm = payload.get("r_norm_by_idx", {})
        delta_norm = payload.get("delta_norm_by_idx", {})
        for count in counts:
            idxs = [
                int(idx)
                for idx in test_indices
                if int(records[int(idx)].gold_count) == int(count) and int(idx) in r_perp
            ]
            perp_values = [float(r_perp[idx]) for idx in idxs]
            r_values = [float(r_norm[idx]) for idx in idxs if idx in r_norm]
            delta_values = [float(delta_norm[idx]) for idx in idxs if idx in delta_norm]
            rows.append(
                {
                    "method": method,
                    "stage": "stage3",
                    "variant": method_variant(method, method_configs),
                    "config_id": method_config_id(method, method_configs),
                    "lambda_count": config.get("lambda_count", ""),
                    "lambda_res": config.get("lambda_res", ""),
                    "split": "test",
                    "evidence_count": int(count),
                    "n": len(perp_values),
                    "mean_residual_norm": float(np.mean(perp_values)) if perp_values else math.nan,
                    "residual_norm_ci95": ci95(perp_values),
                    "mean_r_norm": float(np.mean(r_values)) if r_values else math.nan,
                    "mean_delta_norm": float(np.mean(delta_values)) if delta_values else math.nan,
                }
            )
    return rows


def mean_for_counts(rows: Sequence[Dict[str, Any]], method: str, key: str, counts: Sequence[int] = MIDDLE_COUNTS) -> float:
    values = [
        float(row[key])
        for row in rows
        if row.get("method") == method
        and int(row.get("evidence_count", -1)) in set(int(c) for c in counts)
        and math.isfinite(float(row.get(key, math.nan)))
    ]
    return float(np.mean(values)) if values else math.nan


def overall_for_method(rows: Sequence[Dict[str, Any]], method: str, key: str) -> float:
    for row in rows:
        if row.get("method") == method:
            return float(row.get(key, math.nan))
    return math.nan


def best_calibrated_method(
    *,
    stage3_methods: Sequence[str],
    metric_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    calibration_rows: Sequence[Dict[str, Any]],
) -> Optional[str]:
    calibrated = [method for method in stage3_methods if is_calibrated_method(method)]
    if not calibrated:
        return None

    def score(method: str) -> Tuple[float, float, float, float]:
        mid_acc = mean_for_counts(metric_rows, method, "accuracy")
        mid_delta = mean_for_counts(gold_rows, method, "mean_gold_score_delta_vs_base")
        cal_err = mean_for_counts(calibration_rows, method, "mean_abs_calibration_error", counts=range(9))
        overall_acc = overall_for_method(metric_rows, method, "accuracy")
        return (
            -math.inf if not math.isfinite(mid_acc) else mid_acc,
            -math.inf if not math.isfinite(mid_delta) else mid_delta,
            math.inf if not math.isfinite(cal_err) else -cal_err,
            -math.inf if not math.isfinite(overall_acc) else overall_acc,
        )

    return max(calibrated, key=score)


def plot_overall_acc_mae(
    output_dir: Path,
    overall_rows: Sequence[Dict[str, Any]],
    stage3_methods: Sequence[str],
    method_configs: Dict[str, Dict[str, Any]],
) -> None:
    rows = [row for row in overall_rows if row.get("method") in set(stage3_methods)]
    if not rows:
        return
    labels = [short_label(str(row["method"]), method_configs) for row in rows]
    acc = [float(row["accuracy"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]
    xs = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(rows) * 1.2), 4.5))
    axes[0].bar(xs, acc, color="#4078a8")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, max(0.75, max(acc) * 1.15 if acc else 0.75))
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(xs, mae, color="#b85c38")
    axes[1].set_ylabel("MAE")
    axes[1].set_ylim(0, max(1.0, max(mae) * 1.15 if mae else 1.0))
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.suptitle("Overall Accuracy and MAE by Variant")
    fig.tight_layout()
    fig.savefig(output_dir / "overall_acc_mae_by_variant.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_line_with_methods(
    *,
    output_dir: Path,
    filename: str,
    rows: Sequence[Dict[str, Any]],
    stage3_methods: Sequence[str],
    method_configs: Dict[str, Dict[str, Any]],
    y_key: str,
    ylabel: str,
    title: str,
    ci_key: Optional[str] = None,
) -> None:
    counts = sorted({int(row["evidence_count"]) for row in rows if str(row.get("evidence_count", "")).lstrip("-").isdigit()})
    if not counts:
        return
    plt.figure(figsize=(9, 5.5))
    for method in stage3_methods:
        by_count = {int(row["evidence_count"]): row for row in rows if row.get("method") == method}
        ys: List[float] = []
        cis: List[float] = []
        for count in counts:
            row = by_count.get(count, {})
            ys.append(float(row.get(y_key, math.nan)))
            cis.append(float(row.get(ci_key, math.nan)) if ci_key else math.nan)
        if not any(math.isfinite(v) for v in ys):
            continue
        label = short_label(method, method_configs)
        plt.plot(counts, ys, marker="o", linewidth=1.8, label=label)
        if ci_key and any(math.isfinite(v) for v in cis):
            y_arr = np.array(ys, dtype=float)
            ci_arr = np.array([0.0 if not math.isfinite(v) else v for v in cis], dtype=float)
            plt.fill_between(counts, y_arr - ci_arr, y_arr + ci_arr, alpha=0.12)
    plt.xlabel("True evidence count")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(counts)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close()


def plot_middle_accuracy(
    output_dir: Path,
    metric_rows: Sequence[Dict[str, Any]],
    stage3_methods: Sequence[str],
    method_configs: Dict[str, Dict[str, Any]],
) -> None:
    labels = [short_label(method, method_configs) for method in stage3_methods]
    vals = [mean_for_counts(metric_rows, method, "accuracy") for method in stage3_methods]
    xs = np.arange(len(stage3_methods))
    plt.figure(figsize=(max(8, len(stage3_methods) * 1.1), 4.8))
    plt.bar(xs, vals, color="#4f7f52")
    plt.ylabel("Mean accuracy, counts 3-6")
    plt.title("Middle-Count Accuracy by Variant")
    plt.xticks(xs, labels, rotation=35, ha="right")
    plt.ylim(0, max(0.75, max([v for v in vals if math.isfinite(v)] or [0.0]) * 1.15))
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "middle_count_accuracy_3_6_by_variant.png", dpi=180, bbox_inches="tight")
    plt.close()


def make_required_plots(
    *,
    output_dir: Path,
    metric_rows: Sequence[Dict[str, Any]],
    overall_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    calibration_rows: Sequence[Dict[str, Any]],
    residual_rows: Sequence[Dict[str, Any]],
    stage3_methods: Sequence[str],
    method_configs: Dict[str, Dict[str, Any]],
) -> None:
    plot_overall_acc_mae(output_dir, overall_rows, stage3_methods, method_configs)
    plot_line_with_methods(
        output_dir=output_dir,
        filename="per_count_accuracy_by_variant.png",
        rows=metric_rows,
        stage3_methods=stage3_methods,
        method_configs=method_configs,
        y_key="accuracy",
        ylabel="Accuracy",
        title="Per-Count Accuracy by Variant",
    )
    plot_middle_accuracy(output_dir, metric_rows, stage3_methods, method_configs)
    plot_line_with_methods(
        output_dir=output_dir,
        filename="gold_score_delta_by_count.png",
        rows=gold_rows,
        stage3_methods=stage3_methods,
        method_configs=method_configs,
        y_key="mean_gold_score_delta_vs_base",
        ylabel="Mean gold-score delta vs base",
        title="Gold-Score Delta by Evidence Count",
    )
    plot_line_with_methods(
        output_dir=output_dir,
        filename="count_projection_by_true_count.png",
        rows=calibration_rows,
        stage3_methods=stage3_methods,
        method_configs=method_configs,
        y_key="mean_count_projection",
        ci_key="count_projection_ci95",
        ylabel="dot(r, c_hat)",
        title="Count Projection by True Evidence Count",
    )
    plot_line_with_methods(
        output_dir=output_dir,
        filename="calibration_error_by_count.png",
        rows=calibration_rows,
        stage3_methods=stage3_methods,
        method_configs=method_configs,
        y_key="mean_abs_calibration_error",
        ci_key="calibration_error_ci95",
        ylabel="abs(dot(r, c_hat) - true count)",
        title="Calibration Error by Evidence Count",
    )
    plot_line_with_methods(
        output_dir=output_dir,
        filename="residual_norm_by_count.png",
        rows=residual_rows,
        stage3_methods=stage3_methods,
        method_configs=method_configs,
        y_key="mean_residual_norm",
        ci_key="residual_norm_ci95",
        ylabel="Mean ||r_perp||",
        title="Residual Norm by Evidence Count",
    )


def format_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> List[str]:
    table = [list(map(str, headers))] + [[str(cell) for cell in row] for row in rows]
    widths = [max(len(row[col]) for row in table) for col in range(len(headers))]
    lines = []
    for idx, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)))
        if idx == 0:
            lines.append("  ".join("-" * widths[col] for col in range(len(headers))))
    return lines


def write_summary(
    *,
    output_dir: Path,
    stage3_methods: Sequence[str],
    method_configs: Dict[str, Dict[str, Any]],
    overall_rows: Sequence[Dict[str, Any]],
    metric_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    calibration_rows: Sequence[Dict[str, Any]],
    residual_rows: Sequence[Dict[str, Any]],
    checkpoints: Dict[str, Path],
) -> None:
    best_method = best_calibrated_method(
        stage3_methods=stage3_methods,
        metric_rows=metric_rows,
        gold_rows=gold_rows,
        calibration_rows=calibration_rows,
    )
    baseline = STAGE3_SHARED_RESIDUAL
    lines = [
        "Shared count-direction calibrated memory seq_len=8",
        "",
        "Overall and middle-count metrics:",
    ]
    overview_rows = []
    for method in stage3_methods:
        config = method_configs.get(method, {})
        overview_rows.append(
            [
                short_label(method, method_configs),
                f"{overall_for_method(overall_rows, method, 'accuracy'):.4f}",
                f"{overall_for_method(overall_rows, method, 'mae'):.4f}",
                f"{mean_for_counts(metric_rows, method, 'accuracy'):.4f}",
                f"{mean_for_counts(gold_rows, method, 'mean_gold_score_delta_vs_base'):.4f}",
                f"{mean_for_counts(calibration_rows, method, 'mean_abs_calibration_error', counts=range(9)):.4f}",
                f"{mean_for_counts(residual_rows, method, 'mean_residual_norm', counts=range(9)):.4f}",
                f"{config.get('lambda_count', '')}",
                f"{config.get('lambda_res', '')}",
            ]
        )
    lines.extend(
        format_table(
            overview_rows,
            [
                "method",
                "acc",
                "mae",
                "mid_acc_3_6",
                "mid_gold_delta",
                "cal_mae",
                "res_norm",
                "lambda_count",
                "lambda_res",
            ],
        )
    )
    lines.append("")
    lines.append("Per-count accuracy:")
    for method in stage3_methods:
        rows = [
            [
                int(row["evidence_count"]),
                f"{float(row['accuracy']):.4f}",
                f"{float(row['mae']):.4f}",
                int(row["n"]),
            ]
            for row in metric_rows
            if row.get("method") == method
        ]
        lines.append(f"{short_label(method, method_configs)}:")
        lines.extend(format_table(rows, ["count", "acc", "mae", "n"]))
    lines.append("")
    lines.append("Count-direction diagnostics:")
    diag_rows = []
    for method in stage3_methods:
        corr_values = [
            float(row.get("corr_dot_true_count", math.nan))
            for row in calibration_rows
            if row.get("method") == method
        ]
        corr = next((value for value in corr_values if math.isfinite(value)), math.nan)
        diag_rows.append(
            [
                short_label(method, method_configs),
                f"{corr:.4f}",
                f"{mean_for_counts(calibration_rows, method, 'mean_abs_calibration_error', counts=range(9)):.4f}",
                f"{mean_for_counts(residual_rows, method, 'mean_residual_norm', counts=range(9)):.4f}",
                f"{mean_for_counts(gold_rows, method, 'mean_gold_score_drop_vs_base'):.4f}",
            ]
        )
    lines.extend(format_table(diag_rows, ["method", "corr(dot,true)", "mean_abs_cal_error", "mean_res_norm", "mid_gold_drop"]))
    lines.append("")
    if best_method is not None:
        config = method_configs[best_method]
        lines.append(
            "Best lambda_count/lambda_res config: "
            f"lambda_count={config['lambda_count']}, lambda_res={config['lambda_res']} "
            f"({short_label(best_method, method_configs)})."
        )
        base_mid = mean_for_counts(metric_rows, baseline, "accuracy")
        best_mid = mean_for_counts(metric_rows, best_method, "accuracy")
        base_delta = mean_for_counts(gold_rows, baseline, "mean_gold_score_delta_vs_base")
        best_delta = mean_for_counts(gold_rows, best_method, "mean_gold_score_delta_vs_base")
        base_corr = next(
            (
                float(row.get("corr_dot_true_count", math.nan))
                for row in calibration_rows
                if row.get("method") == baseline and math.isfinite(float(row.get("corr_dot_true_count", math.nan)))
            ),
            math.nan,
        )
        best_corr = next(
            (
                float(row.get("corr_dot_true_count", math.nan))
                for row in calibration_rows
                if row.get("method") == best_method and math.isfinite(float(row.get("corr_dot_true_count", math.nan)))
            ),
            math.nan,
        )
        base_res = mean_for_counts(residual_rows, baseline, "mean_residual_norm", counts=range(9))
        best_res = mean_for_counts(residual_rows, best_method, "mean_residual_norm", counts=range(9))
        success = best_mid > base_mid and best_delta > base_delta and best_corr >= 0.9 and best_res > 1e-6 and best_res < 10.0 * max(base_res, 1e-6)
        lines.append(
            "Calibration improves over the rerun shared-count residual baseline: "
            f"{'Yes' if success else 'No'} "
            f"(mid_acc {best_mid:.4f} vs {base_mid:.4f}, "
            f"mid_gold_delta {best_delta:.4f} vs {base_delta:.4f}, "
            f"corr {best_corr:.4f}, residual_norm {best_res:.4f} vs {base_res:.4f})."
        )
        lines.append("")
        lines.append("Interpretation:")
        lines.append(
            f"- Middle counts 3-6: {'improved' if best_mid > base_mid else 'did not improve'} "
            f"relative to shared residual ({best_mid:.4f} vs {base_mid:.4f})."
        )
        lines.append(
            f"- Linear count direction: {'preserved' if best_corr >= 0.9 else 'weakened'} "
            f"with corr(dot(r,c_hat), true_count)={best_corr:.4f}."
        )
        lines.append(
            f"- Residual norm: {'controlled' if best_res > 1e-6 and best_res < 10.0 * max(base_res, 1e-6) else 'not controlled'} "
            f"(best={best_res:.4f}, baseline={base_res:.4f})."
        )
        stage3_usage = best_mid > base_mid or best_delta > base_delta
        offline_clean = (
            mean_for_counts(calibration_rows, best_method, "mean_abs_calibration_error", counts=range(9))
            < mean_for_counts(calibration_rows, baseline, "mean_abs_calibration_error", counts=range(9))
        )
        if stage3_usage and offline_clean:
            usage_text = "improved Stage 3 Qwen usage and offline count cleanliness"
        elif stage3_usage:
            usage_text = "helped Stage 3 Qwen usage more than offline count cleanliness"
        elif offline_clean:
            usage_text = "mainly improved offline count cleanliness"
        else:
            usage_text = "did not clearly improve Stage 3 usage or offline cleanliness"
        lines.append(f"- Stage 3 usage vs offline cleanliness: {usage_text}.")
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
    args.lambda_counts = [float(v) for v in prev.split_tokens(args.lambda_counts)]
    args.lambda_res = [float(v) for v in prev.split_tokens(args.lambda_res)]
    output_dir = (args.output_dir if args.output_dir is not None else default_output_dir()).resolve()
    log_handle, old_stdout, old_stderr = prev.setup_logging(output_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        method_configs: Dict[str, Dict[str, Any]] = {}
        calibrated_methods: List[str] = []
        for lambda_count in args.lambda_counts:
            for lambda_res in args.lambda_res:
                method = calibrated_method_name(lambda_count, lambda_res)
                calibrated_methods.append(method)
                method_configs[method] = {
                    "variant": "shared_count_direction_plus_small_residual_calibrated",
                    "config_id": f"lc{float_tag(lambda_count)}_lres{float_tag(lambda_res)}",
                    "lambda_count": float(lambda_count),
                    "lambda_res": float(lambda_res),
                }
        stage3_methods = list(BASE_STAGE3_VARIANTS) + calibrated_methods
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
            "stage3_variants": list(stage3_methods),
            "calibration_grid": list(method_configs.values()),
            "bottleneck_dim": int(args.bottleneck_dim),
            "key_dim": int(args.key_dim),
            "value_dim": int(args.value_dim),
            "dropout": float(args.dropout),
            "frame_gate_bce_weight": float(args.frame_gate_bce_weight),
            "count_direction_mse_weight": float(args.count_direction_mse_weight),
            "count_direction_mse_weight_stage3_baseline": float(args.count_direction_mse_weight_stage3),
            "aux_mem_ce_weight_baseline": float(args.aux_mem_ce_weight),
            "residual_scale": float(args.residual_scale),
            "residual_l2_weight_baseline": float(args.residual_l2_weight),
            "huber_delta": float(args.huber_delta),
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

        stage1_models: Dict[str, shared.Stage1SharedCountReadout] = {}
        stage1_histories: Dict[str, Any] = {}
        stage1_eval_all: Dict[str, Dict[str, Any]] = {}
        stage1_checkpoints: Dict[str, Path] = {}
        eval_device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
        for variant in STAGE1_VARIANTS:
            print(f"Training {variant}")
            model_stage1, history, checkpoint = shared.train_stage1_variant(
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
                stage1_eval_all[variant][split_name] = shared.evaluate_stage1_variant(
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
        for variant, evals in stage1_eval_all.items():
            method_predictions[variant] = {
                idx: pred for split_eval in evals.values() for idx, pred in split_eval["pred_by_idx"].items()
            }
            stage1_logits_by_method[variant] = {
                idx: value for split_eval in evals.values() for idx, value in split_eval["logits_by_idx"].items()
            }

        device = prev.resolve_device(str(args.device))
        dtype = prev.resolve_dtype(str(args.dtype), device)
        model, processor = prev.load_model_and_processor(args, device=device, dtype=dtype)
        candidate_format, count_ids = prev.candidate_token_ids(
            processor.tokenizer, int(args.candidate_min), int(args.candidate_max)
        )
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
        for method in stage3_methods:
            print(f"Training {method}")
            source_stage1 = stage3_to_stage1(method)
            adapter, history, checkpoint = train_stage3_method(
                args=args,
                output_dir=output_dir,
                stage3_method=method,
                method_configs=method_configs,
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
            stage3_histories[method] = history
            stage3_checkpoints[method] = checkpoint
            print(f"Evaluating {method} on test split")
            stage3_eval = evaluate_qwen_count_channel(
                method=method,
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
            stage3_eval_by_method[method] = stage3_eval
            method_predictions[method] = stage3_eval["pred_by_idx"]
            adapter.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raw_metric_rows, raw_overall_rows, raw_mean_rows = prev.metric_tables(
            records=records,
            test_indices=splits["test"],
            method_predictions=method_predictions,
            counts=counts,
        )
        metric_rows = add_method_metadata(raw_metric_rows, method_configs)
        overall_rows = add_method_metadata(raw_overall_rows, method_configs)
        mean_rows = add_method_metadata(raw_mean_rows, method_configs)
        stage3_scores_by_method = {
            method: payload.get("gold_score_by_idx", {}) for method, payload in stage3_eval_by_method.items()
        }
        stage3_logits_by_method = {
            method: payload.get("logits_by_idx", {}) for method, payload in stage3_eval_by_method.items()
        }
        gold_rows = build_gold_score_rows(
            records=records,
            test_indices=splits["test"],
            base_scores=base_eval.get("gold_score_by_idx", {}),
            stage3_scores_by_method=stage3_scores_by_method,
            counts=counts,
            method_configs=method_configs,
        )
        calibration_rows = build_calibration_rows(
            records=records,
            test_indices=splits["test"],
            stage3_eval_by_method=stage3_eval_by_method,
            counts=counts,
            method_configs=method_configs,
        )
        residual_rows = build_residual_norm_rows(
            records=records,
            test_indices=splits["test"],
            stage3_eval_by_method=stage3_eval_by_method,
            counts=counts,
            method_configs=method_configs,
        )
        per_sample_rows = shared.build_per_sample_rows(
            records=records,
            splits=splits,
            method_predictions=method_predictions,
            stage1_logits_by_method=stage1_logits_by_method,
            base_scores=base_eval.get("gold_score_by_idx", {}),
            stage3_scores_by_method=stage3_scores_by_method,
            stage3_logits_by_method=stage3_logits_by_method,
            candidate_min=int(args.candidate_min),
        )
        debug = {
            "x_messages_shape": list(x_messages.shape),
            "D_msg": int(x_messages.shape[-1]),
            "split_counts": {split: {str(k): int(v) for k, v in row.items()} for split, row in counts_by_split.items()},
            "stage1": {
                method: {
                    "history": stage1_histories[method],
                    "checkpoint": os.fspath(stage1_checkpoints[method]),
                }
                for method in STAGE1_VARIANTS
            },
            "stage3": {
                method: {
                    "history": stage3_histories[method],
                    "checkpoint": os.fspath(stage3_checkpoints[method]),
                    "config": method_configs.get(method, {}),
                }
                for method in stage3_methods
            },
            "source_cache": os.fspath(feature_data["cache_path"]),
            "runtime_seconds": time.time() - started,
        }

        metric_fields = [
            "method",
            "stage",
            "variant",
            "config_id",
            "lambda_count",
            "lambda_res",
            "split",
            "evidence_count",
            "n",
            "accuracy",
            "mae",
        ]
        write_csv(output_dir / "metrics.csv", metric_fields, metric_rows)
        write_csv(output_dir / "per_count_accuracy.csv", metric_fields, metric_rows)
        write_csv(
            output_dir / "overall_metrics.csv",
            [
                "method",
                "stage",
                "variant",
                "config_id",
                "lambda_count",
                "lambda_res",
                "split",
                "n",
                "accuracy",
                "mae",
                "mean_predicted_count",
            ],
            overall_rows,
        )
        write_csv(
            output_dir / "mean_predicted_count_by_evidence_count.csv",
            [
                "method",
                "stage",
                "variant",
                "config_id",
                "lambda_count",
                "lambda_res",
                "split",
                "evidence_count",
                "n",
                "mean_predicted_count",
            ],
            mean_rows,
        )
        write_csv(
            output_dir / "gold_score_deltas_by_count.csv",
            [
                "method",
                "stage",
                "variant",
                "config_id",
                "lambda_count",
                "lambda_res",
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
        write_csv(
            output_dir / "calibration_by_count.csv",
            [
                "method",
                "stage",
                "variant",
                "config_id",
                "lambda_count",
                "lambda_res",
                "split",
                "evidence_count",
                "n",
                "mean_count_projection",
                "count_projection_ci95",
                "mean_abs_calibration_error",
                "calibration_error_ci95",
                "corr_dot_true_count",
            ],
            calibration_rows,
        )
        write_csv(
            output_dir / "residual_norm_by_count.csv",
            [
                "method",
                "stage",
                "variant",
                "config_id",
                "lambda_count",
                "lambda_res",
                "split",
                "evidence_count",
                "n",
                "mean_residual_norm",
                "residual_norm_ci95",
                "mean_r_norm",
                "mean_delta_norm",
            ],
            residual_rows,
        )
        per_sample_fields = sorted({key for row in per_sample_rows for key in row.keys()})
        leading = ["split", "sample_index", "sample_id", "sample_dir", "evidence_count", "gold_count"]
        per_sample_fields = leading + [field for field in per_sample_fields if field not in leading]
        write_csv(output_dir / "per_sample_predictions.csv", per_sample_fields, per_sample_rows)
        prev.write_json(output_dir / "adapter_debug.json", debug)
        checkpoints = {**stage1_checkpoints, **stage3_checkpoints}
        write_summary(
            output_dir=output_dir,
            stage3_methods=stage3_methods,
            method_configs=method_configs,
            overall_rows=overall_rows,
            metric_rows=metric_rows,
            gold_rows=gold_rows,
            calibration_rows=calibration_rows,
            residual_rows=residual_rows,
            checkpoints=checkpoints,
        )
        if not bool(args.no_plots):
            make_required_plots(
                output_dir=output_dir,
                metric_rows=metric_rows,
                overall_rows=overall_rows,
                gold_rows=gold_rows,
                calibration_rows=calibration_rows,
                residual_rows=residual_rows,
                stage3_methods=stage3_methods,
                method_configs=method_configs,
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

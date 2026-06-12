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

from scripts.experiments import evidence_only_layer_local_seq1_8_7b as base
from scripts.experiments import evidence_only_sum_evidence_adapter_seq1_8_7b as sum_base
from scripts.experiments import translator_ablation_gold_count_seq8_7b as trans
from scripts.probes import run_message_memory_adapter_stage1_stage3_seq8 as prev
from scripts.probes import run_message_memory_carrier_update_seq8_7b as carrier


EXPERIMENT_NAME = "distractor_oracle_mask_sum_adapter_seq8_7b"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images_park"

BASELINE = "baseline"
ORACLE_MASK_SUM = "oracle_mask_sum_adapter"
ORACLE_MASK_SUM_READOUT = "oracle_mask_sum"
NUM_FRAMES = 8
COUNT_VALUES = list(range(9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distractor seq_len=8 diagnostic upper bound: frozen Qwen baseline vs an "
            "oracle evidence-mask additive memory adapter. The adapter is allowed to "
            "use gold evidence-frame labels and is not a valid inference method."
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
    parser.add_argument("--run-oracle-mask-sum", action="store_true", default=False)
    parser.add_argument("--run-all", action="store_true", default=False)

    parser.add_argument("--d-mem", type=int, default=256)
    parser.add_argument("--layer-start", type=int, default=14)
    parser.add_argument("--layer-end", type=int, default=17)
    parser.add_argument("--gamma-init", type=float, default=0.05)
    parser.add_argument("--message-mode", default="auto", choices=["auto", "exact", "proxy"])
    parser.add_argument("--message-token-group", default="all_question", choices=sorted(carrier.TOKEN_GROUP_ALIASES))
    parser.add_argument("--inject-token-group", default="last_token", choices=sorted(carrier.TOKEN_GROUP_ALIASES))

    parser.add_argument("--lambda-margin", type=float, default=0.2)
    parser.add_argument("--margin-target", type=float, default=1.0)
    parser.add_argument("--lambda-update-energy", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
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
    name = str(args.run_name).strip() or f"oracle_mask_sum_{time.strftime('%Y%m%d_%H%M%S')}"
    return Path(args.output_root).resolve() / base.safe_name(name)


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
        query_token_group=str(args.inject_token_group),
        inject_token_group=str(args.inject_token_group),
    )


def oracle_masks_for_records(records: Sequence[prev.SampleRecord], device: str) -> torch.Tensor:
    masks = [evidence_mask_for_record(record, seq_len=NUM_FRAMES) for record in records]
    return torch.tensor(masks, device=device, dtype=torch.float32)


class OracleMaskSumEvidenceAdapter(sum_base.SimpleSumEvidenceAdapter):
    """Sum exact frame messages only where the gold evidence-frame mask is one."""

    def __init__(
        self,
        *,
        hidden_size: int,
        d_mem: int,
        inject_layers: Sequence[int],
        gamma_init: float,
        message_mode: str,
    ) -> None:
        super().__init__(
            hidden_size=int(hidden_size),
            d_mem=int(d_mem),
            inject_layers=[int(layer) for layer in inject_layers],
            gamma_init=float(gamma_init),
            message_mode=str(message_mode),
        )
        self.readout_mode = ORACLE_MASK_SUM_READOUT
        self._evidence_frame_masks: Optional[torch.Tensor] = None
        self._gold_counts: Optional[torch.Tensor] = None

    def set_context(
        self,
        *,
        message_target_positions: Sequence[Sequence[int]],
        inject_positions: Sequence[Sequence[int]],
        frame_groups: Sequence[Sequence[Sequence[int]]],
        evidence_frame_masks: torch.Tensor,
        gold_counts: torch.Tensor,
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
        self._last_stats["oracle_mask_by_layer"] = {}
        self._last_stats["oracle_mask_count_by_layer"] = {}
        self._last_stats["selected_message_norm_by_layer"] = {}

    def clear_context(self) -> None:
        super().clear_context()
        self._evidence_frame_masks = None
        self._gold_counts = None

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
            or self._evidence_frame_masks is None
            or int(layer_idx) not in self.layer_to_pos
        ):
            return hidden_states

        layer_pos = self.layer_to_pos[int(layer_idx)]
        self.hook_fire_counts[int(layer_idx)] += 1
        raw_messages, mode = self._message_contribution(module, hidden_states, int(layer_idx), kwargs)
        self.message_mode_counts[f"{int(layer_idx)}:{mode}"] += int(hidden_states.shape[0])

        projected_messages = self.message_to_memory[layer_pos](raw_messages.float())
        oracle_mask = self._evidence_frame_masks.to(device=projected_messages.device, dtype=projected_messages.dtype)
        masked_messages = projected_messages.float() * oracle_mask.unsqueeze(-1).float()
        summed = masked_messages.sum(dim=1)
        delta = self.w_o[layer_pos](summed).float()
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
        self._last_stats["selected_message_norm_by_layer"][layer_key] = (
            masked_messages.detach().float().norm(dim=-1).cpu().tolist()
        )
        self._last_stats["oracle_mask_by_layer"][layer_key] = oracle_mask.detach().float().cpu().tolist()
        self._last_stats["oracle_mask_count_by_layer"][layer_key] = oracle_mask.detach().float().sum(dim=1).cpu().tolist()
        self._last_stats["message_mode_by_layer"][layer_key] = [mode for _ in range(batch)]
        return out


def adapter_set_context(adapter: OracleMaskSumEvidenceAdapter, batch: carrier.MemoryBatch, records: Sequence[prev.SampleRecord]) -> None:
    adapter.set_context(
        message_target_positions=batch.message_target_positions,
        inject_positions=batch.inject_positions,
        frame_groups=batch.frame_groups,
        evidence_frame_masks=oracle_masks_for_records(records, str(batch.prompt_last_indices.device)),
        gold_counts=batch.gold_counts,
    )


def blank_diagnostics(layers: Sequence[int]) -> Dict[str, Any]:
    payload = sum_base.blank_diagnostics(layers)
    payload["oracle_mask_by_layer"] = {str(int(layer)): [] for layer in layers}
    payload["oracle_mask_count_by_layer"] = {str(int(layer)): 0.0 for layer in layers}
    payload["selected_message_norm_by_layer"] = {str(int(layer)): [] for layer in layers}
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
    adapter: Optional[OracleMaskSumEvidenceAdapter],
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
                diag = adapter.stats_for_row(row_idx) if adapter is not None else blank_diagnostics(inject_layers)
                update_norm_by_layer = diag.get("update_norm_by_layer", {})
                message_norm_by_layer = diag.get("message_norm_by_layer", {})
                summed_message_norm_by_layer = diag.get("summed_message_norm_by_layer", {})
                oracle_mask_count_by_layer = diag.get("oracle_mask_count_by_layer", {})
                update_norm = base.finite_mean(update_norm_by_layer.values(), default=0.0)
                message_norm = sum_base.mean_layer_frame_value(message_norm_by_layer)
                summed_message_norm = base.finite_mean(summed_message_norm_by_layer.values(), default=0.0)
                oracle_layer_count = base.finite_mean(oracle_mask_count_by_layer.values(), default=float(sum(oracle_mask)))
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
                        "summed_evidence_message_norm": float(summed_message_norm) if adapter is not None else "",
                        "update_norm_by_layer_json": base.json_compact(update_norm_by_layer) if adapter is not None else "",
                        "message_norm_by_layer_json": base.json_compact(message_norm_by_layer) if adapter is not None else "",
                        "raw_message_norm_by_layer_json": base.json_compact(diag.get("raw_message_norm_by_layer", {}))
                        if adapter is not None
                        else "",
                        "summed_evidence_message_norm_by_layer_json": base.json_compact(summed_message_norm_by_layer)
                        if adapter is not None
                        else "",
                        "selected_message_norm_by_layer_json": base.json_compact(
                            diag.get("selected_message_norm_by_layer", {})
                        )
                        if adapter is not None
                        else "",
                        "oracle_mask_by_layer_json": base.json_compact(diag.get("oracle_mask_by_layer", {}))
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
    inject_layers: Sequence[int],
    device: str,
) -> Tuple[OracleMaskSumEvidenceAdapter, List[Dict[str, Any]], Dict[str, Any], Path]:
    adapter = OracleMaskSumEvidenceAdapter(
        hidden_size=int(hidden_size),
        d_mem=int(args.d_mem),
        inject_layers=[int(x) for x in inject_layers],
        gamma_init=float(args.gamma_init),
        message_mode=str(args.message_mode),
    ).to(device)
    carrier.verify_trainable_parameters(model, adapter)
    optimizer = torch.optim.AdamW(
        [param for param in adapter.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "oracle_mask_sum_adapter_best.pt"
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
                        f"  {ORACLE_MASK_SUM} epoch={epoch} step={step}/{len(train_batches)} "
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
            method=ORACLE_MASK_SUM,
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
            "method": ORACLE_MASK_SUM,
            "readout_mode": ORACLE_MASK_SUM_READOUT,
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
            f"  {ORACLE_MASK_SUM} epoch={epoch} train_ce={row['train_ce']:.4f} "
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
                    "variant": ORACLE_MASK_SUM,
                    "method": ORACLE_MASK_SUM,
                    "message_mode": str(args.message_mode),
                    "readout_mode": ORACLE_MASK_SUM_READOUT,
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
        if method == ORACLE_MASK_SUM
        else 0.0,
        "mean_message_norm": base.finite_mean((row.get("message_norm") for row in rows), default=0.0)
        if method == ORACLE_MASK_SUM
        else 0.0,
        "mean_summed_evidence_message_norm": base.finite_mean(
            (row.get("summed_evidence_message_norm") for row in rows), default=0.0
        )
        if method == ORACLE_MASK_SUM
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
                if method == ORACLE_MASK_SUM
                else 0.0,
                "mean_message_norm": base.finite_mean((row.get("message_norm") for row in count_rows), default=0.0)
                if method == ORACLE_MASK_SUM
                else 0.0,
                "mean_summed_evidence_message_norm": base.finite_mean(
                    (row.get("summed_evidence_message_norm") for row in count_rows),
                    default=0.0,
                )
                if method == ORACLE_MASK_SUM
                else 0.0,
            }
        )
    return out


def comparison_by_evidence_count(accuracy_rows: Sequence[Dict[str, Any]], counts: Sequence[int]) -> List[Dict[str, Any]]:
    by_key = {(row["method"], int(row["evidence_count"])): row for row in accuracy_rows}
    out: List[Dict[str, Any]] = []
    for count in counts:
        base_row = by_key.get((BASELINE, int(count)), {})
        oracle_row = by_key.get((ORACLE_MASK_SUM, int(count)), {})
        base_acc = base.finite_float(base_row.get("accuracy"))
        oracle_acc = base.finite_float(oracle_row.get("accuracy"))
        out.append(
            {
                "evidence_count": int(count),
                "gold_count": int(count),
                "baseline_accuracy": "" if base_acc is None else float(base_acc),
                "oracle_mask_sum_accuracy": "" if oracle_acc is None else float(oracle_acc),
                "delta_accuracy": ""
                if base_acc is None or oracle_acc is None
                else float(oracle_acc) - float(base_acc),
                "baseline_mean_pred": base_row.get("mean_pred_count", ""),
                "oracle_mask_sum_mean_pred": oracle_row.get("mean_pred_count", ""),
                "baseline_mean_margin": base_row.get("mean_margin", ""),
                "oracle_mask_sum_mean_margin": oracle_row.get("mean_margin", ""),
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
    for method in [BASELINE, ORACLE_MASK_SUM]:
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
    for ax, method in zip(axes, [BASELINE, ORACLE_MASK_SUM]):
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
    for method in [BASELINE, ORACLE_MASK_SUM]:
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
    oracle_rows = [row for row in metrics_rows if row["method"] == ORACLE_MASK_SUM]
    save_confusion(plots_dir / "predicted_count_confusion_matrix_baseline.png", base_rows, counts, "Baseline Confusion Matrix")
    save_confusion(
        plots_dir / "predicted_count_confusion_matrix_oracle_mask_sum_adapter.png",
        oracle_rows,
        counts,
        "Oracle Mask Sum Adapter Confusion Matrix",
    )
    save_combined_confusions(plots_dir / "combined_confusion_matrices.png", metrics_rows, counts)

    plt.figure(figsize=(7.2, 4.3))
    xs = [int(row["evidence_count"]) for row in comparison_rows]
    ys = [float(row["delta_accuracy"]) if base.finite_float(row.get("delta_accuracy")) is not None else math.nan for row in comparison_rows]
    colors = ["#2ca02c" if base.finite_float(y) is not None and float(y) >= 0 else "#d62728" for y in ys]
    plt.bar(xs, ys, color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Evidence count")
    plt.ylabel("Oracle mask sum minus baseline accuracy")
    plt.title("Delta Accuracy")
    plt.xticks(counts)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "delta_accuracy_oracle_mask_sum_minus_baseline.png", dpi=180, bbox_inches="tight")
    plt.close()

    by_count = {int(row["evidence_count"]): row for row in accuracy_rows if row["method"] == ORACLE_MASK_SUM}
    diagnostic_specs = [
        ("mean_update_norm", "update_norm_vs_evidence_count.png", "Mean update norm", "Oracle Mask Sum Update Norm"),
        ("mean_message_norm", "message_norm_vs_evidence_count.png", "Mean message norm", "Projected Message Norm"),
        (
            "mean_summed_evidence_message_norm",
            "summed_evidence_message_norm_vs_evidence_count.png",
            "Mean summed evidence message norm",
            "Summed Evidence Message Norm",
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
        plots_dir / "candidate_logit_curves_by_evidence_count_oracle_mask_sum_adapter.png",
        metrics_rows,
        ORACLE_MASK_SUM,
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
    adapter: Optional[OracleMaskSumEvidenceAdapter],
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
    oracle_rows = method_rows(metrics_rows, ORACLE_MASK_SUM)
    localization_rows = oracle_rows or metrics_rows
    message_counts = [len(base.parse_json_field(row, "message_target_positions_json", [])) for row in localization_rows]
    inject_counts = [len(base.parse_json_field(row, "inject_positions_json", [])) for row in localization_rows]
    update_values = numeric_values_from_json_field(oracle_rows, "update_norm_by_layer_json")
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
    param_names = [] if adapter is None else [name for name, _param in adapter.named_parameters()]
    forbidden_param_names = [
        name
        for name in param_names
        if any(part in name for part in ("w_q", "w_k", "w_v", "gate", "readout", "key", "query"))
    ]
    allowed_trainable_ok = True
    trainable_param_names = [] if adapter is None else [name for name, param in adapter.named_parameters() if param.requires_grad]
    for name in trainable_param_names:
        if not (name == "gamma" or name.startswith("message_to_memory.") or name.startswith("w_o.")):
            allowed_trainable_ok = False
    mode_counts = {} if adapter is None else dict(adapter.message_mode_counts)
    exact_message_rows = sum(int(value) for key, value in mode_counts.items() if str(key).endswith(":exact"))
    proxy_message_rows = sum(int(value) for key, value in mode_counts.items() if str(key).endswith(":proxy"))
    hooks_ok = bool(
        adapter is None
        or all(int(adapter.hook_fire_counts.get(int(layer), 0)) > 0 for layer in adapter.inject_layers)
    )
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
            "summed_evidence_message_norm",
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
        "only_w_m_w_o_gamma_trainable": int(bool(adapter is not None and allowed_trainable_ok)),
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
        "oracle_mask_validation": oracle_validation,
        "oracle_masks_found_for_all_samples": int(len(mask_rows) == len(metrics_rows) and bool(mask_rows)),
        "mean_selected_evidence_frames": float(mean_selected),
        "mean_gold_count": float(mean_gold),
        "mean_selected_evidence_frames_equals_mean_gold_count": int(abs(float(mean_selected) - float(mean_gold)) <= 1e-9),
        "per_sample_oracle_mask_count_equals_gold_count": int(mask_count_equals_gold and bool(mask_rows)),
        "oracle_masks_binary": int(masks_binary and bool(mask_rows)),
        "distractor_frames_have_mask_zero": int(distractor_frames_zero and bool(mask_rows)),
        "finite_update_norms": int(finite_updates),
        "nonzero_updates": int(nonzero_updates),
        "query_key_readout_scores_used": 0,
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
    oracle_acc = base.finite_float(summary.get(ORACLE_MASK_SUM, {}).get("accuracy"))
    improved = base_acc is not None and oracle_acc is not None and float(oracle_acc) > float(base_acc)
    base_mae = mean_pred_mae(accuracy_rows, BASELINE)
    oracle_mae = mean_pred_mae(accuracy_rows, ORACLE_MASK_SUM)
    better_diagonal = base.finite_float(base_mae) is not None and base.finite_float(oracle_mae) is not None and oracle_mae < base_mae
    update_norm = base.finite_mean(
        (row.get("update_norm") for row in metrics_rows if row.get("method") == ORACLE_MASK_SUM), default=0.0
    )
    update_reasonable = base.finite_float(update_norm) is not None and 0.0 < float(update_norm) < 100.0
    mode_counts = diagnostics.get("message_mode_counts", {})
    metric_mode_counts = diagnostics.get("message_mode_resolution_from_metrics", {})

    lines = [
        "# Distractor Oracle Mask Sum Adapter seq_len=8 7B",
        "",
        "This is a diagnostic upper bound, not a valid inference method.",
        "",
        "It answers:",
        "",
        "\"If evidence selection were perfect, can a simple additive memory injection make frozen Qwen solve the distractor task?\"",
        "",
        "For each injection layer, the adapter extracts exact frame-to-all-question-token messages, applies the gold evidence-frame mask y_f, sums only evidence-frame memory vectors, and injects the sum into the last prompt token.",
        "",
        "s^l = sum_f y_f * W_m m_f^l",
        "",
        "h_last^l <- h_last^l + gamma_l * W_o s^l",
        "",
        "There is no learned gate, query/key readout, softmax, raw-matrix readout, sigmoid, or learned frame selection.",
        "",
        "## Automatic Interpretation",
        "",
        (
            f"- Did oracle mask sum improve over baseline? {bool(improved)} "
            f"(baseline={base_acc if base_acc is not None else math.nan:.4f}, "
            f"oracle-mask-sum={oracle_acc if oracle_acc is not None else math.nan:.4f})."
        ),
        (
            f"- Does mean predicted count follow y=x better than baseline? {bool(better_diagonal)} "
            f"(baseline mean-pred MAE={base_mae:.4f}, oracle-mask-sum={oracle_mae:.4f})."
        ),
        (
            f"- Are update norms active? {bool(update_reasonable)} "
            f"(mean update norm={update_norm:.6f}, finite={bool(diagnostics.get('finite_update_norms'))}, "
            f"nonzero={bool(diagnostics.get('nonzero_updates'))})."
        ),
        f"- Were oracle masks found for all samples? {bool(diagnostics.get('oracle_masks_found_for_all_samples'))}.",
        f"- Does selected-frame count equal gold_count? {bool(diagnostics.get('per_sample_oracle_mask_count_equals_gold_count'))}.",
        f"- Do distractor frames have mask 0? {bool(diagnostics.get('distractor_frames_have_mask_zero'))}.",
        f"- Were query/key/readout/gate components avoided? {bool(diagnostics.get('no_gate_query_key_readout_used'))}.",
        f"- Did message_mode=auto resolve to exact or proxy? adapter_counts={base.json_compact(mode_counts)}, metric_counts={base.json_compact(metric_mode_counts)}.",
        f"- Did Qwen remain frozen? {bool(diagnostics.get('qwen_frozen'))}.",
        f"- Were only adapter parameters trainable? {bool(diagnostics.get('only_adapter_params_trainable'))}.",
        "",
        "## Interpretation Rules",
        "",
        "- If oracle_mask_sum_adapter gets high accuracy, the injection mechanism works and the remaining problem is learning the evidence selector/gate.",
        "- If oracle_mask_sum_adapter stays low, selection is not the main issue; the injected representation is still not compatible enough with Qwen.",
        "",
        "## Files",
        "",
        "- `metrics.csv`: per-sample logits, predictions, oracle masks, token positions, and additive-sum diagnostics.",
        "- `summary.csv`: overall frozen baseline and oracle-mask sum summary.",
        "- `accuracy_by_evidence_count.csv`: accuracy and prediction histograms by evidence/gold count.",
        "- `comparison_by_evidence_count.csv`: baseline vs oracle-mask sum deltas.",
        "- `train_history.csv`: adapter training and validation history.",
        "- `diagnostics.json`: frozen-model, trainability, hook, exact-message, oracle-mask, update-norm, and no-query/key/gate checks.",
        "- `plots/`: requested comparison, confusion, candidate-logit, update/message-norm, and oracle-mask plots.",
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
            "update_norm",
            "message_norm",
            "summed_evidence_message_norm",
            "update_norm_by_layer_json",
            "message_norm_by_layer_json",
            "raw_message_norm_by_layer_json",
            "summed_evidence_message_norm_by_layer_json",
            "selected_message_norm_by_layer_json",
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
            "mean_summed_evidence_message_norm",
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
            "mean_summed_evidence_message_norm",
        ],
    )
    base.write_csv_dynamic(
        output_dir / "comparison_by_evidence_count.csv",
        comparison_rows,
        [
            "evidence_count",
            "gold_count",
            "baseline_accuracy",
            "oracle_mask_sum_accuracy",
            "delta_accuracy",
            "baseline_mean_pred",
            "oracle_mask_sum_mean_pred",
            "baseline_mean_margin",
            "oracle_mask_sum_mean_margin",
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
    args = parse_args()
    if int(args.seq_len) != NUM_FRAMES:
        raise ValueError("This diagnostic is intentionally seq_len=8 only.")
    if int(args.layer_end) < int(args.layer_start):
        raise ValueError("--layer-end must be >= --layer-start")
    if int(args.candidate_min) != 0 or int(args.candidate_max) != 8:
        raise ValueError("This runner expects candidate counts 0-8.")
    args.message_token_group = canonical_group(str(args.message_token_group))
    args.inject_token_group = canonical_group(str(args.inject_token_group))
    args.evidence_counts = prev.parse_int_tokens(args.evidence_counts)
    if not args.evidence_counts:
        raise ValueError("--evidence-counts cannot be empty")
    if not (args.run_baseline or args.run_oracle_mask_sum or args.run_all):
        args.run_all = True

    should_run_baseline = bool(args.run_baseline or args.run_all)
    should_run_oracle = bool(args.run_oracle_mask_sum or args.run_all)
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else default_output_dir(args)
    log_handle, old_stdout, old_stderr = base.setup_logging(output_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        inject_layers = list(range(int(args.layer_start), int(args.layer_end) + 1))
        run_config = {
            "experiment_name": EXPERIMENT_NAME,
            "diagnostic_upper_bound": True,
            "model_name": str(args.model_name),
            "dataset_root": os.fspath(Path(args.dataset_root).resolve()),
            "source_run": os.fspath(Path(args.source_run).resolve()),
            "seq_len": NUM_FRAMES,
            "split": str(args.split),
            "evidence_counts": [int(x) for x in args.evidence_counts],
            "output_root": os.fspath(Path(args.output_root).resolve()),
            "output_dir": os.fspath(output_dir),
            "run_baseline": bool(should_run_baseline),
            "run_oracle_mask_sum": bool(should_run_oracle),
            "d_mem": int(args.d_mem),
            "layer_start": int(args.layer_start),
            "layer_end": int(args.layer_end),
            "inject_layers": inject_layers,
            "message_mode": str(args.message_mode),
            "readout_mode": ORACLE_MASK_SUM_READOUT,
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
            raise RuntimeError("Oracle-mask adapter training requires non-empty train and val splits")
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
        adapter: Optional[OracleMaskSumEvidenceAdapter] = None
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
                inject_layers=inject_layers,
            )
            metrics_rows.extend(baseline_eval["rows"])

        if should_run_oracle:
            print("Training oracle-mask sum adapter")
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
                inject_layers=inject_layers,
                device=device,
            )
            base.write_json(
                output_dir / "checkpoint.json",
                {
                    "oracle_mask_sum_best_checkpoint": os.fspath(checkpoint_path),
                    "readout_mode": ORACLE_MASK_SUM_READOUT,
                    "message_token_group": str(args.message_token_group),
                    "inject_token_group": str(args.inject_token_group),
                },
            )
            print("Evaluating oracle-mask sum adapter on test split")
            oracle_eval = evaluate_model(
                args=args,
                method=ORACLE_MASK_SUM,
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
                inject_layers=inject_layers,
            )
            metrics_rows.extend(oracle_eval["rows"])

        summary_rows: List[Dict[str, Any]] = []
        if should_run_baseline:
            summary_rows.append(summarize_method(method_rows(metrics_rows, BASELINE), method=BASELINE))
        if should_run_oracle:
            summary_rows.append(
                summarize_method(method_rows(metrics_rows, ORACLE_MASK_SUM), method=ORACLE_MASK_SUM, train_history=train_history)
            )
        accuracy_rows: List[Dict[str, Any]] = []
        for method in [BASELINE, ORACLE_MASK_SUM]:
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
        write_readme(output_dir, summary_rows, accuracy_rows, metrics_rows, diagnostics)
        base.write_json(
            output_dir / "run_done.json",
            {
                "completed": True,
                "elapsed_seconds": time.time() - started,
                "output_dir": os.fspath(output_dir),
                "readout_mode": ORACLE_MASK_SUM_READOUT,
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

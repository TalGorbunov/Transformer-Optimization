#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
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

from experiments.evidence_only import evidence_only_layer_local_seq1_8_7b as base
from experiments.carrier_probes import run_message_memory_adapter_stage1_stage3_seq8 as prev
from experiments.carrier_probes import run_message_memory_carrier_update_seq8_7b as carrier


EXPERIMENT_NAME = "evidence_only_all_question_to_last_seq1_8_7b"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images_park_evidence_only_seq1_8"
DEFAULT_SOURCE_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images_park"

BASELINE = "baseline"
LAYER_LOCAL = "layer_local_all_question_to_last_raw_matrix"
RAW_MATRIX_READOUT = carrier.RAW_MATRIX_READOUT
SIGMOID_GATE_READOUT = carrier.SIGMOID_GATE_READOUT
COUNT_VALUES = list(range(9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evidence-only seq_len 1..8 Qwen2.5-VL-7B experiment: all-question exact "
            "frame messages build memory, the last prompt token reads raw matrix memory, "
            "and the update is injected into the last prompt token only."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--seq-lens", nargs="+", default=[str(x) for x in range(1, 9)])
    parser.add_argument("--samples-per-seq-len", type=int, default=100)
    parser.add_argument("--force-generate", action="store_true", default=False)

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="")

    parser.add_argument("--generate-dataset", action="store_true", default=False)
    parser.add_argument("--run-baseline", action="store_true", default=False)
    parser.add_argument("--run-layer-local", action="store_true", default=False)
    parser.add_argument("--run-all", action="store_true", default=False)

    parser.add_argument("--d-mem", type=int, default=256)
    parser.add_argument("--layer-start", type=int, default=14)
    parser.add_argument("--layer-end", type=int, default=17)
    parser.add_argument("--gamma-init", type=float, default=0.05)
    parser.add_argument("--message-mode", default="auto", choices=["auto", "exact", "proxy"])
    parser.add_argument("--readout-mode", default=RAW_MATRIX_READOUT, choices=sorted(carrier.READOUT_MODES))
    parser.add_argument("--message-token-group", default="all_question", choices=sorted(carrier.TOKEN_GROUP_ALIASES))
    parser.add_argument("--query-token-group", default="last_token", choices=sorted(carrier.TOKEN_GROUP_ALIASES))
    parser.add_argument("--inject-token-group", default="last_token", choices=sorted(carrier.TOKEN_GROUP_ALIASES))

    parser.add_argument("--lambda-margin", type=float, default=0.2)
    parser.add_argument("--margin-target", type=float, default=1.0)
    parser.add_argument("--lambda-update-energy", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
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
        name = f"all_question_to_last_raw_matrix_{time.strftime('%Y%m%d_%H%M%S')}"
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
        query_token_group=str(args.query_token_group),
        inject_token_group=str(args.inject_token_group),
    )


def adapter_set_context(adapter: carrier.MessageMemoryCarrierAdapter, batch: carrier.MemoryBatch) -> None:
    adapter.set_context(
        message_target_positions=batch.message_target_positions,
        query_positions=batch.query_positions,
        inject_positions=batch.inject_positions,
        frame_groups=batch.frame_groups,
    )


def mean_layer_frame_value(layer_json: Dict[str, Any]) -> float:
    return base.mean_layer_frame_value(layer_json)


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
    adapter: Optional[carrier.MessageMemoryCarrierAdapter],
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
    query_group = canonical_group(str(args.query_token_group))
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
                diag = adapter.stats_for_row(row_idx) if adapter is not None else base.blank_diagnostics(inject_layers, seq_len)
                readout_mode = str(getattr(adapter, "readout_mode", "none")) if adapter is not None else "none"
                matrix_scores_by_layer = diag.get("matrix_scores_by_layer", {})
                matrix_score_sum_by_layer = diag.get("matrix_score_sum_by_layer", {})
                matrix_score_abs_sum_by_layer = diag.get("matrix_score_abs_sum_by_layer", {})
                matrix_score_mean_by_layer = diag.get("matrix_score_mean_by_layer", {})
                matrix_score_abs_mean_by_layer = diag.get("matrix_score_abs_mean_by_layer", {})
                update_norm_by_layer = diag.get("update_norm_by_layer", {})
                message_norm_by_layer = diag.get("message_norm_by_layer", {})
                memory_norm_by_layer = diag.get("memory_norm_by_layer", {})
                update_norm = base.finite_mean(update_norm_by_layer.values(), default=0.0)
                message_norm = mean_layer_frame_value(message_norm_by_layer)
                memory_norm = mean_layer_frame_value(memory_norm_by_layer)
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
                        "query_token_group": query_group,
                        "inject_token_group": inject_group,
                        "message_target_positions_json": base.json_compact(batch.message_target_positions[row_idx]),
                        "query_positions_json": base.json_compact(batch.query_positions[row_idx]),
                        "inject_positions_json": base.json_compact(batch.inject_positions[row_idx]),
                        "mean_matrix_score_sum": base.finite_mean(matrix_score_sum_by_layer.values(), default=0.0)
                        if adapter is not None
                        else "",
                        "mean_matrix_score_abs_sum": base.finite_mean(matrix_score_abs_sum_by_layer.values(), default=0.0)
                        if adapter is not None
                        else "",
                        "mean_matrix_score_mean": base.finite_mean(matrix_score_mean_by_layer.values(), default=0.0)
                        if adapter is not None
                        else "",
                        "mean_matrix_score_abs_mean": base.finite_mean(matrix_score_abs_mean_by_layer.values(), default=0.0)
                        if adapter is not None
                        else "",
                        "matrix_scores_by_layer_json": base.json_compact(matrix_scores_by_layer) if adapter is not None else "",
                        "matrix_score_sum_by_layer_json": base.json_compact(matrix_score_sum_by_layer)
                        if adapter is not None
                        else "",
                        "matrix_score_abs_sum_by_layer_json": base.json_compact(matrix_score_abs_sum_by_layer)
                        if adapter is not None
                        else "",
                        "matrix_score_mean_by_layer_json": base.json_compact(matrix_score_mean_by_layer)
                        if adapter is not None
                        else "",
                        "matrix_score_abs_mean_by_layer_json": base.json_compact(matrix_score_abs_mean_by_layer)
                        if adapter is not None
                        else "",
                        "update_norm": float(update_norm) if adapter is not None else "",
                        "message_norm": float(message_norm) if adapter is not None else "",
                        "memory_norm": float(memory_norm) if adapter is not None else "",
                        "update_norm_by_layer_json": base.json_compact(update_norm_by_layer) if adapter is not None else "",
                        "message_norm_by_layer_json": base.json_compact(message_norm_by_layer) if adapter is not None else "",
                        "memory_norm_by_layer_json": base.json_compact(memory_norm_by_layer) if adapter is not None else "",
                        "raw_message_norm_by_layer_json": base.json_compact(diag.get("raw_message_norm_by_layer", {}))
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
) -> Tuple[carrier.MessageMemoryCarrierAdapter, List[Dict[str, Any]], Dict[str, Any], Path]:
    adapter = carrier.MessageMemoryCarrierAdapter(
        variant=carrier.LAYER_LOCAL,
        hidden_size=int(hidden_size),
        d_mem=int(args.d_mem),
        inject_layers=[int(x) for x in inject_layers],
        gamma_init=float(args.gamma_init),
        message_mode=str(args.message_mode),
        readout_mode=str(args.readout_mode),
    ).to(device)
    carrier.verify_trainable_parameters(model, adapter)
    optimizer = torch.optim.AdamW(
        [param for param in adapter.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "layer_local_all_question_to_last_raw_matrix_best.pt"
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
                    or not any(batch.query_positions)
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
                        f"  {LAYER_LOCAL} epoch={epoch} step={step}/{len(train_batches)} "
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
            method=LAYER_LOCAL,
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
            "method": LAYER_LOCAL,
            "readout_mode": str(args.readout_mode),
            "message_token_group": canonical_group(str(args.message_token_group)),
            "query_token_group": canonical_group(str(args.query_token_group)),
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
            f"  {LAYER_LOCAL} epoch={epoch} train_ce={row['train_ce']:.4f} "
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
                    "variant": carrier.LAYER_LOCAL,
                    "method": LAYER_LOCAL,
                    "message_mode": str(args.message_mode),
                    "readout_mode": str(args.readout_mode),
                    "message_token_group": canonical_group(str(args.message_token_group)),
                    "query_token_group": canonical_group(str(args.query_token_group)),
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
        "query_token_group": infer_column(rows, "query_token_group", default=canonical_group("last_token")),
        "inject_token_group": infer_column(rows, "inject_token_group", default=canonical_group("last_token")),
        "n": len(rows),
        "accuracy": float(np.mean(correct)) if correct else math.nan,
        "mean_margin": base.finite_mean(row.get("margin") for row in rows),
        "mean_gold_logit": base.finite_mean(row.get("gold_logit") for row in rows),
        "mean_pred_count": base.finite_mean(row.get("pred_count") for row in rows),
        "mean_matrix_score_sum": base.finite_mean((row.get("mean_matrix_score_sum") for row in rows), default=0.0)
        if method == LAYER_LOCAL
        else 0.0,
        "mean_matrix_score_abs_sum": base.finite_mean((row.get("mean_matrix_score_abs_sum") for row in rows), default=0.0)
        if method == LAYER_LOCAL
        else 0.0,
        "mean_update_norm": base.finite_mean((row.get("update_norm") for row in rows), default=0.0)
        if method == LAYER_LOCAL
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
                "mean_matrix_score_sum": base.finite_mean(
                    (row.get("mean_matrix_score_sum") for row in seq_rows),
                    default=0.0,
                )
                if method == LAYER_LOCAL
                else 0.0,
                "mean_matrix_score_abs_sum": base.finite_mean(
                    (row.get("mean_matrix_score_abs_sum") for row in seq_rows),
                    default=0.0,
                )
                if method == LAYER_LOCAL
                else 0.0,
                "mean_matrix_score_mean": base.finite_mean(
                    (row.get("mean_matrix_score_mean") for row in seq_rows),
                    default=0.0,
                )
                if method == LAYER_LOCAL
                else 0.0,
                "mean_matrix_score_abs_mean": base.finite_mean(
                    (row.get("mean_matrix_score_abs_mean") for row in seq_rows),
                    default=0.0,
                )
                if method == LAYER_LOCAL
                else 0.0,
                "mean_update_norm": base.finite_mean((row.get("update_norm") for row in seq_rows), default=0.0)
                if method == LAYER_LOCAL
                else 0.0,
                "mean_message_norm": base.finite_mean((row.get("message_norm") for row in seq_rows), default=0.0)
                if method == LAYER_LOCAL
                else 0.0,
                "mean_memory_norm": base.finite_mean((row.get("memory_norm") for row in seq_rows), default=0.0)
                if method == LAYER_LOCAL
                else 0.0,
            }
        )
    return out


def comparison_by_seq_len(accuracy_rows: Sequence[Dict[str, Any]], seq_lens: Sequence[int]) -> List[Dict[str, Any]]:
    by_key = {(row["method"], int(row["seq_len"])): row for row in accuracy_rows}
    out: List[Dict[str, Any]] = []
    for seq_len in seq_lens:
        base_row = by_key.get((BASELINE, int(seq_len)), {})
        local_row = by_key.get((LAYER_LOCAL, int(seq_len)), {})
        base_acc = base.finite_float(base_row.get("accuracy"))
        local_acc = base.finite_float(local_row.get("accuracy"))
        out.append(
            {
                "seq_len": int(seq_len),
                "gold_count": int(seq_len),
                "baseline_accuracy": "" if base_acc is None else float(base_acc),
                "layer_local_accuracy": "" if local_acc is None else float(local_acc),
                "delta_accuracy": ""
                if base_acc is None or local_acc is None
                else float(local_acc) - float(base_acc),
                "baseline_mean_pred": base_row.get("mean_pred_count", ""),
                "layer_local_mean_pred": local_row.get("mean_pred_count", ""),
                "baseline_mean_margin": base_row.get("mean_margin", ""),
                "layer_local_mean_margin": local_row.get("mean_margin", ""),
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
    for method in [BASELINE, LAYER_LOCAL]:
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
    for ax, method in zip(axes, [BASELINE, LAYER_LOCAL]):
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
    for method in [BASELINE, LAYER_LOCAL]:
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
    local_rows = [row for row in metrics_rows if row["method"] == LAYER_LOCAL]
    save_confusion(plots_dir / "predicted_count_confusion_matrix_baseline.png", base_rows, seq_lens, "Baseline Confusion Matrix")
    save_confusion(
        plots_dir / "predicted_count_confusion_matrix_layer_local.png",
        local_rows,
        seq_lens,
        "Layer-Local All-Question-to-Last Confusion Matrix",
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
    plt.ylabel("Layer-local minus baseline accuracy")
    plt.title("Delta Accuracy")
    plt.xticks(seq_lens)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "delta_accuracy_layer_local_minus_baseline.png", dpi=180, bbox_inches="tight")
    plt.close()

    diagnostic_specs = [
        ("mean_matrix_score_sum", "matrix_score_sum_vs_seq_len.png", "Mean matrix score sum", "Raw Matrix Score Sum vs Seq Len"),
        (
            "mean_matrix_score_abs_sum",
            "matrix_score_abs_sum_vs_seq_len.png",
            "Mean abs matrix score sum",
            "Raw Matrix Abs Score Sum vs Seq Len",
        ),
        ("mean_matrix_score_mean", "matrix_score_mean_vs_seq_len.png", "Mean matrix score", "Raw Matrix Score Mean vs Seq Len"),
        (
            "mean_matrix_score_abs_mean",
            "matrix_score_abs_mean_vs_seq_len.png",
            "Mean abs matrix score",
            "Raw Matrix Abs Score Mean vs Seq Len",
        ),
        ("mean_update_norm", "update_norm_vs_seq_len.png", "Mean update norm", "Layer-Local Update Norm vs Seq Len"),
        ("mean_message_norm", "message_norm_vs_seq_len.png", "Mean message norm", "Layer-Local Message Norm vs Seq Len"),
        ("mean_memory_norm", "memory_norm_vs_seq_len.png", "Mean memory norm", "Layer-Local Memory Norm vs Seq Len"),
    ]
    rows = sorted([row for row in accuracy_rows if row["method"] == LAYER_LOCAL], key=lambda row: int(row["seq_len"]))
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
        plots_dir / "candidate_logit_curves_by_seq_len_layer_local.png",
        metrics_rows,
        LAYER_LOCAL,
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
    adapter: Optional[carrier.MessageMemoryCarrierAdapter],
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
    query_counts = [len(base.parse_json_field(row, "query_positions_json", [])) for row in metrics_rows]
    inject_counts = [len(base.parse_json_field(row, "inject_positions_json", [])) for row in metrics_rows]
    matrix_values = numeric_values_from_json_field(metrics_rows, "matrix_scores_by_layer_json")
    update_values = numeric_values_from_json_field(metrics_rows, "update_norm_by_layer_json")
    finite_matrix = bool(matrix_values) and all(math.isfinite(float(value)) for value in matrix_values)
    nonzero_matrix = any(abs(float(value)) > 1e-12 for value in matrix_values)
    nonzero_updates = any(abs(float(value)) > 1e-12 for value in update_values)
    populated = bool(matrix_values)
    nonfinite_fields: List[Dict[str, Any]] = []
    for row in metrics_rows:
        for field in [
            "margin",
            "gold_logit",
            "pred_logit",
            "ce",
            "mean_matrix_score_sum",
            "mean_matrix_score_abs_sum",
            "mean_matrix_score_mean",
            "mean_matrix_score_abs_mean",
            "update_norm",
            "message_norm",
            "memory_norm",
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
        "qwen_frozen": int(model_trainable_tensors == 0),
        "model_trainable_tensors": int(model_trainable_tensors),
        "adapter_trainable_tensors": int(adapter_trainable_tensors),
        "adapter_trainable_params": int(adapter_trainable_params),
        "only_adapter_params_trainable": int(model_trainable_tensors == 0 and adapter_trainable_tensors > 0)
        if adapter is not None
        else "",
        "message_token_group": canonical_group(str(args.message_token_group)),
        "query_token_group": canonical_group(str(args.query_token_group)),
        "inject_token_group": canonical_group(str(args.inject_token_group)),
        "readout_mode": "none" if adapter is None else str(getattr(adapter, "readout_mode", "unknown")),
        "avg_num_message_target_positions": base.finite_mean(message_counts, default=0.0),
        "avg_num_query_positions": base.finite_mean(query_counts, default=0.0),
        "avg_num_inject_positions": base.finite_mean(inject_counts, default=0.0),
        "hooks_fire_counts": {} if adapter is None else {str(k): int(v) for k, v in sorted(adapter.hook_fire_counts.items())},
        "message_mode_counts": {} if adapter is None else dict(adapter.message_mode_counts),
        "message_mode_resolution_from_metrics": message_mode_resolution(metrics_rows),
        "exact_failure_counts": {} if adapter is None else dict(adapter.exact_failure_counts),
        "exact_failure_examples": [] if adapter is None else list(adapter.exact_failure_examples),
        "backward_diagnostics": backward_diag,
        "train_history_last": dict(train_history[-1]) if train_history else {},
        "matrix_score_diagnostics_populated": int(populated),
        "finite_matrix_scores": int(finite_matrix),
        "nonzero_matrix_scores": int(nonzero_matrix),
        "nonzero_updates": int(nonzero_updates),
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
    local_acc = base.finite_float(summary.get(LAYER_LOCAL, {}).get("accuracy"))
    improved = base_acc is not None and local_acc is not None and float(local_acc) > float(base_acc)
    base_high = high_count_accuracy(accuracy_rows, BASELINE)
    local_high = high_count_accuracy(accuracy_rows, LAYER_LOCAL)
    high_delta = local_high - base_high if base.finite_float(local_high) is not None and base.finite_float(base_high) is not None else math.nan
    base_mae = mean_pred_mae(accuracy_rows, BASELINE)
    local_mae = mean_pred_mae(accuracy_rows, LAYER_LOCAL)
    better_diagonal = base.finite_float(base_mae) is not None and base.finite_float(local_mae) is not None and local_mae < base_mae
    update_norm = base.finite_mean((row.get("update_norm") for row in metrics_rows if row.get("method") == LAYER_LOCAL), default=0.0)
    update_reasonable = base.finite_float(update_norm) is not None and 0.0 < float(update_norm) < 100.0
    mode_counts = diagnostics.get("message_mode_counts", {})
    metric_mode_counts = diagnostics.get("message_mode_resolution_from_metrics", {})

    lines = [
        "# Evidence-Only All-Question-to-Last seq_len 1..8 7B",
        "",
        "This experiment tests a more task-agnostic adapter:",
        "",
        "all-question-token frame messages -> memory -> last-token readout/injection",
        "",
        "Every frame is evidence, so gold_count=evidence_count=seq_len.",
        "",
        "Memory slot f contains the exact attention-value contribution from frame f into all question tokens:",
        "",
        "m_f^l = (1 / |Q|) sum_{q in Q} W_O [ sum_{j in I_f} A^l_{q,j} V^l_j ]",
        "",
        "The last token then queries this memory using raw matrix readout:",
        "",
        "r = sum_f (k_f^T q_last) v_f",
        "",
        "and the update is injected into the last token:",
        "",
        "h_last <- h_last + gamma W_o r",
        "",
        "No softmax and no sigmoid are used in the default configuration.",
        "",
        "## Automatic Interpretation",
        "",
        (
            f"- Did last-token memory injection improve over baseline? {bool(improved)} "
            f"(baseline={base_acc if base_acc is not None else math.nan:.4f}, "
            f"layer-local={local_acc if local_acc is not None else math.nan:.4f})."
        ),
        (
            f"- Does it improve high counts 4..8? {base.finite_float(high_delta) is not None and high_delta > 0.0} "
            f"(baseline high={base_high:.4f}, layer-local high={local_high:.4f}, delta={high_delta:.4f})."
        ),
        (
            f"- Does mean predicted count follow y=x better than baseline? {bool(better_diagonal)} "
            f"(baseline mean-pred MAE={base_mae:.4f}, layer-local={local_mae:.4f})."
        ),
        (
            f"- Are matrix scores finite/nonzero? "
            f"finite={bool(diagnostics.get('finite_matrix_scores'))}, "
            f"nonzero={bool(diagnostics.get('nonzero_matrix_scores'))}."
        ),
        (
            f"- Are update norms reasonable? {bool(update_reasonable)} "
            f"(mean update norm={update_norm:.6f}, nonzero_updates={bool(diagnostics.get('nonzero_updates'))})."
        ),
        f"- Did message_mode=auto resolve to exact or proxy? adapter_counts={base.json_compact(mode_counts)}, metric_counts={base.json_compact(metric_mode_counts)}.",
        f"- Did Qwen remain frozen? {bool(diagnostics.get('qwen_frozen'))}.",
        f"- Were only adapter parameters trainable? {bool(diagnostics.get('only_adapter_params_trainable'))}.",
        "",
        "## Interpretation Notes",
        "",
        "- If this works, it is stronger evidence for a task-agnostic memory/mixing mechanism because it does not inject into hand-picked room/char or question tokens.",
        "- If all-question-to-question works but all-question-to-last fails, that suggests the last token cannot use the memory early enough or needs later injection layers.",
        "- If it fails at layers 14..17, a follow-up should try later injection layers, e.g. 18..27 or 20..27.",
        "",
        "## Files",
        "",
        "- `metrics.csv`: per-sample logits, predictions, positions, and raw-matrix diagnostics.",
        "- `summary.csv`: overall baseline and layer-local summary.",
        "- `accuracy_by_seq_len.csv`: accuracy and prediction histograms by count.",
        "- `comparison_by_seq_len.csv`: baseline vs layer-local deltas.",
        "- `diagnostics.json`: frozen-model, trainability, token-position, hook, exact/proxy, and matrix-score checks.",
        "- `plots/`: combined accuracy, margins, mean predicted counts, confusion matrices, deltas, candidate logits, and raw-matrix diagnostics.",
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
    args.query_token_group = canonical_group(str(args.query_token_group))
    args.inject_token_group = canonical_group(str(args.inject_token_group))
    if not (args.generate_dataset or args.run_baseline or args.run_layer_local or args.run_all):
        args.run_all = True

    should_generate = bool(args.generate_dataset or args.run_all)
    should_run_baseline = bool(args.run_baseline or args.run_all)
    should_run_layer_local = bool(args.run_layer_local or args.run_all)

    if should_generate:
        base.generate_evidence_only_dataset(
            dataset_root=Path(args.dataset_root),
            source_dataset_root=Path(args.source_dataset_root),
            seq_lens=seq_lens,
            samples_per_seq_len=int(args.samples_per_seq_len),
            force=bool(args.force_generate),
        )

    if not (should_run_baseline or should_run_layer_local):
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
            "output_root": os.fspath(Path(args.output_root).resolve()),
            "output_dir": os.fspath(output_dir),
            "run_baseline": bool(should_run_baseline),
            "run_layer_local": bool(should_run_layer_local),
            "d_mem": int(args.d_mem),
            "layer_start": int(args.layer_start),
            "layer_end": int(args.layer_end),
            "inject_layers": inject_layers,
            "message_mode": str(args.message_mode),
            "readout_mode": str(args.readout_mode),
            "message_token_group": str(args.message_token_group),
            "query_token_group": str(args.query_token_group),
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
        base.print_split_counts(records, splits, seq_lens)
        if should_run_layer_local and (not splits["train"] or not splits["val"]):
            raise RuntimeError("Layer-local training requires non-empty train and val splits")
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
        adapter: Optional[carrier.MessageMemoryCarrierAdapter] = None
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

        if should_run_layer_local:
            print("Training shared all-question-to-last raw-matrix layer-local adapter")
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
                    "layer_local_best_checkpoint": os.fspath(checkpoint_path),
                    "readout_mode": str(args.readout_mode),
                    "message_token_group": str(args.message_token_group),
                    "query_token_group": str(args.query_token_group),
                    "inject_token_group": str(args.inject_token_group),
                },
            )
            print("Evaluating shared all-question-to-last layer-local adapter on test split")
            layer_eval = evaluate_model(
                args=args,
                method=LAYER_LOCAL,
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
        if should_run_layer_local:
            summary_rows.append(
                summarize_method(method_rows(metrics_rows, LAYER_LOCAL), method=LAYER_LOCAL, train_history=train_history)
            )
        accuracy_rows: List[Dict[str, Any]] = []
        for method in [BASELINE, LAYER_LOCAL]:
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
                "query_token_group",
                "inject_token_group",
                "message_target_positions_json",
                "query_positions_json",
                "inject_positions_json",
                "mean_matrix_score_sum",
                "mean_matrix_score_abs_sum",
                "mean_matrix_score_mean",
                "mean_matrix_score_abs_mean",
                "matrix_scores_by_layer_json",
                "matrix_score_sum_by_layer_json",
                "matrix_score_abs_sum_by_layer_json",
                "matrix_score_mean_by_layer_json",
                "matrix_score_abs_mean_by_layer_json",
                "update_norm",
                "message_norm",
                "memory_norm",
                "update_norm_by_layer_json",
                "message_norm_by_layer_json",
                "memory_norm_by_layer_json",
                "raw_message_norm_by_layer_json",
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
                "query_token_group",
                "inject_token_group",
                "n",
                "accuracy",
                "mean_margin",
                "mean_gold_logit",
                "mean_pred_count",
                "mean_matrix_score_sum",
                "mean_matrix_score_abs_sum",
                "mean_update_norm",
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
                "mean_matrix_score_sum",
                "mean_matrix_score_abs_sum",
                "mean_update_norm",
                "mean_message_norm",
                "mean_memory_norm",
            ],
        )
        base.write_csv_dynamic(
            output_dir / "comparison_by_seq_len.csv",
            comparison_rows,
            [
                "seq_len",
                "gold_count",
                "baseline_accuracy",
                "layer_local_accuracy",
                "delta_accuracy",
                "baseline_mean_pred",
                "layer_local_mean_pred",
                "baseline_mean_margin",
                "layer_local_mean_margin",
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
                    "query_token_group",
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
                "readout_mode": str(args.readout_mode),
                "message_token_group": str(args.message_token_group),
                "query_token_group": str(args.query_token_group),
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

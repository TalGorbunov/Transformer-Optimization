#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
from PIL import Image
from torch import nn

from scripts.probes import run_message_memory_adapter_stage1_stage3_seq8 as prev
from scripts.probes import run_shared_count_direction_memory_seq8 as shared


STAGE3_SHARED_RESIDUAL = shared.STAGE3_SHARED_RESIDUAL
STAGE3_CALIBRATED_PREFIX = "stage3_shared_count_direction_plus_small_residual_calibrated"
DEFAULT_BASELINE_RUN = PROJECT_ROOT / "outputs" / "shared_count_direction_memory_seq8_7b_20260527_203756"
DEFAULT_BASELINE_CHECKPOINT = DEFAULT_BASELINE_RUN / "checkpoints" / f"{STAGE3_SHARED_RESIDUAL}_best.pt"
PREVIOUS_BASELINE_ACCURACY = 0.5481481481481482
MIDDLE_COUNTS = (3, 4, 5, 6)
EASY_COUNTS = (0, 1, 2, 7, 8)
TARGET_ORDER = (
    "room_char",
    "all_question_tokens",
    "last_token",
    "room_char_plus_last",
    "question_plus_last",
)
TARGET_LABELS = {
    "room_char": "room_char",
    "all_question_tokens": "all_question_tokens",
    "last_token": "last_token",
    "room_char_plus_last": "room_char_plus_last",
    "question_plus_last": "question_plus_last",
}


class SweepStage3Adapter(shared.Stage3SharedCountAdapter):
    def __init__(self, *args: Any, inject_layers: Sequence[int], **kwargs: Any) -> None:
        first_layer = int(list(inject_layers)[0])
        kwargs["inject_layer"] = first_layer
        super().__init__(*args, **kwargs)
        self.inject_layers = [int(layer) for layer in inject_layers]

    def set_inject_layers(self, inject_layers: Sequence[int]) -> None:
        layers = [int(layer) for layer in inject_layers]
        if not layers:
            raise ValueError("inject_layers must not be empty")
        self.inject_layers = layers
        self.inject_layer = int(layers[0])

    def register_hooks(self, model: Any) -> None:
        self.remove_hooks()
        layers = prev.get_layers(model)
        for layer_idx in self.inject_layers:
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(f"inject_layer={layer_idx} outside [0, {len(layers) - 1}]")

        def hook(_module: Any, _args: Any, output: Any) -> Any:
            hidden = self._hidden_from_output(output)
            if hidden is None:
                return output
            return self._replace_hidden(output, self.inject(hidden))

        for layer_idx in self.inject_layers:
            self._handles.append(layers[int(layer_idx)].register_forward_hook(hook))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation-only injection-site and injection-layer sweep for seq_len=8 shared count memory."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/mmred_images_park")
    parser.add_argument("--source-run", type=Path, default=prev.DEFAULT_SOURCE_RUN)
    parser.add_argument("--base-source-run", type=Path, default=prev.DEFAULT_BASE_SOURCE_RUN)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--stage3-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--calibrated-source-run", type=Path, default=None)
    parser.add_argument("--no-auto-calibrated", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="all_uniform")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in range(9)])
    parser.add_argument("--layers", nargs="+", default=[str(x) for x in prev.DEFAULT_LAYERS])
    parser.add_argument("--max-samples-per-count", type=int, default=100)
    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--targets", nargs="+", default=list(TARGET_ORDER))
    parser.add_argument("--single-layers", nargs="+", default=["14", "16", "18", "20", "22", "24"])
    parser.add_argument("--layer-windows", nargs="+", default=["14-17", "16-19", "18-21", "20-23"])

    parser.add_argument("--bottleneck-dim", "--d-m", type=int, default=256)
    parser.add_argument("--key-dim", "--d-k", type=int, default=64)
    parser.add_argument("--value-dim", "--d-v", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--gamma-init", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=1)

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
    parser.add_argument("--smoke-limit-configs", type=int, default=0)
    parser.add_argument("--smoke-limit-samples", type=int, default=0)
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / f"memory_injection_site_sweep_seq8_7b_{stamp}"


def safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(".", "p")


def parse_layer_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for raw in prev.parse_int_tokens(args.single_layers):
        specs.append({"layer_label": str(int(raw)), "layer_kind": "single", "inject_layers": [int(raw)]})
    for raw in prev.split_tokens(args.layer_windows):
        text = str(raw)
        if "-" not in text:
            raise ValueError(f"Layer window must look like 14-17, got {text!r}")
        left, right = text.split("-", 1)
        start, end = int(left), int(right)
        if end < start:
            raise ValueError(f"Layer window end before start: {text!r}")
        specs.append(
            {
                "layer_label": f"{start}-{end}",
                "layer_kind": "window",
                "inject_layers": list(range(start, end + 1)),
            }
        )
    return specs


def read_previous_baseline(baseline_run: Path) -> Tuple[float, float]:
    path = Path(baseline_run) / "overall_metrics.csv"
    if not path.is_file():
        return PREVIOUS_BASELINE_ACCURACY, math.nan
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") == STAGE3_SHARED_RESIDUAL:
                return float(row.get("accuracy", PREVIOUS_BASELINE_ACCURACY)), float(row.get("mae", math.nan))
    return PREVIOUS_BASELINE_ACCURACY, math.nan


def latest_calibrated_dirs(explicit: Optional[Path]) -> List[Path]:
    if explicit is not None:
        return [Path(explicit)]
    dirs = [path for path in (PROJECT_ROOT / "outputs").glob("shared_count_direction_calibrated_seq8_7b_*") if path.is_dir()]
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def find_best_completed_calibrated(
    *, explicit: Optional[Path], previous_baseline_accuracy: float
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for run_dir in latest_calibrated_dirs(explicit):
        overall_path = run_dir / "overall_metrics.csv"
        if not overall_path.is_file():
            continue
        with overall_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                method = str(row.get("method", ""))
                if not method.startswith(STAGE3_CALIBRATED_PREFIX):
                    continue
                try:
                    acc = float(row.get("accuracy", "nan"))
                    mae = float(row.get("mae", "nan"))
                except ValueError:
                    continue
                ckpt = run_dir / "checkpoints" / f"{safe_name(method)}_best.pt"
                if not ckpt.is_file() or acc <= float(previous_baseline_accuracy):
                    continue
                if best is None or acc > float(best["accuracy"]):
                    best = {
                        "method": method,
                        "checkpoint": ckpt,
                        "source_run": run_dir,
                        "accuracy": acc,
                        "mae": mae,
                    }
    return best


def make_adapter_from_checkpoint(
    *,
    checkpoint_path: Path,
    method: str,
    initial_layers: Sequence[int],
) -> SweepStage3Adapter:
    ckpt = prev.load_torch(Path(checkpoint_path))
    state = ckpt["adapter_state_dict"]
    input_dim = int(state["core.norm.weight"].numel())
    bottleneck_dim = int(state["core.w_p.weight"].shape[0])
    key_dim = int(state["core.q0"].numel())
    value_dim = int(state["core.count_direction"].numel())
    hidden_size = int(state["w_o.weight"].shape[0])
    num_classes = int(state["count_head.weight"].shape[0])
    core = shared.SharedCountMemoryCore(
        input_dim=input_dim,
        bottleneck_dim=bottleneck_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        dropout=0.1,
        residual_scale=0.1,
        variant="shared_count_direction_plus_small_residual",
    )
    count_head = nn.Linear(value_dim, num_classes)
    adapter = SweepStage3Adapter(
        core=core,
        count_head=count_head,
        scalar_scale=state.get("scalar_scale", torch.tensor(1.0)),
        scalar_bias=state.get("scalar_bias", torch.tensor(0.0)),
        variant=method,
        hidden_size=hidden_size,
        inject_layers=list(initial_layers),
        gamma_init=float(ckpt.get("gamma_init", 1.0)),
        train_gamma=False,
    )
    adapter.load_state_dict(state, strict=True)
    adapter.set_inject_layers(initial_layers)
    adapter.eval()
    return adapter


def _prompt_text_bounds(record: prev.SampleRecord, processor: Any, input_ids_1d: torch.Tensor) -> Tuple[str, int]:
    input_ids = [int(token_id) for token_id in input_ids_1d.detach().cpu().tolist()]
    prompt_text = prev.core.build_prompt(record.question, num_frames=len(record.frame_paths))
    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_text_start = prev.find_subsequence(input_ids, [int(token_id) for token_id in prompt_ids])
    if prompt_text_start is None:
        raise RuntimeError(f"sample_id={record.sample_id}: failed to locate prompt text")
    return prompt_text, int(prompt_text_start)


def locate_positions_for_target(
    *,
    input_ids_1d: torch.Tensor,
    attention_mask_1d: Optional[torch.Tensor],
    record: prev.SampleRecord,
    processor: Any,
    target: str,
) -> Tuple[List[int], int, Dict[str, Any]]:
    room_char_positions, prompt_last, debug = prev.locate_target_positions(
        input_ids_1d=input_ids_1d,
        attention_mask_1d=attention_mask_1d,
        record=record,
        processor=processor,
        carriers=("target_char", "target_room"),
    )
    prompt_text, prompt_text_start = _prompt_text_bounds(record, processor, input_ids_1d)
    question_fragment = f"Question: {record.question}\n"
    question_start = prompt_text.index(question_fragment) + len("Question: ")
    question_end = question_start + len(record.question)
    question_positions = prev._positions_from_token_span(
        prompt_text_start,
        prev._token_span_from_char_span(prompt_text, (question_start, question_end), processor),
    )
    last_positions = [int(prompt_last)]
    by_target = {
        "room_char": room_char_positions,
        "all_question_tokens": question_positions,
        "last_token": last_positions,
        "room_char_plus_last": room_char_positions + last_positions,
        "question_plus_last": question_positions + last_positions,
    }
    if target not in by_target:
        raise ValueError(f"Unknown injection target: {target!r}")
    requested = sorted({int(pos) for pos in by_target[target] if 0 <= int(pos) <= int(prompt_last)})
    if not requested:
        raise RuntimeError(f"sample_id={record.sample_id}: no positions for target={target}")
    debug = dict(debug)
    debug["question_positions"] = question_positions
    debug["last_token_position"] = int(prompt_last)
    debug["target"] = target
    debug["selected_positions"] = requested
    return requested, int(prompt_last), debug


def load_frames(record: prev.SampleRecord) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for frame_path in record.frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))
    return frames


def prepare_sweep_batch(
    *,
    records: Sequence[prev.SampleRecord],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
    target: str,
) -> prev.QwenBatch:
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
    for batch_idx, record in enumerate(records):
        positions, prompt_last, _debug = locate_positions_for_target(
            input_ids_1d=input_ids[batch_idx],
            attention_mask_1d=attention_mask[batch_idx] if torch.is_tensor(attention_mask) else None,
            record=record,
            processor=processor,
            target=target,
        )
        target_positions.append(positions)
        prompt_last_indices.append(prompt_last)
    return prev.QwenBatch(
        inputs=prev.move_inputs_to_device(dict(raw_inputs), device),
        target_positions=target_positions,
        prompt_last_indices=torch.tensor(prompt_last_indices, device=device, dtype=torch.long),
        gold_counts=torch.tensor([int(record.gold_count) for record in records], device=device, dtype=torch.long),
        sample_indices=[int(idx) for idx in sample_indices],
    )


def evaluate_qwen_config(
    *,
    method: str,
    model: Any,
    processor: Any,
    adapter: Optional[SweepStage3Adapter],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    x_messages: torch.Tensor,
    count_token_ids: Dict[int, int],
    target: str,
    inject_layers: Sequence[int],
    device: str,
    batch_size: int,
) -> Dict[str, Any]:
    if adapter is not None:
        adapter.to(device)
        adapter.eval()
        adapter.enabled = True
        adapter.set_inject_layers(inject_layers)
        adapter.register_hooks(model)
    pred_by_idx: Dict[int, int] = {}
    logits_by_idx: Dict[int, List[float]] = {}
    gold_score_by_idx: Dict[int, float] = {}
    count_projection_by_idx: Dict[int, float] = {}
    count_scalar_by_idx: Dict[int, float] = {}
    r_norm_by_idx: Dict[int, float] = {}
    delta_norm_by_idx: Dict[int, float] = {}
    ce_total = 0.0
    n = 0
    try:
        for batch_num, batch_indices in enumerate(prev.chunked(list(indices), int(batch_size)), start=1):
            batch_records = [records[idx] for idx in batch_indices]
            batch = prepare_sweep_batch(
                records=batch_records,
                sample_indices=batch_indices,
                processor=processor,
                device=device,
                target=target,
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
                    for row, idx in enumerate(batch_indices):
                        if adapter.last_count_projection is not None:
                            count_projection_by_idx[int(idx)] = float(adapter.last_count_projection[row].item())
                        if adapter.last_count_scalar is not None:
                            count_scalar_by_idx[int(idx)] = float(adapter.last_count_scalar[row].item())
                        if adapter.last_r_norm is not None:
                            r_norm_by_idx[int(idx)] = float(adapter.last_r_norm[row].item())
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
        "mae": float(np.mean([abs(a - b) for a, b in zip(y_true, y_pred)])) if y_true else math.nan,
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "gold_score_by_idx": gold_score_by_idx,
        "count_projection_by_idx": count_projection_by_idx,
        "count_scalar_by_idx": count_scalar_by_idx,
        "r_norm_by_idx": r_norm_by_idx,
        "delta_norm_by_idx": delta_norm_by_idx,
    }


def metrics_for_predictions(
    *,
    method: str,
    metadata: Dict[str, Any],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    pred_by_idx: Dict[int, int],
    counts: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    per_count_rows: List[Dict[str, Any]] = []
    mean_rows: List[Dict[str, Any]] = []
    for count in counts:
        idxs = [int(idx) for idx in indices if int(records[int(idx)].gold_count) == int(count)]
        y_true = [int(records[idx].gold_count) for idx in idxs if idx in pred_by_idx]
        y_pred = [int(pred_by_idx[idx]) for idx in idxs if idx in pred_by_idx]
        row = {
            **metadata,
            "method": method,
            "split": "test",
            "evidence_count": int(count),
            "n": len(y_true),
            "accuracy": prev.accuracy(y_true, y_pred) if y_true else math.nan,
            "mae": float(np.mean([abs(a - b) for a, b in zip(y_true, y_pred)])) if y_true else math.nan,
        }
        per_count_rows.append(row)
        mean_rows.append(
            {
                **metadata,
                "method": method,
                "split": "test",
                "evidence_count": int(count),
                "n": len(y_pred),
                "mean_predicted_count": float(np.mean(y_pred)) if y_pred else math.nan,
            }
        )
    all_true = [int(records[int(idx)].gold_count) for idx in indices if int(idx) in pred_by_idx]
    all_pred = [int(pred_by_idx[int(idx)]) for idx in indices if int(idx) in pred_by_idx]
    overall = {
        **metadata,
        "method": method,
        "split": "test",
        "n": len(all_true),
        "accuracy": prev.accuracy(all_true, all_pred) if all_true else math.nan,
        "mae": float(np.mean([abs(a - b) for a, b in zip(all_true, all_pred)])) if all_true else math.nan,
        "mean_predicted_count": float(np.mean(all_pred)) if all_pred else math.nan,
        "ce": metadata.get("ce", ""),
    }
    return per_count_rows, overall, mean_rows


def aggregate_count_accuracy(
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    pred_by_idx: Dict[int, int],
    selected_counts: Sequence[int],
) -> Tuple[float, float, int]:
    count_set = {int(count) for count in selected_counts}
    y_true = [int(records[int(idx)].gold_count) for idx in indices if int(records[int(idx)].gold_count) in count_set]
    y_pred = [int(pred_by_idx[int(idx)]) for idx in indices if int(records[int(idx)].gold_count) in count_set]
    if not y_true:
        return math.nan, math.nan, 0
    return (
        prev.accuracy(y_true, y_pred),
        float(np.mean([abs(a - b) for a, b in zip(y_true, y_pred)])),
        len(y_true),
    )


def build_gold_rows(
    *,
    method: str,
    metadata: Dict[str, Any],
    records: Sequence[prev.SampleRecord],
    test_indices: Sequence[int],
    base_scores: Dict[int, float],
    stage_scores: Dict[int, float],
    counts: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for count in counts:
        base_values: List[float] = []
        stage_values: List[float] = []
        drops: List[float] = []
        deltas: List[float] = []
        for idx in test_indices:
            idx = int(idx)
            if int(records[idx].gold_count) != int(count) or idx not in base_scores or idx not in stage_scores:
                continue
            base = float(base_scores[idx])
            stage = float(stage_scores[idx])
            base_values.append(base)
            stage_values.append(stage)
            drops.append(base - stage)
            deltas.append(stage - base)
        rows.append(
            {
                **metadata,
                "method": method,
                "split": "test",
                "evidence_count": int(count),
                "n": len(deltas),
                "mean_base_gold_score": float(np.mean(base_values)) if base_values else math.nan,
                "mean_stage3_gold_score": float(np.mean(stage_values)) if stage_values else math.nan,
                "mean_gold_score_drop_vs_base": float(np.mean(drops)) if drops else math.nan,
                "median_gold_score_drop_vs_base": float(np.median(drops)) if drops else math.nan,
                "mean_gold_score_delta_vs_base": float(np.mean(deltas)) if deltas else math.nan,
                "median_gold_score_delta_vs_base": float(np.median(deltas)) if deltas else math.nan,
            }
        )
    return rows


def select_summary_row(rows: Sequence[Dict[str, Any]], predicate: Any, *, key: str = "accuracy") -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if predicate(row) and math.isfinite(float(row.get(key, math.nan)))]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get(key, -math.inf)))


def heatmap_matrix(
    rows: Sequence[Dict[str, Any]],
    *,
    value_key: str,
    targets: Sequence[str],
    layer_labels: Sequence[str],
) -> np.ndarray:
    matrix = np.full((len(targets), len(layer_labels)), np.nan, dtype=float)
    for i, target in enumerate(targets):
        for j, layer_label in enumerate(layer_labels):
            vals = [
                float(row.get(value_key, math.nan))
                for row in rows
                if row.get("injection_target") == target and row.get("layer_label") == layer_label
            ]
            vals = [value for value in vals if math.isfinite(value)]
            if vals:
                matrix[i, j] = max(vals)
    return matrix


def plot_heatmap(
    *,
    output_dir: Path,
    filename: str,
    rows: Sequence[Dict[str, Any]],
    value_key: str,
    title: str,
    targets: Sequence[str],
    layer_labels: Sequence[str],
    cmap: str,
) -> None:
    matrix = heatmap_matrix(rows, value_key=value_key, targets=targets, layer_labels=layer_labels)
    fig, ax = plt.subplots(figsize=(max(9, len(layer_labels) * 0.95), max(4.8, len(targets) * 0.75)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(targets)))
    ax.set_yticklabels([TARGET_LABELS.get(target, target) for target in targets])
    ax.set_xlabel("Injection layer/window")
    ax.set_ylabel("Injection target")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(float(value)):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color="black")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def rows_for_methods(rows: Sequence[Dict[str, Any]], methods: Sequence[str]) -> List[Dict[str, Any]]:
    method_set = set(methods)
    return [row for row in rows if str(row.get("method")) in method_set]


def plot_per_count_best_configs(
    *,
    output_dir: Path,
    per_count_rows: Sequence[Dict[str, Any]],
    selected: Sequence[Tuple[str, str]],
    filename: str,
    y_key: str,
    ylabel: str,
    title: str,
) -> None:
    counts = sorted({int(row["evidence_count"]) for row in per_count_rows})
    plt.figure(figsize=(8.5, 5))
    for method, label in selected:
        by_count = {int(row["evidence_count"]): row for row in per_count_rows if row.get("method") == method}
        values = [float(by_count.get(count, {}).get(y_key, math.nan)) for count in counts]
        if any(math.isfinite(value) for value in values):
            plt.plot(counts, values, marker="o", linewidth=1.9, label=label)
    plt.xlabel("Evidence count")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(counts)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close()


def make_required_plots(
    *,
    output_dir: Path,
    summary_rows: Sequence[Dict[str, Any]],
    per_count_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    targets: Sequence[str],
    layer_labels: Sequence[str],
    previous_room_char_method: Optional[str],
    best_last_method: Optional[str],
    best_combined_method: Optional[str],
) -> None:
    plot_heatmap(
        output_dir=output_dir,
        filename="overall_acc_heatmap_target_by_layer.png",
        rows=summary_rows,
        value_key="accuracy",
        title="Overall Accuracy by Injection Target and Layer",
        targets=targets,
        layer_labels=layer_labels,
        cmap="YlGnBu",
    )
    plot_heatmap(
        output_dir=output_dir,
        filename="middle_count_acc_heatmap_target_by_layer.png",
        rows=summary_rows,
        value_key="middle_count_accuracy_3_6",
        title="Middle-Count Accuracy by Injection Target and Layer",
        targets=targets,
        layer_labels=layer_labels,
        cmap="YlGnBu",
    )
    plot_heatmap(
        output_dir=output_dir,
        filename="mae_heatmap_target_by_layer.png",
        rows=summary_rows,
        value_key="mae",
        title="MAE by Injection Target and Layer",
        targets=targets,
        layer_labels=layer_labels,
        cmap="YlOrRd",
    )
    selected: List[Tuple[str, str]] = []
    if previous_room_char_method is not None:
        selected.append((previous_room_char_method, "prev room_char L18"))
    if best_last_method is not None and best_last_method not in {method for method, _ in selected}:
        selected.append((best_last_method, "best last_token"))
    if best_combined_method is not None and best_combined_method not in {method for method, _ in selected}:
        selected.append((best_combined_method, "best combined"))
    plot_per_count_best_configs(
        output_dir=output_dir,
        per_count_rows=per_count_rows,
        selected=selected,
        filename="per_count_accuracy_best_configs.png",
        y_key="accuracy",
        ylabel="Accuracy",
        title="Per-Count Accuracy for Best Injection Configs",
    )
    plot_per_count_best_configs(
        output_dir=output_dir,
        per_count_rows=gold_rows,
        selected=selected,
        filename="gold_score_delta_by_count_best_configs.png",
        y_key="mean_gold_score_delta_vs_base",
        ylabel="Mean gold-score delta vs base",
        title="Gold-Score Delta by Count for Best Injection Configs",
    )


def format_top(rows: Sequence[Dict[str, Any]], key: str, n: int = 5) -> List[str]:
    candidates = [row for row in rows if math.isfinite(float(row.get(key, math.nan)))]
    candidates = sorted(candidates, key=lambda row: float(row[key]), reverse=True)[:n]
    lines: List[str] = []
    for rank, row in enumerate(candidates, start=1):
        lines.append(
            f"{rank}. {row['method']}: {key}={float(row[key]):.4f}, "
            f"acc={float(row['accuracy']):.4f}, mid={float(row['middle_count_accuracy_3_6']):.4f}, "
            f"mae={float(row['mae']):.4f}"
        )
    return lines


def write_summary(
    *,
    output_dir: Path,
    summary_rows: Sequence[Dict[str, Any]],
    previous_baseline_accuracy: float,
    previous_baseline_mae: float,
    previous_room_char: Optional[Dict[str, Any]],
    best_overall: Optional[Dict[str, Any]],
    best_middle: Optional[Dict[str, Any]],
    best_last: Optional[Dict[str, Any]],
    best_combined: Optional[Dict[str, Any]],
    best_room_char: Optional[Dict[str, Any]],
    calibrated_info: Optional[Dict[str, Any]],
) -> None:
    def acc(row: Optional[Dict[str, Any]], key: str = "accuracy") -> float:
        return float(row.get(key, math.nan)) if row is not None else math.nan

    single = [row for row in summary_rows if row.get("layer_kind") == "single"]
    multi = [row for row in summary_rows if row.get("layer_kind") == "window"]
    early = [row for row in single if int(row.get("layer_start", 999)) <= 18]
    late = [row for row in single if int(row.get("layer_start", -1)) >= 20]
    early_mean = float(np.mean([float(row["accuracy"]) for row in early])) if early else math.nan
    late_mean = float(np.mean([float(row["accuracy"]) for row in late])) if late else math.nan
    best_single = select_summary_row(summary_rows, lambda row: row.get("layer_kind") == "single")
    best_multi = select_summary_row(summary_rows, lambda row: row.get("layer_kind") == "window")

    last_improves = acc(best_last) > acc(best_room_char)
    combined_improves = acc(best_combined) > acc(best_last)
    best_vs_prev = acc(best_overall) - float(previous_baseline_accuracy)
    middle_delta = acc(best_overall, "middle_count_accuracy_3_6") - acc(previous_room_char, "middle_count_accuracy_3_6")
    easy_delta = acc(best_overall, "easy_count_accuracy") - acc(previous_room_char, "easy_count_accuracy")
    if math.isfinite(middle_delta) and math.isfinite(easy_delta):
        improvement_source = "middle counts 3-6" if middle_delta > easy_delta else "easy/non-middle counts"
    else:
        improvement_source = "unclear"

    lines = [
        "Memory injection-site / last-token sweep seq_len=8",
        "",
        f"Previous baseline {STAGE3_SHARED_RESIDUAL}: accuracy={previous_baseline_accuracy:.4f}, mae={previous_baseline_mae:.4f}",
    ]
    if calibrated_info is None:
        lines.append("Calibrated variant included: no completed calibrated run beat the previous baseline at launch time.")
    else:
        lines.append(
            "Calibrated variant included: "
            f"{calibrated_info['method']} from {calibrated_info['source_run']} "
            f"(acc={float(calibrated_info['accuracy']):.4f})."
        )
    lines.extend(
        [
            "",
            "Best configs:",
            f"- Best overall config: {best_overall['method'] if best_overall else 'none'} "
            f"acc={acc(best_overall):.4f}, mae={acc(best_overall, 'mae'):.4f}",
            f"- Best middle-count 3-6 config: {best_middle['method'] if best_middle else 'none'} "
            f"mid_acc={acc(best_middle, 'middle_count_accuracy_3_6'):.4f}, acc={acc(best_middle):.4f}",
            f"- Previous room_char layer-18 config in this sweep: {previous_room_char['method'] if previous_room_char else 'missing'} "
            f"acc={acc(previous_room_char):.4f}, mid_acc={acc(previous_room_char, 'middle_count_accuracy_3_6'):.4f}",
            "",
            "Interpretation:",
            f"- Last-token injection improves over best room_char injection: {'Yes' if last_improves else 'No'} "
            f"(best_last={acc(best_last):.4f}, best_room_char={acc(best_room_char):.4f}).",
            f"- Combined injection improves over last-token-only: {'Yes' if combined_improves else 'No'} "
            f"(best_combined={acc(best_combined):.4f}, best_last={acc(best_last):.4f}).",
            f"- Earlier vs later single layers: early_mean_14_18={early_mean:.4f}, late_mean_20_24={late_mean:.4f}; "
            f"{'earlier' if early_mean >= late_mean else 'later'} layers work better on average.",
            f"- Best single-layer vs multi-layer: single={acc(best_single):.4f}, multi={acc(best_multi):.4f}; "
            f"multi-layer injection is {'better' if acc(best_multi) > acc(best_single) else 'not better'}.",
            f"- Improvement source versus previous room_char layer-18: middle_delta={middle_delta:.4f}, "
            f"easy_delta={easy_delta:.4f}; mainly {improvement_source}.",
            f"- Best overall delta versus 54.81% baseline: {best_vs_prev:+.4f}.",
            "",
            "Key questions:",
            "1. Is the 54.81% ceiling mostly an injection-site problem? "
            + ("Yes, this sweep found a higher-accuracy delivery site." if best_vs_prev > 0 else "No clear evidence from this sweep."),
            "2. Does last-token injection make the memory more directly usable? "
            + ("Yes." if acc(best_last) > float(previous_baseline_accuracy) else "Not by itself in this sweep."),
            "3. Is last-token-only enough, or does Qwen need room/character/question tokens too? "
            + ("Last-token-only was enough." if acc(best_last) >= acc(best_combined) else "Combined token delivery worked better."),
            "4. Does injecting too late fail because there are not enough layers left to process the memory? "
            + ("Likely yes." if early_mean > late_mean else "Not supported by the single-layer averages."),
            "5. Is multi-layer injection better than single-layer injection? "
            + ("Yes." if acc(best_multi) > acc(best_single) else "No."),
            "",
            "Top 5 by overall accuracy:",
        ]
    )
    lines.extend(format_top(summary_rows, "accuracy"))
    lines.append("")
    lines.append("Top 5 by middle-count accuracy:")
    lines.extend(format_top(summary_rows, "middle_count_accuracy_3_6"))
    (output_dir / "results_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    prev.write_csv(path, fieldnames, rows)


def main() -> int:
    args = parse_args()
    args.evidence_counts = prev.parse_int_tokens(args.evidence_counts)
    args.layers = prev.parse_int_tokens(args.layers)
    targets = [target for target in prev.split_tokens(args.targets)]
    unknown_targets = [target for target in targets if target not in TARGET_ORDER]
    if unknown_targets:
        raise ValueError(f"Unknown targets: {unknown_targets}")
    layer_specs = parse_layer_specs(args)
    if int(args.smoke_limit_configs) > 0:
        layer_specs = layer_specs[: max(1, int(args.smoke_limit_configs))]
        targets = targets[:1]
    output_dir = (args.output_dir if args.output_dir is not None else default_output_dir()).resolve()
    log_handle, old_stdout, old_stderr = prev.setup_logging(output_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        previous_baseline_accuracy, previous_baseline_mae = read_previous_baseline(args.baseline_run)
        calibrated_info = None
        memory_sources = [
            {
                "memory_variant": STAGE3_SHARED_RESIDUAL,
                "method_prefix": "shared_residual",
                "checkpoint": Path(args.stage3_checkpoint),
                "source_run": Path(args.baseline_run),
                "source_accuracy": previous_baseline_accuracy,
            }
        ]
        if not bool(args.no_auto_calibrated):
            calibrated_info = find_best_completed_calibrated(
                explicit=args.calibrated_source_run,
                previous_baseline_accuracy=previous_baseline_accuracy,
            )
            if calibrated_info is not None:
                memory_sources.append(
                    {
                        "memory_variant": str(calibrated_info["method"]),
                        "method_prefix": "calibrated",
                        "checkpoint": Path(calibrated_info["checkpoint"]),
                        "source_run": Path(calibrated_info["source_run"]),
                        "source_accuracy": float(calibrated_info["accuracy"]),
                    }
                )
        for source in memory_sources:
            if not Path(source["checkpoint"]).is_file():
                raise FileNotFoundError(f"Missing adapter checkpoint: {source['checkpoint']}")

        config = {
            "model_name": str(args.model_name),
            "dataset_root": os.fspath(args.dataset_root),
            "source_run": os.fspath(args.source_run),
            "base_source_run": os.fspath(args.base_source_run),
            "baseline_run": os.fspath(args.baseline_run),
            "stage3_checkpoint": os.fspath(args.stage3_checkpoint),
            "output_dir": os.fspath(output_dir),
            "split": str(args.split),
            "seq_len": int(args.seq_len),
            "evidence_counts": list(args.evidence_counts),
            "layers": list(args.layers),
            "targets": list(targets),
            "layer_specs": list(layer_specs),
            "max_samples_per_count": int(args.max_samples_per_count),
            "candidate_min": int(args.candidate_min),
            "candidate_max": int(args.candidate_max),
            "memory_sources": [
                {
                    **source,
                    "checkpoint": os.fspath(source["checkpoint"]),
                    "source_run": os.fspath(source["source_run"]),
                }
                for source in memory_sources
            ],
            "dtype": str(args.dtype),
            "attn_implementation": str(args.attn_implementation),
            "load_in_4bit": bool(args.load_in_4bit),
            "seed": int(args.seed),
            "eval_only": True,
        }
        prev.write_json(output_dir / "run_config.json", config)
        print(f"Output dir: {output_dir}")
        print(f"Config: {json.dumps(config, sort_keys=True)}")

        feature_data = prev.load_message_features(args, args.layers, args.evidence_counts)
        sample_ids = feature_data["sample_ids"]
        labels = feature_data["labels"]
        x_messages = feature_data["x_messages"]
        records = prev.load_records(args.dataset_root, args.split, args.seq_len, sample_ids)
        counts = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
        splits = prev.stratified_split(sample_ids, labels, int(args.seed))
        test_indices = list(splits["test"])
        if int(args.smoke_limit_samples) > 0:
            test_indices = test_indices[: max(1, int(args.smoke_limit_samples))]
        counts_by_split = prev.split_counts(splits, labels, counts)
        print(f"x_messages shape={tuple(x_messages.shape)} D_msg={int(x_messages.shape[-1])}")
        for split, row in counts_by_split.items():
            print(f"  {split}: " + ", ".join(f"{count}:{row.get(count, 0)}" for count in counts))

        device = prev.resolve_device(str(args.device))
        dtype = prev.resolve_dtype(str(args.dtype), device)
        model, processor = prev.load_model_and_processor(args, device, dtype)
        hidden_size = prev.hidden_size_from_model(model)
        candidate_format, count_ids = prev.candidate_token_ids(processor.tokenizer, int(args.candidate_min), int(args.candidate_max))
        print(f"Loaded Qwen hidden_size={hidden_size} candidate_format={candidate_format} count_ids={count_ids}")

        print("Evaluating base frozen Qwen on test split")
        base_eval = evaluate_qwen_config(
            method="base_frozen_qwen",
            model=model,
            processor=processor,
            adapter=None,
            records=records,
            indices=test_indices,
            x_messages=x_messages,
            count_token_ids=count_ids,
            target="room_char",
            inject_layers=[18],
            device=device,
            batch_size=int(args.eval_batch_size),
        )

        per_count_rows: List[Dict[str, Any]] = []
        overall_rows: List[Dict[str, Any]] = []
        mean_rows: List[Dict[str, Any]] = []
        middle_rows: List[Dict[str, Any]] = []
        gold_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        per_sample_rows: List[Dict[str, Any]] = []
        eval_by_method: Dict[str, Dict[str, Any]] = {}

        base_meta = {
            "memory_variant": "base_frozen_qwen",
            "source_checkpoint": "",
            "injection_target": "none",
            "layer_label": "",
            "layer_kind": "",
            "inject_layers": "",
            "layer_start": "",
            "layer_end": "",
            "ce": float(base_eval["ce"]),
        }
        base_metric, base_overall, base_mean = metrics_for_predictions(
            method="base_frozen_qwen",
            metadata=base_meta,
            records=records,
            indices=test_indices,
            pred_by_idx=base_eval["pred_by_idx"],
            counts=counts,
        )
        per_count_rows.extend(base_metric)
        overall_rows.append(base_overall)
        mean_rows.extend(base_mean)
        eval_by_method["base_frozen_qwen"] = base_eval

        first_layers = layer_specs[0]["inject_layers"]
        adapters = {
            str(source["memory_variant"]): make_adapter_from_checkpoint(
                checkpoint_path=Path(source["checkpoint"]),
                method=str(source["memory_variant"]),
                initial_layers=first_layers,
            )
            for source in memory_sources
        }

        for source in memory_sources:
            memory_variant = str(source["memory_variant"])
            adapter = adapters[memory_variant]
            for target in targets:
                for spec in layer_specs:
                    layer_label = str(spec["layer_label"])
                    inject_layers = [int(layer) for layer in spec["inject_layers"]]
                    method = f"{source['method_prefix']}__{target}__L{layer_label}"
                    metadata = {
                        "memory_variant": memory_variant,
                        "source_checkpoint": os.fspath(source["checkpoint"]),
                        "injection_target": target,
                        "layer_label": layer_label,
                        "layer_kind": str(spec["layer_kind"]),
                        "inject_layers": " ".join(str(layer) for layer in inject_layers),
                        "layer_start": int(inject_layers[0]),
                        "layer_end": int(inject_layers[-1]),
                    }
                    print(f"Evaluating {method}: target={target} layers={inject_layers}")
                    eval_payload = evaluate_qwen_config(
                        method=method,
                        model=model,
                        processor=processor,
                        adapter=adapter,
                        records=records,
                        indices=test_indices,
                        x_messages=x_messages,
                        count_token_ids=count_ids,
                        target=target,
                        inject_layers=inject_layers,
                        device=device,
                        batch_size=int(args.eval_batch_size),
                    )
                    eval_by_method[method] = eval_payload
                    metadata_with_ce = {**metadata, "ce": float(eval_payload["ce"])}
                    metrics, overall, means = metrics_for_predictions(
                        method=method,
                        metadata=metadata_with_ce,
                        records=records,
                        indices=test_indices,
                        pred_by_idx=eval_payload["pred_by_idx"],
                        counts=counts,
                    )
                    mid_acc, mid_mae, mid_n = aggregate_count_accuracy(
                        records, test_indices, eval_payload["pred_by_idx"], MIDDLE_COUNTS
                    )
                    easy_acc, easy_mae, easy_n = aggregate_count_accuracy(
                        records, test_indices, eval_payload["pred_by_idx"], EASY_COUNTS
                    )
                    summary = {
                        **overall,
                        "middle_count_accuracy_3_6": mid_acc,
                        "middle_count_mae_3_6": mid_mae,
                        "middle_count_n_3_6": mid_n,
                        "easy_count_accuracy": easy_acc,
                        "easy_count_mae": easy_mae,
                        "easy_count_n": easy_n,
                        "delta_vs_previous_baseline": float(overall["accuracy"]) - float(previous_baseline_accuracy),
                    }
                    summary_rows.append(summary)
                    middle_rows.append(
                        {
                            **metadata,
                            "method": method,
                            "split": "test",
                            "counts": "3 4 5 6",
                            "n": mid_n,
                            "accuracy": mid_acc,
                            "mae": mid_mae,
                        }
                    )
                    per_count_rows.extend(metrics)
                    overall_rows.append(overall)
                    mean_rows.extend(means)
                    gold_rows.extend(
                        build_gold_rows(
                            method=method,
                            metadata=metadata,
                            records=records,
                            test_indices=test_indices,
                            base_scores=base_eval["gold_score_by_idx"],
                            stage_scores=eval_payload["gold_score_by_idx"],
                            counts=counts,
                        )
                    )
                    for idx in test_indices:
                        idx = int(idx)
                        row = {
                            **metadata,
                            "method": method,
                            "split": "test",
                            "sample_index": idx,
                            "sample_id": records[idx].sample_id,
                            "evidence_count": int(records[idx].evidence_count),
                            "gold_count": int(records[idx].gold_count),
                            "pred_count": eval_payload["pred_by_idx"].get(idx, ""),
                            "correct": int(eval_payload["pred_by_idx"].get(idx, -999) == int(records[idx].gold_count)),
                            "base_gold_score": base_eval["gold_score_by_idx"].get(idx, ""),
                            "stage_gold_score": eval_payload["gold_score_by_idx"].get(idx, ""),
                            "gold_score_delta_vs_base": (
                                float(eval_payload["gold_score_by_idx"][idx]) - float(base_eval["gold_score_by_idx"][idx])
                                if idx in eval_payload["gold_score_by_idx"] and idx in base_eval["gold_score_by_idx"]
                                else ""
                            ),
                            "delta_norm": eval_payload["delta_norm_by_idx"].get(idx, ""),
                            "r_norm": eval_payload["r_norm_by_idx"].get(idx, ""),
                            "count_scalar": eval_payload["count_scalar_by_idx"].get(idx, ""),
                        }
                        per_sample_rows.append(row)
                    adapter.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        previous_room_char = select_summary_row(
            summary_rows,
            lambda row: row.get("memory_variant") == STAGE3_SHARED_RESIDUAL
            and row.get("injection_target") == "room_char"
            and row.get("layer_label") == "18",
        )
        best_overall = select_summary_row(summary_rows, lambda _row: True)
        best_middle = select_summary_row(summary_rows, lambda _row: True, key="middle_count_accuracy_3_6")
        best_last = select_summary_row(summary_rows, lambda row: row.get("injection_target") == "last_token")
        best_room_char = select_summary_row(summary_rows, lambda row: row.get("injection_target") == "room_char")
        best_combined = select_summary_row(
            summary_rows,
            lambda row: row.get("injection_target") in {"room_char_plus_last", "question_plus_last"},
        )

        common_fields = [
            "method",
            "memory_variant",
            "source_checkpoint",
            "injection_target",
            "layer_label",
            "layer_kind",
            "inject_layers",
            "layer_start",
            "layer_end",
            "split",
        ]
        per_count_fields = common_fields + ["evidence_count", "n", "accuracy", "mae"]
        write_csv(output_dir / "metrics.csv", per_count_fields, per_count_rows)
        write_csv(output_dir / "per_count_accuracy.csv", per_count_fields, per_count_rows)
        write_csv(
            output_dir / "overall_metrics.csv",
            common_fields + ["n", "accuracy", "mae", "mean_predicted_count", "ce"],
            overall_rows,
        )
        write_csv(
            output_dir / "middle_count_accuracy_3_6.csv",
            common_fields + ["counts", "n", "accuracy", "mae"],
            middle_rows,
        )
        write_csv(
            output_dir / "mean_predicted_count_by_evidence_count.csv",
            common_fields + ["evidence_count", "n", "mean_predicted_count"],
            mean_rows,
        )
        write_csv(
            output_dir / "gold_score_deltas_by_count.csv",
            common_fields
            + [
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
            output_dir / "injection_site_summary.csv",
            common_fields
            + [
                "n",
                "accuracy",
                "mae",
                "mean_predicted_count",
                "ce",
                "middle_count_accuracy_3_6",
                "middle_count_mae_3_6",
                "middle_count_n_3_6",
                "easy_count_accuracy",
                "easy_count_mae",
                "easy_count_n",
                "delta_vs_previous_baseline",
            ],
            summary_rows,
        )
        per_sample_fields = sorted({key for row in per_sample_rows for key in row.keys()})
        leading = [
            "method",
            "memory_variant",
            "injection_target",
            "layer_label",
            "split",
            "sample_index",
            "sample_id",
            "evidence_count",
            "gold_count",
        ]
        per_sample_fields = leading + [field for field in per_sample_fields if field not in leading]
        write_csv(output_dir / "per_sample_predictions.csv", per_sample_fields, per_sample_rows)

        debug = {
            "runtime_seconds": time.time() - started,
            "source_cache": os.fspath(feature_data["cache_path"]),
            "x_messages_shape": list(x_messages.shape),
            "candidate_format": candidate_format,
            "count_token_ids": {str(k): int(v) for k, v in count_ids.items()},
            "previous_baseline_accuracy": previous_baseline_accuracy,
            "previous_baseline_mae": previous_baseline_mae,
            "calibrated_info": (
                {
                    **calibrated_info,
                    "checkpoint": os.fspath(calibrated_info["checkpoint"]),
                    "source_run": os.fspath(calibrated_info["source_run"]),
                }
                if calibrated_info is not None
                else None
            ),
            "best_overall": best_overall,
            "best_middle": best_middle,
            "previous_room_char": previous_room_char,
        }
        prev.write_json(output_dir / "adapter_debug.json", debug)

        if not bool(args.no_plots):
            make_required_plots(
                output_dir=output_dir,
                summary_rows=summary_rows,
                per_count_rows=per_count_rows,
                gold_rows=gold_rows,
                targets=targets,
                layer_labels=[str(spec["layer_label"]) for spec in layer_specs],
                previous_room_char_method=str(previous_room_char["method"]) if previous_room_char else None,
                best_last_method=str(best_last["method"]) if best_last else None,
                best_combined_method=str(best_combined["method"]) if best_combined else None,
            )

        write_summary(
            output_dir=output_dir,
            summary_rows=summary_rows,
            previous_baseline_accuracy=previous_baseline_accuracy,
            previous_baseline_mae=previous_baseline_mae,
            previous_room_char=previous_room_char,
            best_overall=best_overall,
            best_middle=best_middle,
            best_last=best_last,
            best_combined=best_combined,
            best_room_char=best_room_char,
            calibrated_info=calibrated_info,
        )
        print(f"Finished memory injection-site sweep: {output_dir}")
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

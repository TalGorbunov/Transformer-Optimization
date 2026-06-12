#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
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

from scripts.probes import run_answer_aligned_count_codebook_memory_seq8 as codebook_prev
from scripts.probes import run_message_memory_adapter_stage1_stage3_seq8 as prev


EXPERIMENT_FAMILY = "oracle_count_multilayer_injection"
DEFAULT_RUN_NAME = "oracle_count_multilayer_injection_seq8_7b"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_FAMILY
DEFAULT_PREVIOUS_SHARED_RUN = PROJECT_ROOT / "outputs" / "shared_count_direction_memory_seq8_7b_20260527_203756"
MIDDLE_COUNTS = (3, 4, 5, 6)

STATIC_CODEBOOK = "static_count_codebook"
LAYER_CODEBOOK = "layer_specific_count_codebook"
LOW_RANK_REFT = "low_rank_reft_translator"
OLD_ORACLE = "old_qwen_oracle_codebook"
BASELINE = "frozen_qwen_baseline"

TOKEN_GROUPS = (
    "room_char",
    "all_question_tokens",
    "semantic_question_tokens",
    "last_token",
    "room_char_plus_question",
    "question_plus_last",
    "room_char_question_last",
)

PRIORITY_COMBOS = {
    "room_char": ("14-17",),
    "all_question_tokens": ("18-21", "14-24"),
    "last_token": ("20-28",),
    "question_plus_last": ("18-24", "20-28"),
    "room_char_question_last": ("14-24",),
}


class CountInjectionBatch:
    def __init__(
        self,
        *,
        inputs: Dict[str, Any],
        target_positions: List[List[int]],
        prompt_last_indices: torch.Tensor,
        gold_counts: torch.Tensor,
        sample_indices: List[int],
    ) -> None:
        self.inputs = inputs
        self.target_positions = target_positions
        self.prompt_last_indices = prompt_last_indices
        self.gold_counts = gold_counts
        self.sample_indices = sample_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle-count multi-layer residual injection experiment for MMReD seq_len=8 Qwen2.5-VL-7B."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/mmred_images_park")
    parser.add_argument("--source-run", type=Path, default=prev.DEFAULT_SOURCE_RUN)
    parser.add_argument("--previous-shared-run", type=Path, default=DEFAULT_PREVIOUS_SHARED_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--split", default="all_uniform")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in range(9)])
    parser.add_argument("--sample-limit", type=int, default=0, help="Optional max samples per split after stratified split.")
    parser.add_argument("--max-samples-per-count", type=int, default=100)
    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)

    parser.add_argument("--alphas", nargs="+", default=["0.25", "0.5", "1.0", "2.0"])
    parser.add_argument("--train-alpha", type=float, default=1.0)
    parser.add_argument("--token-groups", nargs="+", default=["room_char", "all_question_tokens", "last_token", "question_plus_last", "room_char_question_last"])
    parser.add_argument("--layer-windows", nargs="+", default=["14-17", "18-21", "18-24", "20-28", "14-24"])
    parser.add_argument("--param-types", nargs="+", default=[LAYER_CODEBOOK, STATIC_CODEBOOK])
    parser.add_argument("--margin-lambda", nargs="+", default=["0.0", "0.5"])
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--residual-l2", type=float, default=1e-4)
    parser.add_argument("--normalize-injection-energy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full-grid", action="store_true", default=False)
    parser.add_argument("--reft-rank", type=int, default=8)
    parser.add_argument("--include-old-oracle-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qwen-codebook-init-norm", type=float, default=None)
    parser.add_argument("--smoke", action="store_true", default=False)

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


def split_tokens(raw_values: Sequence[str]) -> List[str]:
    return prev.split_tokens(raw_values)


def parse_float_list(raw_values: Sequence[str]) -> List[float]:
    return [float(x) for x in split_tokens(raw_values)]


def parse_layer_window(raw: str) -> List[int]:
    text = str(raw).strip()
    if "-" not in text:
        return [int(text)]
    left, right = text.split("-", 1)
    start, end = int(left), int(right)
    if end < start:
        raise ValueError(f"Layer window end before start: {raw!r}")
    return list(range(start, end + 1))


def layer_label(layers: Sequence[int]) -> str:
    values = [int(x) for x in layers]
    if not values:
        return ""
    if len(values) == 1:
        return str(values[0])
    return f"{values[0]}-{values[-1]}"


def float_tag(value: Any) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(".", "p").replace(" ", "_").replace(",", "_")


def make_output_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"smoke_{args.run_name}" if bool(args.smoke) else str(args.run_name)
    return Path(args.output_root).resolve() / f"{prefix}_{stamp}"


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not bool(args.smoke):
        return
    args.epochs = min(int(args.epochs), 1)
    args.sample_limit = int(args.sample_limit) if int(args.sample_limit) > 0 else 2
    args.max_train_steps = int(args.max_train_steps) if int(args.max_train_steps) > 0 else 2
    args.max_eval_samples = int(args.max_eval_samples) if int(args.max_eval_samples) > 0 else 2
    args.token_groups = split_tokens(args.token_groups)[:1]
    args.layer_windows = split_tokens(args.layer_windows)[:1]
    args.param_types = split_tokens(args.param_types)[:1]
    args.alphas = split_tokens(args.alphas)[:1]
    args.margin_lambda = split_tokens(args.margin_lambda)[:1]


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def finite_mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else math.nan


def finite_median(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.median(vals)) if vals else math.nan


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return prev.accuracy(y_true, y_pred)


def mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return prev.mae(y_true, y_pred)


def load_sample_index_payload(args: argparse.Namespace) -> Dict[str, Any]:
    cache_path = prev.source_cache_path(Path(args.source_run))
    if not cache_path.is_file():
        raise FileNotFoundError(f"Could not find source cache at {cache_path}")
    print(f"Loading sample ids from source cache: {cache_path}")
    payload = prev.load_torch(cache_path)
    sample_ids = [str(x) for x in payload["sample_ids"]]
    labels = payload["labels"].long()
    frame_labels = payload.get("frame_labels")
    if torch.is_tensor(frame_labels):
        frame_labels = frame_labels.float()
    counts = set(int(x) for x in args.evidence_counts)
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
    sample_ids = [sample_ids[idx] for idx in keep]
    labels = labels[keep]
    if torch.is_tensor(frame_labels):
        frame_labels = frame_labels[keep]
    return {
        "cache_path": cache_path,
        "sample_ids": sample_ids,
        "labels": labels,
        "frame_labels": frame_labels,
    }


def limit_split_indices(indices: Sequence[int], records: Sequence[prev.SampleRecord], limit: int, seed: int) -> List[int]:
    if int(limit) <= 0 or len(indices) <= int(limit):
        return [int(x) for x in indices]
    by_count: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        by_count[int(records[int(idx)].gold_count)].append(int(idx))
    rng = random.Random(int(seed) + 17)
    out: List[int] = []
    while len(out) < int(limit) and any(by_count.values()):
        for count in sorted(by_count):
            if len(out) >= int(limit):
                break
            bucket = by_count[count]
            if not bucket:
                continue
            bucket.sort(key=lambda idx: prev.stable_hash_int(f"{seed}:limit:{idx}:{records[idx].sample_id}"))
            out.append(bucket.pop(0))
    rng.shuffle(out)
    return sorted(out)


def build_config_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    token_groups = [str(x) for x in split_tokens(args.token_groups)]
    layer_texts = [str(x) for x in split_tokens(args.layer_windows)]
    param_types = [str(x) for x in split_tokens(args.param_types)]
    margin_values = parse_float_list(args.margin_lambda)
    unknown_groups = [group for group in token_groups if group not in TOKEN_GROUPS]
    if unknown_groups:
        raise ValueError(f"Unknown token groups: {unknown_groups}; valid={list(TOKEN_GROUPS)}")
    unknown_params = [param for param in param_types if param not in {STATIC_CODEBOOK, LAYER_CODEBOOK, LOW_RANK_REFT}]
    if unknown_params:
        raise ValueError(f"Unknown param types: {unknown_params}")
    layer_specs = {text: parse_layer_window(text) for text in layer_texts}

    group_window_pairs: List[Tuple[str, str]] = []
    if bool(args.full_grid):
        group_window_pairs = [(group, text) for group in token_groups for text in layer_texts]
    else:
        selected_windows = set(layer_texts)
        for group in token_groups:
            preferred = PRIORITY_COMBOS.get(group, tuple(layer_texts))
            for text in preferred:
                if text in selected_windows:
                    group_window_pairs.append((group, text))
            if not any(existing_group == group for existing_group, _text in group_window_pairs):
                for text in layer_texts:
                    group_window_pairs.append((group, text))
                    break

    configs: List[Dict[str, Any]] = []
    for group, text in group_window_pairs:
        for param_type in param_types:
            for margin_lambda in margin_values:
                layers = layer_specs[text]
                config_id = (
                    f"{param_type}__{group}__L{layer_label(layers)}"
                    f"__ml{float_tag(margin_lambda)}"
                    f"__{'norm' if bool(args.normalize_injection_energy) else 'raw'}"
                )
                configs.append(
                    {
                        "config_id": config_id,
                        "param_type": param_type,
                        "token_group": group,
                        "layer_window": layer_label(layers),
                        "inject_layers": layers,
                        "margin_lambda": float(margin_lambda),
                        "normalize_injection_energy": bool(args.normalize_injection_energy),
                    }
                )
    return configs


def _prompt_text_bounds(record: prev.SampleRecord, processor: Any, input_ids_1d: torch.Tensor) -> Tuple[str, int]:
    input_ids = [int(token_id) for token_id in input_ids_1d.detach().cpu().tolist()]
    prompt_text = prev.core.build_prompt(record.question, num_frames=len(record.frame_paths))
    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_text_start = prev.find_subsequence(input_ids, [int(token_id) for token_id in prompt_ids])
    if prompt_text_start is None:
        raise RuntimeError(f"sample_id={record.sample_id}: failed to locate prompt text")
    return prompt_text, int(prompt_text_start)


def _dedupe_positions(values: Iterable[int], prompt_last: int) -> List[int]:
    return sorted({int(pos) for pos in values if 0 <= int(pos) <= int(prompt_last)})


def locate_positions_for_group(
    *,
    input_ids_1d: torch.Tensor,
    attention_mask_1d: Optional[torch.Tensor],
    record: prev.SampleRecord,
    processor: Any,
    token_group: str,
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
    input_ids = [int(token_id) for token_id in input_ids_1d.detach().cpu().tolist()]
    semantic_positions: List[int] = []
    for pos in question_positions:
        if not (0 <= int(pos) < len(input_ids)):
            continue
        piece = processor.tokenizer.decode([int(input_ids[int(pos)])], skip_special_tokens=True)
        stripped = str(piece).strip()
        if stripped and any(ch.isalnum() for ch in stripped):
            semantic_positions.append(int(pos))
    last_positions = [int(prompt_last)]
    by_group = {
        "room_char": room_char_positions,
        "all_question_tokens": question_positions,
        "semantic_question_tokens": semantic_positions,
        "last_token": last_positions,
        "room_char_plus_question": list(room_char_positions) + list(question_positions),
        "question_plus_last": list(question_positions) + last_positions,
        "room_char_question_last": list(room_char_positions) + list(question_positions) + last_positions,
    }
    if token_group not in by_group:
        raise ValueError(f"Unknown token group: {token_group!r}")
    selected = _dedupe_positions(by_group[token_group], int(prompt_last))
    if not selected:
        raise RuntimeError(f"sample_id={record.sample_id}: no positions for token_group={token_group}")
    out_debug = dict(debug)
    out_debug.update(
        {
            "question_positions": question_positions,
            "semantic_question_positions": semantic_positions,
            "last_token_position": int(prompt_last),
            "token_group": str(token_group),
            "selected_positions": selected,
        }
    )
    return selected, int(prompt_last), out_debug


def load_frames(record: prev.SampleRecord) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for frame_path in record.frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))
    return frames


def prepare_injection_batch(
    *,
    records: Sequence[prev.SampleRecord],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
    token_group: str,
) -> CountInjectionBatch:
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
        positions, prompt_last, _debug = locate_positions_for_group(
            input_ids_1d=input_ids[batch_idx],
            attention_mask_1d=attention_mask[batch_idx] if torch.is_tensor(attention_mask) else None,
            record=record,
            processor=processor,
            token_group=token_group,
        )
        target_positions.append(positions)
        prompt_last_indices.append(prompt_last)
    return CountInjectionBatch(
        inputs=prev.move_inputs_to_device(dict(raw_inputs), device),
        target_positions=target_positions,
        prompt_last_indices=torch.tensor(prompt_last_indices, device=device, dtype=torch.long),
        gold_counts=torch.tensor([int(record.gold_count) for record in records], device=device, dtype=torch.long),
        sample_indices=[int(idx) for idx in sample_indices],
    )


class CountResidualInjectionAdapter(nn.Module):
    def __init__(
        self,
        *,
        count_values: Sequence[int],
        hidden_size: int,
        inject_layers: Sequence[int],
        param_type: str,
        alpha: float,
        normalize_injection_energy: bool,
        reft_rank: int,
    ) -> None:
        super().__init__()
        self.count_values = [int(x) for x in count_values]
        self.count_min = min(self.count_values)
        self.count_max = max(self.count_values)
        self.hidden_size = int(hidden_size)
        self.inject_layers = [int(x) for x in inject_layers]
        self.layer_to_pos = {int(layer): pos for pos, layer in enumerate(self.inject_layers)}
        self.param_type = str(param_type)
        self.alpha = float(alpha)
        self.normalize_injection_energy = bool(normalize_injection_energy)
        self.reft_rank = int(reft_rank)
        self.enabled = True
        self._gold_counts: Optional[torch.Tensor] = None
        self._target_positions: Optional[List[List[int]]] = None
        self._cached_residuals: Optional[torch.Tensor] = None
        self._handles: List[Any] = []

        num_counts = len(self.count_values)
        num_layers = len(self.inject_layers)
        if self.param_type == STATIC_CODEBOOK:
            self.codebook = nn.Parameter(torch.zeros(num_counts, self.hidden_size, dtype=torch.float32))
        elif self.param_type == LAYER_CODEBOOK:
            self.codebook = nn.Parameter(torch.zeros(num_layers, num_counts, self.hidden_size, dtype=torch.float32))
        elif self.param_type == LOW_RANK_REFT:
            rank = max(1, int(self.reft_rank))
            self.count_code = nn.Parameter(torch.randn(num_counts, self.hidden_size, dtype=torch.float32) * 0.01)
            self.reft_down = nn.Parameter(torch.randn(num_layers, rank, self.hidden_size, dtype=torch.float32) * 0.01)
            self.reft_up = nn.Parameter(torch.zeros(num_layers, self.hidden_size, rank, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown param_type: {self.param_type!r}")

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def set_context(self, gold_counts: torch.Tensor, target_positions: Sequence[Sequence[int]]) -> None:
        self._gold_counts = gold_counts.detach().long()
        self._target_positions = [[int(pos) for pos in positions] for positions in target_positions]
        self._cached_residuals = None

    def clear_context(self) -> None:
        self._gold_counts = None
        self._target_positions = None
        self._cached_residuals = None

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

    def residuals(self, device: torch.device) -> torch.Tensor:
        if self._cached_residuals is not None and self._cached_residuals.device == device:
            return self._cached_residuals
        offsets = self._offsets(device)
        if self.param_type == STATIC_CODEBOOK:
            selected = self.codebook.to(device=device).index_select(0, offsets)
            residuals = selected.unsqueeze(1).expand(-1, len(self.inject_layers), -1).contiguous()
        elif self.param_type == LAYER_CODEBOOK:
            codebook = self.codebook.to(device=device)
            residuals = codebook[:, offsets, :].permute(1, 0, 2).contiguous()
        elif self.param_type == LOW_RANK_REFT:
            count_code = self.count_code.to(device=device).index_select(0, offsets)
            low = torch.einsum("bh,lrh->blr", count_code, self.reft_down.to(device=device))
            residuals = torch.einsum("blr,lhr->blh", low, self.reft_up.to(device=device))
        else:
            raise ValueError(f"Unknown param_type: {self.param_type!r}")
        self._cached_residuals = residuals.float()
        return self._cached_residuals

    def _sample_scales(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._target_positions is None:
            raise RuntimeError("Adapter target positions are not set")
        scales: List[float] = []
        for positions in self._target_positions:
            scale = float(self.alpha)
            if self.normalize_injection_energy:
                denom = math.sqrt(max(1, len(positions)) * max(1, len(self.inject_layers)))
                scale = scale / denom
            scales.append(scale)
        return torch.tensor(scales, device=device, dtype=dtype)

    def selected_residual_l2(self, device: torch.device) -> torch.Tensor:
        residuals = self.residuals(device)
        return residuals.pow(2).sum(dim=-1).mean()

    def energy_stats(self, device: torch.device) -> Dict[str, Any]:
        if self._target_positions is None:
            return {}
        with torch.no_grad():
            residuals = self.residuals(device).detach().float()
            scales = self._sample_scales(device, torch.float32).view(-1, 1)
            per_layer_norm = (residuals * scales.view(-1, 1, 1)).norm(dim=-1)
            position_counts = torch.tensor([len(x) for x in self._target_positions], device=device, dtype=torch.float32)
            energy_by_sample = (per_layer_norm.pow(2).sum(dim=-1) * position_counts).detach().cpu()
            total_norm_by_sample = torch.sqrt(energy_by_sample.clamp_min(0.0))
            layer_mean_norm = per_layer_norm.mean(dim=0).detach().cpu()
            layer_mean_energy = (per_layer_norm.pow(2) * position_counts.view(-1, 1)).mean(dim=0).detach().cpu()
        return {
            "energy_by_sample": [float(x) for x in energy_by_sample.tolist()],
            "total_norm_by_sample": [float(x) for x in total_norm_by_sample.tolist()],
            "position_counts": [int(len(x)) for x in self._target_positions],
            "layer_mean_norm": {str(layer): float(layer_mean_norm[pos].item()) for pos, layer in enumerate(self.inject_layers)},
            "layer_mean_energy": {str(layer): float(layer_mean_energy[pos].item()) for pos, layer in enumerate(self.inject_layers)},
        }

    def inject_for_layer(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not self.enabled or self._gold_counts is None or self._target_positions is None:
            return hidden_states
        if int(layer_idx) not in self.layer_to_pos:
            return hidden_states
        residuals = self.residuals(hidden_states.device)
        layer_pos = int(self.layer_to_pos[int(layer_idx)])
        delta = residuals[:, layer_pos, :].to(dtype=hidden_states.dtype)
        scales = self._sample_scales(hidden_states.device, hidden_states.dtype)
        delta = delta * scales.unsqueeze(-1)
        seq_len = int(hidden_states.shape[1])
        updates: List[torch.Tensor] = []
        for batch_idx, positions in enumerate(self._target_positions):
            valid = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len})
            mask = hidden_states.new_zeros((seq_len, 1))
            if valid:
                pos_idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
                mask.index_fill_(0, pos_idx, 1.0)
            updates.append(mask * delta[batch_idx].unsqueeze(0))
        return hidden_states + torch.stack(updates, dim=0)

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


def margin_loss(count_logits: torch.Tensor, gold_offsets: torch.Tensor, margin: float) -> torch.Tensor:
    batch_idx = torch.arange(int(count_logits.shape[0]), device=count_logits.device, dtype=torch.long)
    gold = count_logits[batch_idx, gold_offsets]
    wrong_logits = count_logits.clone()
    wrong_logits[batch_idx, gold_offsets] = -torch.inf
    best_wrong = wrong_logits.max(dim=-1).values
    return F.relu(float(margin) - (gold - best_wrong)).mean()


def make_qwen_codebook(model: Any, count_token_ids: Dict[int, int], hidden_size: int, norm_scale: float) -> torch.Tensor:
    vectors = codebook_prev.get_output_head_vectors(model, count_token_ids, int(hidden_size))
    centered = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    return F.normalize(centered, dim=-1) * float(norm_scale)


def evaluate_adapter(
    *,
    method: str,
    model: Any,
    processor: Any,
    adapter: Optional[CountResidualInjectionAdapter],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    count_token_ids: Dict[int, int],
    token_group: str,
    alpha: float,
    device: str,
    batch_size: int,
    max_eval_samples: int,
) -> Dict[str, Any]:
    eval_indices = list(indices)
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
    gold_margin_by_idx: Dict[int, float] = {}
    best_wrong_margin_by_idx: Dict[int, float] = {}
    ce_by_idx: Dict[int, float] = {}
    energy_by_idx: Dict[int, float] = {}
    total_norm_by_idx: Dict[int, float] = {}
    position_count_by_idx: Dict[int, int] = {}
    layer_norm_accum: Dict[int, List[float]] = defaultdict(list)
    layer_energy_accum: Dict[int, List[float]] = defaultdict(list)
    ce_total = 0.0
    n = 0
    try:
        for batch_num, batch_indices in enumerate(prev.chunked(eval_indices, int(batch_size)), start=1):
            batch_records = [records[int(idx)] for idx in batch_indices]
            batch = prepare_injection_batch(
                records=batch_records,
                sample_indices=batch_indices,
                processor=processor,
                device=device,
                token_group=token_group,
            )
            if adapter is not None:
                adapter.set_context(batch.gold_counts, batch.target_positions)
            with torch.no_grad():
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - min(count_token_ids.keys())
                ce_vec = F.cross_entropy(count_logits, gold_offsets, reduction="none")
                ce_total += float(ce_vec.sum().detach().cpu().item())
                n += int(batch.gold_counts.numel())
                pred_offsets = count_logits.argmax(dim=-1)
                logits_cpu = count_logits.detach().float().cpu()
                batch_idx_tensor = torch.arange(int(count_logits.shape[0]), device=count_logits.device, dtype=torch.long)
                gold_logits = count_logits[batch_idx_tensor, gold_offsets].float()
                wrong_logits = count_logits.float().clone()
                wrong_logits[batch_idx_tensor, gold_offsets] = -torch.inf
                best_wrong = wrong_logits.max(dim=-1).values
                for row, idx in enumerate(batch_indices):
                    idx = int(idx)
                    pred_by_idx[idx] = int(pred_offsets[row].detach().cpu().item()) + min(count_token_ids.keys())
                    values = [float(v) for v in logits_cpu[row].tolist()]
                    logits_by_idx[idx] = values
                    margin = float((gold_logits[row] - best_wrong[row]).detach().cpu().item())
                    gold_margin_by_idx[idx] = margin
                    best_wrong_margin_by_idx[idx] = -margin
                    ce_by_idx[idx] = float(ce_vec[row].detach().cpu().item())
                if adapter is not None:
                    stats = adapter.energy_stats(torch.device(device))
                    for row, idx in enumerate(batch_indices):
                        idx = int(idx)
                        energy_by_idx[idx] = float(stats["energy_by_sample"][row])
                        total_norm_by_idx[idx] = float(stats["total_norm_by_sample"][row])
                        position_count_by_idx[idx] = int(stats["position_counts"][row])
                    for layer, value in stats["layer_mean_norm"].items():
                        layer_norm_accum[int(layer)].append(float(value))
                    for layer, value in stats["layer_mean_energy"].items():
                        layer_energy_accum[int(layer)].append(float(value))
            if adapter is not None:
                adapter.clear_context()
            if batch_num == 1 or batch_num % 50 == 0:
                print(f"  eval {method}: {min(len(eval_indices), batch_num * int(batch_size))}/{len(eval_indices)}")
    finally:
        if adapter is not None:
            adapter.remove_hooks()
    y_true = [int(records[int(idx)].gold_count) for idx in eval_indices if int(idx) in pred_by_idx]
    y_pred = [pred_by_idx[int(idx)] for idx in eval_indices if int(idx) in pred_by_idx]
    layer_mean_norm = {str(layer): finite_mean(values) for layer, values in sorted(layer_norm_accum.items())}
    layer_mean_energy = {str(layer): finite_mean(values) for layer, values in sorted(layer_energy_accum.items())}
    return {
        "indices": eval_indices,
        "ce": ce_total / max(1, n),
        "accuracy": accuracy(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "gold_margin_by_idx": gold_margin_by_idx,
        "best_wrong_margin_by_idx": best_wrong_margin_by_idx,
        "ce_by_idx": ce_by_idx,
        "energy_by_idx": energy_by_idx,
        "total_norm_by_idx": total_norm_by_idx,
        "position_count_by_idx": position_count_by_idx,
        "layer_mean_norm": layer_mean_norm,
        "layer_mean_energy": layer_mean_energy,
    }


def train_one_config(
    *,
    config: Dict[str, Any],
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
    device: str,
) -> Tuple[CountResidualInjectionAdapter, List[Dict[str, Any]], Path]:
    adapter = CountResidualInjectionAdapter(
        count_values=count_values,
        hidden_size=int(hidden_size),
        inject_layers=config["inject_layers"],
        param_type=str(config["param_type"]),
        alpha=float(args.train_alpha),
        normalize_injection_energy=bool(config["normalize_injection_energy"]),
        reft_rank=int(args.reft_rank),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in adapter.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"{safe_name(config['config_id'])}_best.pt"
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_acc = -math.inf
    best_val_ce = math.inf
    for epoch in range(1, int(args.epochs) + 1):
        adapter.train()
        adapter.set_alpha(float(args.train_alpha))
        rng = random.Random(int(args.seed) + epoch * 997 + prev.stable_hash_int(str(config["config_id"])) % 100000)
        shuffled = [int(x) for x in train_indices]
        rng.shuffle(shuffled)
        optimizer.zero_grad(set_to_none=True)
        train_ce_total = 0.0
        train_loss_total = 0.0
        train_steps = 0
        try:
            adapter.register_hooks(model)
            for step, batch_indices in enumerate(prev.chunked(shuffled, int(args.train_batch_size)), start=1):
                if int(args.max_train_steps) > 0 and step > int(args.max_train_steps):
                    break
                batch_records = [records[int(idx)] for idx in batch_indices]
                batch = prepare_injection_batch(
                    records=batch_records,
                    sample_indices=batch_indices,
                    processor=processor,
                    device=device,
                    token_group=str(config["token_group"]),
                )
                adapter.set_context(batch.gold_counts, batch.target_positions)
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - min(count_token_ids.keys())
                ce = F.cross_entropy(count_logits, gold_offsets)
                m_loss = margin_loss(count_logits, gold_offsets, float(args.margin))
                r_l2 = adapter.selected_residual_l2(torch.device(device))
                loss = ce + float(config["margin_lambda"]) * m_loss + float(args.residual_l2) * r_l2
                (loss / max(1, int(args.grad_accum))).backward()
                train_ce_total += float(ce.detach().cpu().item())
                train_loss_total += float(loss.detach().cpu().item())
                train_steps += 1
                adapter.clear_context()
                if step % max(1, int(args.grad_accum)) == 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 50 == 0:
                    print(
                        f"  {config['config_id']} epoch={epoch} step={step} "
                        f"train_ce={train_ce_total / max(1, train_steps):.4f}"
                    )
            if train_steps % max(1, int(args.grad_accum)) != 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        finally:
            adapter.remove_hooks()
        val_eval = evaluate_adapter(
            method=f"{config['config_id']}__val",
            model=model,
            processor=processor,
            adapter=adapter,
            records=records,
            indices=val_indices,
            count_token_ids=count_token_ids,
            token_group=str(config["token_group"]),
            alpha=float(args.train_alpha),
            device=device,
            batch_size=int(args.eval_batch_size),
            max_eval_samples=int(args.max_eval_samples),
        )
        row = {
            "train_config_id": str(config["config_id"]),
            "param_type": str(config["param_type"]),
            "token_group": str(config["token_group"]),
            "layer_window": str(config["layer_window"]),
            "inject_layers": " ".join(str(x) for x in config["inject_layers"]),
            "margin_lambda": float(config["margin_lambda"]),
            "normalize_injection_energy": int(bool(config["normalize_injection_energy"])),
            "train_alpha": float(args.train_alpha),
            "epoch": int(epoch),
            "train_ce": train_ce_total / max(1, train_steps),
            "train_loss": train_loss_total / max(1, train_steps),
            "val_ce": float(val_eval["ce"]),
            "val_accuracy": float(val_eval["accuracy"]),
            "val_mae": float(val_eval["mae"]),
        }
        history.append(row)
        print(
            f"  {config['config_id']} epoch={epoch} train_ce={row['train_ce']:.4f} "
            f"val_ce={row['val_ce']:.4f} val_acc={row['val_accuracy']:.4f}"
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
                    "config": config,
                    "history": history,
                    "hidden_size": int(hidden_size),
                    "count_values": list(count_values),
                },
                checkpoint_path,
            )
    if best_state is not None:
        adapter.load_state_dict(best_state)
    return adapter.cpu(), history, checkpoint_path


def count_subset_metrics(
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    pred_by_idx: Dict[int, int],
    counts: Sequence[int],
) -> Tuple[float, float, int]:
    wanted = set(int(x) for x in counts)
    y_true: List[int] = []
    y_pred: List[int] = []
    for idx in indices:
        idx = int(idx)
        if idx in pred_by_idx and int(records[idx].gold_count) in wanted:
            y_true.append(int(records[idx].gold_count))
            y_pred.append(int(pred_by_idx[idx]))
    return accuracy(y_true, y_pred), mae(y_true, y_pred), len(y_true)


def per_count_rows(
    *,
    metadata: Dict[str, Any],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    pred_by_idx: Dict[int, int],
    count_values: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for count in count_values:
        idxs = [int(idx) for idx in indices if int(records[int(idx)].gold_count) == int(count)]
        y_true = [int(records[idx].gold_count) for idx in idxs if idx in pred_by_idx]
        y_pred = [int(pred_by_idx[idx]) for idx in idxs if idx in pred_by_idx]
        rows.append(
            {
                **metadata,
                "evidence_count": int(count),
                "gold_count": int(count),
                "n": len(y_true),
                "accuracy": accuracy(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "mean_predicted_count": float(np.mean(y_pred)) if y_pred else math.nan,
            }
        )
    return rows


def predicted_distribution_rows(
    *,
    metadata: Dict[str, Any],
    pred_by_idx: Dict[int, int],
    count_values: Sequence[int],
) -> List[Dict[str, Any]]:
    preds = [int(x) for x in pred_by_idx.values()]
    rows: List[Dict[str, Any]] = []
    for count in count_values:
        n = sum(1 for pred in preds if int(pred) == int(count))
        rows.append(
            {
                **metadata,
                "predicted_count": int(count),
                "n": int(n),
                "fraction": n / max(1, len(preds)),
            }
        )
    return rows


def summarize_eval_row(
    *,
    metadata: Dict[str, Any],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    eval_payload: Dict[str, Any],
    train_history: Sequence[Dict[str, Any]],
    count_values: Sequence[int],
) -> Dict[str, Any]:
    eval_indices = eval_payload["indices"]
    pred_by_idx = eval_payload["pred_by_idx"]
    y_true = [int(records[int(idx)].gold_count) for idx in eval_indices if int(idx) in pred_by_idx]
    y_pred = [int(pred_by_idx[int(idx)]) for idx in eval_indices if int(idx) in pred_by_idx]
    mid_acc, mid_mae, mid_n = count_subset_metrics(records, eval_indices, pred_by_idx, MIDDLE_COUNTS)
    count_rows = per_count_rows(
        metadata=metadata,
        records=records,
        indices=eval_indices,
        pred_by_idx=pred_by_idx,
        count_values=count_values,
    )
    pred_dist = predicted_distribution_rows(metadata=metadata, pred_by_idx=pred_by_idx, count_values=count_values)
    last_history = train_history[-1] if train_history else {}
    best_history = max(
        train_history,
        key=lambda row: (
            float(row.get("val_accuracy", -math.inf)),
            -float(row.get("val_ce", math.inf)),
        ),
        default={},
    )
    energy_values = [float(x) for x in eval_payload["energy_by_idx"].values()]
    norm_values = [float(x) for x in eval_payload["total_norm_by_idx"].values()]
    row = {
        **metadata,
        "split": "test",
        "n": len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "middle_count_accuracy_3_6": mid_acc,
        "middle_count_mae_3_6": mid_mae,
        "middle_count_n_3_6": mid_n,
        "ce": float(eval_payload["ce"]),
        "mean_predicted_count": float(np.mean(y_pred)) if y_pred else math.nan,
        "gold_logit_margin_mean": finite_mean(list(eval_payload["gold_margin_by_idx"].values())),
        "gold_logit_margin_median": finite_median(list(eval_payload["gold_margin_by_idx"].values())),
        "best_wrong_logit_margin_mean": finite_mean(list(eval_payload["best_wrong_margin_by_idx"].values())),
        "best_wrong_logit_margin_median": finite_median(list(eval_payload["best_wrong_margin_by_idx"].values())),
        "mean_total_injection_energy": finite_mean(energy_values),
        "sum_total_injection_energy": float(np.sum(energy_values)) if energy_values else 0.0,
        "mean_total_injection_norm": finite_mean(norm_values),
        "median_total_injection_norm": finite_median(norm_values),
        "residual_norm_per_layer_json": json_dumps_compact(eval_payload["layer_mean_norm"]),
        "injection_energy_per_layer_json": json_dumps_compact(eval_payload["layer_mean_energy"]),
        "accuracy_by_gold_count_json": json_dumps_compact({str(r["gold_count"]): r["accuracy"] for r in count_rows}),
        "predicted_count_distribution_json": json_dumps_compact({str(r["predicted_count"]): r["fraction"] for r in pred_dist}),
        "train_ce_last": last_history.get("train_ce", ""),
        "val_ce_last": last_history.get("val_ce", ""),
        "val_accuracy_last": last_history.get("val_accuracy", ""),
        "best_val_ce": best_history.get("val_ce", ""),
        "best_val_accuracy": best_history.get("val_accuracy", ""),
    }
    return row


def build_per_sample_rows(
    *,
    metadata: Dict[str, Any],
    records: Sequence[prev.SampleRecord],
    eval_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx in eval_payload["indices"]:
        idx = int(idx)
        pred = eval_payload["pred_by_idx"].get(idx, "")
        rows.append(
            {
                **metadata,
                "split": "test",
                "sample_index": idx,
                "sample_id": records[idx].sample_id,
                "sample_dir": os.fspath(records[idx].sample_dir),
                "evidence_count": int(records[idx].evidence_count),
                "gold_count": int(records[idx].gold_count),
                "predicted_count": pred,
                "correct": int(pred == int(records[idx].gold_count)) if pred != "" else "",
                "ce": eval_payload["ce_by_idx"].get(idx, ""),
                "gold_logit_margin": eval_payload["gold_margin_by_idx"].get(idx, ""),
                "best_wrong_logit_margin": eval_payload["best_wrong_margin_by_idx"].get(idx, ""),
                "total_injection_energy": eval_payload["energy_by_idx"].get(idx, 0.0),
                "total_injection_norm": eval_payload["total_norm_by_idx"].get(idx, 0.0),
                "num_injected_tokens": eval_payload["position_count_by_idx"].get(idx, 0),
                "candidate_logits_json": json_dumps_compact(eval_payload["logits_by_idx"].get(idx, [])),
            }
        )
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    prev.write_csv(path, fieldnames, rows)


def write_dynamic_csv(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str]) -> None:
    fields = list(leading)
    for key in sorted({key for row in rows for key in row.keys()}):
        if key not in fields:
            fields.append(key)
    write_csv(path, fields, rows)


def heatmap_values(
    rows: Sequence[Dict[str, Any]],
    token_groups: Sequence[str],
    layer_windows: Sequence[str],
    key: str,
    mode: str,
) -> np.ndarray:
    matrix = np.full((len(token_groups), len(layer_windows)), np.nan, dtype=float)
    for i, group in enumerate(token_groups):
        for j, window in enumerate(layer_windows):
            values = [
                float(row.get(key, math.nan))
                for row in rows
                if row.get("token_group") == group and row.get("layer_window") == window
            ]
            values = [value for value in values if math.isfinite(value)]
            if values:
                matrix[i, j] = min(values) if mode == "min" else max(values)
    return matrix


def plot_heatmap(path: Path, rows: Sequence[Dict[str, Any]], token_groups: Sequence[str], layer_windows: Sequence[str], key: str, title: str, mode: str, cmap: str) -> None:
    if not rows or not token_groups or not layer_windows:
        return
    matrix = heatmap_values(rows, token_groups, layer_windows, key, mode)
    fig, ax = plt.subplots(figsize=(max(7, len(layer_windows) * 1.1), max(4, len(token_groups) * 0.7)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(layer_windows)))
    ax.set_xticklabels(layer_windows, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(token_groups)))
    ax.set_yticklabels(token_groups)
    ax.set_xlabel("Layer window")
    ax.set_ylabel("Token group")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if math.isfinite(float(matrix[i, j])):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def short_label(row: Dict[str, Any]) -> str:
    if row.get("method") == BASELINE:
        return "baseline"
    if row.get("param_type") == OLD_ORACLE:
        return f"old oracle a={row.get('alpha')}"
    return f"{row.get('param_type')} {row.get('token_group')} L{row.get('layer_window')} a={row.get('alpha')}"


def make_plots(
    *,
    output_dir: Path,
    metrics_rows: Sequence[Dict[str, Any]],
    per_count_metric_rows: Sequence[Dict[str, Any]],
    dist_rows: Sequence[Dict[str, Any]],
    epoch_rows: Sequence[Dict[str, Any]],
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    injection_rows = [row for row in metrics_rows if row.get("method") != BASELINE and row.get("param_type") != OLD_ORACLE]
    token_groups = list(dict.fromkeys(str(row["token_group"]) for row in injection_rows if row.get("token_group")))
    layer_windows = list(dict.fromkeys(str(row["layer_window"]) for row in injection_rows if row.get("layer_window")))
    plot_heatmap(
        plots_dir / "heatmap_best_accuracy_token_group_by_layer_window.png",
        injection_rows,
        token_groups,
        layer_windows,
        "accuracy",
        "Best Accuracy by Token Group and Layer Window",
        "max",
        "YlGnBu",
    )
    plot_heatmap(
        plots_dir / "heatmap_middle_count_accuracy_token_group_by_layer_window.png",
        injection_rows,
        token_groups,
        layer_windows,
        "middle_count_accuracy_3_6",
        "Best Middle-Count Accuracy by Token Group and Layer Window",
        "max",
        "YlGnBu",
    )
    plot_heatmap(
        plots_dir / "heatmap_mae_token_group_by_layer_window.png",
        injection_rows,
        token_groups,
        layer_windows,
        "mae",
        "Best MAE by Token Group and Layer Window",
        "min",
        "YlOrRd",
    )

    best_by_param: Dict[str, Dict[str, Any]] = {}
    for row in injection_rows:
        param = str(row.get("param_type", ""))
        if param and (param not in best_by_param or float(row.get("accuracy", -math.inf)) > float(best_by_param[param].get("accuracy", -math.inf))):
            best_by_param[param] = row
    if best_by_param:
        params = list(best_by_param)
        vals = [float(best_by_param[param]["accuracy"]) for param in params]
        plt.figure(figsize=(7.5, 4.5))
        plt.bar(np.arange(len(params)), vals)
        plt.xticks(np.arange(len(params)), params, rotation=20, ha="right")
        plt.ylabel("Best accuracy")
        plt.title("Best Accuracy by Injection Parameterization")
        plt.ylim(0, max(1.0, max(vals) * 1.1))
        plt.tight_layout()
        plt.savefig(plots_dir / "bar_param_type_best_accuracy.png", dpi=180, bbox_inches="tight")
        plt.close()

    top_rows = sorted(
        [row for row in metrics_rows if math.isfinite(float(row.get("accuracy", math.nan)))],
        key=lambda row: float(row["accuracy"]),
        reverse=True,
    )[:5]
    top_methods = [str(row["method"]) for row in top_rows]
    counts = sorted({int(row["gold_count"]) for row in per_count_metric_rows if str(row.get("gold_count", "")).lstrip("-").isdigit()})
    if top_methods and counts:
        plt.figure(figsize=(8.5, 5.0))
        for row in top_rows:
            method = str(row["method"])
            by_count = {int(r["gold_count"]): r for r in per_count_metric_rows if r.get("method") == method}
            ys = [float(by_count.get(count, {}).get("accuracy", math.nan)) for count in counts]
            plt.plot(counts, ys, marker="o", linewidth=1.8, label=short_label(row))
        plt.xlabel("Gold evidence count")
        plt.ylabel("Accuracy")
        plt.title("Per-Count Accuracy Curves for Top Variants")
        plt.xticks(counts)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plots_dir / "per_evidence_count_accuracy_top5.png", dpi=180, bbox_inches="tight")
        plt.close()

    baseline_rows = [row for row in metrics_rows if row.get("method") == BASELINE]
    display_rows = (baseline_rows[:1] + [row for row in top_rows if row.get("method") != BASELINE][:4])[:5]
    if display_rows and counts:
        width = 0.8 / max(1, len(display_rows))
        xs = np.arange(len(counts))
        plt.figure(figsize=(9.0, 5.0))
        for pos, row in enumerate(display_rows):
            method = str(row["method"])
            by_count = {int(r["predicted_count"]): r for r in dist_rows if r.get("method") == method}
            ys = [float(by_count.get(count, {}).get("fraction", 0.0)) for count in counts]
            plt.bar(xs + (pos - (len(display_rows) - 1) / 2.0) * width, ys, width=width, label=short_label(row))
        plt.xlabel("Predicted count")
        plt.ylabel("Fraction")
        plt.title("Predicted Count Distribution")
        plt.xticks(xs, counts)
        plt.grid(axis="y", alpha=0.25)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plots_dir / "predicted_count_distribution_baseline_vs_best.png", dpi=180, bbox_inches="tight")
        plt.close()

    top_train_ids = list(dict.fromkeys(str(row.get("train_config_id", "")) for row in top_rows if row.get("train_config_id")))
    for y_key, ylabel, filename in (
        ("val_accuracy", "Validation accuracy", "line_val_accuracy_by_epoch_top_variants.png"),
        ("val_ce", "Validation CE", "line_val_ce_by_epoch_top_variants.png"),
    ):
        if top_train_ids and epoch_rows:
            plt.figure(figsize=(8.5, 5.0))
            for train_id in top_train_ids[:5]:
                rows = [row for row in epoch_rows if row.get("train_config_id") == train_id]
                rows = sorted(rows, key=lambda row: int(row["epoch"]))
                if not rows:
                    continue
                plt.plot([int(r["epoch"]) for r in rows], [float(r[y_key]) for r in rows], marker="o", label=train_id)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(ylabel + " by Epoch for Top Variants")
            plt.grid(alpha=0.25)
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=180, bbox_inches="tight")
            plt.close()

    scatter_rows = [row for row in metrics_rows if row.get("method") != BASELINE and math.isfinite(float(row.get("mean_total_injection_norm", math.nan)))]
    if scatter_rows:
        plt.figure(figsize=(7.0, 4.8))
        xs = [float(row["mean_total_injection_norm"]) for row in scatter_rows]
        ys = [float(row["accuracy"]) for row in scatter_rows]
        plt.scatter(xs, ys, alpha=0.8)
        plt.xlabel("Mean total injection norm")
        plt.ylabel("Accuracy")
        plt.title("Injection Norm vs Accuracy")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / "scatter_total_injection_norm_vs_accuracy.png", dpi=180, bbox_inches="tight")
        plt.close()


def best_row(rows: Sequence[Dict[str, Any]], predicate: Any, key: str = "accuracy", mode: str = "max") -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if predicate(row) and math.isfinite(float(row.get(key, math.nan)))]
    if not candidates:
        return None
    return (min if mode == "min" else max)(candidates, key=lambda row: float(row.get(key, math.nan)))


def write_summary(output_dir: Path, metrics_rows: Sequence[Dict[str, Any]]) -> None:
    injection_rows = [row for row in metrics_rows if row.get("method") != BASELINE]
    learned_rows = [row for row in injection_rows if row.get("param_type") != OLD_ORACLE]
    baseline = best_row(metrics_rows, lambda row: row.get("method") == BASELINE)
    best = best_row(injection_rows, lambda _row: True)
    best_learned = best_row(learned_rows, lambda _row: True)
    best_static = best_row(learned_rows, lambda row: row.get("param_type") == STATIC_CODEBOOK)
    best_layer = best_row(learned_rows, lambda row: row.get("param_type") == LAYER_CODEBOOK)
    best_single = best_row(learned_rows, lambda row: int(row.get("num_layers", 0)) == 1)
    best_multi = best_row(learned_rows, lambda row: int(row.get("num_layers", 0)) > 1)
    best_room = best_row(learned_rows, lambda row: row.get("token_group") == "room_char")
    best_question = best_row(learned_rows, lambda row: row.get("token_group") in {"all_question_tokens", "semantic_question_tokens"})
    best_last = best_row(learned_rows, lambda row: row.get("token_group") == "last_token")
    best_combo = best_row(learned_rows, lambda row: row.get("token_group") in {"question_plus_last", "room_char_plus_question", "room_char_question_last"})
    broad = best_row(learned_rows, lambda row: row.get("token_group") == "room_char_question_last" and row.get("layer_window") == "14-24")
    raw_best = best_row(learned_rows, lambda row: not bool(int(row.get("normalize_injection_energy", 0))))
    norm_best = best_row(learned_rows, lambda row: bool(int(row.get("normalize_injection_energy", 0))))

    def fmt(row: Optional[Dict[str, Any]], key: str = "accuracy") -> str:
        if row is None:
            return "not run"
        value = float(row.get(key, math.nan))
        return f"{value:.4f}" if math.isfinite(value) else "nan"

    def label(row: Optional[Dict[str, Any]]) -> str:
        return str(row.get("method", "not run")) if row is not None else "not run"

    best_acc = float(best.get("accuracy", math.nan)) if best is not None else math.nan
    upper_bound = (
        "Yes: this is high enough to make the memory/counter the next bottleneck."
        if math.isfinite(best_acc) and best_acc >= 0.80
        else (
            "Maybe: it clears 70%, but still leaves injection headroom."
            if math.isfinite(best_acc) and best_acc >= 0.70
            else "No: count-conditioned residuals alone did not reach the hoped-for upper bound."
        )
    )
    lines = [
        "Oracle count multilayer injection seq_len=8 7B",
        "",
        f"Frozen Qwen baseline accuracy: {fmt(baseline)} ({label(baseline)})",
        f"Best injection-only accuracy: {fmt(best)} ({label(best)})",
        f"Best learned injection accuracy: {fmt(best_learned)} ({label(best_learned)})",
        "",
        "Requested questions:",
        f"1. Best injection-only accuracy: {fmt(best)} from {label(best)}.",
        f"2. Layer-specific beats static: {'Yes' if best_layer is not None and best_static is not None and float(best_layer['accuracy']) > float(best_static['accuracy']) else 'No or not established'} "
        f"(layer={fmt(best_layer)}, static={fmt(best_static)}).",
        f"3. Multi-layer beats single-layer: {'Yes' if best_multi is not None and best_single is not None and float(best_multi['accuracy']) > float(best_single['accuracy']) else 'No single-layer comparison was run' if best_single is None else 'No'} "
        f"(multi={fmt(best_multi)}, single={fmt(best_single)}).",
        f"4. Best token group family: room_char={fmt(best_room)}, question={fmt(best_question)}, last_token={fmt(best_last)}, combinations={fmt(best_combo)}.",
        f"5. Late last-token beats room+char: {'Yes' if best_last is not None and best_room is not None and float(best_last['accuracy']) > float(best_room['accuracy']) else 'No or not established'} "
        f"(last={fmt(best_last)}, room_char={fmt(best_room)}).",
        f"6. Broad room_char_question_last over 14-24 helps: {fmt(broad)} for {label(broad)}.",
        f"7. Normalization changes ranking: raw_best={fmt(raw_best)}, normalized_best={fmt(norm_best)}; "
        + ("both modes were run." if raw_best is not None and norm_best is not None else "only one normalization mode was run."),
        f"8. Is the injection upper bound high enough to justify improving memory/counter next? {upper_bound}",
        "",
        "Top 10 variants:",
    ]
    for rank, row in enumerate(sorted(metrics_rows, key=lambda item: float(item.get("accuracy", -math.inf)), reverse=True)[:10], start=1):
        lines.append(
            f"{rank}. {row['method']} acc={float(row.get('accuracy', math.nan)):.4f} "
            f"mid={float(row.get('middle_count_accuracy_3_6', math.nan)):.4f} "
            f"mae={float(row.get('mae', math.nan)):.4f} "
            f"norm={float(row.get('mean_total_injection_norm', 0.0)):.4f}"
        )
    (output_dir / "results_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.evidence_counts = prev.parse_int_tokens(args.evidence_counts)
    apply_smoke_overrides(args)
    alphas = parse_float_list(args.alphas)
    count_values = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
    configs = build_config_grid(args)
    output_dir = make_output_dir(args)
    log_handle, old_stdout, old_stderr = prev.setup_logging(output_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        config_payload = {
            "experiment_family": EXPERIMENT_FAMILY,
            "run_name": str(args.run_name),
            "output_root": os.fspath(Path(args.output_root).resolve()),
            "output_dir": os.fspath(output_dir),
            "model_name": str(args.model_name),
            "dataset_root": os.fspath(args.dataset_root),
            "source_run": os.fspath(args.source_run),
            "previous_shared_run": os.fspath(args.previous_shared_run),
            "split": str(args.split),
            "seq_len": int(args.seq_len),
            "evidence_counts": list(args.evidence_counts),
            "sample_limit": int(args.sample_limit),
            "max_samples_per_count": int(args.max_samples_per_count),
            "candidate_min": int(args.candidate_min),
            "candidate_max": int(args.candidate_max),
            "epochs": int(args.epochs),
            "train_batch_size": int(args.train_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "grad_accum": int(args.grad_accum),
            "lr": float(args.lr),
            "alphas": list(alphas),
            "train_alpha": float(args.train_alpha),
            "token_groups": split_tokens(args.token_groups),
            "layer_windows": split_tokens(args.layer_windows),
            "param_types": split_tokens(args.param_types),
            "margin_lambda": parse_float_list(args.margin_lambda),
            "margin": float(args.margin),
            "residual_l2": float(args.residual_l2),
            "normalize_injection_energy": bool(args.normalize_injection_energy),
            "full_grid": bool(args.full_grid),
            "configs": configs,
            "device": str(args.device),
            "dtype": str(args.dtype),
            "attn_implementation": str(args.attn_implementation),
            "load_in_4bit": bool(args.load_in_4bit),
            "smoke": bool(args.smoke),
            "seed": int(args.seed),
        }
        prev.write_json(output_dir / "configs.json", config_payload)
        prev.write_json(output_dir / "run_config.json", config_payload)
        print(f"Output dir: {output_dir}")
        print(f"Config: {json.dumps(config_payload, sort_keys=True)}")

        sample_payload = load_sample_index_payload(args)
        sample_ids = sample_payload["sample_ids"]
        labels = sample_payload["labels"].long()
        records = prev.load_records(args.dataset_root, args.split, args.seq_len, sample_ids)
        splits = prev.stratified_split(sample_ids, labels, int(args.seed))
        train_indices = limit_split_indices(splits["train"], records, int(args.sample_limit), int(args.seed))
        val_indices = limit_split_indices(splits["val"] or splits["train"], records, int(args.sample_limit), int(args.seed) + 1)
        test_indices = limit_split_indices(splits["test"] or splits["val"] or splits["train"], records, int(args.sample_limit), int(args.seed) + 2)
        counts_by_split = prev.split_counts(
            {"train": train_indices, "val": val_indices, "test": test_indices},
            labels,
            count_values,
        )
        for split, row in counts_by_split.items():
            print(f"  {split}: " + ", ".join(f"{count}:{row.get(count, 0)}" for count in count_values))
        if not train_indices or not val_indices or not test_indices:
            raise RuntimeError("Train/val/test split is empty after sample limiting")

        device = prev.resolve_device(str(args.device))
        dtype = prev.resolve_dtype(str(args.dtype), device)
        model, processor = prev.load_model_and_processor(args, device=device, dtype=dtype)
        hidden_size = prev.hidden_size_from_model(model)
        candidate_format, count_token_ids = prev.candidate_token_ids(
            processor.tokenizer,
            int(args.candidate_min),
            int(args.candidate_max),
        )
        chosen_count_ids, token_rows = codebook_prev.tokenization_report(processor.tokenizer, count_values, candidate_format)
        write_csv(output_dir / "token_ids.csv", ["count", "plain_ids", "leading_space_ids", "chosen_format", "chosen_token_id"], token_rows)
        print(f"Loaded Qwen hidden_size={hidden_size} candidate_format={candidate_format} count_ids={count_token_ids}")

        metrics_rows: List[Dict[str, Any]] = []
        per_sample_rows: List[Dict[str, Any]] = []
        per_count_metric_rows: List[Dict[str, Any]] = []
        dist_rows: List[Dict[str, Any]] = []
        epoch_rows: List[Dict[str, Any]] = []
        residual_rows: List[Dict[str, Any]] = []

        print("Evaluating frozen Qwen baseline")
        base_eval = evaluate_adapter(
            method=BASELINE,
            model=model,
            processor=processor,
            adapter=None,
            records=records,
            indices=test_indices,
            count_token_ids=count_token_ids,
            token_group="room_char",
            alpha=0.0,
            device=device,
            batch_size=int(args.eval_batch_size),
            max_eval_samples=int(args.max_eval_samples),
        )
        base_meta = {
            "method": BASELINE,
            "train_config_id": "",
            "param_type": "baseline",
            "token_group": "none",
            "layer_window": "",
            "inject_layers": "",
            "num_layers": 0,
            "alpha": 0.0,
            "train_alpha": "",
            "margin_lambda": "",
            "normalize_injection_energy": int(bool(args.normalize_injection_energy)),
            "checkpoint": "",
        }
        metrics_rows.append(
            summarize_eval_row(
                metadata=base_meta,
                records=records,
                indices=test_indices,
                eval_payload=base_eval,
                train_history=[],
                count_values=count_values,
            )
        )
        per_sample_rows.extend(build_per_sample_rows(metadata=base_meta, records=records, eval_payload=base_eval))
        per_count_metric_rows.extend(
            per_count_rows(metadata=base_meta, records=records, indices=base_eval["indices"], pred_by_idx=base_eval["pred_by_idx"], count_values=count_values)
        )
        dist_rows.extend(predicted_distribution_rows(metadata=base_meta, pred_by_idx=base_eval["pred_by_idx"], count_values=count_values))

        if bool(args.include_old_oracle_baseline):
            previous_scale = codebook_prev.previous_residual_norm_scale(Path(args.previous_shared_run))
            codebook_norm = (
                float(args.qwen_codebook_init_norm)
                if args.qwen_codebook_init_norm is not None
                else (previous_scale if math.isfinite(previous_scale) and previous_scale > 0 else 50.0)
            )
            old_codebook = make_qwen_codebook(model, chosen_count_ids, int(hidden_size), codebook_norm)
            old_adapter = CountResidualInjectionAdapter(
                count_values=count_values,
                hidden_size=int(hidden_size),
                inject_layers=parse_layer_window("18-21"),
                param_type=STATIC_CODEBOOK,
                alpha=1.0,
                normalize_injection_energy=bool(args.normalize_injection_energy),
                reft_rank=int(args.reft_rank),
            )
            with torch.no_grad():
                old_adapter.codebook.copy_(old_codebook)
            for param in old_adapter.parameters():
                param.requires_grad_(False)
            for alpha in alphas:
                method = f"{OLD_ORACLE}__all_question_tokens__L18-21__a{float_tag(alpha)}"
                print(f"Evaluating {method}")
                eval_payload = evaluate_adapter(
                    method=method,
                    model=model,
                    processor=processor,
                    adapter=old_adapter,
                    records=records,
                    indices=test_indices,
                    count_token_ids=count_token_ids,
                    token_group="all_question_tokens",
                    alpha=float(alpha),
                    device=device,
                    batch_size=int(args.eval_batch_size),
                    max_eval_samples=int(args.max_eval_samples),
                )
                meta = {
                    "method": method,
                    "train_config_id": "",
                    "param_type": OLD_ORACLE,
                    "token_group": "all_question_tokens",
                    "layer_window": "18-21",
                    "inject_layers": "18 19 20 21",
                    "num_layers": 4,
                    "alpha": float(alpha),
                    "train_alpha": "",
                    "margin_lambda": "",
                    "normalize_injection_energy": int(bool(args.normalize_injection_energy)),
                    "checkpoint": "",
                }
                metrics_rows.append(
                    summarize_eval_row(
                        metadata=meta,
                        records=records,
                        indices=test_indices,
                        eval_payload=eval_payload,
                        train_history=[],
                        count_values=count_values,
                    )
                )
                per_sample_rows.extend(build_per_sample_rows(metadata=meta, records=records, eval_payload=eval_payload))
                per_count_metric_rows.extend(
                    per_count_rows(metadata=meta, records=records, indices=eval_payload["indices"], pred_by_idx=eval_payload["pred_by_idx"], count_values=count_values)
                )
                dist_rows.extend(predicted_distribution_rows(metadata=meta, pred_by_idx=eval_payload["pred_by_idx"], count_values=count_values))
                for layer, value in eval_payload["layer_mean_norm"].items():
                    residual_rows.append({**meta, "layer": int(layer), "mean_residual_norm": float(value), "mean_injection_energy": eval_payload["layer_mean_energy"].get(layer, "")})
            old_adapter.cpu()

        for config_num, config in enumerate(configs, start=1):
            print(f"Training config {config_num}/{len(configs)}: {config['config_id']}")
            adapter, history, checkpoint_path = train_one_config(
                config=config,
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
                device=device,
            )
            epoch_rows.extend(history)
            for alpha in alphas:
                method = f"{config['config_id']}__a{float_tag(alpha)}"
                print(f"Evaluating {method}")
                eval_payload = evaluate_adapter(
                    method=method,
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    records=records,
                    indices=test_indices,
                    count_token_ids=count_token_ids,
                    token_group=str(config["token_group"]),
                    alpha=float(alpha),
                    device=device,
                    batch_size=int(args.eval_batch_size),
                    max_eval_samples=int(args.max_eval_samples),
                )
                meta = {
                    "method": method,
                    "train_config_id": str(config["config_id"]),
                    "param_type": str(config["param_type"]),
                    "token_group": str(config["token_group"]),
                    "layer_window": str(config["layer_window"]),
                    "inject_layers": " ".join(str(x) for x in config["inject_layers"]),
                    "num_layers": len(config["inject_layers"]),
                    "alpha": float(alpha),
                    "train_alpha": float(args.train_alpha),
                    "margin_lambda": float(config["margin_lambda"]),
                    "normalize_injection_energy": int(bool(config["normalize_injection_energy"])),
                    "checkpoint": os.fspath(checkpoint_path),
                }
                metrics_rows.append(
                    summarize_eval_row(
                        metadata=meta,
                        records=records,
                        indices=test_indices,
                        eval_payload=eval_payload,
                        train_history=history,
                        count_values=count_values,
                    )
                )
                per_sample_rows.extend(build_per_sample_rows(metadata=meta, records=records, eval_payload=eval_payload))
                per_count_metric_rows.extend(
                    per_count_rows(metadata=meta, records=records, indices=eval_payload["indices"], pred_by_idx=eval_payload["pred_by_idx"], count_values=count_values)
                )
                dist_rows.extend(predicted_distribution_rows(metadata=meta, pred_by_idx=eval_payload["pred_by_idx"], count_values=count_values))
                for layer, value in eval_payload["layer_mean_norm"].items():
                    residual_rows.append({**meta, "layer": int(layer), "mean_residual_norm": float(value), "mean_injection_energy": eval_payload["layer_mean_energy"].get(layer, "")})
            adapter.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_dynamic_csv(
            output_dir / "metrics.csv",
            metrics_rows,
            [
                "method",
                "param_type",
                "token_group",
                "layer_window",
                "alpha",
                "margin_lambda",
                "normalize_injection_energy",
                "accuracy",
                "mae",
                "middle_count_accuracy_3_6",
                "ce",
            ],
        )
        write_dynamic_csv(
            output_dir / "per_sample_metrics.csv",
            per_sample_rows,
            ["method", "split", "sample_index", "sample_id", "gold_count", "predicted_count", "correct"],
        )
        write_dynamic_csv(
            output_dir / "accuracy_by_evidence_count.csv",
            per_count_metric_rows,
            ["method", "evidence_count", "gold_count", "n", "accuracy", "mae", "mean_predicted_count"],
        )
        write_dynamic_csv(
            output_dir / "predicted_count_distribution.csv",
            dist_rows,
            ["method", "predicted_count", "n", "fraction"],
        )
        write_dynamic_csv(
            output_dir / "epoch_metrics.csv",
            epoch_rows,
            ["train_config_id", "epoch", "train_ce", "train_loss", "val_ce", "val_accuracy", "val_mae"],
        )
        write_dynamic_csv(
            output_dir / "residual_norms_by_layer.csv",
            residual_rows,
            ["method", "param_type", "token_group", "layer_window", "alpha", "layer", "mean_residual_norm", "mean_injection_energy"],
        )

        if not bool(args.no_plots):
            make_plots(
                output_dir=output_dir,
                metrics_rows=metrics_rows,
                per_count_metric_rows=per_count_metric_rows,
                dist_rows=dist_rows,
                epoch_rows=epoch_rows,
            )
        write_summary(output_dir, metrics_rows)
        debug = {
            "runtime_seconds": time.time() - started,
            "source_cache": os.fspath(sample_payload["cache_path"]),
            "num_records": len(records),
            "splits": {key: len(value) for key, value in splits.items()},
            "limited_splits": {"train": len(train_indices), "val": len(val_indices), "test": len(test_indices)},
            "best_metric": max(metrics_rows, key=lambda row: float(row.get("accuracy", -math.inf))) if metrics_rows else None,
        }
        prev.write_json(output_dir / "debug.json", debug)
        print(f"Finished oracle count multilayer injection run in {time.time() - started:.1f}s")
        print(f"Results: {output_dir}")
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

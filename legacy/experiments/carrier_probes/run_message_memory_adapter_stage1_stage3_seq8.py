#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
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
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from evaluations.helpers import patching_core as core
from evaluations.helpers import utils as eval_utils
from models.model import find_subsequence, get_layers


DEFAULT_SOURCE_RUN = PROJECT_ROOT / "outputs" / "frame_to_carrier_evidence_sum_probe_seq8_7b_20260521_164621"
DEFAULT_BASE_SOURCE_RUN = (
    PROJECT_ROOT / "outputs" / "frame_to_carrier_message_memory_probe_seq8_7b_multilayer_20260521_154136"
)
DEFAULT_LAYERS = (14, 15, 16, 17)
DEFAULT_EVIDENCE_COUNTS = tuple(range(9))
INTEGER_RE = re.compile(r"[+-]?\d+")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    sample_dir: Path
    frame_paths: Tuple[Path, ...]
    question: str
    states: Tuple[Dict[str, Any], ...]
    gold_count: int
    evidence_count: int


@dataclass
class QwenBatch:
    inputs: Dict[str, Any]
    target_positions: List[List[int]]
    prompt_last_indices: torch.Tensor
    gold_counts: torch.Tensor
    sample_indices: List[int]


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1/Stage 3 gLSTM-style message-memory adapter for MMReD seq_len=8 "
            "using cached frame-to-target-character/room messages."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/mmred_images_park")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--base-source-run", type=Path, default=DEFAULT_BASE_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="all_uniform")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in DEFAULT_EVIDENCE_COUNTS])
    parser.add_argument("--layers", nargs="+", default=[str(x) for x in DEFAULT_LAYERS])
    parser.add_argument("--carriers", nargs="+", default=["target_char", "target_room"])
    parser.add_argument("--max-samples-per-count", type=int, default=100)
    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--run-stage1", action="store_true", default=False)
    parser.add_argument("--run-stage3", action="store_true", default=False)
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--stage3-checkpoint", type=Path, default=None)
    parser.add_argument("--no-per-frame-linear", action="store_true", default=False)
    parser.add_argument("--no-plots", action="store_true", default=False)

    parser.add_argument("--bottleneck-dim", "--d-m", type=int, default=256)
    parser.add_argument("--key-dim", "--d-k", type=int, default=64)
    parser.add_argument("--value-dim", "--d-v", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--frame-gate-bce-weight", type=float, default=0.2)

    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage1-weight-decay", type=float, default=1e-2)
    parser.add_argument("--stage1-patience", type=int, default=5)
    parser.add_argument("--stage1-batch-size", type=int, default=64)

    parser.add_argument("--linear-epochs", type=int, default=30)
    parser.add_argument("--linear-lr", type=float, default=1e-3)
    parser.add_argument("--linear-weight-decay", type=float, default=1e-2)
    parser.add_argument("--linear-patience", type=int, default=5)

    parser.add_argument("--stage3-epochs", type=int, default=5)
    parser.add_argument("--stage3-lr", type=float, default=1e-4)
    parser.add_argument("--stage3-weight-decay", type=float, default=1e-2)
    parser.add_argument("--stage3-batch-size", type=int, default=1)
    parser.add_argument("--stage3-grad-accum", type=int, default=8)
    parser.add_argument("--stage3-patience", type=int, default=0)
    parser.add_argument("--stage3-grad-clip", type=float, default=1.0)
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
    return parser.parse_args()


def split_tokens(raw_values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in raw_values:
        out.extend(part.strip() for part in str(raw).replace(",", " ").split() if part.strip())
    return out


def parse_int_tokens(raw_values: Sequence[str]) -> List[int]:
    values = []
    for part in split_tokens(raw_values):
        values.append(int(part))
    return sorted(dict.fromkeys(values))


def default_output_dir() -> Path:
    return PROJECT_ROOT / "outputs" / f"message_memory_adapter_stage1_stage3_seq8_7b_{time.strftime('%Y%m%d_%H%M%S')}"


def setup_logging(output_dir: Path) -> Tuple[Any, Any, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (output_dir / "run.log").open("a", encoding="utf-8", buffering=1)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)
    return log_handle, old_stdout, old_stderr


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_torch(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16], 16)


def stratified_split(sample_ids: Sequence[str], labels: torch.Tensor, seed: int) -> Dict[str, List[int]]:
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        by_label[int(label)].append(int(idx))
    splits = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        indices = sorted(by_label[label], key=lambda idx: stable_hash_int(f"{seed}:{sample_ids[idx]}"))
        n = len(indices)
        if n == 1:
            splits["train"].extend(indices)
            continue
        if n == 2:
            splits["train"].extend(indices[:1])
            splits["val"].extend(indices[1:])
            continue
        n_val = max(1, int(round(0.15 * n)))
        n_test = max(1, int(round(0.15 * n)))
        if n_val + n_test >= n:
            n_val = 1
            n_test = 1
        n_train = n - n_val - n_test
        splits["train"].extend(indices[:n_train])
        splits["val"].extend(indices[n_train : n_train + n_val])
        splits["test"].extend(indices[n_train + n_val :])
    for split in splits:
        splits[split] = sorted(splits[split], key=lambda idx: sample_ids[idx])
    return splits


def split_counts(splits: Dict[str, List[int]], labels: torch.Tensor, counts: Sequence[int]) -> Dict[str, Dict[int, int]]:
    out: Dict[str, Dict[int, int]] = {}
    for split, indices in splits.items():
        row = {int(count): 0 for count in counts}
        for idx in indices:
            row[int(labels[int(idx)].item())] = row.get(int(labels[int(idx)].item()), 0) + 1
        out[split] = row
    return out


def resolve_existing_path(path: Path, *, want_file: Optional[str] = None) -> Path:
    candidates = [path]
    parts = list(path.parts)
    for replacement in ("outputs_oh_man", "outputs_yeah_baby", "outputs_best", "outputs_kitkat", "outputs_least_oldest"):
        if "outputs" in parts:
            idx = parts.index("outputs")
            new_parts = parts[:]
            new_parts[idx] = replacement
            candidates.append(Path(*new_parts))
    for candidate in candidates:
        target = candidate / want_file if want_file is not None and candidate.is_dir() else candidate
        if target.exists():
            return candidate
    return path


def source_cache_path(source_run: Path) -> Path:
    source_run = resolve_existing_path(source_run)
    if source_run.is_file():
        return source_run
    focused = source_run / "cache" / "evidence_sum_features.pt"
    if focused.is_file():
        return focused
    return source_run / "cache" / "features.pt"


def base_cache_path(base_source_run: Path) -> Path:
    base_source_run = resolve_existing_path(base_source_run)
    if base_source_run.is_file():
        return base_source_run
    return base_source_run / "cache" / "features.pt"


def build_condition_features(payload: Dict[str, Any], condition: str, layers: Sequence[int]) -> torch.Tensor:
    if condition != "target_char_room":
        raise ValueError(f"This experiment expects target_char_room messages, got {condition!r}")
    carrier_actual = payload.get("carrier_actual", {})
    parts: List[torch.Tensor] = []
    for layer in layers:
        for carrier in ("target_char", "target_room"):
            layer_map = carrier_actual.get(carrier, {})
            tensor = layer_map.get(int(layer), layer_map.get(str(int(layer))))
            if not torch.is_tensor(tensor):
                raise RuntimeError(f"Missing cached {carrier} actual messages for layer {layer}")
            parts.append(tensor.float())
    if not parts:
        raise RuntimeError("No message feature parts were selected")
    return torch.cat(parts, dim=-1).contiguous()


def load_message_features(args: argparse.Namespace, layers: Sequence[int], counts: Sequence[int]) -> Dict[str, Any]:
    cache_path = source_cache_path(args.source_run)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Could not find message feature cache at {cache_path}. "
            "Pass --source-run to an evidence-sum run with cache/evidence_sum_features.pt."
        )
    print(f"Loading message feature cache: {cache_path}")
    payload = load_torch(cache_path)
    sample_ids = [str(x) for x in payload["sample_ids"]]
    labels = payload["labels"].long()
    frame_labels = payload["frame_labels"].float()
    keep = [
        idx
        for idx, label in enumerate(labels.tolist())
        if int(label) in set(int(x) for x in counts)
    ]
    if args.max_samples_per_count > 0:
        per_count_seen: Dict[int, int] = defaultdict(int)
        limited: List[int] = []
        for idx in keep:
            count = int(labels[idx].item())
            if per_count_seen[count] < int(args.max_samples_per_count):
                limited.append(idx)
                per_count_seen[count] += 1
        keep = limited
    if len(keep) != len(sample_ids):
        sample_ids = [sample_ids[idx] for idx in keep]
        labels = labels[keep]
        frame_labels = frame_labels[keep]
        sliced_payload = dict(payload)
        sliced_payload["sample_ids"] = sample_ids
        sliced_payload["labels"] = labels
        sliced_payload["frame_labels"] = frame_labels
        carrier_actual: Dict[str, Dict[int, torch.Tensor]] = {}
        for carrier, layer_map in payload.get("carrier_actual", {}).items():
            carrier_actual[str(carrier)] = {}
            for layer, tensor in layer_map.items():
                carrier_actual[str(carrier)][int(layer)] = tensor[keep]
        sliced_payload["carrier_actual"] = carrier_actual
        payload = sliced_payload
    x_messages = build_condition_features(payload, "target_char_room", layers)
    print(f"x_messages shape={tuple(x_messages.shape)}")
    return {
        "payload": payload,
        "cache_path": cache_path,
        "sample_ids": sample_ids,
        "labels": labels,
        "frame_labels": frame_labels,
        "x_messages": x_messages,
    }


def load_base_predictions(args: argparse.Namespace, sample_ids: Sequence[str]) -> Tuple[Dict[str, int], Optional[Path], List[str]]:
    warnings: List[str] = []
    path = base_cache_path(args.base_source_run)
    if not path.is_file():
        warnings.append(f"Base cache not found: {path}")
        return {}, None, warnings
    payload = load_torch(path)
    if "base_pred" not in payload:
        warnings.append(f"Base cache has no base_pred: {path}")
        return {}, path, warnings
    source_ids = [str(x) for x in payload.get("sample_ids", [])]
    source_pred = [int(x) for x in payload["base_pred"].tolist()]
    pred_by_id = dict(zip(source_ids, source_pred))
    missing = [sample_id for sample_id in sample_ids if sample_id not in pred_by_id]
    if missing:
        warnings.append(f"Base cache missing {len(missing)} requested sample ids")
    return pred_by_id, path, warnings


def compute_standardizer(features: torch.Tensor, train_indices: Sequence[int], chunk_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    if not train_indices:
        raise ValueError("Cannot compute standardizer with no train indices")
    total: Optional[torch.Tensor] = None
    total_sq: Optional[torch.Tensor] = None
    n = 0
    for start in range(0, len(train_indices), int(chunk_size)):
        idx = list(train_indices[start : start + int(chunk_size)])
        x = features[idx].float()
        flat = x.reshape(-1, x.shape[-1])
        s = flat.sum(dim=0)
        ss = (flat * flat).sum(dim=0)
        total = s if total is None else total + s
        total_sq = ss if total_sq is None else total_sq + ss
        n += int(flat.shape[0])
    assert total is not None and total_sq is not None
    mean = total / max(1, n)
    var = (total_sq / max(1, n)) - mean * mean
    std = torch.sqrt(var.clamp_min(1e-6))
    return mean.float(), std.float()


class MessageDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, frame_labels: torch.Tensor, indices: Sequence[int]) -> None:
        self.features = features
        self.labels = labels
        self.frame_labels = frame_labels
        self.indices = [int(x) for x in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.indices[int(item)]
        return (
            torch.tensor(idx, dtype=torch.long),
            self.features[idx],
            self.labels[idx].long(),
            self.frame_labels[idx].float(),
        )


class MessageMemoryCore(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int, key_dim: int, value_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.norm = nn.LayerNorm(self.input_dim)
        self.w_p = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.w_k = nn.Linear(self.bottleneck_dim, self.key_dim, bias=False)
        self.w_v = nn.Linear(self.bottleneck_dim, self.value_dim, bias=False)
        self.w_alpha = nn.Linear(self.bottleneck_dim, 1)
        self.q0 = nn.Parameter(torch.randn(self.key_dim) / math.sqrt(float(self.key_dim)))
        self.register_buffer("x_mean", torch.zeros(self.input_dim), persistent=True)
        self.register_buffer("x_std", torch.ones(self.input_dim), persistent=True)

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.x_mean.data.copy_(mean.float().reshape(-1))
        self.x_std.data.copy_(std.float().reshape(-1).clamp_min(1e-6))

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def forward(self, x_messages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.standardize(x_messages)
        z = F.gelu(self.w_p(self.norm(x)))
        z = self.dropout(z)
        k = F.normalize(self.w_k(z), dim=-1)
        v = self.w_v(z)
        alpha = torch.sigmoid(self.w_alpha(z)).squeeze(-1)
        q = F.normalize(self.q0, dim=0)
        scores = torch.matmul(k, q)
        r = torch.sum(alpha.unsqueeze(-1) * v * scores.unsqueeze(-1), dim=1)
        return r, alpha


class Stage1MemoryReadout(nn.Module):
    def __init__(self, core: MessageMemoryCore, num_classes: int) -> None:
        super().__init__()
        self.core = core
        self.count_head = nn.Linear(core.value_dim, int(num_classes))

    def forward(self, x_messages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r, alpha = self.core(x_messages)
        logits = self.count_head(r)
        return logits, alpha, r


class SharedFrameLinearProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(input_dim), 1)

    def forward(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        x_norm = (x.float() - mean.to(x.device)) / std.to(x.device).clamp_min(1e-6)
        return self.linear(x_norm).squeeze(-1)


class Stage3ResidualAdapter(nn.Module):
    def __init__(
        self,
        *,
        core: MessageMemoryCore,
        hidden_size: int,
        inject_layer: int,
        gamma_init: float,
        train_gamma: bool,
    ) -> None:
        super().__init__()
        self.core = core
        self.w_o = nn.Linear(core.value_dim, int(hidden_size), bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32), requires_grad=bool(train_gamma))
        self.inject_layer = int(inject_layer)
        self.enabled = True
        self._x_messages: Optional[torch.Tensor] = None
        self._target_positions: Optional[List[List[int]]] = None
        self._handles: List[Any] = []
        self.last_alpha: Optional[torch.Tensor] = None
        self.last_delta_norm: Optional[torch.Tensor] = None
        nn.init.zeros_(self.w_o.weight)

    def set_context(self, x_messages: torch.Tensor, target_positions: Sequence[Sequence[int]]) -> None:
        self._x_messages = x_messages
        self._target_positions = [[int(pos) for pos in positions] for positions in target_positions]
        self.last_alpha = None
        self.last_delta_norm = None

    def clear_context(self) -> None:
        self._x_messages = None
        self._target_positions = None

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
        delta = self.w_o(r)
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
        update = torch.stack(updates, dim=0)
        return hidden_states + update

    def register_hooks(self, model: Any) -> None:
        self.remove_hooks()
        layers = get_layers(model)
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


def make_stage1_model(args: argparse.Namespace, input_dim: int, num_classes: int, mean: torch.Tensor, std: torch.Tensor) -> Stage1MemoryReadout:
    core = MessageMemoryCore(
        input_dim=int(input_dim),
        bottleneck_dim=int(args.bottleneck_dim),
        key_dim=int(args.key_dim),
        value_dim=int(args.value_dim),
        dropout=float(args.dropout),
    )
    core.set_standardizer(mean, std)
    return Stage1MemoryReadout(core, num_classes=int(num_classes))


def batch_to_device(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(x.to(device) for x in batch)  # type: ignore[return-value]


def train_stage1(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    x_messages: torch.Tensor,
    labels: torch.Tensor,
    frame_labels: torch.Tensor,
    splits: Dict[str, List[int]],
    candidate_min: int,
    candidate_max: int,
    checkpoint_path: Optional[Path],
) -> Tuple[Stage1MemoryReadout, Dict[str, Any], Path]:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "stage1_best.pt"
    num_classes = int(candidate_max) - int(candidate_min) + 1
    input_dim = int(x_messages.shape[-1])
    train_device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if checkpoint_path is not None and checkpoint_path.is_file():
        print(f"Loading Stage 1 checkpoint: {checkpoint_path}")
        ckpt = load_torch(checkpoint_path)
        mean = ckpt["x_mean"].float()
        std = ckpt["x_std"].float()
        model = make_stage1_model(args, input_dim, num_classes, mean, std)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, dict(ckpt.get("history", {})), checkpoint_path
    if best_path.is_file():
        print(f"Reusing existing Stage 1 checkpoint: {best_path}")
        ckpt = load_torch(best_path)
        mean = ckpt["x_mean"].float()
        std = ckpt["x_std"].float()
        model = make_stage1_model(args, input_dim, num_classes, mean, std)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, dict(ckpt.get("history", {})), best_path

    mean, std = compute_standardizer(x_messages, splits["train"])
    model = make_stage1_model(args, input_dim, num_classes, mean, std).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.stage1_lr), weight_decay=float(args.stage1_weight_decay))
    train_loader = DataLoader(
        MessageDataset(x_messages, labels, frame_labels, splits["train"]),
        batch_size=int(args.stage1_batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(args.seed) + 11),
    )
    val_loader = DataLoader(
        MessageDataset(x_messages, labels, frame_labels, splits["val"]),
        batch_size=int(args.stage1_batch_size),
        shuffle=False,
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metric = -math.inf
    best_val_ce = math.inf
    best_epoch = 0
    bad_epochs = 0
    history_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.stage1_epochs) + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            _idx, x, y, y_frame = batch_to_device(batch, train_device)
            y_offset = (y - int(candidate_min)).long()
            optimizer.zero_grad(set_to_none=True)
            logits, alpha, _r = model(x)
            count_loss = F.cross_entropy(logits, y_offset)
            gate_loss = F.binary_cross_entropy(alpha, y_frame.float())
            loss = count_loss + float(args.frame_gate_bce_weight) * gate_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            total_batches += 1
        val = evaluate_stage1_model(
            model=model,
            x_messages=x_messages,
            labels=labels,
            frame_labels=frame_labels,
            indices=splits["val"],
            candidate_min=candidate_min,
            batch_size=int(args.stage1_batch_size),
            device=train_device,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": total_loss / max(1, total_batches),
            "val_ce": float(val["ce"]),
            "val_accuracy": float(val["accuracy"]),
            "val_gate_bce": float(val["gate_bce"]),
        }
        history_rows.append(row)
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
            f"  stage1 epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_ce={row['val_ce']:.4f} val_acc={row['val_accuracy']:.4f}"
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
        "config": {
            "bottleneck_dim": int(args.bottleneck_dim),
            "key_dim": int(args.key_dim),
            "value_dim": int(args.value_dim),
            "dropout": float(args.dropout),
        },
        "history": {
            "rows": history_rows,
            "best_epoch": int(best_epoch),
            "best_val_accuracy": float(best_metric),
            "best_val_ce": float(best_val_ce),
        },
    }
    torch.save(ckpt, best_path)
    print(f"Saved Stage 1 checkpoint: {best_path}")
    return model_cpu, ckpt["history"], best_path


@torch.no_grad()
def evaluate_stage1_model(
    *,
    model: Stage1MemoryReadout,
    x_messages: torch.Tensor,
    labels: torch.Tensor,
    frame_labels: torch.Tensor,
    indices: Sequence[int],
    candidate_min: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    model = model.to(device)
    model.eval()
    loader = DataLoader(MessageDataset(x_messages, labels, frame_labels, indices), batch_size=int(batch_size), shuffle=False)
    pred_by_idx: Dict[int, int] = {}
    logits_by_idx: Dict[int, List[float]] = {}
    alpha_by_idx: Dict[int, List[float]] = {}
    ce_total = 0.0
    bce_total = 0.0
    n = 0
    for batch in loader:
        idx, x, y, y_frame = batch_to_device(batch, device)
        y_offset = (y - int(candidate_min)).long()
        logits, alpha, _r = model(x)
        ce = F.cross_entropy(logits, y_offset, reduction="sum")
        bce = F.binary_cross_entropy(alpha, y_frame.float(), reduction="sum")
        ce_total += float(ce.detach().cpu().item())
        bce_total += float(bce.detach().cpu().item())
        n += int(y.numel())
        pred = logits.argmax(dim=-1) + int(candidate_min)
        for row, sample_idx in enumerate(idx.detach().cpu().tolist()):
            pred_by_idx[int(sample_idx)] = int(pred[row].detach().cpu().item())
            logits_by_idx[int(sample_idx)] = [float(v) for v in logits[row].detach().float().cpu().tolist()]
            alpha_by_idx[int(sample_idx)] = [float(v) for v in alpha[row].detach().float().cpu().tolist()]
    y_true = [int(labels[int(idx)].item()) for idx in indices]
    y_pred = [pred_by_idx[int(idx)] for idx in indices]
    return {
        "ce": ce_total / max(1, n),
        "gate_bce": bce_total / max(1, n * int(frame_labels.shape[1])),
        "accuracy": accuracy(y_true, y_pred),
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "alpha_by_idx": alpha_by_idx,
    }


def train_linear_probe(
    *,
    args: argparse.Namespace,
    x_messages: torch.Tensor,
    labels: torch.Tensor,
    frame_labels: torch.Tensor,
    splits: Dict[str, List[int]],
    candidate_min: int,
    candidate_max: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[SharedFrameLinearProbe, Dict[str, Any]]:
    del labels, candidate_min, candidate_max
    device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model = SharedFrameLinearProbe(int(x_messages.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.linear_lr), weight_decay=float(args.linear_weight_decay))
    train_loader = DataLoader(
        MessageDataset(x_messages, torch.zeros(len(x_messages), dtype=torch.long), frame_labels, splits["train"]),
        batch_size=int(args.stage1_batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(args.seed) + 23),
    )
    val_loader = DataLoader(
        MessageDataset(x_messages, torch.zeros(len(x_messages), dtype=torch.long), frame_labels, splits["val"]),
        batch_size=int(args.stage1_batch_size),
        shuffle=False,
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = math.inf
    bad_epochs = 0
    rows: List[Dict[str, Any]] = []
    mean_d = mean.to(device)
    std_d = std.to(device)
    for epoch in range(1, int(args.linear_epochs) + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for _idx, x, _y, y_frame in train_loader:
            x = x.to(device)
            y_frame = y_frame.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mean_d, std_d)
            loss = F.binary_cross_entropy_with_logits(logits, y_frame.float())
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu().item())
            batches += 1
        val_loss = 0.0
        val_batches = 0
        model.eval()
        with torch.no_grad():
            for _idx, x, _y, y_frame in val_loader:
                x = x.to(device)
                y_frame = y_frame.to(device)
                val_loss += float(F.binary_cross_entropy_with_logits(model(x, mean_d, std_d), y_frame.float()).cpu().item())
                val_batches += 1
        val_loss = val_loss / max(1, val_batches)
        rows.append({"epoch": int(epoch), "train_bce": train_loss / max(1, batches), "val_bce": float(val_loss)})
        if val_loss < best_val - 1e-9:
            best_val = float(val_loss)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"  linear probe epoch={epoch} train_bce={rows[-1]['train_bce']:.4f} val_bce={val_loss:.4f}")
        if bad_epochs >= int(args.linear_patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), {"rows": rows, "best_val_bce": float(best_val)}


@torch.no_grad()
def evaluate_linear_probe(
    *,
    model: SharedFrameLinearProbe,
    x_messages: torch.Tensor,
    frame_labels: torch.Tensor,
    indices: Sequence[int],
    mean: torch.Tensor,
    std: torch.Tensor,
    candidate_min: int,
    candidate_max: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    model = model.to(device)
    model.eval()
    loader = DataLoader(
        MessageDataset(x_messages, torch.zeros(len(x_messages), dtype=torch.long), frame_labels, indices),
        batch_size=int(batch_size),
        shuffle=False,
    )
    pred_by_idx: Dict[int, int] = {}
    prob_by_idx: Dict[int, List[float]] = {}
    mean_d = mean.to(device)
    std_d = std.to(device)
    for idx, x, _y, _y_frame in loader:
        logits = model(x.to(device), mean_d, std_d)
        probs = torch.sigmoid(logits)
        pred = probs.sum(dim=-1).round().long().clamp(int(candidate_min), int(candidate_max))
        for row, sample_idx in enumerate(idx.tolist()):
            pred_by_idx[int(sample_idx)] = int(pred[row].detach().cpu().item())
            prob_by_idx[int(sample_idx)] = [float(v) for v in probs[row].detach().float().cpu().tolist()]
    return {"pred_by_idx": pred_by_idx, "prob_by_idx": prob_by_idx}


def parse_qa_file(sample_dir: Path) -> Tuple[str, List[Dict[str, Any]], int]:
    lines = (sample_dir / "qa.txt").read_text(encoding="utf-8").splitlines()
    q_idx = next((i for i, line in enumerate(lines) if line.strip() == "question:"), -1)
    a_idx = next((i for i, line in enumerate(lines) if line.strip() == "answer:"), -1)
    if q_idx < 0 or a_idx <= q_idx:
        raise RuntimeError(f"Bad qa.txt format: {sample_dir / 'qa.txt'}")
    states: List[Dict[str, Any]] = []
    question: Optional[str] = None
    for line in lines[q_idx + 1 : a_idx]:
        text = line.strip()
        if not text:
            continue
        if text.startswith("{") and text.endswith("}"):
            states.append(ast.literal_eval(text))
        elif question is None:
            question = text
    answer_text = next((line.strip() for line in lines[a_idx + 1 :] if line.strip()), "")
    match = INTEGER_RE.search(answer_text)
    if question is None or match is None:
        raise RuntimeError(f"Could not parse question/answer: {sample_dir / 'qa.txt'}")
    return question, states, int(match.group(0))


def load_records(dataset_root: Path, split: str, seq_len: int, sample_ids: Sequence[str]) -> List[SampleRecord]:
    split_root = Path(dataset_root) / f"seq_len_{int(seq_len)}" / str(split)
    records: List[SampleRecord] = []
    missing: List[str] = []
    for sample_id in sample_ids:
        sample_dir = split_root / str(sample_id)
        if not sample_dir.is_dir():
            missing.append(str(sample_id))
            continue
        question, states, gold_count = parse_qa_file(sample_dir)
        evidence_count = len(eval_utils.collect_evidence_frame_indices(question, states))
        frame_paths = tuple(sample_dir / f"{idx:03d}.png" for idx in range(len(states)))
        records.append(
            SampleRecord(
                sample_id=str(sample_id),
                sample_dir=sample_dir,
                frame_paths=frame_paths,
                question=question,
                states=tuple(states),
                gold_count=int(gold_count),
                evidence_count=int(evidence_count),
            )
        )
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} sample dirs under {split_root}; first={missing[:3]}")
    return records


def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {raw!r}, but CUDA is not available")
    return raw


def resolve_dtype(raw: str, device: str) -> torch.dtype:
    key = str(raw).lower()
    if key == "auto":
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.startswith("cuda"):
            return torch.float16
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[key]


def load_model_and_processor(args: argparse.Namespace, device: str, dtype: torch.dtype) -> Tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    if args.max_pixels is not None and hasattr(processor, "image_processor"):
        setattr(processor.image_processor, "max_pixels", int(args.max_pixels))
    if args.min_pixels is not None and hasattr(processor, "image_processor"):
        setattr(processor.image_processor, "min_pixels", int(args.min_pixels))
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": str(args.attn_implementation),
    }
    if bool(args.load_in_4bit):
        if not device.startswith("cuda"):
            raise ValueError("--load-in-4bit requires CUDA")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = {"": device}
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(args.model_name, **model_kwargs)
    if not bool(args.load_in_4bit):
        model.to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)
    return model, processor


def hidden_size_from_model(model: Any) -> int:
    candidates = [
        getattr(getattr(model, "config", None), "hidden_size", None),
        getattr(getattr(getattr(model, "config", None), "text_config", None), "hidden_size", None),
        getattr(getattr(getattr(model, "language_model", None), "config", None), "hidden_size", None),
        getattr(getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None), "hidden_size", None),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    embed = model.get_input_embeddings()
    return int(embed.embedding_dim)


def candidate_token_ids(tokenizer: Any, candidate_min: int, candidate_max: int) -> Tuple[str, Dict[int, int]]:
    for name, fmt in (("plain", lambda x: str(x)), ("leading_space", lambda x: f" {x}")):
        ids: Dict[int, int] = {}
        ok = True
        for value in range(int(candidate_min), int(candidate_max) + 1):
            token_ids = tokenizer.encode(fmt(value), add_special_tokens=False)
            if len(token_ids) != 1:
                ok = False
                break
            ids[int(value)] = int(token_ids[0])
        if ok:
            return name, ids
    raise RuntimeError("Candidate counts are not single-token under plain or leading-space formatting")


def load_frames(record: SampleRecord) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for frame_path in record.frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))
    return frames


def build_conversation(record: SampleRecord, frames: Sequence[Image.Image]) -> List[Dict[str, Any]]:
    prompt = core.build_prompt(record.question, num_frames=len(frames))
    return [
        {
            "role": "user",
            "content": ([{"type": "image", "image": image} for image in frames] + [{"type": "text", "text": prompt}]),
        }
    ]


def _token_span_from_char_span(text: str, char_span: Tuple[int, int], processor: Any) -> Tuple[int, int]:
    start_char, end_char = int(char_span[0]), int(char_span[1])
    if start_char > 0 and text[start_char - 1].isspace():
        start_char -= 1
    start_token = len(processor.tokenizer(text[:start_char], add_special_tokens=False)["input_ids"])
    end_token = len(processor.tokenizer(text[:end_char], add_special_tokens=False)["input_ids"])
    return start_token, end_token


def _positions_from_token_span(base_start: int, token_span: Tuple[int, int]) -> List[int]:
    return list(range(int(base_start) + int(token_span[0]), int(base_start) + int(token_span[1])))


def locate_target_positions(
    *,
    input_ids_1d: torch.Tensor,
    attention_mask_1d: Optional[torch.Tensor],
    record: SampleRecord,
    processor: Any,
    carriers: Sequence[str],
) -> Tuple[List[int], int, Dict[str, Any]]:
    input_ids = [int(token_id) for token_id in input_ids_1d.detach().cpu().tolist()]
    if attention_mask_1d is None:
        prompt_last_index = len(input_ids) - 1
    else:
        active = attention_mask_1d.detach().cpu().nonzero(as_tuple=True)[0]
        prompt_last_index = int(active[-1].item()) if active.numel() else len(input_ids) - 1
    prompt_text = core.build_prompt(record.question, num_frames=len(record.frame_paths))
    prompt_text_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_text_start = find_subsequence(input_ids, [int(token_id) for token_id in prompt_text_ids])
    if prompt_text_start is None:
        raise RuntimeError(f"sample_id={record.sample_id}: failed to locate prompt text")
    question_fragment = f"Question: {record.question}\n"
    question_start_in_prompt = prompt_text.index(question_fragment)
    question_text_start = question_start_in_prompt + len("Question: ")
    parsed = eval_utils.parse_target_character_room_with_spans(record.question)
    if parsed is None:
        raise RuntimeError(f"sample_id={record.sample_id}: failed to parse target character/room")
    _char, _room, character_span, room_span = parsed
    positions_by_carrier: Dict[str, List[int]] = {}
    character_span_in_prompt = (
        question_text_start + int(character_span[0]),
        question_text_start + int(character_span[1]),
    )
    room_span_in_prompt = (
        question_text_start + int(room_span[0]),
        question_text_start + int(room_span[1]),
    )
    positions_by_carrier["target_char"] = _positions_from_token_span(
        prompt_text_start,
        _token_span_from_char_span(prompt_text, character_span_in_prompt, processor),
    )
    positions_by_carrier["target_room"] = _positions_from_token_span(
        prompt_text_start,
        _token_span_from_char_span(prompt_text, room_span_in_prompt, processor),
    )
    requested: List[int] = []
    for carrier in carriers:
        requested.extend(positions_by_carrier.get(str(carrier), []))
    requested = sorted({int(pos) for pos in requested if int(pos) >= 0})
    if not requested:
        raise RuntimeError(f"sample_id={record.sample_id}: no target positions for carriers={list(carriers)}")
    return requested, int(prompt_last_index), {"positions_by_carrier": positions_by_carrier}


def move_inputs_to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


def prepare_qwen_batch(
    *,
    records: Sequence[SampleRecord],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
    carriers: Sequence[str],
) -> QwenBatch:
    frames_by_record = [load_frames(record) for record in records]
    conversations = [build_conversation(record, frames) for record, frames in zip(records, frames_by_record)]
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
        positions, prompt_last, _debug = locate_target_positions(
            input_ids_1d=input_ids[batch_idx],
            attention_mask_1d=attention_mask[batch_idx] if torch.is_tensor(attention_mask) else None,
            record=record,
            processor=processor,
            carriers=carriers,
        )
        target_positions.append(positions)
        prompt_last_indices.append(prompt_last)
    return QwenBatch(
        inputs=move_inputs_to_device(dict(raw_inputs), device),
        target_positions=target_positions,
        prompt_last_indices=torch.tensor(prompt_last_indices, device=device, dtype=torch.long),
        gold_counts=torch.tensor([int(record.gold_count) for record in records], device=device, dtype=torch.long),
        sample_indices=[int(idx) for idx in sample_indices],
    )


def select_count_logits(logits: torch.Tensor, prompt_last_indices: torch.Tensor, count_token_ids: Dict[int, int]) -> torch.Tensor:
    batch_idx = torch.arange(int(logits.shape[0]), device=logits.device, dtype=torch.long)
    selected = logits[batch_idx, prompt_last_indices, :].float()
    ordered_ids = [int(count_token_ids[count]) for count in sorted(count_token_ids)]
    token_idx = torch.tensor(ordered_ids, device=selected.device, dtype=torch.long)
    return selected.index_select(dim=-1, index=token_idx)


def chunked(values: Sequence[int], chunk_size: int) -> Iterable[List[int]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(values), size):
        yield [int(x) for x in values[start : start + size]]


def make_stage3_adapter(
    *,
    args: argparse.Namespace,
    stage1_model: Stage1MemoryReadout,
    hidden_size: int,
) -> Stage3ResidualAdapter:
    core = MessageMemoryCore(
        input_dim=stage1_model.core.input_dim,
        bottleneck_dim=stage1_model.core.bottleneck_dim,
        key_dim=stage1_model.core.key_dim,
        value_dim=stage1_model.core.value_dim,
        dropout=float(args.dropout),
    )
    core.load_state_dict(stage1_model.core.state_dict())
    return Stage3ResidualAdapter(
        core=core,
        hidden_size=int(hidden_size),
        inject_layer=int(args.inject_layer),
        gamma_init=float(args.gamma_init),
        train_gamma=bool(args.train_gamma),
    )


def train_stage3(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    model: Any,
    processor: Any,
    stage1_model: Stage1MemoryReadout,
    records: Sequence[SampleRecord],
    x_messages: torch.Tensor,
    splits: Dict[str, List[int]],
    count_token_ids: Dict[int, int],
    device: str,
    hidden_size: int,
    checkpoint_path: Optional[Path],
) -> Tuple[Stage3ResidualAdapter, Dict[str, Any], Path]:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "stage3_best.pt"
    adapter = make_stage3_adapter(args=args, stage1_model=stage1_model, hidden_size=hidden_size)
    if checkpoint_path is not None and checkpoint_path.is_file():
        print(f"Loading Stage 3 checkpoint: {checkpoint_path}")
        ckpt = load_torch(checkpoint_path)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        return adapter, dict(ckpt.get("history", {})), checkpoint_path
    if best_path.is_file():
        print(f"Reusing existing Stage 3 checkpoint: {best_path}")
        ckpt = load_torch(best_path)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        return adapter, dict(ckpt.get("history", {})), best_path

    adapter.to(device)
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
            rng = random.Random(int(args.seed) + 1000 + epoch)
            shuffled = list(train_indices)
            rng.shuffle(shuffled)
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            train_steps = 0
            for step, batch_indices in enumerate(chunked(shuffled, int(args.stage3_batch_size)), start=1):
                batch_records = [records[idx] for idx in batch_indices]
                batch = prepare_qwen_batch(
                    records=batch_records,
                    sample_indices=batch_indices,
                    processor=processor,
                    device=device,
                    carriers=args.carriers,
                )
                x_batch = x_messages[batch_indices].to(device)
                adapter.set_context(x_batch, batch.target_positions)
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - min(count_token_ids.keys())
                loss = F.cross_entropy(count_logits, gold_offsets) / max(1, int(args.stage3_grad_accum))
                loss.backward()
                train_loss += float(loss.detach().cpu().item()) * max(1, int(args.stage3_grad_accum))
                train_steps += 1
                adapter.clear_context()
                if step % max(1, int(args.stage3_grad_accum)) == 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.stage3_grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 50 == 0:
                    print(f"  stage3 epoch={epoch} step={step} train_ce={train_loss / max(1, train_steps):.4f}")
            if train_steps % max(1, int(args.stage3_grad_accum)) != 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.stage3_grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            val_eval = evaluate_qwen_method(
                method="stage3_room_char_residual",
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
                "train_ce": train_loss / max(1, train_steps),
                "val_ce": float(val_eval["ce"]),
                "val_accuracy": float(val_eval["accuracy"]),
                "gamma": float(adapter.gamma.detach().cpu().item()),
            }
            rows.append(row)
            print(
                f"  stage3 epoch={epoch} train_ce={row['train_ce']:.4f} "
                f"val_ce={row['val_ce']:.4f} val_acc={row['val_accuracy']:.4f} gamma={row['gamma']:.4f}"
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
    print(f"Saved Stage 3 checkpoint: {best_path}")
    return adapter_cpu, ckpt["history"], best_path


def evaluate_qwen_method(
    *,
    method: str,
    model: Any,
    processor: Any,
    adapter: Optional[Stage3ResidualAdapter],
    records: Sequence[SampleRecord],
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
    delta_norm_by_idx: Dict[int, float] = {}
    ce_total = 0.0
    n = 0
    try:
        for batch_num, batch_indices in enumerate(chunked(list(indices), int(batch_size)), start=1):
            batch_records = [records[idx] for idx in batch_indices]
            batch = prepare_qwen_batch(
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
                count_logits = select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
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
                if adapter is not None and save_gates and adapter.last_alpha is not None:
                    for row, idx in enumerate(batch_indices):
                        gate_by_idx[int(idx)] = [float(v) for v in adapter.last_alpha[row].tolist()]
                    if adapter.last_delta_norm is not None:
                        for row, idx in enumerate(batch_indices):
                            delta_norm_by_idx[int(idx)] = float(adapter.last_delta_norm[row].item())
            if adapter is not None:
                adapter.clear_context()
            if batch_num == 1 or batch_num % 25 == 0:
                print(f"  eval {method}: {min(len(indices), batch_num * int(batch_size))}/{len(indices)}")
    finally:
        if adapter is not None:
            adapter.remove_hooks()
    y_true = [int(records[int(idx)].gold_count) for idx in indices]
    y_pred = [pred_by_idx[int(idx)] for idx in indices]
    return {
        "ce": ce_total / max(1, n),
        "accuracy": accuracy(y_true, y_pred),
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "gold_score_by_idx": gold_score_by_idx,
        "gate_by_idx": gate_by_idx,
        "delta_norm_by_idx": delta_norm_by_idx,
    }


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return math.nan
    return sum(int(a) == int(b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return math.nan
    return sum(abs(int(a) - int(b)) for a, b in zip(y_true, y_pred)) / len(y_true)


def rank_of_gold(logits: Sequence[float], gold_offset: int) -> int:
    gold = float(logits[int(gold_offset)])
    return 1 + sum(float(value) > gold for value in logits)


def auroc_binary(y_true: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(float(score), int(label)) for label, score in zip(y_true, scores)]
    pos = sum(label == 1 for _score, label in pairs)
    neg = sum(label == 0 for _score, label in pairs)
    if pos == 0 or neg == 0:
        return math.nan
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def metric_tables(
    *,
    records: Sequence[SampleRecord],
    test_indices: Sequence[int],
    method_predictions: Dict[str, Dict[int, int]],
    counts: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metrics_rows: List[Dict[str, Any]] = []
    overall_rows: List[Dict[str, Any]] = []
    mean_rows: List[Dict[str, Any]] = []
    for method, pred_by_idx in method_predictions.items():
        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        for idx in test_indices:
            if int(idx) not in pred_by_idx:
                continue
            y_true_all.append(int(records[int(idx)].gold_count))
            y_pred_all.append(int(pred_by_idx[int(idx)]))
        overall_rows.append(
            {
                "method": method,
                "split": "test",
                "n": len(y_true_all),
                "accuracy": accuracy(y_true_all, y_pred_all),
                "mae": mae(y_true_all, y_pred_all),
                "mean_predicted_count": float(np.mean(y_pred_all)) if y_pred_all else math.nan,
            }
        )
        for count in counts:
            items = [int(idx) for idx in test_indices if int(records[int(idx)].gold_count) == int(count) and int(idx) in pred_by_idx]
            y_true = [int(records[idx].gold_count) for idx in items]
            y_pred = [int(pred_by_idx[idx]) for idx in items]
            metrics_rows.append(
                {
                    "method": method,
                    "split": "test",
                    "evidence_count": int(count),
                    "n": len(items),
                    "accuracy": accuracy(y_true, y_pred),
                    "mae": mae(y_true, y_pred),
                }
            )
            mean_rows.append(
                {
                    "method": method,
                    "split": "test",
                    "evidence_count": int(count),
                    "n": len(items),
                    "mean_predicted_count": float(np.mean(y_pred)) if y_pred else math.nan,
                }
            )
    return metrics_rows, overall_rows, mean_rows


def build_gold_drop_rows(
    *,
    records: Sequence[SampleRecord],
    test_indices: Sequence[int],
    base_scores: Dict[int, float],
    stage3_scores: Dict[int, float],
    counts: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
                "method": "stage3_room_char_residual",
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


def make_per_sample_rows(
    *,
    records: Sequence[SampleRecord],
    splits: Dict[str, List[int]],
    method_predictions: Dict[str, Dict[int, int]],
    stage1_logits: Dict[int, List[float]],
    base_scores: Dict[int, float],
    stage3_scores: Dict[int, float],
    stage3_logits: Dict[int, List[float]],
    candidate_min: int,
) -> List[Dict[str, Any]]:
    split_by_idx = {idx: split for split, indices in splits.items() for idx in indices}
    rows: List[Dict[str, Any]] = []
    all_indices = sorted(split_by_idx)
    for idx in all_indices:
        record = records[int(idx)]
        row: Dict[str, Any] = {
            "split": split_by_idx[int(idx)],
            "sample_index": int(idx),
            "sample_id": record.sample_id,
            "evidence_count": int(record.evidence_count),
            "gold_count": int(record.gold_count),
            "sample_dir": os.fspath(record.sample_dir),
        }
        for method, preds in method_predictions.items():
            if int(idx) in preds:
                row[f"{method}_pred_count"] = int(preds[int(idx)])
                row[f"{method}_correct"] = int(preds[int(idx)] == record.gold_count)
        if int(idx) in base_scores:
            row["base_gold_score"] = float(base_scores[int(idx)])
        if int(idx) in stage3_scores:
            row["stage3_gold_score"] = float(stage3_scores[int(idx)])
        if int(idx) in base_scores and int(idx) in stage3_scores:
            row["stage3_gold_score_drop_vs_base"] = float(base_scores[int(idx)] - stage3_scores[int(idx)])
            row["stage3_gold_score_delta_vs_base"] = float(stage3_scores[int(idx)] - base_scores[int(idx)])
        if int(idx) in stage1_logits:
            logits = stage1_logits[int(idx)]
            gold_offset = int(record.gold_count) - int(candidate_min)
            row["stage1_gold_head_logit"] = float(logits[gold_offset])
            row["stage1_gold_head_rank"] = int(rank_of_gold(logits, gold_offset))
            row["stage1_gold_score_drop_vs_base"] = "NA"
            row["stage1_count_head_logits_json"] = json.dumps(logits)
        if int(idx) in stage3_logits:
            row["stage3_candidate_logits_json"] = json.dumps(stage3_logits[int(idx)])
        rows.append(row)
    return rows


def gate_rows(
    *,
    stage: str,
    records: Sequence[SampleRecord],
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


def summarize_gate_debug(gates_by_idx: Dict[int, List[float]], frame_labels: torch.Tensor, indices: Sequence[int]) -> Dict[str, Any]:
    labels: List[int] = []
    scores: List[float] = []
    sums_by_count: Dict[int, List[float]] = defaultdict(list)
    for idx in indices:
        idx = int(idx)
        gates = gates_by_idx.get(idx)
        if gates is None:
            continue
        frame_y = [int(v) for v in frame_labels[idx].int().tolist()]
        labels.extend(frame_y)
        scores.extend([float(v) for v in gates])
        sums_by_count[sum(frame_y)].append(float(sum(gates)))
    pred = [1 if score >= 0.5 else 0 for score in scores]
    acc = accuracy(labels, pred) if labels else math.nan
    return {
        "frame_gate_accuracy_at_0_5": acc,
        "frame_gate_auc": auroc_binary(labels, scores) if labels else math.nan,
        "mean_sum_alpha_by_evidence_count": {
            str(count): float(np.mean(values)) for count, values in sorted(sums_by_count.items()) if values
        },
    }


def plot_line(path_base: Path, rows: Sequence[Dict[str, Any]], y_key: str, ylabel: str, title: str, methods: Optional[Sequence[str]] = None) -> None:
    if not rows:
        return
    methods = list(methods) if methods is not None else sorted({str(row["method"]) for row in rows})
    counts = sorted({int(row["evidence_count"]) for row in rows if str(row.get("evidence_count", "")).lstrip("-").isdigit()})
    plt.figure(figsize=(8, 5))
    for method in methods:
        by_count = {int(row["evidence_count"]): row for row in rows if row.get("method") == method}
        ys = []
        for count in counts:
            value = by_count.get(count, {}).get(y_key, math.nan)
            try:
                ys.append(float(value))
            except Exception:
                ys.append(math.nan)
        if any(math.isfinite(v) for v in ys):
            plt.plot(counts, ys, marker="o", linewidth=1.8, label=method)
    plt.xlabel("Evidence count")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(counts)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(path_base.with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight")
    plt.close()


def make_plots(
    *,
    output_dir: Path,
    metric_rows: Sequence[Dict[str, Any]],
    mean_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    stage1_gate_rows: Sequence[Dict[str, Any]],
    stage3_delta_by_idx: Dict[int, float],
    records: Sequence[SampleRecord],
    test_indices: Sequence[int],
    counts: Sequence[int],
) -> None:
    plot_line(
        output_dir / "accuracy_by_evidence_count_all_methods",
        metric_rows,
        "accuracy",
        "Accuracy",
        "Accuracy by Evidence Count",
    )
    plot_line(
        output_dir / "mean_predicted_count_by_evidence_count_all_methods",
        mean_rows,
        "mean_predicted_count",
        "Mean predicted count",
        "Mean Predicted Count by Evidence Count",
    )
    plot_line(
        output_dir / "gold_score_delta_by_evidence_count_stage3_vs_base",
        gold_rows,
        "mean_gold_score_delta_vs_base",
        "Mean gold-score delta vs base",
        "Stage 3 Gold-Score Delta vs Base",
        methods=["stage3_room_char_residual"],
    )
    gate_sum_rows: List[Dict[str, Any]] = []
    grouped: Dict[int, List[float]] = defaultdict(list)
    for row in stage1_gate_rows:
        if row.get("split") == "test":
            grouped[int(row["evidence_count"])].append(float(row["gate"]))
    for count in counts:
        values = [sum(float(row["gate"]) for row in stage1_gate_rows if row.get("split") == "test" and int(row["sample_index"]) == int(idx)) for idx in test_indices if int(records[int(idx)].evidence_count) == int(count)]
        gate_sum_rows.append({"method": "stage1_memory_readout", "evidence_count": int(count), "mean_gate_sum": float(np.mean(values)) if values else math.nan})
    plot_line(
        output_dir / "gate_sum_by_evidence_count_stage1",
        gate_sum_rows,
        "mean_gate_sum",
        "Mean sum alpha_i",
        "Stage 1 Gate Sum by Evidence Count",
        methods=["stage1_memory_readout"],
    )
    gate_by_label: Dict[int, List[float]] = defaultdict(list)
    for row in stage1_gate_rows:
        if row.get("split") == "test":
            gate_by_label[int(row["frame_label"])].append(float(row["gate"]))
    plt.figure(figsize=(5, 4))
    xs = [0, 1]
    ys = [float(np.mean(gate_by_label[x])) if gate_by_label[x] else math.nan for x in xs]
    plt.bar(xs, ys)
    plt.xticks(xs, ["non-evidence", "evidence"])
    plt.ylabel("Mean gate")
    plt.title("Stage 1 Gate by Gold Frame Label")
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig((output_dir / "gate_score_by_gold_frame_label_stage1").with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight")
    plt.close()
    delta_rows: List[Dict[str, Any]] = []
    for count in counts:
        values = [float(stage3_delta_by_idx[int(idx)]) for idx in test_indices if int(records[int(idx)].evidence_count) == int(count) and int(idx) in stage3_delta_by_idx]
        delta_rows.append({"method": "stage3_room_char_residual", "evidence_count": int(count), "mean_delta_norm": float(np.mean(values)) if values else math.nan})
    plot_line(
        output_dir / "delta_norm_by_evidence_count_stage3",
        delta_rows,
        "mean_delta_norm",
        "Mean ||delta_h||",
        "Stage 3 Delta Norm by Evidence Count",
        methods=["stage3_room_char_residual"],
    )


def write_summary(
    *,
    output_dir: Path,
    overall_rows: Sequence[Dict[str, Any]],
    stage1_ckpt: Optional[Path],
    stage3_ckpt: Optional[Path],
    debug: Dict[str, Any],
) -> None:
    lines = [
        "Message-memory adapter Stage 1/Stage 3 seq_len=8",
        "",
        "Overall test accuracy:",
    ]
    for row in overall_rows:
        lines.append(f"- {row['method']}: accuracy={float(row['accuracy']):.4f} n={int(row['n'])} mae={float(row['mae']):.4f}")
    lines.extend(
        [
            "",
            "Notes:",
            "- Stage 1 logits come from an offline count head, not Qwen, so Stage 1 gold-score drop vs base is intentionally NA.",
            f"- Stage 1 checkpoint: {stage1_ckpt if stage1_ckpt is not None else 'none'}",
            f"- Stage 3 checkpoint: {stage3_ckpt if stage3_ckpt is not None else 'none'}",
            f"- Injection: layer {debug.get('inject_layer')} into {debug.get('injection_target')}",
        ]
    )
    (output_dir / "results_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.run_stage1 and not args.run_stage3:
        args.run_stage1 = True
        args.run_stage3 = True
    args.layers = parse_int_tokens(args.layers)
    args.evidence_counts = parse_int_tokens(args.evidence_counts)
    args.carriers = split_tokens(args.carriers)
    output_dir = (args.output_dir if args.output_dir is not None else default_output_dir()).resolve()
    log_handle, old_stdout, old_stderr = setup_logging(output_dir)
    started = time.time()
    try:
        set_seed(int(args.seed))
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
            "run_stage1": bool(args.run_stage1),
            "run_stage3": bool(args.run_stage3),
            "bottleneck_dim": int(args.bottleneck_dim),
            "key_dim": int(args.key_dim),
            "value_dim": int(args.value_dim),
            "dropout": float(args.dropout),
            "frame_gate_bce_weight": float(args.frame_gate_bce_weight),
            "stage1_epochs": int(args.stage1_epochs),
            "stage3_epochs": int(args.stage3_epochs),
            "inject_layer": int(args.inject_layer),
            "gamma_init": float(args.gamma_init),
            "train_gamma": bool(args.train_gamma),
            "dtype": str(args.dtype),
            "attn_implementation": str(args.attn_implementation),
            "load_in_4bit": bool(args.load_in_4bit),
            "seed": int(args.seed),
        }
        write_json(output_dir / "run_config.json", config)
        print(f"Output dir: {output_dir}")
        print(f"Config: {json.dumps(config, sort_keys=True)}")

        feature_data = load_message_features(args, args.layers, args.evidence_counts)
        sample_ids = feature_data["sample_ids"]
        labels = feature_data["labels"]
        frame_labels = feature_data["frame_labels"]
        x_messages = feature_data["x_messages"]
        records = load_records(args.dataset_root, args.split, args.seq_len, sample_ids)
        counts = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
        splits = stratified_split(sample_ids, labels, int(args.seed))
        counts_by_split = split_counts(splits, labels, counts)
        print(f"x_messages shape={tuple(x_messages.shape)} D_msg={int(x_messages.shape[-1])}")
        print(f"d_m={args.bottleneck_dim} d_k={args.key_dim} d_v={args.value_dim}")
        for split, row in counts_by_split.items():
            print(f"  {split}: " + ", ".join(f"{count}:{row.get(count, 0)}" for count in counts))
        write_json(
            output_dir / "probe_feature_shapes.json",
            {
                "x_messages_shape": list(x_messages.shape),
                "D_msg": int(x_messages.shape[-1]),
                "labels_shape": list(labels.shape),
                "frame_labels_shape": list(frame_labels.shape),
                "layers": list(args.layers),
                "carriers": ["target_char", "target_room"],
                "source_cache": os.fspath(feature_data["cache_path"]),
            },
        )

        base_pred_by_id, base_cache, base_warnings = load_base_predictions(args, sample_ids)
        for warning in base_warnings:
            print(f"[warn] {warning}")

        stage1_model: Optional[Stage1MemoryReadout] = None
        stage1_history: Dict[str, Any] = {}
        stage1_ckpt: Optional[Path] = None
        if args.run_stage1 or args.run_stage3:
            stage1_model, stage1_history, stage1_ckpt = train_stage1(
                args=args,
                output_dir=output_dir,
                x_messages=x_messages,
                labels=labels,
                frame_labels=frame_labels,
                splits=splits,
                candidate_min=int(args.candidate_min),
                candidate_max=int(args.candidate_max),
                checkpoint_path=args.stage1_checkpoint,
            )
        if stage1_model is None:
            raise RuntimeError("Stage 1 model is required for this experiment")

        eval_device = torch.device("cuda" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
        stage1_eval_all: Dict[str, Any] = {}
        for split_name, split_indices in splits.items():
            stage1_eval_all[split_name] = evaluate_stage1_model(
                model=stage1_model,
                x_messages=x_messages,
                labels=labels,
                frame_labels=frame_labels,
                indices=split_indices,
                candidate_min=int(args.candidate_min),
                batch_size=int(args.stage1_batch_size),
                device=eval_device,
            )
        stage1_pred = {idx: pred for split_eval in stage1_eval_all.values() for idx, pred in split_eval["pred_by_idx"].items()}
        stage1_logits = {idx: value for split_eval in stage1_eval_all.values() for idx, value in split_eval["logits_by_idx"].items()}
        stage1_gates = {idx: value for split_eval in stage1_eval_all.values() for idx, value in split_eval["alpha_by_idx"].items()}
        stage1_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        linear_pred: Dict[int, int] = {}
        linear_probs: Dict[int, List[float]] = {}
        linear_history: Dict[str, Any] = {}
        if not bool(args.no_per_frame_linear):
            print("Training per-frame message linear sum diagnostic")
            linear_model, linear_history = train_linear_probe(
                args=args,
                x_messages=x_messages,
                labels=labels,
                frame_labels=frame_labels,
                splits=splits,
                candidate_min=int(args.candidate_min),
                candidate_max=int(args.candidate_max),
                mean=stage1_model.core.x_mean.detach().cpu(),
                std=stage1_model.core.x_std.detach().cpu(),
            )
            for split_name, split_indices in splits.items():
                result = evaluate_linear_probe(
                    model=linear_model,
                    x_messages=x_messages,
                    frame_labels=frame_labels,
                    indices=split_indices,
                    mean=stage1_model.core.x_mean.detach().cpu(),
                    std=stage1_model.core.x_std.detach().cpu(),
                    candidate_min=int(args.candidate_min),
                    candidate_max=int(args.candidate_max),
                    batch_size=int(args.stage1_batch_size),
                    device=eval_device,
                )
                linear_pred.update(result["pred_by_idx"])
                linear_probs.update(result["prob_by_idx"])

        method_predictions: Dict[str, Dict[int, int]] = {
            "stage1_memory_readout": stage1_pred,
        }
        if linear_pred:
            method_predictions["per_frame_message_linear_sum"] = linear_pred
        cached_base_pred = {idx: int(base_pred_by_id[sample_id]) for idx, sample_id in enumerate(sample_ids) if sample_id in base_pred_by_id}
        if cached_base_pred:
            method_predictions["base_frozen_qwen"] = cached_base_pred

        base_eval: Dict[str, Any] = {"pred_by_idx": {}, "gold_score_by_idx": {}, "logits_by_idx": {}}
        stage3_eval: Dict[str, Any] = {"pred_by_idx": {}, "gold_score_by_idx": {}, "logits_by_idx": {}, "gate_by_idx": {}, "delta_norm_by_idx": {}}
        stage3_history: Dict[str, Any] = {}
        stage3_ckpt: Optional[Path] = None
        if args.run_stage3:
            device = resolve_device(str(args.device))
            dtype = resolve_dtype(str(args.dtype), device)
            model, processor = load_model_and_processor(args, device=device, dtype=dtype)
            candidate_format, count_ids = candidate_token_ids(processor.tokenizer, int(args.candidate_min), int(args.candidate_max))
            hidden_size = hidden_size_from_model(model)
            print(f"Loaded Qwen hidden_size={hidden_size} candidate_format={candidate_format} count_ids={count_ids}")
            print("Evaluating base frozen Qwen on test split")
            base_eval = evaluate_qwen_method(
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
            method_predictions["base_frozen_qwen"].update(base_eval["pred_by_idx"]) if "base_frozen_qwen" in method_predictions else method_predictions.update({"base_frozen_qwen": base_eval["pred_by_idx"]})
            stage3_adapter, stage3_history, stage3_ckpt = train_stage3(
                args=args,
                output_dir=output_dir,
                model=model,
                processor=processor,
                stage1_model=stage1_model,
                records=records,
                x_messages=x_messages,
                splits=splits,
                count_token_ids=count_ids,
                device=device,
                hidden_size=hidden_size,
                checkpoint_path=args.stage3_checkpoint,
            )
            print("Evaluating Stage 3 residual adapter on test split")
            stage3_eval = evaluate_qwen_method(
                method="stage3_room_char_residual",
                model=model,
                processor=processor,
                adapter=stage3_adapter,
                records=records,
                indices=splits["test"],
                x_messages=x_messages,
                count_token_ids=count_ids,
                args=args,
                device=device,
                batch_size=int(args.stage3_batch_size),
                save_gates=True,
            )
            method_predictions["stage3_room_char_residual"] = stage3_eval["pred_by_idx"]
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        metric_rows, overall_rows, mean_rows = metric_tables(
            records=records,
            test_indices=splits["test"],
            method_predictions=method_predictions,
            counts=counts,
        )
        gold_rows = build_gold_drop_rows(
            records=records,
            test_indices=splits["test"],
            base_scores=base_eval.get("gold_score_by_idx", {}),
            stage3_scores=stage3_eval.get("gold_score_by_idx", {}),
            counts=counts,
        )
        per_sample_rows = make_per_sample_rows(
            records=records,
            splits=splits,
            method_predictions=method_predictions,
            stage1_logits=stage1_logits,
            base_scores=base_eval.get("gold_score_by_idx", {}),
            stage3_scores=stage3_eval.get("gold_score_by_idx", {}),
            stage3_logits=stage3_eval.get("logits_by_idx", {}),
            candidate_min=int(args.candidate_min),
        )
        stage1_gate_rows = gate_rows(stage="stage1", records=records, splits=splits, gates_by_idx=stage1_gates, frame_labels=frame_labels)
        stage3_gate_rows = gate_rows(
            stage="stage3",
            records=records,
            splits={"test": splits["test"]},
            gates_by_idx=stage3_eval.get("gate_by_idx", {}),
            frame_labels=frame_labels,
        )
        debug = {
            "x_messages_shape": list(x_messages.shape),
            "D_msg": int(x_messages.shape[-1]),
            "d_m": int(args.bottleneck_dim),
            "d_k": int(args.key_dim),
            "d_v": int(args.value_dim),
            "split_counts": {split: {str(k): int(v) for k, v in row.items()} for split, row in counts_by_split.items()},
            "stage1_gate_debug_test": summarize_gate_debug(stage1_gates, frame_labels, splits["test"]),
            "stage3_gate_debug_test": summarize_gate_debug(stage3_eval.get("gate_by_idx", {}), frame_labels, splits["test"]),
            "stage3_mean_delta_norm_by_evidence_count": {},
            "inject_layer": int(args.inject_layer),
            "injection_target": "target_char + target_room tokens",
            "base_cache": os.fspath(base_cache) if base_cache is not None else None,
            "source_cache": os.fspath(feature_data["cache_path"]),
            "stage1_history": stage1_history,
            "stage3_history": stage3_history,
            "linear_history": linear_history,
            "runtime_seconds": time.time() - started,
        }
        for count in counts:
            values = [
                float(stage3_eval.get("delta_norm_by_idx", {})[int(idx)])
                for idx in splits["test"]
                if int(records[int(idx)].evidence_count) == int(count) and int(idx) in stage3_eval.get("delta_norm_by_idx", {})
            ]
            debug["stage3_mean_delta_norm_by_evidence_count"][str(count)] = float(np.mean(values)) if values else math.nan

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
            output_dir / "per_frame_gates_stage1.csv",
            ["stage", "split", "sample_index", "sample_id", "evidence_count", "frame_idx", "frame_label", "gate"],
            stage1_gate_rows,
        )
        write_csv(
            output_dir / "per_frame_gates_stage3.csv",
            ["stage", "split", "sample_index", "sample_id", "evidence_count", "frame_idx", "frame_label", "gate"],
            stage3_gate_rows,
        )
        write_json(output_dir / "adapter_debug.json", debug)
        write_summary(output_dir=output_dir, overall_rows=overall_rows, stage1_ckpt=stage1_ckpt, stage3_ckpt=stage3_ckpt, debug=debug)
        if not bool(args.no_plots):
            make_plots(
                output_dir=output_dir,
                metric_rows=metric_rows,
                mean_rows=mean_rows,
                gold_rows=gold_rows,
                stage1_gate_rows=stage1_gate_rows,
                stage3_delta_by_idx=stage3_eval.get("delta_norm_by_idx", {}),
                records=records,
                test_indices=splits["test"],
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

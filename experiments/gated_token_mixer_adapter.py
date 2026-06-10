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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from scripts.experiments import evidence_only_layer_local_seq1_8_7b as base
from scripts.experiments import translator_ablation_gold_count_seq8_7b as trans
from scripts.probes import run_message_memory_adapter_stage1_stage3_seq8 as prev
from scripts.probes import run_message_memory_carrier_update_seq8_7b as carrier


EXPERIMENT_NAME = "gated_token_mixer_adapter"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images_park"
EVIDENCE_ONLY_EXPERIMENT_NAME = "evidence_only_gated_token_mixer_adapter_seq1_8_7b"
DEFAULT_EVIDENCE_ONLY_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EVIDENCE_ONLY_EXPERIMENT_NAME
DEFAULT_EVIDENCE_ONLY_DATASET_ROOT = base.DEFAULT_DATASET_ROOT
DEFAULT_SOURCE_DATASET_ROOT = base.DEFAULT_SOURCE_DATASET_ROOT

FROZEN_QWEN = "frozen_qwen"
MLP_ADAPTER = "mlp_adapter"
LORA_ATTENTION = "lora_attention"
GATED_TOKEN_MIXER = "gated_token_mixer"
METHODS = (FROZEN_QWEN, MLP_ADAPTER, LORA_ATTENTION, GATED_TOKEN_MIXER)
METHOD_ALIASES = {
    "frozen": FROZEN_QWEN,
    "baseline": FROZEN_QWEN,
    "frozen_qwen": FROZEN_QWEN,
    "mlp": MLP_ADAPTER,
    "mlp_adapter": MLP_ADAPTER,
    "lora": LORA_ATTENTION,
    "lora_attention": LORA_ATTENTION,
    "gated": GATED_TOKEN_MIXER,
    "gated_token_mixer": GATED_TOKEN_MIXER,
}
COUNT_VALUES = list(range(9))
EPS = 1e-6


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
            "Frozen Qwen evidence-counting comparison: no adapter, per-token MLP, "
            "attention LoRA, and a non-softmax gated token-mixing adapter."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-run", type=Path, default=prev.DEFAULT_SOURCE_RUN)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument(
        "--evidence-only-seq1-8",
        "--evidence_only_seq1_8",
        dest="evidence_only_seq1_8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the evidence-only seq_len 1..8 dataset/splits instead of the cached seq8 count-balanced dataset.",
    )
    parser.add_argument("--seq-lens", "--seq_lens", dest="seq_lens", nargs="+", default=[",".join(str(x) for x in range(1, 9))])
    parser.add_argument("--source-dataset-root", "--source_dataset_root", dest="source_dataset_root", type=Path, default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--samples-per-seq-len", "--samples_per_seq_len", dest="samples_per_seq_len", type=int, default=100)
    parser.add_argument("--generate-dataset", "--generate_dataset", dest="generate_dataset", action="store_true", default=False)
    parser.add_argument("--force-generate", "--force_generate", dest="force_generate", action="store_true", default=False)
    parser.add_argument("--split", default="all_uniform")
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in COUNT_VALUES])
    parser.add_argument("--max-samples-per-count", "--max_samples_per_count", dest="max_samples_per_count", type=int, default=100)
    parser.add_argument("--max-train-samples", "--max_train_samples", dest="max_train_samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", "--max_eval_samples", dest="max_eval_samples", type=int, default=0)
    parser.add_argument("--max-train-samples-per-seq-len", "--max_train_samples_per_seq_len", dest="max_train_samples_per_seq_len", type=int, default=100)
    parser.add_argument("--max-eval-samples-per-seq-len", "--max_eval_samples_per_seq_len", dest="max_eval_samples_per_seq_len", type=int, default=100)

    parser.add_argument(
        "--methods",
        nargs="+",
        default=[",".join(METHODS)],
        help="Comma and/or space separated subset of frozen_qwen, mlp_adapter, lora_attention, gated_token_mixer.",
    )
    parser.add_argument("--layers", nargs="+", default=["14,15,16,17"])
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--grad-accum", "--grad_accum", dest="grad_accum", type=int, default=8)
    parser.add_argument("--grad-clip", "--grad_clip", dest="grad_clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda-margin", "--lambda_margin", dest="lambda_margin", type=float, default=0.2)
    parser.add_argument("--margin-target", "--margin_target", dest="margin_target", type=float, default=1.0)
    parser.add_argument("--lambda-update-energy", "--lambda_update_energy", dest="lambda_update_energy", type=float, default=1e-4)
    parser.add_argument("--lambda-init", "--lambda_init", dest="lambda_init", type=float, default=1e-3)
    parser.add_argument("--train-lambda", "--train_lambda", dest="train_lambda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", "--run_name", dest="run_name", default="")
    parser.add_argument("--skip-lora", "--skip_lora", dest="skip_lora", action="store_true", default=False)
    parser.add_argument("--lora-alpha", "--lora_alpha", dest="lora_alpha", type=float, default=None)
    parser.add_argument("--lora-dropout", "--lora_dropout", dest="lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-targets",
        "--lora_targets",
        dest="lora_targets",
        nargs="+",
        default=["q_proj,k_proj,v_proj,o_proj"],
    )

    parser.add_argument("--candidate-min", "--candidate_min", dest="candidate_min", type=int, default=0)
    parser.add_argument("--candidate-max", "--candidate_max", dest="candidate_max", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--attn-implementation", "--attn_implementation", dest="attn_implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", "--load_in_4bit", dest="load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pixels", "--max_pixels", dest="max_pixels", type=int, default=None)
    parser.add_argument("--min-pixels", "--min_pixels", dest="min_pixels", type=int, default=None)
    parser.add_argument("--no-plots", "--no_plots", dest="no_plots", action="store_true", default=False)
    parser.add_argument("--submit-mode", "--submit_mode", dest="submit_mode", default="local")
    return parser.parse_args()


def split_tokens(raw_values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for raw in raw_values:
        out.extend(part.strip() for part in str(raw).replace(",", " ").split() if part.strip())
    return out


def parse_int_tokens(raw_values: Sequence[Any]) -> List[int]:
    return sorted(dict.fromkeys(int(part) for part in split_tokens(raw_values)))


def parse_methods(raw_values: Sequence[Any]) -> List[str]:
    methods: List[str] = []
    for token in split_tokens(raw_values):
        key = str(token).strip()
        if key not in METHOD_ALIASES:
            raise ValueError(f"Unknown method {token!r}; valid values are {sorted(METHOD_ALIASES)}")
        methods.append(METHOD_ALIASES[key])
    return list(dict.fromkeys(methods))


def safe_name(value: Any) -> str:
    safe = str(value)
    for old, new in (("/", "_"), ("+", "_"), (" ", "_"), (":", "_"), (".", "p"), (",", "_")):
        safe = safe.replace(old, new)
    return safe


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_mean(values: Iterable[Any], default: float = math.nan) -> float:
    vals = [float(v) for v in values if finite_float(v) is not None]
    return float(np.mean(vals)) if vals else float(default)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return math.nan
    return sum(int(a) == int(b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return math.nan
    return sum(abs(int(a) - int(b)) for a, b in zip(y_true, y_pred)) / len(y_true)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_dynamic(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(leading)
    for key in sorted({key for row in rows for key in row.keys()}):
        if key not in fields:
            fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def setup_logging(output_dir: Path) -> Tuple[Any, Any, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (output_dir / "run.log").open("a", encoding="utf-8", buffering=1)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)
    return log_handle, old_stdout, old_stderr


def restore_logging(log_handle: Any, old_stdout: Any, old_stderr: Any) -> None:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_handle.close()


def default_run_name() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_run_dir(args: argparse.Namespace) -> Path:
    name = str(args.run_name).strip() or default_run_name()
    return Path(args.output_dir).resolve() / safe_name(name)


def configure_evidence_only_defaults(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "evidence_only_seq1_8", False)):
        return
    if Path(args.dataset_root) == DEFAULT_DATASET_ROOT:
        args.dataset_root = DEFAULT_EVIDENCE_ONLY_DATASET_ROOT
    if Path(args.output_dir) == DEFAULT_OUTPUT_ROOT:
        args.output_dir = DEFAULT_EVIDENCE_ONLY_OUTPUT_ROOT


def split_hist_by_seq_len(records: Sequence[prev.SampleRecord], indices: Sequence[int]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for idx in indices:
        seq_len = len(records[int(idx)].frame_paths)
        key = str(int(seq_len))
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda item: int(item[0])))


def build_evidence_only_sample_manifest(
    *,
    args: argparse.Namespace,
    seq_lens: Sequence[int],
    records: Sequence[prev.SampleRecord],
    splits: Dict[str, List[int]],
    dataset_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "source": "evidence_only_seq1_8_dataset",
        "dataset_root": os.fspath(Path(args.dataset_root).resolve()),
        "source_dataset_root": os.fspath(Path(args.source_dataset_root).resolve()),
        "seq_lens": [int(x) for x in seq_lens],
        "n": len(records),
        "split_sizes": {split: len(indices) for split, indices in splits.items()},
        "split_seq_len_histogram": {
            split: split_hist_by_seq_len(records, indices)
            for split, indices in splits.items()
        },
        "dataset_manifest": dataset_manifest,
    }


def load_evidence_only_records_and_splits(
    *,
    args: argparse.Namespace,
    seq_lens: Sequence[int],
) -> Tuple[List[prev.SampleRecord], Dict[str, List[int]], Dict[str, Any]]:
    if bool(args.generate_dataset):
        base.generate_evidence_only_dataset(
            dataset_root=Path(args.dataset_root),
            source_dataset_root=Path(args.source_dataset_root),
            seq_lens=seq_lens,
            samples_per_seq_len=int(args.samples_per_seq_len),
            force=bool(args.force_generate),
        )
    ok, dataset_manifest = base.validate_evidence_only_dataset(
        Path(args.dataset_root),
        seq_lens,
        int(args.samples_per_seq_len),
    )
    if not ok:
        raise RuntimeError(
            f"Evidence-only dataset failed validation: {Path(args.dataset_root).resolve()}. "
            "Pass --generate-dataset or --force-generate if it needs to be rebuilt."
        )
    records, by_seq = base.load_all_records(Path(args.dataset_root), seq_lens)
    splits = base.make_splits(
        records,
        by_seq,
        seed=int(args.seed),
        max_train_per_seq=int(args.max_train_samples_per_seq_len),
        max_eval_per_seq=int(args.max_eval_samples_per_seq_len),
    )
    base.print_split_counts(records, splits, seq_lens)
    return records, splits, dataset_manifest


def load_sample_ids_and_labels(
    *,
    args: argparse.Namespace,
    count_values: Sequence[int],
) -> Tuple[List[str], torch.Tensor, Dict[str, Any]]:
    counts = set(int(x) for x in count_values)
    manifest: Dict[str, Any] = {
        "source": "source_run_cache",
        "source_run": os.fspath(Path(args.source_run)),
        "dataset_root": os.fspath(Path(args.dataset_root)),
        "split": str(args.split),
        "seq_len": int(args.seq_len),
    }
    cache_path = prev.source_cache_path(Path(args.source_run))
    if cache_path.is_file():
        print(f"Loading sample ids from source cache: {cache_path}")
        payload = prev.load_torch(cache_path)
        source_ids = [str(x) for x in payload["sample_ids"]]
        source_labels = payload["labels"].long()
        keep = [idx for idx, label in enumerate(source_labels.tolist()) if int(label) in counts]
        if int(args.max_samples_per_count) > 0:
            seen: Dict[int, int] = defaultdict(int)
            limited: List[int] = []
            for idx in keep:
                label = int(source_labels[int(idx)].item())
                if seen[label] < int(args.max_samples_per_count):
                    limited.append(int(idx))
                    seen[label] += 1
            keep = limited
        sample_ids = [source_ids[int(idx)] for idx in keep]
        labels = source_labels[keep].long()
        manifest.update(
            {
                "cache_path": os.fspath(cache_path),
                "n": len(sample_ids),
                "label_histogram": {
                    str(k): int(v)
                    for k, v in sorted(
                        {int(x): int((labels == int(x)).sum().item()) for x in counts}.items()
                    )
                },
            }
        )
        return sample_ids, labels, manifest

    print(f"Source cache not found at {cache_path}; falling back to dataset scan.")
    split_root = Path(args.dataset_root) / f"seq_len_{int(args.seq_len)}" / str(args.split)
    if not split_root.is_dir():
        raise FileNotFoundError(f"Could not find source cache or dataset split directory: {split_root}")
    by_count: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for sample_dir in sorted(split_root.iterdir(), key=lambda path: path.name):
        if not sample_dir.is_dir() or not (sample_dir / "qa.txt").is_file():
            continue
        try:
            _question, _states, gold_count = prev.parse_qa_file(sample_dir)
        except Exception as exc:
            print(f"  warning: failed to parse {sample_dir}: {exc}")
            continue
        if int(gold_count) in counts:
            by_count[int(gold_count)].append((sample_dir.name, int(gold_count)))
    sample_ids = []
    labels_list = []
    rng = random.Random(int(args.seed))
    for count in sorted(by_count):
        values = list(by_count[count])
        rng.shuffle(values)
        if int(args.max_samples_per_count) > 0:
            values = values[: int(args.max_samples_per_count)]
        values.sort(key=lambda item: item[0])
        for sample_id, label in values:
            sample_ids.append(str(sample_id))
            labels_list.append(int(label))
    labels = torch.tensor(labels_list, dtype=torch.long)
    manifest.update(
        {
            "source": "dataset_scan",
            "split_root": os.fspath(split_root),
            "n": len(sample_ids),
            "label_histogram": {
                str(k): int(v)
                for k, v in sorted({int(x): int((labels == int(x)).sum().item()) for x in counts}.items())
            },
        }
    )
    return sample_ids, labels, manifest


def make_splits(
    *,
    sample_ids: Sequence[str],
    labels: torch.Tensor,
    records: Sequence[prev.SampleRecord],
    args: argparse.Namespace,
    count_values: Sequence[int],
) -> Dict[str, List[int]]:
    splits = prev.stratified_split(sample_ids, labels, int(args.seed))
    splits["train"] = trans.split_limited_indices(
        splits["train"],
        records,
        int(args.max_train_samples),
        int(args.seed) + 101,
    )
    splits["val"] = trans.split_limited_indices(
        splits["val"],
        records,
        int(args.max_eval_samples),
        int(args.seed) + 202,
    )
    splits["test"] = trans.split_limited_indices(
        splits["test"],
        records,
        int(args.max_eval_samples),
        int(args.seed) + 303,
    )
    for split in splits:
        splits[split] = sorted(splits[split], key=lambda idx: (int(records[idx].gold_count), records[idx].sample_id))
    print("Split counts:")
    for split in ("train", "val", "test"):
        hist = {int(count): 0 for count in count_values}
        for idx in splits[split]:
            hist[int(records[int(idx)].gold_count)] = hist.get(int(records[int(idx)].gold_count), 0) + 1
        print(f"  {split}: n={len(splits[split])} hist={json_compact(hist)}")
    return splits


def prepare_batch(
    *,
    records: Sequence[prev.SampleRecord],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
) -> carrier.MemoryBatch:
    if not records:
        raise ValueError("records cannot be empty")
    seq_lens = {len(record.frame_paths) for record in records}
    if len(seq_lens) != 1:
        raise ValueError(f"Expected a homogeneous seq_len batch, got {sorted(seq_lens)}")
    carrier.NUM_FRAMES = int(next(iter(seq_lens)))
    return carrier.prepare_memory_batch(
        records=records,
        sample_indices=sample_indices,
        processor=processor,
        device=device,
        token_group=carrier.ALL_QUESTION_TOKEN_GROUP,
        message_token_group=carrier.ALL_QUESTION_TOKEN_GROUP,
        query_token_group=carrier.ALL_QUESTION_TOKEN_GROUP,
        inject_token_group=carrier.LAST_TOKEN_GROUP,
    )


def homogeneous_batches(
    indices: Sequence[int],
    records: Sequence[prev.SampleRecord],
    batch_size: int,
    *,
    seed: int,
    shuffle_batches: bool,
) -> List[List[int]]:
    return base.homogeneous_batches(
        indices,
        records,
        int(batch_size),
        seed=int(seed),
        shuffle_batches=bool(shuffle_batches),
    )


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def freeze_qwen(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False


def verify_trainable_scope(model: Any, adapter: Optional[nn.Module], method: str) -> int:
    adapter_ids = {id(param) for param in adapter.parameters() if param.requires_grad} if adapter is not None else set()
    rogue = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and id(param) not in adapter_ids
    ]
    if rogue:
        raise RuntimeError(f"{method}: frozen Qwen has unexpected trainable parameters: {rogue[:10]}")
    adapter_count = count_trainable_parameters(adapter) if adapter is not None else 0
    model_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if adapter is None and model_trainable:
        raise RuntimeError(f"{method}: baseline has {model_trainable} trainable model parameters")
    print(
        f"Trainable parameters for {method}: adapter={adapter_count:,} "
        f"model-visible={model_trainable:,}"
    )
    return int(adapter_count)


def _hidden_from_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
    if args and torch.is_tensor(args[0]):
        return args[0]
    hidden = kwargs.get("hidden_states")
    return hidden if torch.is_tensor(hidden) else None


def _first_tensor_from_output(output: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    return None


def _replace_first_tensor(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    if isinstance(output, list):
        return [hidden] + list(output[1:])
    return output


class AttentionOutputResidualAdapter(nn.Module):
    """Base class for adapters added in parallel to frozen self-attention output."""

    def __init__(self, *, hidden_size: int, inject_layers: Sequence[int], lambda_init: float, train_lambda: bool) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.inject_layers = [int(layer) for layer in inject_layers]
        self.layer_to_pos = {int(layer): pos for pos, layer in enumerate(self.inject_layers)}
        self.lambdas = nn.Parameter(
            torch.full((len(self.inject_layers),), float(lambda_init), dtype=torch.float32),
            requires_grad=bool(train_lambda),
        )
        self.enabled = True
        self._handles: List[Any] = []
        self._attention_mask: Optional[torch.Tensor] = None
        self._prompt_last_indices: Optional[List[int]] = None
        self._query_positions: Optional[List[List[int]]] = None
        self._frame_groups: Optional[List[List[List[int]]]] = None
        self._evidence_masks: Optional[List[List[int]]] = None
        self._loss_update_energies: List[torch.Tensor] = []
        self._last_stats: Dict[str, Dict[str, Any]] = {}

    def attach(self, model: Any) -> None:
        self.detach()
        layers = prev.get_layers(model)
        for layer_idx in self.inject_layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"layer={layer_idx} outside [0, {len(layers) - 1}]")
            attn = getattr(layers[int(layer_idx)], "self_attn", None)
            if attn is None:
                raise RuntimeError(f"layer={layer_idx} does not expose self_attn")

            def hook(module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any, *, layer: int = int(layer_idx)) -> Any:
                if not self.enabled:
                    return output
                hidden = _hidden_from_args(args, kwargs)
                attn_output = _first_tensor_from_output(output)
                if hidden is None or attn_output is None:
                    return output
                update = self.compute_scaled_update(hidden, layer)
                return _replace_first_tensor(output, attn_output + update.to(dtype=attn_output.dtype))

            self._handles.append(attn.register_forward_hook(hook, with_kwargs=True))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def set_context(self, batch: carrier.MemoryBatch, records: Sequence[prev.SampleRecord]) -> None:
        attention_mask = batch.inputs.get("attention_mask")
        self._attention_mask = attention_mask.detach() if torch.is_tensor(attention_mask) else None
        self._prompt_last_indices = [int(x) for x in batch.prompt_last_indices.detach().cpu().tolist()]
        self._query_positions = [[int(pos) for pos in positions] for positions in batch.query_positions]
        self._frame_groups = [
            [[int(pos) for pos in group] for group in sample_groups]
            for sample_groups in batch.frame_groups
        ]
        self._evidence_masks = [
            carrier.evidence_frame_mask(record, len(record.frame_paths))
            for record in records
        ]
        self._loss_update_energies = []
        self._last_stats = {
            "adapter_residual_norm_by_layer": {},
            "lambda_by_layer": {},
        }

    def clear_context(self) -> None:
        self._attention_mask = None
        self._prompt_last_indices = None
        self._query_positions = None
        self._frame_groups = None
        self._evidence_masks = None
        self._loss_update_energies = []

    def valid_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        if self._attention_mask is None:
            return torch.ones((batch, seq_len), device=hidden_states.device, dtype=torch.bool)
        mask = self._attention_mask.to(device=hidden_states.device)
        if mask.dim() != 2 or int(mask.shape[0]) != batch:
            return torch.ones((batch, seq_len), device=hidden_states.device, dtype=torch.bool)
        if int(mask.shape[1]) < seq_len:
            pad = torch.ones((batch, seq_len - int(mask.shape[1])), device=hidden_states.device, dtype=mask.dtype)
            mask = torch.cat([mask, pad], dim=1)
        return mask[:, :seq_len].bool()

    def compute_scaled_update(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        layer_pos = self.layer_to_pos[int(layer_idx)]
        valid = self.valid_mask(hidden_states)
        raw_update = self.compute_raw_update(hidden_states, layer_pos, valid, layer_idx=int(layer_idx))
        scaled = self.lambdas[layer_pos].to(raw_update.device).float() * raw_update.float()
        scaled = scaled * valid.unsqueeze(-1).to(dtype=scaled.dtype)
        self.record_update_stats(int(layer_idx), layer_pos, scaled, valid)
        return scaled.to(dtype=hidden_states.dtype)

    def compute_raw_update(
        self,
        hidden_states: torch.Tensor,
        layer_pos: int,
        valid_mask: torch.Tensor,
        *,
        layer_idx: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def record_update_stats(
        self,
        layer_idx: int,
        layer_pos: int,
        scaled_update: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        norms = scaled_update.float().norm(dim=-1)
        per_sample: List[float] = []
        energies: List[torch.Tensor] = []
        for row in range(int(norms.shape[0])):
            row_mask = valid_mask[row].bool()
            if bool(row_mask.any().item()):
                per_sample.append(float(norms[row][row_mask].detach().mean().cpu().item()))
                energies.append(scaled_update[row][row_mask].float().pow(2).sum(dim=-1).mean())
            else:
                per_sample.append(0.0)
                energies.append(scaled_update[row].float().new_zeros(()))
        self._loss_update_energies.append(torch.stack(energies, dim=0).mean())
        self._last_stats["adapter_residual_norm_by_layer"][str(int(layer_idx))] = per_sample
        self._last_stats["lambda_by_layer"][str(int(layer_idx))] = [
            float(self.lambdas[layer_pos].detach().float().cpu().item())
            for _ in range(int(scaled_update.shape[0]))
        ]

    def update_energy_for_loss(self, device: torch.device) -> torch.Tensor:
        if not self._loss_update_energies:
            return torch.zeros((), device=device)
        return torch.stack([item.to(device) for item in self._loss_update_energies], dim=0).mean()

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
        out["adapter_residual_norm"] = finite_mean(out.get("adapter_residual_norm_by_layer", {}).values(), default=0.0)
        out["lambda_mean"] = finite_mean(out.get("lambda_by_layer", {}).values(), default=math.nan)
        return out

    def mean_lambda(self) -> float:
        return float(self.lambdas.detach().float().mean().cpu().item()) if self.lambdas.numel() else math.nan


class PerTokenMLPAdapter(AttentionOutputResidualAdapter):
    """Per-token residual MLP: controls for trainable parameters without token mixing."""

    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        inject_layers: Sequence[int],
        lambda_init: float,
        train_lambda: bool,
    ) -> None:
        super().__init__(
            hidden_size=int(hidden_size),
            inject_layers=inject_layers,
            lambda_init=float(lambda_init),
            train_lambda=bool(train_lambda),
        )
        self.rank = int(rank)
        self.down = nn.ModuleList([nn.Linear(self.hidden_size, self.rank, bias=False) for _ in self.inject_layers])
        self.up = nn.ModuleList([nn.Linear(self.rank, self.hidden_size, bias=False) for _ in self.inject_layers])
        for down, up in zip(self.down, self.up):
            nn.init.xavier_uniform_(down.weight, gain=0.5)
            nn.init.zeros_(up.weight)

    def compute_raw_update(
        self,
        hidden_states: torch.Tensor,
        layer_pos: int,
        valid_mask: torch.Tensor,
        *,
        layer_idx: int,
    ) -> torch.Tensor:
        del valid_mask, layer_idx
        x = hidden_states.float()
        return self.up[layer_pos](F.gelu(self.down[layer_pos](x))).float()


class GatedTokenMixerAdapter(AttentionOutputResidualAdapter):
    """Independent sigmoid-gated low-rank token message passing.

    For each selected layer and token q:
      Q = H Wq, K = H Wk, V = H Wv
      g[q, i] = sigmoid(Q[q] dot K[i] / sqrt(rank))
      M[q] = sum_i g[q, i] V[i] / sqrt(sum_i g[q, i] + eps)
      delta[q] = M[q] Wo
    The scaled delta is added in parallel to the frozen attention output.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        inject_layers: Sequence[int],
        lambda_init: float,
        train_lambda: bool,
    ) -> None:
        super().__init__(
            hidden_size=int(hidden_size),
            inject_layers=inject_layers,
            lambda_init=float(lambda_init),
            train_lambda=bool(train_lambda),
        )
        self.rank = int(rank)
        self.w_q = nn.ModuleList([nn.Linear(self.hidden_size, self.rank, bias=False) for _ in self.inject_layers])
        self.w_k = nn.ModuleList([nn.Linear(self.hidden_size, self.rank, bias=False) for _ in self.inject_layers])
        self.w_v = nn.ModuleList([nn.Linear(self.hidden_size, self.rank, bias=False) for _ in self.inject_layers])
        self.w_o = nn.ModuleList([nn.Linear(self.rank, self.hidden_size, bias=False) for _ in self.inject_layers])
        for modules in (self.w_q, self.w_k, self.w_v):
            for module in modules:
                nn.init.xavier_uniform_(module.weight, gain=0.05)
        for module in self.w_o:
            nn.init.normal_(module.weight, mean=0.0, std=1e-5)

    def set_context(self, batch: carrier.MemoryBatch, records: Sequence[prev.SampleRecord]) -> None:
        super().set_context(batch, records)
        self._last_stats.update(
            {
                "gate_mass_by_layer": {},
                "gate_mass_into_final_token_by_layer": {},
                "gate_mass_into_question_tokens_by_layer": {},
                "gate_mass_from_evidence_frame_tokens_by_layer": {},
                "gate_mass_from_distractor_frame_tokens_by_layer": {},
                "mean_gate_value_by_layer": {},
                "gate_diagnostic_note_by_layer": {},
            }
        )

    def compute_raw_update(
        self,
        hidden_states: torch.Tensor,
        layer_pos: int,
        valid_mask: torch.Tensor,
        *,
        layer_idx: int,
    ) -> torch.Tensor:
        x = hidden_states.float()
        q = self.w_q[layer_pos](x)
        k = self.w_k[layer_pos](x)
        v = self.w_v[layer_pos](x)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.rank))
        key_mask = valid_mask.unsqueeze(1)
        query_mask = valid_mask.unsqueeze(2)
        scores = scores.masked_fill(~key_mask, -30.0)
        gates = torch.sigmoid(scores) * key_mask.to(dtype=scores.dtype) * query_mask.to(dtype=scores.dtype)
        mass = gates.sum(dim=-1).clamp_min(EPS)
        mixed = torch.matmul(gates, v) / torch.sqrt(mass + EPS).unsqueeze(-1)
        self.record_gate_stats(int(layer_idx), gates.detach(), valid_mask)
        return self.w_o[layer_pos](mixed).float()

    @staticmethod
    def _mean_query_mass(mass_per_query: torch.Tensor, positions: Sequence[int], valid: torch.Tensor) -> float:
        seq_len = int(mass_per_query.shape[0])
        clean = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len and bool(valid[int(pos)].item())})
        if not clean:
            return math.nan
        idx = torch.tensor(clean, device=mass_per_query.device, dtype=torch.long)
        return float(mass_per_query.index_select(0, idx).float().mean().cpu().item())

    @staticmethod
    def _source_mass(gates: torch.Tensor, key_positions: Sequence[int], valid: torch.Tensor) -> float:
        seq_len = int(gates.shape[0])
        keys = sorted({int(pos) for pos in key_positions if 0 <= int(pos) < seq_len and bool(valid[int(pos)].item())})
        query_idx = valid.nonzero(as_tuple=True)[0]
        if not keys or int(query_idx.numel()) == 0:
            return math.nan
        key_idx = torch.tensor(keys, device=gates.device, dtype=torch.long)
        selected = gates.index_select(0, query_idx).index_select(1, key_idx).sum(dim=-1)
        return float(selected.float().mean().cpu().item())

    def record_gate_stats(self, layer_idx: int, gates: torch.Tensor, valid_mask: torch.Tensor) -> None:
        batch = int(gates.shape[0])
        mean_mass: List[float] = []
        final_mass: List[float] = []
        question_mass: List[float] = []
        evidence_source_mass: List[float] = []
        distractor_source_mass: List[float] = []
        mean_gate_value: List[float] = []
        notes: List[str] = []

        for row in range(batch):
            valid = valid_mask[row].detach().bool().cpu()
            gate = gates[row].float().cpu()
            mass_per_query = gate.sum(dim=-1)
            if bool(valid.any().item()):
                mean_mass.append(float(mass_per_query[valid].mean().item()))
                valid_gate = gate[valid][:, valid]
                mean_gate_value.append(float(valid_gate.mean().item()) if valid_gate.numel() else math.nan)
            else:
                mean_mass.append(math.nan)
                mean_gate_value.append(math.nan)

            final_pos = self._prompt_last_indices[row] if self._prompt_last_indices and row < len(self._prompt_last_indices) else -1
            final_mass.append(self._mean_query_mass(mass_per_query, [final_pos], valid))
            q_positions = self._query_positions[row] if self._query_positions and row < len(self._query_positions) else []
            question_mass.append(self._mean_query_mass(mass_per_query, q_positions, valid))

            frame_groups = self._frame_groups[row] if self._frame_groups and row < len(self._frame_groups) else []
            evidence_mask = self._evidence_masks[row] if self._evidence_masks and row < len(self._evidence_masks) else []
            if frame_groups and evidence_mask and len(evidence_mask) >= len(frame_groups):
                evidence_positions: List[int] = []
                distractor_positions: List[int] = []
                for frame_idx, group in enumerate(frame_groups):
                    if int(evidence_mask[frame_idx]):
                        evidence_positions.extend(int(pos) for pos in group)
                    else:
                        distractor_positions.extend(int(pos) for pos in group)
                evidence_source_mass.append(self._source_mass(gate, evidence_positions, valid))
                distractor_source_mass.append(self._source_mass(gate, distractor_positions, valid))
                notes.append("")
            else:
                evidence_source_mass.append(math.nan)
                distractor_source_mass.append(math.nan)
                notes.append("evidence/non-evidence frame token labels unavailable")

        key = str(int(layer_idx))
        self._last_stats["gate_mass_by_layer"][key] = mean_mass
        self._last_stats["gate_mass_into_final_token_by_layer"][key] = final_mass
        self._last_stats["gate_mass_into_question_tokens_by_layer"][key] = question_mass
        self._last_stats["gate_mass_from_evidence_frame_tokens_by_layer"][key] = evidence_source_mass
        self._last_stats["gate_mass_from_distractor_frame_tokens_by_layer"][key] = distractor_source_mass
        self._last_stats["mean_gate_value_by_layer"][key] = mean_gate_value
        self._last_stats["gate_diagnostic_note_by_layer"][key] = notes

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        out = super().stats_for_row(row)
        aggregate_keys = {
            "gate_mass_by_layer": "gate_mass",
            "gate_mass_into_final_token_by_layer": "gate_mass_into_final_token",
            "gate_mass_into_question_tokens_by_layer": "gate_mass_into_question_tokens",
            "gate_mass_from_evidence_frame_tokens_by_layer": "gate_mass_from_evidence_frame_tokens",
            "gate_mass_from_distractor_frame_tokens_by_layer": "gate_mass_from_distractor_frame_tokens",
            "mean_gate_value_by_layer": "mean_gate_value",
        }
        for layer_key, aggregate_key in aggregate_keys.items():
            out[aggregate_key] = finite_mean(out.get(layer_key, {}).values(), default=math.nan)
        notes = [str(value) for value in out.get("gate_diagnostic_note_by_layer", {}).values() if str(value)]
        out["gate_diagnostic_note"] = "; ".join(sorted(set(notes)))
        return out


class LoRALinearWrapper(nn.Module):
    """Small LoRA module that keeps the frozen base layer out of adapter checkpoints."""

    def __init__(self, base_layer: nn.Module, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        object.__setattr__(self, "base_layer", base_layer)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(1, int(rank))
        self.dropout = float(dropout)
        in_features = getattr(base_layer, "in_features", None)
        out_features = getattr(base_layer, "out_features", None)
        weight = getattr(base_layer, "weight", None)
        if in_features is None and torch.is_tensor(weight):
            in_features = int(weight.shape[1])
        if out_features is None and torch.is_tensor(weight):
            out_features = int(weight.shape[0])
        if in_features is None or out_features is None:
            raise ValueError(f"Cannot infer LoRA dimensions for {type(base_layer).__name__}")
        device = weight.device if torch.is_tensor(weight) else torch.device("cpu")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_A = nn.Parameter(torch.empty((self.rank, self.in_features), device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty((self.out_features, self.rank), device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_B)
        self.last_delta_norm: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_float = x.float()
        if self.dropout > 0.0:
            x_float = F.dropout(x_float, p=float(self.dropout), training=self.training)
        delta = F.linear(F.linear(x_float, self.lora_A), self.lora_B) * float(self.scaling)
        if delta.dim() >= 3:
            self.last_delta_norm = [
                float(v)
                for v in delta.detach().float().norm(dim=-1).mean(dim=-1).cpu().tolist()
            ]
        elif delta.dim() == 2:
            self.last_delta_norm = [float(v) for v in delta.detach().float().norm(dim=-1).cpu().tolist()]
        else:
            self.last_delta_norm = []
        return base_out + delta.to(dtype=base_out.dtype)


class MinimalAttentionLoRAAdapter(nn.Module):
    def __init__(
        self,
        *,
        inject_layers: Sequence[int],
        rank: int,
        alpha: float,
        dropout: float,
        target_modules: Sequence[str],
    ) -> None:
        super().__init__()
        self.inject_layers = [int(layer) for layer in inject_layers]
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.target_modules = [str(name) for name in target_modules]
        self.wrappers = nn.ModuleList()
        self._wrapped: List[Tuple[Any, str, nn.Module, LoRALinearWrapper, int]] = []
        self._last_stats: Dict[str, Any] = {}

    def attach(self, model: Any) -> None:
        self.detach()
        layers = prev.get_layers(model)
        for layer_idx in self.inject_layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"layer={layer_idx} outside [0, {len(layers) - 1}]")
            attn = getattr(layers[int(layer_idx)], "self_attn", None)
            if attn is None:
                raise RuntimeError(f"layer={layer_idx} does not expose self_attn")
            for name in self.target_modules:
                base_layer = getattr(attn, name, None)
                if base_layer is None:
                    raise RuntimeError(f"layer={layer_idx}.self_attn does not expose {name}")
                wrapper = LoRALinearWrapper(
                    base_layer,
                    rank=int(self.rank),
                    alpha=float(self.alpha),
                    dropout=float(self.dropout),
                )
                setattr(attn, name, wrapper)
                self.wrappers.append(wrapper)
                self._wrapped.append((attn, name, base_layer, wrapper, int(layer_idx)))

    def detach(self) -> None:
        for parent, name, original, _wrapper, _layer in reversed(self._wrapped):
            setattr(parent, name, original)
        self._wrapped = []
        self.wrappers = nn.ModuleList()

    def set_context(self, batch: carrier.MemoryBatch, records: Sequence[prev.SampleRecord]) -> None:
        del batch, records
        for wrapper in self.wrappers:
            wrapper.last_delta_norm = []
        self._last_stats = {}

    def clear_context(self) -> None:
        pass

    def update_energy_for_loss(self, device: torch.device) -> torch.Tensor:
        return torch.zeros((), device=device)

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        layer_values: Dict[str, List[float]] = defaultdict(list)
        for _parent, name, _original, wrapper, layer_idx in self._wrapped:
            if row < len(wrapper.last_delta_norm):
                layer_values[str(int(layer_idx))].append(float(wrapper.last_delta_norm[row]))
        by_layer = {
            layer: finite_mean(values, default=0.0)
            for layer, values in layer_values.items()
        }
        return {
            "adapter_residual_norm": finite_mean(by_layer.values(), default=0.0),
            "adapter_residual_norm_by_layer": by_layer,
            "lambda_mean": math.nan,
            "lambda_by_layer": {},
            "lora_delta_norm_by_layer": by_layer,
        }

    def mean_lambda(self) -> float:
        return math.nan


def make_adapter(
    *,
    method: str,
    args: argparse.Namespace,
    hidden_size: int,
    layers: Sequence[int],
) -> Optional[nn.Module]:
    if method == MLP_ADAPTER:
        return PerTokenMLPAdapter(
            hidden_size=int(hidden_size),
            rank=int(args.rank),
            inject_layers=layers,
            lambda_init=float(args.lambda_init),
            train_lambda=bool(args.train_lambda),
        )
    if method == GATED_TOKEN_MIXER:
        return GatedTokenMixerAdapter(
            hidden_size=int(hidden_size),
            rank=int(args.rank),
            inject_layers=layers,
            lambda_init=float(args.lambda_init),
            train_lambda=bool(args.train_lambda),
        )
    if method == LORA_ATTENTION:
        targets = split_tokens(args.lora_targets)
        return MinimalAttentionLoRAAdapter(
            inject_layers=layers,
            rank=int(args.rank),
            alpha=float(args.lora_alpha if args.lora_alpha is not None else args.rank),
            dropout=float(args.lora_dropout),
            target_modules=targets,
        )
    return None


def attach_adapter(adapter: Optional[nn.Module], model: Any, device: str) -> None:
    if adapter is None:
        return
    if isinstance(adapter, MinimalAttentionLoRAAdapter):
        adapter.attach(model)
        adapter.to(device)
    elif isinstance(adapter, AttentionOutputResidualAdapter):
        adapter.to(device)
        adapter.attach(model)
    else:
        raise TypeError(f"Unknown adapter type: {type(adapter).__name__}")


def detach_adapter(adapter: Optional[nn.Module]) -> None:
    if adapter is None:
        return
    if hasattr(adapter, "detach"):
        adapter.detach()  # type: ignore[misc]


def adapter_set_context(adapter: Optional[nn.Module], batch: carrier.MemoryBatch, records: Sequence[prev.SampleRecord]) -> None:
    if adapter is not None and hasattr(adapter, "set_context"):
        adapter.set_context(batch, records)  # type: ignore[misc]


def adapter_clear_context(adapter: Optional[nn.Module]) -> None:
    if adapter is not None and hasattr(adapter, "clear_context"):
        adapter.clear_context()  # type: ignore[misc]


def adapter_update_energy(adapter: Optional[nn.Module], device: torch.device) -> torch.Tensor:
    if adapter is None or not hasattr(adapter, "update_energy_for_loss"):
        return torch.zeros((), device=device)
    return adapter.update_energy_for_loss(device)  # type: ignore[misc]


def adapter_stats_for_row(adapter: Optional[nn.Module], row: int) -> Dict[str, Any]:
    if adapter is None or not hasattr(adapter, "stats_for_row"):
        return {
            "adapter_residual_norm": 0.0,
            "adapter_residual_norm_by_layer": {},
            "lambda_mean": math.nan,
            "lambda_by_layer": {},
        }
    return adapter.stats_for_row(row)  # type: ignore[misc]


def adapter_mean_lambda(adapter: Optional[nn.Module]) -> float:
    if adapter is None or not hasattr(adapter, "mean_lambda"):
        return math.nan
    return float(adapter.mean_lambda())  # type: ignore[misc]


def select_logits_and_scores(
    outputs: Any,
    prompt_last_indices: torch.Tensor,
    count_token_ids: Dict[int, int],
) -> torch.Tensor:
    return prev.select_count_logits(outputs.logits, prompt_last_indices, count_token_ids)


def candidate_logits_json(count_logits: torch.Tensor, count_values: Sequence[int]) -> str:
    values = [float(v) for v in count_logits.detach().float().cpu().tolist()]
    return json_compact({str(int(count)): values[pos] for pos, count in enumerate(count_values) if pos < len(values)})


@torch.no_grad()
def evaluate_split(
    *,
    method: str,
    split_name: str,
    model: Any,
    processor: Any,
    adapter: Optional[nn.Module],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    count_token_ids: Dict[int, int],
    count_values: Sequence[int],
    device: str,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    model.eval()
    if adapter is not None:
        adapter.eval()
    rows: List[Dict[str, Any]] = []
    ce_total = 0.0
    n = 0
    count_min = min(int(x) for x in count_values)
    batches = homogeneous_batches(indices, records, int(batch_size), seed=int(seed), shuffle_batches=False)
    for batch_num, batch_indices in enumerate(batches, start=1):
        batch_records = [records[int(idx)] for idx in batch_indices]
        batch = prepare_batch(records=batch_records, sample_indices=batch_indices, processor=processor, device=device)
        adapter_set_context(adapter, batch, batch_records)
        outputs = model(**batch.inputs, use_cache=False)
        count_logits = select_logits_and_scores(outputs, batch.prompt_last_indices, count_token_ids)
        gold_offsets = batch.gold_counts.long() - int(count_min)
        ce_vec = F.cross_entropy(count_logits, gold_offsets, reduction="none")
        ce_total += float(ce_vec.sum().detach().cpu().item())
        n += int(batch.gold_counts.numel())
        pred_offsets = count_logits.argmax(dim=-1)
        gold_logits, best_non_gold_logits, margins = carrier.select_gold_logits_and_margins(count_logits, gold_offsets)

        for row, idx in enumerate(batch_indices):
            idx = int(idx)
            record = records[idx]
            pred = int(pred_offsets[row].detach().cpu().item()) + int(count_min)
            gold = int(record.gold_count)
            stats = adapter_stats_for_row(adapter, row)
            row_payload: Dict[str, Any] = {
                "method": str(method),
                "split": str(split_name),
                "sample_id": record.sample_id,
                "sample_index": int(idx),
                "seq_len": len(record.frame_paths),
                "true_count": int(gold),
                "gold_count": int(gold),
                "evidence_count": int(record.evidence_count),
                "pred_count": int(pred),
                "correct": int(pred == gold),
                "abs_error": abs(int(pred) - int(gold)),
                "ce": float(ce_vec[row].detach().cpu().item()),
                "gold_logit": float(gold_logits[row].detach().cpu().item()),
                "gold_score": float(gold_logits[row].detach().cpu().item()),
                "best_non_gold_logit": float(best_non_gold_logits[row].detach().cpu().item()),
                "best_non_gold_score": float(best_non_gold_logits[row].detach().cpu().item()),
                "gold_margin": float(margins[row].detach().cpu().item()),
                "gold_vs_best_nongold_margin": float(margins[row].detach().cpu().item()),
                "candidate_logits_json": candidate_logits_json(count_logits[row], count_values),
                "adapter_residual_norm": float(finite_float(stats.get("adapter_residual_norm")) or 0.0),
                "adapter_residual_norm_by_layer_json": json_compact(stats.get("adapter_residual_norm_by_layer", {})),
                "lambda_mean": "" if finite_float(stats.get("lambda_mean")) is None else float(stats.get("lambda_mean")),
                "lambda_by_layer_json": json_compact(stats.get("lambda_by_layer", {})),
                "token_selection_ok": int(bool(batch.token_selection_ok[row])),
                "token_selection_error": str(batch.token_selection_errors[row]),
                "frame_grouping_ok": int(bool(batch.frame_grouping_ok[row])),
                "frame_grouping_error": str(batch.frame_grouping_errors[row]),
                "frame_token_counts_json": json_compact(batch.frame_token_counts[row]),
                "question_positions_json": json_compact(batch.query_positions[row]),
                "final_token_index": int(batch.prompt_last_indices[row].detach().cpu().item()),
                "evidence_frame_mask_json": json_compact(carrier.evidence_frame_mask(record, len(record.frame_paths))),
            }
            if method == GATED_TOKEN_MIXER:
                for key in (
                    "gate_mass",
                    "gate_mass_into_final_token",
                    "gate_mass_into_question_tokens",
                    "gate_mass_from_evidence_frame_tokens",
                    "gate_mass_from_distractor_frame_tokens",
                    "mean_gate_value",
                    "gate_diagnostic_note",
                ):
                    row_payload[key] = stats.get(key, "")
                for key in (
                    "gate_mass_by_layer",
                    "gate_mass_into_final_token_by_layer",
                    "gate_mass_into_question_tokens_by_layer",
                    "gate_mass_from_evidence_frame_tokens_by_layer",
                    "gate_mass_from_distractor_frame_tokens_by_layer",
                    "mean_gate_value_by_layer",
                    "gate_diagnostic_note_by_layer",
                ):
                    row_payload[f"{key}_json"] = json_compact(stats.get(key, {}))
            rows.append(row_payload)
        adapter_clear_context(adapter)
        if batch_num == 1 or batch_num % 25 == 0:
            print(f"  eval {method} {split_name}: {min(len(indices), batch_num * int(batch_size))}/{len(indices)}")
    return {
        "rows": rows,
        "ce": ce_total / max(1, n),
        "accuracy": accuracy([int(row["true_count"]) for row in rows], [int(row["pred_count"]) for row in rows]),
    }


def train_adapter(
    *,
    method: str,
    args: argparse.Namespace,
    output_dir: Path,
    model: Any,
    processor: Any,
    adapter: nn.Module,
    records: Sequence[prev.SampleRecord],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    count_token_ids: Dict[int, int],
    count_values: Sequence[int],
    device: str,
) -> Tuple[List[Dict[str, Any]], Path]:
    trainable = [param for param in adapter.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError(f"{method}: adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"{method}_best.pt"
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_acc = -math.inf
    best_val_ce = math.inf
    count_min = min(int(x) for x in count_values)

    for epoch in range(1, int(args.epochs) + 1):
        model.eval()
        adapter.train()
        train_batches = homogeneous_batches(
            train_indices,
            records,
            int(args.batch_size),
            seed=int(args.seed) + epoch * 9973,
            shuffle_batches=True,
        )
        optimizer.zero_grad(set_to_none=True)
        train_ce_total = 0.0
        train_loss_total = 0.0
        train_correct = 0
        train_n = 0
        train_steps = 0
        backward_steps = 0
        for step, batch_indices in enumerate(train_batches, start=1):
            batch_records = [records[int(idx)] for idx in batch_indices]
            batch = prepare_batch(records=batch_records, sample_indices=batch_indices, processor=processor, device=device)
            adapter_set_context(adapter, batch, batch_records)
            outputs = model(**batch.inputs, use_cache=False)
            count_logits = select_logits_and_scores(outputs, batch.prompt_last_indices, count_token_ids)
            gold_offsets = batch.gold_counts.long() - int(count_min)
            ce = F.cross_entropy(count_logits, gold_offsets)
            margin = carrier.margin_loss(count_logits, gold_offsets, float(args.margin_target))
            update_energy = adapter_update_energy(adapter, count_logits.device)
            loss = ce + float(args.lambda_margin) * margin + float(args.lambda_update_energy) * update_energy
            torch.autograd.backward(loss / max(1, int(args.grad_accum)))

            preds = count_logits.argmax(dim=-1) + int(count_min)
            train_correct += int((preds == batch.gold_counts.long()).sum().detach().cpu().item())
            train_n += int(batch.gold_counts.numel())
            train_ce_total += float(ce.detach().cpu().item())
            train_loss_total += float(loss.detach().cpu().item())
            train_steps += 1
            backward_steps += 1
            adapter_clear_context(adapter)

            if backward_steps % max(1, int(args.grad_accum)) == 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 25 == 0:
                print(
                    f"  {method} epoch={epoch} step={step}/{len(train_batches)} "
                    f"train_ce={train_ce_total / max(1, train_steps):.4f} "
                    f"train_acc={train_correct / max(1, train_n):.4f}"
                )

        if backward_steps and backward_steps % max(1, int(args.grad_accum)) != 0:
            torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        val_eval = evaluate_split(
            method=method,
            split_name="val",
            model=model,
            processor=processor,
            adapter=adapter,
            records=records,
            indices=val_indices,
            count_token_ids=count_token_ids,
            count_values=count_values,
            device=device,
            batch_size=int(args.batch_size),
            seed=int(args.seed) + 444 + epoch,
        )
        row = {
            "method": method,
            "epoch": int(epoch),
            "train_ce": train_ce_total / max(1, train_steps),
            "train_loss": train_loss_total / max(1, train_steps),
            "train_accuracy": train_correct / max(1, train_n),
            "train_steps": int(train_steps),
            "val_ce": float(val_eval["ce"]),
            "val_accuracy": float(val_eval["accuracy"]),
            "adapter_parameter_norm": float(
                math.sqrt(
                    sum(
                        float(param.detach().float().pow(2).sum().cpu().item())
                        for param in adapter.parameters()
                        if param.requires_grad
                    )
                )
            ),
            "lambda_mean": adapter_mean_lambda(adapter),
        }
        history.append(row)
        print(
            f"  {method} epoch={epoch} train_ce={row['train_ce']:.4f} "
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
                    "method": method,
                    "rank": int(args.rank),
                    "layers": [int(x) for x in getattr(adapter, "inject_layers", [])],
                    "lambda_init": float(args.lambda_init),
                    "train_lambda": bool(args.train_lambda),
                    "note": "Adapter-only checkpoint; frozen Qwen weights are not stored.",
                },
                checkpoint_path,
            )
    if best_state is not None:
        adapter.load_state_dict(best_state)
    return history, checkpoint_path


def rows_for_split(rows: Sequence[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get("split")) == str(split_name)]


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [int(row["true_count"]) for row in rows]
    y_pred = [int(row["pred_count"]) for row in rows]
    return {
        "n": len(rows),
        "accuracy": accuracy(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mean_predicted_count": finite_mean((row.get("pred_count") for row in rows), default=math.nan),
        "mean_gold_logit": finite_mean((row.get("gold_logit") for row in rows), default=math.nan),
        "mean_gold_score": finite_mean((row.get("gold_score") for row in rows), default=math.nan),
        "mean_best_non_gold_logit": finite_mean((row.get("best_non_gold_logit") for row in rows), default=math.nan),
        "mean_best_non_gold_score": finite_mean((row.get("best_non_gold_score") for row in rows), default=math.nan),
        "mean_gold_margin": finite_mean((row.get("gold_margin") for row in rows), default=math.nan),
        "mean_adapter_residual_norm": finite_mean((row.get("adapter_residual_norm") for row in rows), default=0.0),
        "mean_lambda": finite_mean((row.get("lambda_mean") for row in rows), default=math.nan),
    }


def build_per_count_metrics(
    rows: Sequence[Dict[str, Any]],
    *,
    methods: Sequence[str],
    count_values: Sequence[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for method in methods:
        for split_name in ("train", "val", "test"):
            split_rows = [row for row in rows if row.get("method") == method and row.get("split") == split_name]
            for count in count_values:
                count_rows = [row for row in split_rows if int(row.get("true_count", -999)) == int(count)]
                summary = summarize_rows(count_rows)
                out.append(
                    {
                        "method": method,
                        "split": split_name,
                        "true_count": int(count),
                        "evidence_count": int(count),
                        **summary,
                    }
                )
    return out


def build_per_seq_len_metrics(
    rows: Sequence[Dict[str, Any]],
    *,
    methods: Sequence[str],
    seq_lens: Sequence[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for method in methods:
        for split_name in ("train", "val", "test"):
            split_rows = [row for row in rows if row.get("method") == method and row.get("split") == split_name]
            for seq_len in seq_lens:
                seq_rows = [row for row in split_rows if int(row.get("seq_len", -999)) == int(seq_len)]
                summary = summarize_rows(seq_rows)
                out.append(
                    {
                        "method": method,
                        "split": split_name,
                        "seq_len": int(seq_len),
                        "gold_count": int(seq_len),
                        "evidence_count": int(seq_len),
                        **summary,
                    }
                )
    return out


def prediction_histogram(rows: Sequence[Dict[str, Any]], count_values: Sequence[int]) -> Dict[str, int]:
    hist = {str(int(count)): 0 for count in count_values}
    for row in rows:
        key = str(int(row["pred_count"]))
        hist[key] = hist.get(key, 0) + 1
    return hist


def build_method_summaries(
    *,
    rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    count_values: Sequence[int],
    trainable_counts: Dict[str, int],
    checkpoint_paths: Dict[str, str],
    histories: Dict[str, List[Dict[str, Any]]],
    skipped: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metrics_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"methods": {}}
    for method in methods:
        if method in skipped:
            row = {
                "method": method,
                "status": "skipped",
                "skip_reason": skipped[method],
                "trainable_parameters": 0,
            }
            metrics_rows.append(row)
            summary["methods"][method] = row
            continue
        method_rows = [row for row in rows if row.get("method") == method]
        train_summary = summarize_rows(rows_for_split(method_rows, "train"))
        val_summary = summarize_rows(rows_for_split(method_rows, "val"))
        test_rows = rows_for_split(method_rows, "test")
        test_summary = summarize_rows(test_rows)
        high_rows = [row for row in test_rows if int(row.get("true_count", -999)) >= 5]
        per_count_accuracy: Dict[str, float] = {}
        mean_pred_by_count: Dict[str, float] = {}
        for count in count_values:
            count_rows = [row for row in test_rows if int(row.get("true_count", -999)) == int(count)]
            count_summary = summarize_rows(count_rows)
            per_count_accuracy[str(int(count))] = float(count_summary["accuracy"]) if finite_float(count_summary["accuracy"]) is not None else math.nan
            mean_pred_by_count[str(int(count))] = (
                float(count_summary["mean_predicted_count"])
                if finite_float(count_summary["mean_predicted_count"]) is not None
                else math.nan
            )
        train_last = histories.get(method, [])[-1] if histories.get(method) else {}
        row = {
            "method": method,
            "status": "ok",
            "trainable_parameters": int(trainable_counts.get(method, 0)),
            "checkpoint": checkpoint_paths.get(method, ""),
            "train_accuracy": train_summary["accuracy"],
            "val_accuracy": val_summary["accuracy"],
            "test_accuracy": test_summary["accuracy"],
            "test_mae": test_summary["mae"],
            "test_high_count_accuracy_k_ge_5": summarize_rows(high_rows)["accuracy"],
            "test_mean_gold_logit": test_summary["mean_gold_logit"],
            "test_mean_gold_score": test_summary["mean_gold_score"],
            "test_mean_best_non_gold_logit": test_summary["mean_best_non_gold_logit"],
            "test_mean_best_non_gold_score": test_summary["mean_best_non_gold_score"],
            "test_mean_gold_margin": test_summary["mean_gold_margin"],
            "test_mean_adapter_residual_norm": test_summary["mean_adapter_residual_norm"],
            "mean_lambda": test_summary["mean_lambda"],
            "per_count_accuracy_json": json_compact(per_count_accuracy),
            "mean_predicted_count_per_true_count_json": json_compact(mean_pred_by_count),
            "prediction_histogram_json": json_compact(prediction_histogram(test_rows, count_values)),
            "last_train_loss": train_last.get("train_loss", ""),
            "last_val_ce": train_last.get("val_ce", ""),
        }
        metrics_rows.append(row)
        summary["methods"][method] = {
            **row,
            "per_count_accuracy": per_count_accuracy,
            "mean_predicted_count_per_true_count": mean_pred_by_count,
            "train": train_summary,
            "val": val_summary,
            "test": test_summary,
            "history": histories.get(method, []),
        }
    return metrics_rows, summary


def build_gate_diagnostics(rows: Sequence[Dict[str, Any]], count_values: Sequence[int]) -> List[Dict[str, Any]]:
    gated_rows = [row for row in rows if row.get("method") == GATED_TOKEN_MIXER]
    out: List[Dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        split_rows = [row for row in gated_rows if row.get("split") == split_name]
        all_summary = {
            "method": GATED_TOKEN_MIXER,
            "split": split_name,
            "true_count": "all",
            "evidence_count": "all",
            "n": len(split_rows),
            "mean_gate_mass": finite_mean((row.get("gate_mass") for row in split_rows), default=math.nan),
            "gate_mass_into_final_token": finite_mean((row.get("gate_mass_into_final_token") for row in split_rows), default=math.nan),
            "gate_mass_into_question_tokens": finite_mean((row.get("gate_mass_into_question_tokens") for row in split_rows), default=math.nan),
            "gate_mass_from_evidence_frame_tokens": finite_mean((row.get("gate_mass_from_evidence_frame_tokens") for row in split_rows), default=math.nan),
            "gate_mass_from_distractor_frame_tokens": finite_mean((row.get("gate_mass_from_distractor_frame_tokens") for row in split_rows), default=math.nan),
            "mean_gate_value": finite_mean((row.get("mean_gate_value") for row in split_rows), default=math.nan),
            "note": "; ".join(sorted({str(row.get("gate_diagnostic_note", "")) for row in split_rows if str(row.get("gate_diagnostic_note", ""))})),
        }
        out.append(all_summary)
        for count in count_values:
            count_rows = [row for row in split_rows if int(row.get("true_count", -999)) == int(count)]
            out.append(
                {
                    **all_summary,
                    "true_count": int(count),
                    "evidence_count": int(count),
                    "n": len(count_rows),
                    "mean_gate_mass": finite_mean((row.get("gate_mass") for row in count_rows), default=math.nan),
                    "gate_mass_into_final_token": finite_mean((row.get("gate_mass_into_final_token") for row in count_rows), default=math.nan),
                    "gate_mass_into_question_tokens": finite_mean((row.get("gate_mass_into_question_tokens") for row in count_rows), default=math.nan),
                    "gate_mass_from_evidence_frame_tokens": finite_mean((row.get("gate_mass_from_evidence_frame_tokens") for row in count_rows), default=math.nan),
                    "gate_mass_from_distractor_frame_tokens": finite_mean((row.get("gate_mass_from_distractor_frame_tokens") for row in count_rows), default=math.nan),
                    "mean_gate_value": finite_mean((row.get("mean_gate_value") for row in count_rows), default=math.nan),
                    "note": "; ".join(sorted({str(row.get("gate_diagnostic_note", "")) for row in count_rows if str(row.get("gate_diagnostic_note", ""))})),
                }
            )
    return out


def plot_line_by_count(
    path: Path,
    per_count_rows: Sequence[Dict[str, Any]],
    *,
    y_key: str,
    ylabel: str,
    title: str,
    methods: Sequence[str],
) -> None:
    test_rows = [row for row in per_count_rows if row.get("split") == "test"]
    plt.figure(figsize=(7.4, 4.8))
    for method in methods:
        rows = sorted(
            [row for row in test_rows if row.get("method") == method],
            key=lambda row: int(row["true_count"]),
        )
        if not rows:
            continue
        xs = [int(row["true_count"]) for row in rows]
        ys = [float(row.get(y_key, math.nan)) for row in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=method)
    plt.xlabel("True evidence count")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted({int(row["true_count"]) for row in test_rows if str(row["true_count"]).isdigit()}))
    plt.grid(alpha=0.25)
    handles, _labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_accuracy_heatmap(path: Path, per_count_rows: Sequence[Dict[str, Any]], methods: Sequence[str], count_values: Sequence[int]) -> None:
    mat = np.full((len(methods), len(count_values)), np.nan, dtype=float)
    by_key = {
        (row.get("method"), int(row["true_count"])): row
        for row in per_count_rows
        if row.get("split") == "test" and str(row.get("true_count", "")).lstrip("-").isdigit()
    }
    for i, method in enumerate(methods):
        for j, count in enumerate(count_values):
            row = by_key.get((method, int(count)))
            if row is not None and finite_float(row.get("accuracy")) is not None:
                mat[i, j] = float(row["accuracy"])
    fig, ax = plt.subplots(figsize=(8.5, max(2.8, 0.45 * len(methods) + 1.2)))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(count_values)))
    ax.set_xticklabels([str(int(x)) for x in count_values])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("True evidence count")
    ax.set_title("Test Accuracy By Method And Count")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if math.isfinite(float(mat[i, j])):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if mat[i, j] < 0.55 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gate_mass(path: Path, gate_rows: Sequence[Dict[str, Any]]) -> None:
    rows = sorted(
        [
            row
            for row in gate_rows
            if row.get("split") == "test" and str(row.get("true_count", "")).lstrip("-").isdigit()
        ],
        key=lambda row: int(row["true_count"]),
    )
    if not rows:
        return
    xs = [int(row["true_count"]) for row in rows]
    plt.figure(figsize=(7.4, 4.8))
    for key, label in (
        ("mean_gate_mass", "all tokens"),
        ("gate_mass_into_final_token", "final token"),
        ("gate_mass_into_question_tokens", "question tokens"),
        ("gate_mass_from_evidence_frame_tokens", "from evidence frames"),
        ("gate_mass_from_distractor_frame_tokens", "from distractor frames"),
    ):
        ys = [float(row.get(key, math.nan)) for row in rows]
        if any(math.isfinite(v) for v in ys):
            plt.plot(xs, ys, marker="o", linewidth=1.6, label=label)
    plt.xlabel("True evidence count")
    plt.ylabel("Mean gate mass")
    plt.title("Gated Adapter Gate Mass By Count")
    plt.xticks(xs)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_residual_norm(path: Path, metrics_rows: Sequence[Dict[str, Any]]) -> None:
    rows = [row for row in metrics_rows if row.get("status") == "ok"]
    if not rows:
        return
    labels = [str(row["method"]) for row in rows]
    values = [float(row.get("test_mean_adapter_residual_norm", 0.0) or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(np.arange(len(labels)), values, color="#4c78a8")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean adapter residual norm")
    ax.set_title("Adapter Residual Norm By Method")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    *,
    output_dir: Path,
    metrics_rows: Sequence[Dict[str, Any]],
    per_count_rows: Sequence[Dict[str, Any]],
    gate_rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    count_values: Sequence[int],
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    runnable_methods = [row["method"] for row in metrics_rows if row.get("status") == "ok"]
    ordered_methods = [method for method in methods if method in runnable_methods]
    plot_line_by_count(
        plots_dir / "accuracy_by_count.png",
        per_count_rows,
        y_key="accuracy",
        ylabel="Accuracy",
        title="Test Accuracy By Evidence Count",
        methods=ordered_methods,
    )
    plot_line_by_count(
        plots_dir / "mean_predicted_count_by_true_count.png",
        per_count_rows,
        y_key="mean_predicted_count",
        ylabel="Mean predicted count",
        title="Mean Predicted Count By True Count",
        methods=ordered_methods,
    )
    plot_accuracy_heatmap(plots_dir / "method_by_count_accuracy_heatmap.png", per_count_rows, ordered_methods, count_values)
    plot_line_by_count(
        plots_dir / "gold_margin_by_count.png",
        per_count_rows,
        y_key="mean_gold_margin",
        ylabel="Gold vs best non-gold margin",
        title="Gold Margin By Evidence Count",
        methods=ordered_methods,
    )
    if gate_rows:
        plot_gate_mass(plots_dir / "gated_adapter_gate_mass_by_count.png", gate_rows)
    plot_residual_norm(plots_dir / "adapter_residual_norm_by_method.png", metrics_rows)


def save_all_outputs(
    *,
    output_dir: Path,
    config: Dict[str, Any],
    all_rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    count_values: Sequence[int],
    seq_lens: Sequence[int],
    trainable_counts: Dict[str, int],
    checkpoint_paths: Dict[str, str],
    histories: Dict[str, List[Dict[str, Any]]],
    skipped: Dict[str, str],
    splits: Dict[str, List[int]],
    no_plots: bool,
) -> Dict[str, Any]:
    per_count_rows = build_per_count_metrics(all_rows, methods=methods, count_values=count_values)
    per_seq_len_rows = build_per_seq_len_metrics(all_rows, methods=methods, seq_lens=seq_lens) if seq_lens else []
    gate_rows = build_gate_diagnostics(all_rows, count_values)
    metrics_rows, summary = build_method_summaries(
        rows=all_rows,
        methods=methods,
        count_values=count_values,
        trainable_counts=trainable_counts,
        checkpoint_paths=checkpoint_paths,
        histories=histories,
        skipped=skipped,
    )
    summary.update(
        {
            "experiment_name": config.get("experiment_name", EXPERIMENT_NAME),
            "config": config,
            "splits": {split: [int(idx) for idx in indices] for split, indices in splits.items()},
            "split_sizes": {split: len(indices) for split, indices in splits.items()},
            "seq_lens": [int(x) for x in seq_lens],
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    write_json(output_dir / "summary.json", summary)
    write_csv_dynamic(
        output_dir / "metrics.csv",
        metrics_rows,
        leading=[
            "method",
            "status",
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
            "test_mae",
            "test_high_count_accuracy_k_ge_5",
            "trainable_parameters",
        ],
    )
    write_csv_dynamic(
        output_dir / "per_count_metrics.csv",
        per_count_rows,
        leading=["method", "split", "true_count", "evidence_count", "n", "accuracy", "mae", "mean_predicted_count"],
    )
    if per_seq_len_rows:
        write_csv_dynamic(
            output_dir / "accuracy_by_seq_len.csv",
            per_seq_len_rows,
            leading=["method", "split", "seq_len", "gold_count", "evidence_count", "n", "accuracy", "mae", "mean_predicted_count"],
        )
    write_csv_dynamic(
        output_dir / "gate_diagnostics.csv",
        gate_rows,
        leading=["method", "split", "true_count", "evidence_count", "n", "mean_gate_mass"],
    )
    write_csv_dynamic(
        output_dir / "per_sample_metrics.csv",
        all_rows,
        leading=["method", "split", "sample_id", "sample_index", "true_count", "pred_count", "correct"],
    )
    if not bool(no_plots):
        make_plots(
            output_dir=output_dir,
            metrics_rows=metrics_rows,
            per_count_rows=per_count_rows,
            gate_rows=gate_rows,
            methods=methods,
            count_values=count_values,
        )
    return summary


def run_method(
    *,
    method: str,
    args: argparse.Namespace,
    output_dir: Path,
    model: Any,
    processor: Any,
    records: Sequence[prev.SampleRecord],
    splits: Dict[str, List[int]],
    count_token_ids: Dict[int, int],
    count_values: Sequence[int],
    layers: Sequence[int],
    hidden_size: int,
    device: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, str]:
    if method == FROZEN_QWEN:
        print("Evaluating frozen Qwen baseline.")
        trainable = verify_trainable_scope(model, None, method)
        all_rows: List[Dict[str, Any]] = []
        for split_name in ("train", "val", "test"):
            payload = evaluate_split(
                method=method,
                split_name=split_name,
                model=model,
                processor=processor,
                adapter=None,
                records=records,
                indices=splits[split_name],
                count_token_ids=count_token_ids,
                count_values=count_values,
                device=device,
                batch_size=int(args.batch_size),
                seed=int(args.seed) + {"train": 11, "val": 22, "test": 33}[split_name],
            )
            all_rows.extend(payload["rows"])
        return all_rows, [], trainable, ""

    adapter = make_adapter(method=method, args=args, hidden_size=int(hidden_size), layers=layers)
    if adapter is None:
        raise RuntimeError(f"No adapter implementation for method={method}")
    try:
        attach_adapter(adapter, model, device)
        trainable = verify_trainable_scope(model, adapter, method)
        history, checkpoint_path = train_adapter(
            method=method,
            args=args,
            output_dir=output_dir,
            model=model,
            processor=processor,
            adapter=adapter,
            records=records,
            train_indices=splits["train"],
            val_indices=splits["val"],
            count_token_ids=count_token_ids,
            count_values=count_values,
            device=device,
        )
        all_rows = []
        for split_name in ("train", "val", "test"):
            payload = evaluate_split(
                method=method,
                split_name=split_name,
                model=model,
                processor=processor,
                adapter=adapter,
                records=records,
                indices=splits[split_name],
                count_token_ids=count_token_ids,
                count_values=count_values,
                device=device,
                batch_size=int(args.batch_size),
                seed=int(args.seed) + {"train": 101, "val": 202, "test": 303}[split_name],
            )
            all_rows.extend(payload["rows"])
        return all_rows, history, trainable, os.fspath(checkpoint_path)
    finally:
        detach_adapter(adapter)
        freeze_qwen(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    configure_evidence_only_defaults(args)
    methods = parse_methods(args.methods)
    layers = parse_int_tokens(args.layers)
    count_values = parse_int_tokens(args.evidence_counts)
    seq_lens = parse_int_tokens(args.seq_lens) if bool(args.evidence_only_seq1_8) else [int(args.seq_len)]
    if not methods:
        raise ValueError("--methods cannot be empty")
    if not layers:
        raise ValueError("--layers cannot be empty")
    if bool(args.evidence_only_seq1_8):
        if not seq_lens:
            raise ValueError("--seq-lens cannot be empty in evidence-only mode")
        if any(seq_len < 1 or seq_len > 8 for seq_len in seq_lens):
            raise ValueError("--evidence-only-seq1-8 expects seq_lens within 1..8")
    if int(args.rank) <= 0:
        raise ValueError("--rank must be positive")
    if int(args.candidate_min) != min(count_values) or int(args.candidate_max) != max(count_values):
        print(
            "Warning: candidate range and evidence counts differ; count logits use "
            f"{args.candidate_min}..{args.candidate_max}, metrics use {count_values}."
        )
    run_dir = resolve_run_dir(args)
    log_handle, old_stdout, old_stderr = setup_logging(run_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        config = {
            "experiment_name": EVIDENCE_ONLY_EXPERIMENT_NAME if bool(args.evidence_only_seq1_8) else EXPERIMENT_NAME,
            "script": "experiments/gated_token_mixer_adapter.py",
            "model_name": str(args.model_name),
            "dataset_root": os.fspath(Path(args.dataset_root).resolve()),
            "source_run": os.fspath(Path(args.source_run)),
            "source_dataset_root": os.fspath(Path(args.source_dataset_root).resolve()),
            "dataset_mode": "evidence_only_seq1_8" if bool(args.evidence_only_seq1_8) else "source_cache_seq8",
            "seq_len": int(args.seq_len),
            "seq_lens": [int(x) for x in seq_lens],
            "split": str(args.split),
            "methods": methods,
            "layers": [int(x) for x in layers],
            "rank": int(args.rank),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "grad_accum": int(args.grad_accum),
            "seed": int(args.seed),
            "lambda_init": float(args.lambda_init),
            "train_lambda": bool(args.train_lambda),
            "output_dir": os.fspath(Path(args.output_dir).resolve()),
            "run_dir": os.fspath(run_dir),
            "run_name": run_dir.name,
            "skip_lora": bool(args.skip_lora),
            "candidate_min": int(args.candidate_min),
            "candidate_max": int(args.candidate_max),
            "count_values": [int(x) for x in count_values],
            "samples_per_seq_len": int(args.samples_per_seq_len),
            "max_train_samples_per_seq_len": int(args.max_train_samples_per_seq_len),
            "max_eval_samples_per_seq_len": int(args.max_eval_samples_per_seq_len),
            "generate_dataset": bool(args.generate_dataset),
            "submit_mode": str(args.submit_mode),
        }
        write_json(run_dir / "config.json", config)
        print(f"Output dir: {run_dir}")
        print(f"Config: {json_compact(config)}")

        if bool(args.evidence_only_seq1_8):
            records, splits, dataset_manifest = load_evidence_only_records_and_splits(args=args, seq_lens=seq_lens)
            write_json(run_dir / "dataset_manifest_snapshot.json", dataset_manifest)
            sample_manifest = build_evidence_only_sample_manifest(
                args=args,
                seq_lens=seq_lens,
                records=records,
                splits=splits,
                dataset_manifest=dataset_manifest,
            )
            write_json(run_dir / "sample_manifest.json", sample_manifest)
        else:
            sample_ids, labels, sample_manifest = load_sample_ids_and_labels(args=args, count_values=count_values)
            write_json(run_dir / "sample_manifest.json", sample_manifest)
            records = prev.load_records(Path(args.dataset_root), str(args.split), int(args.seq_len), sample_ids)
            if len(records) != len(sample_ids):
                raise RuntimeError(f"Loaded {len(records)} records for {len(sample_ids)} sample ids")
            splits = make_splits(sample_ids=sample_ids, labels=labels, records=records, args=args, count_values=count_values)
        if any(not splits[split] for split in ("train", "val", "test")):
            raise RuntimeError(f"All train/val/test splits must be non-empty, got { {k: len(v) for k, v in splits.items()} }")

        device = prev.resolve_device(str(args.device))
        dtype = prev.resolve_dtype(str(args.dtype), device)
        model, processor = prev.load_model_and_processor(args, device=device, dtype=dtype)
        freeze_qwen(model)
        hidden_size = prev.hidden_size_from_model(model)
        candidate_format, count_token_ids = prev.candidate_token_ids(
            processor.tokenizer,
            int(args.candidate_min),
            int(args.candidate_max),
        )
        print(
            f"Loaded frozen Qwen hidden_size={hidden_size} candidate_format={candidate_format} "
            f"count_token_ids={json_compact(count_token_ids)}"
        )
        if sum(param.numel() for param in model.parameters() if param.requires_grad) != 0:
            raise RuntimeError("Qwen base model is not frozen before dispatch")

        all_rows: List[Dict[str, Any]] = []
        histories: Dict[str, List[Dict[str, Any]]] = {}
        trainable_counts: Dict[str, int] = {}
        checkpoint_paths: Dict[str, str] = {}
        skipped: Dict[str, str] = {}

        for method in methods:
            if method == LORA_ATTENTION and bool(args.skip_lora):
                reason = "--skip_lora was set"
                print(f"Skipping {LORA_ATTENTION}: {reason}")
                skipped[method] = reason
                continue
            print(f"=== Running method: {method} ===")
            try:
                method_rows, history, trainable, checkpoint_path = run_method(
                    method=method,
                    args=args,
                    output_dir=run_dir,
                    model=model,
                    processor=processor,
                    records=records,
                    splits=splits,
                    count_token_ids=count_token_ids,
                    count_values=count_values,
                    layers=layers,
                    hidden_size=int(hidden_size),
                    device=device,
                )
                all_rows.extend(method_rows)
                histories[method] = history
                trainable_counts[method] = int(trainable)
                if checkpoint_path:
                    checkpoint_paths[method] = checkpoint_path
            except Exception as exc:
                if method == LORA_ATTENTION:
                    reason = f"LoRA skipped after setup failure: {type(exc).__name__}: {exc}"
                    print(reason)
                    skipped[method] = reason
                    freeze_qwen(model)
                    continue
                raise
            save_all_outputs(
                output_dir=run_dir,
                config=config,
                all_rows=all_rows,
                methods=methods,
                count_values=count_values,
                seq_lens=seq_lens if bool(args.evidence_only_seq1_8) else [],
                trainable_counts=trainable_counts,
                checkpoint_paths=checkpoint_paths,
                histories=histories,
                skipped=skipped,
                splits=splits,
                no_plots=bool(args.no_plots),
            )

        summary = save_all_outputs(
            output_dir=run_dir,
            config=config,
            all_rows=all_rows,
            methods=methods,
            count_values=count_values,
            seq_lens=seq_lens if bool(args.evidence_only_seq1_8) else [],
            trainable_counts=trainable_counts,
            checkpoint_paths=checkpoint_paths,
            histories=histories,
            skipped=skipped,
            splits=splits,
            no_plots=bool(args.no_plots),
        )
        write_json(
            run_dir / "run_done.json",
            {
                "ok": True,
                "elapsed_seconds": time.time() - started,
                "output_dir": os.fspath(run_dir),
                "test_accuracy_by_method": {
                    method: summary.get("methods", {}).get(method, {}).get("test_accuracy", "")
                    for method in methods
                },
            },
        )
        print(f"Done. Outputs written to {run_dir}")
        return 0
    finally:
        restore_logging(log_handle, old_stdout, old_stderr)


if __name__ == "__main__":
    raise SystemExit(main())

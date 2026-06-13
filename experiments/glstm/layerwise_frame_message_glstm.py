#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
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

from evaluations.helpers import utils as eval_utils
from models.model import find_subsequence, get_layers, image_token_groups
from experiments.carrier_mixing import pnamix_clean_aggregation_lora as text_base
from experiments.carrier_mixing import visual_fixed8_iid_carrier_slots_lora as visual_base
from experiments.carrier_probes import run_message_memory_adapter_stage1_stage3_seq8 as mmred

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
except Exception:  # pragma: no cover - transformers version dependent
    apply_multimodal_rotary_pos_emb = None  # type: ignore[assignment]


EXPERIMENT_NAME = "layerwise_frame_message_glstm"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_SOURCE_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images_park"
DEFAULT_FALLBACK_SOURCE_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images"
DEFAULT_LAYERS = (14, 15, 16, 17)
TRAIN_SPLIT = "train"
VAL_SPLIT = "iid_val"
IID_TEST_SPLIT = "iid_test"
LENGTH_OOD_SPLIT = "length_ood_test"
COMPOSITION_OOD_SPLIT = "composition_ood_test"
EVAL_SPLITS = (VAL_SPLIT, IID_TEST_SPLIT, LENGTH_OOD_SPLIT, COMPOSITION_OOD_SPLIT)
VISUAL_INPUT_KEYS = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")

GLOBAL_LORA = "global_lora"
CARRIER_LORA = "carrier_lora"
CARRIER_DIRECT_SUM = "carrier_direct_sum"
CARRIER_GLSTM_LAYERWISE = "carrier_glstm_layerwise"
CARRIER_GLSTM_FINAL_ONLY = "carrier_glstm_final_only"
CARRIER_GLSTM_SOFTMAX = "carrier_glstm_layerwise_softmax"
CARRIER_PNA = "carrier_pna"
VARIANTS = (
    GLOBAL_LORA,
    CARRIER_LORA,
    CARRIER_DIRECT_SUM,
    CARRIER_GLSTM_LAYERWISE,
    CARRIER_GLSTM_FINAL_ONLY,
)
VARIANT_ALIASES = {
    "global": GLOBAL_LORA,
    "global_lora": GLOBAL_LORA,
    "carrier": CARRIER_LORA,
    "carrier_lora": CARRIER_LORA,
    "direct": CARRIER_DIRECT_SUM,
    "direct_sum": CARRIER_DIRECT_SUM,
    "carrier_direct_sum": CARRIER_DIRECT_SUM,
    "glstm": CARRIER_GLSTM_LAYERWISE,
    "layerwise": CARRIER_GLSTM_LAYERWISE,
    "carrier_glstm_layerwise": CARRIER_GLSTM_LAYERWISE,
    "final": CARRIER_GLSTM_FINAL_ONLY,
    "final_only": CARRIER_GLSTM_FINAL_ONLY,
    "carrier_glstm_final_only": CARRIER_GLSTM_FINAL_ONLY,
    "softmax": CARRIER_GLSTM_SOFTMAX,
    "softmax_read": CARRIER_GLSTM_SOFTMAX,
    "carrier_glstm_layerwise_softmax": CARRIER_GLSTM_SOFTMAX,
    "pna": CARRIER_PNA,
    "carrier_pna": CARRIER_PNA,
}
MEMORY_VARIANTS = {CARRIER_DIRECT_SUM, CARRIER_GLSTM_LAYERWISE, CARRIER_GLSTM_FINAL_ONLY, CARRIER_GLSTM_SOFTMAX, CARRIER_PNA}
GLSTM_VARIANTS = {CARRIER_GLSTM_LAYERWISE, CARRIER_GLSTM_FINAL_ONLY, CARRIER_GLSTM_SOFTMAX}


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


@dataclass(frozen=True)
class FrameMemoryExample:
    example_id: str
    split: str
    frame_paths: Tuple[str, ...]
    num_frames: int
    gold_count: int
    evidence_frame_indices: Tuple[int, ...]
    question: str
    answer: str
    queried_character: str
    queried_room: str
    template_id: str
    composition_key: str
    source_dataset_info: Tuple[Dict[str, Any], ...]


@dataclass
class FrameMemoryBatch:
    inputs: Dict[str, Any]
    prompt_last_indices: torch.Tensor
    gold_counts: torch.Tensor
    loss_positions: Optional[torch.Tensor]
    loss_targets: Optional[torch.Tensor]
    carrier_positions: List[List[int]]
    carrier_identities: List[List[str]]
    frame_groups: List[List[List[int]]]
    frame_valid_mask: torch.Tensor
    evidence_frame_indices: List[List[int]]
    sample_ids: List[str]
    sample_indices: List[int]
    visual_input_keys: List[str]


def split_tokens(raw_values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for raw in raw_values:
        out.extend(part.strip() for part in str(raw).replace(",", " ").split() if part.strip())
    return out


def parse_int_tokens(raw_values: Sequence[Any]) -> List[int]:
    return sorted(dict.fromkeys(int(part) for part in split_tokens(raw_values)))


def parse_variants(raw_values: Sequence[Any]) -> List[str]:
    variants: List[str] = []
    for token in split_tokens(raw_values):
        if token == "all":
            variants.extend(VARIANTS)
            continue
        if token not in VARIANT_ALIASES:
            raise ValueError(f"Unknown variant {token!r}; valid values are {sorted(VARIANT_ALIASES)} plus 'all'")
        variants.append(VARIANT_ALIASES[token])
    return list(dict.fromkeys(variants))


def safe_name(value: Any) -> str:
    return text_base.safe_name(value)


def finite_float(value: Any) -> Optional[float]:
    return text_base.finite_float(value)


def finite_mean(values: Iterable[Any], default: float = math.nan) -> float:
    return text_base.finite_mean(values, default=default)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return text_base.accuracy(y_true, y_pred)


def mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return text_base.mae(y_true, y_pred)


def json_sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv_dynamic(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(leading)
    for key in sorted({key for row in rows for key in row.keys()}):
        if key not in fields:
            fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(row.get(key, "")) for key in fields})


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, float):
        return f"{value:.8g}" if math.isfinite(value) else ""
    return value


def setup_logging(run_dir: Path) -> Tuple[Any, Any, Any]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    handle = (logs_dir / "run.log").open("a", encoding="utf-8")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(old_stdout, handle)  # type: ignore[assignment]
    sys.stderr = Tee(old_stderr, handle)  # type: ignore[assignment]
    return handle, old_stdout, old_stderr


def restore_logging(handle: Any, old_stdout: Any, old_stderr: Any) -> None:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnosable layerwise frame-message gLSTM memory for Qwen visual counting."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--fallback-source-dataset-root", type=Path, default=DEFAULT_FALLBACK_SOURCE_DATASET_ROOT)
    parser.add_argument("--source-split", default="all_uniform")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--variants", nargs="+", default=[",".join(VARIANTS)])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=24680)
    parser.add_argument("--train-lengths", nargs="+", default=["4,6,8"])
    parser.add_argument("--length-ood-lengths", nargs="+", default=["5,7,10"])
    parser.add_argument("--train-per-count", type=int, default=50)
    parser.add_argument("--val-per-count", type=int, default=20)
    parser.add_argument("--iid-test-per-count", type=int, default=20)
    parser.add_argument("--length-ood-per-count", type=int, default=20)
    parser.add_argument("--composition-ood-per-count", type=int, default=20)
    parser.add_argument("--heldout-compositions", type=int, default=4)
    parser.add_argument(
        "--filler-kind",
        choices=["neutral", "distractor"],
        default="neutral",
        help=(
            "Pool used for non-evidence frames. 'neutral' = queried character absent and queried "
            "room empty (clean task). 'distractor' = queried character elsewhere or queried room "
            "occupied by other characters (hard task)."
        ),
    )
    parser.add_argument("--force-regenerate-dataset", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--max-ablation-examples", type=int, default=64)
    parser.add_argument("--eval-candidate-scores", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--layers", nargs="+", default=["14,15,16,17"])
    parser.add_argument("--carrier-mode", default="room_character", choices=["room_character"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--gamma-init", type=float, default=1e-3)
    parser.add_argument("--projection-sharing", choices=["layer_specific", "shared"], default="layer_specific")
    parser.add_argument("--memory-projection-sharing", choices=["layer_specific", "shared"], default="layer_specific")
    parser.add_argument("--message-mode", choices=["exact", "auto", "proxy"], default="exact")
    parser.add_argument("--frame-kv-lora", action="store_true", default=False)
    parser.add_argument(
        "--no-carrier-lora",
        action="store_true",
        default=False,
        help="Memory-only control: drop the carrier LoRA for memory variants so the memory adapter is the only trainable component (isolates its contribution from the LoRA).",
    )
    parser.add_argument("--reconstruction-tol", type=float, default=5e-3)
    parser.add_argument("--fail-on-reconstruction-error", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-qlora-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=None)

    parser.add_argument("--diagnostic-subset", type=int, default=48)
    parser.add_argument("--probe-epochs", type=int, default=150)
    parser.add_argument("--probe-lr", type=float, default=5e-3)
    parser.add_argument("--no-probes", action="store_true", default=False)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--smoke-test", action="store_true", default=False)
    parser.add_argument("--tiny-debug-model", action="store_true", default=False)
    parser.add_argument("--tiny-num-layers", type=int, default=18)
    parser.add_argument("--tiny-hidden-size", type=int, default=64)
    parser.add_argument("--tiny-num-heads", type=int, default=4)
    parser.add_argument("--submit-mode", default="local")
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    return parser.parse_args()


def resolve_device(raw: str) -> str:
    if str(raw) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(raw).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {raw!r}, but CUDA is unavailable")
    return str(raw)


def dtype_from_arg(raw: str, device: str) -> torch.dtype:
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


def hidden_size_from_model(model: Any) -> int:
    return text_base.hidden_size_from_model(model)


def load_model_and_processor(args: argparse.Namespace, device: str, dtype: torch.dtype) -> Tuple[Any, Any, bool, str]:
    if bool(args.tiny_debug_model):
        processor = visual_base.TinyVisualProcessor()
        model = visual_base.TinyVisualQwen(
            vocab_size=processor.tokenizer.vocab_size,
            hidden_size=int(args.tiny_hidden_size),
            num_layers=int(args.tiny_num_layers),
            num_heads=int(args.tiny_num_heads),
            image_token_id=processor.image_token_id,
        ).to(device)
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
        return model, processor, False, "tiny_visual_debug_model"
    return visual_base.load_model_and_processor(args, device=device, dtype=dtype)


def source_partition(sample_id: str, seed: int) -> str:
    return visual_base.source_partition(sample_id, seed)


def question_for_template(template_id: str, character: str, room: str) -> str:
    return visual_base.question_for_template(template_id, character, room)


def build_prompt(question: str, num_frames: int) -> str:
    return visual_base.build_prompt(question, num_frames)


def example_to_json(example: FrameMemoryExample) -> Dict[str, Any]:
    return {
        "id": example.example_id,
        "split": example.split,
        "frame_paths": list(example.frame_paths),
        "num_frames": int(example.num_frames),
        "gold_count": int(example.gold_count),
        "evidence_frame_indices": list(example.evidence_frame_indices),
        "question": example.question,
        "answer": example.answer,
        "queried_character": example.queried_character,
        "queried_room": example.queried_room,
        "template_id": example.template_id,
        "composition_key": example.composition_key,
        "source_dataset_info": list(example.source_dataset_info),
    }


def example_from_json(row: Dict[str, Any]) -> FrameMemoryExample:
    return FrameMemoryExample(
        example_id=str(row.get("id", row.get("example_id"))),
        split=str(row["split"]),
        frame_paths=tuple(str(path) for path in row["frame_paths"]),
        num_frames=int(row["num_frames"]),
        gold_count=int(row["gold_count"]),
        evidence_frame_indices=tuple(int(x) for x in row["evidence_frame_indices"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        queried_character=str(row["queried_character"]),
        queried_room=str(row["queried_room"]),
        template_id=str(row["template_id"]),
        composition_key=str(row.get("composition_key", f"{row['queried_character']}|{row['queried_room']}")),
        source_dataset_info=tuple(dict(item) for item in row.get("source_dataset_info", [])),
    )


def scan_source_frame_pools(
    source_root: Path,
    fallback_source_root: Optional[Path],
    source_split: str,
    dataset_seed: int,
    lengths: Sequence[int],
) -> Tuple[Dict[int, Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]], Dict[str, Any]]:
    source_root = Path(source_root).resolve()
    fallback_root = Path(fallback_source_root).resolve() if fallback_source_root is not None else None
    all_pools: Dict[int, Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]] = {}
    length_manifests: Dict[str, Any] = {}
    for length in sorted({int(x) for x in lengths}):
        length_source_root = source_root
        split_root = length_source_root / f"seq_len_{int(length)}" / str(source_split)
        if not split_root.is_dir() and fallback_root is not None:
            fallback_split_root = fallback_root / f"seq_len_{int(length)}" / str(source_split)
            if fallback_split_root.is_dir():
                length_source_root = fallback_root
                split_root = fallback_split_root
        sample_dirs = sorted(
            [path for path in split_root.iterdir() if path.is_dir() and (path / "qa.txt").is_file()],
            key=lambda path: path.name,
        )
        if not sample_dirs:
            raise RuntimeError(f"No MMReD samples found under {split_root}")
        parsed_samples: List[Tuple[Path, List[Dict[str, Any]]]] = []
        characters: set[str] = set()
        rooms: set[str] = set()
        for sample_dir in sample_dirs:
            _question, states, _gold = mmred.parse_qa_file(sample_dir)
            parsed_samples.append((sample_dir, states))
            characters.update(eval_utils.extract_characters_from_states(states))
            rooms.update(eval_utils.extract_rooms_from_states(states))
        pools: Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]] = {
            partition: defaultdict(lambda: {"evidence": [], "neutral": [], "distractor": []})
            for partition in ("train", "val", "test")
        }
        for sample_dir, states in parsed_samples:
            partition = source_partition(f"len{int(length)}:{sample_dir.name}", int(dataset_seed))
            for frame_idx, state in enumerate(states):
                room_to_chars = eval_utils.rooms_to_room2chars(state.get("rooms", {}))
                present_chars = {char for values in room_to_chars.values() for char in values}
                frame_path = sample_dir / f"{int(frame_idx):03d}.png"
                if not frame_path.is_file():
                    frame_path = sample_dir / f"frame_{int(frame_idx):03d}.png"
                if not frame_path.is_file():
                    raise FileNotFoundError(frame_path)
                for character in sorted(characters):
                    for room in sorted(rooms):
                        if character in room_to_chars.get(room, []):
                            kind = "evidence"
                        elif character not in present_chars and not room_to_chars.get(room, []):
                            kind = "neutral"
                        else:
                            # Queried character elsewhere, or queried room occupied by
                            # other characters: a non-evidence frame with active content.
                            kind = "distractor"
                        pools[partition][(character, room)][kind].append(
                            {
                                "frame_path": os.fspath(frame_path.relative_to(PROJECT_ROOT)),
                                "source_sample_id": sample_dir.name,
                                "source_frame_index": int(frame_idx),
                                "source_sequence_length": int(length),
                                "source_partition": partition,
                                "selection_type": kind,
                                "state": state,
                            }
                        )
        all_pools[int(length)] = pools
        length_manifests[str(int(length))] = {
            "source_dataset_root": os.fspath(length_source_root),
            "source_sample_count": len(sample_dirs),
            "characters": sorted(characters),
            "rooms": sorted(rooms),
            "pool_counts": {
                partition: {
                    f"{character}|{room}": {
                        "evidence": len(kind_pools["evidence"]),
                        "neutral": len(kind_pools["neutral"]),
                        "distractor": len(kind_pools["distractor"]),
                    }
                    for (character, room), kind_pools in sorted(pair_pools.items())
                }
                for partition, pair_pools in pools.items()
            },
        }
    manifest = {
        "source_dataset_root": os.fspath(source_root),
        "fallback_source_dataset_root": None if fallback_root is None else os.fspath(fallback_root),
        "source_split": str(source_split),
        "lengths_scanned": [int(x) for x in sorted(all_pools)],
        "partition_rule": "sha256(dataset_seed:lenN:sample_id) modulo 10; train=0..6, val=7, test=8..9",
        "neutral_rule": "queried character absent from the frame and queried room empty",
        "evidence_rule": "queried character present in queried room",
        "lengths": length_manifests,
    }
    return all_pools, manifest


def choose_refs(rng: random.Random, pool: Sequence[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return visual_base.choose_refs(rng, pool, n)


def valid_pairs_for_lengths(
    pools: Dict[int, Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]],
    lengths: Sequence[int],
    partition: str,
    filler_kind: str = "neutral",
) -> set[Tuple[str, str]]:
    valid: Optional[set[Tuple[str, str]]] = None
    for length in lengths:
        pairs = {
            pair
            for pair, pair_pools in pools[int(length)][str(partition)].items()
            if pair_pools.get("evidence") and pair_pools.get(str(filler_kind))
        }
        valid = pairs if valid is None else valid & pairs
    return valid or set()


def merged_frame_pools_for_partition(
    pools: Dict[int, Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]],
    source_lengths: Sequence[int],
) -> Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]:
    merged: Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]] = {
        partition: defaultdict(lambda: {"evidence": [], "neutral": [], "distractor": []})
        for partition in ("train", "val", "test")
    }
    for source_length in source_lengths:
        for partition, pair_pools in pools[int(source_length)].items():
            for pair, kind_pools in pair_pools.items():
                for kind in ("evidence", "neutral", "distractor"):
                    for ref in kind_pools.get(kind, []):
                        merged[partition][pair][kind].append(dict(ref))
    return merged


def fill_empty_output_length_pools(
    pools: Dict[int, Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]],
    output_lengths: Sequence[int],
    filler_kind: str = "neutral",
) -> List[int]:
    synthesized: List[int] = []
    source_lengths = sorted(int(length) for length in pools)
    merged = merged_frame_pools_for_partition(pools, source_lengths)
    for length in [int(x) for x in output_lengths]:
        has_test_pairs = bool(valid_pairs_for_lengths(pools, [length], "test", filler_kind))
        has_train_pairs = bool(valid_pairs_for_lengths(pools, [length], "train", filler_kind))
        if has_test_pairs and has_train_pairs:
            continue
        pools[int(length)] = merged
        synthesized.append(int(length))
    return synthesized


def choose_holdout_compositions(
    pools: Dict[int, Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]],
    train_lengths: Sequence[int],
    ood_lengths: Sequence[int],
    count: int,
    seed: int,
    filler_kind: str = "neutral",
) -> List[Tuple[str, str]]:
    del ood_lengths
    train_pairs = valid_pairs_for_lengths(pools, train_lengths, "train", filler_kind)
    test_pairs = valid_pairs_for_lengths(pools, train_lengths, "test", filler_kind)
    candidates = sorted(train_pairs & test_pairs)
    if not candidates:
        raise RuntimeError("No character-room compositions are valid across train and OOD lengths")
    rng = random.Random(int(seed) + 991)
    rng.shuffle(candidates)
    return sorted(candidates[: max(1, min(int(count), max(1, len(candidates) // 4)))])


def generate_split(
    *,
    split: str,
    lengths: Sequence[int],
    examples_per_count: int,
    templates: Sequence[str],
    pools: Dict[int, Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]],
    source_partition_name: str,
    allowed_pairs: Sequence[Tuple[str, str]],
    seed: int,
    filler_kind: str = "neutral",
) -> List[FrameMemoryExample]:
    rng = random.Random(int(seed))
    examples: List[FrameMemoryExample] = []
    allowed = sorted({(str(char), str(room)) for char, room in allowed_pairs})
    if not allowed:
        raise RuntimeError(f"{split}: no allowed character-room pairs")
    for length in [int(x) for x in lengths]:
        schedule = [
            count
            for count in range(int(length) + 1)
            for _ in range(int(examples_per_count))
        ]
        rng.shuffle(schedule)
        for local_idx, gold_count in enumerate(schedule):
            valid_pairs = [
                pair
                for pair in allowed
                if pools[length][source_partition_name].get(pair, {}).get("evidence")
                and pools[length][source_partition_name].get(pair, {}).get(str(filler_kind))
            ]
            if not valid_pairs:
                raise RuntimeError(f"{split}: no valid allowed pairs for length={length}")
            character, room = rng.choice(valid_pairs)
            pair_pool = pools[length][source_partition_name][(character, room)]
            evidence_positions = tuple(sorted(rng.sample(range(length), int(gold_count))))
            evidence_set = set(evidence_positions)
            evidence_refs = choose_refs(rng, pair_pool["evidence"], int(gold_count))
            neutral_refs = choose_refs(rng, pair_pool[str(filler_kind)], int(length) - int(gold_count))
            rng.shuffle(evidence_refs)
            rng.shuffle(neutral_refs)
            ordered_refs: List[Dict[str, Any]] = []
            evidence_cursor = 0
            neutral_cursor = 0
            for frame_idx in range(length):
                if frame_idx in evidence_set:
                    ref = evidence_refs[evidence_cursor]
                    evidence_cursor += 1
                else:
                    ref = neutral_refs[neutral_cursor]
                    neutral_cursor += 1
                ref = dict(ref)
                ref["output_frame_index"] = int(frame_idx)
                ordered_refs.append(ref)
            template_id = rng.choice(tuple(templates))
            question = question_for_template(template_id, character, room)
            sample_hash = hashlib.sha1(
                json.dumps(
                    {
                        "split": split,
                        "length": length,
                        "idx": local_idx,
                        "count": gold_count,
                        "refs": [(r["frame_path"], r["source_frame_index"]) for r in ordered_refs],
                        "pair": [character, room],
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:10]
            examples.append(
                FrameMemoryExample(
                    example_id=f"{split}_len{length}_count{int(gold_count)}_{local_idx:05d}_{sample_hash}",
                    split=split,
                    frame_paths=tuple(str(ref["frame_path"]) for ref in ordered_refs),
                    num_frames=int(length),
                    gold_count=int(gold_count),
                    evidence_frame_indices=evidence_positions,
                    question=question,
                    answer=str(int(gold_count)),
                    queried_character=character,
                    queried_room=room,
                    template_id=template_id,
                    composition_key=f"{character}|{room}",
                    source_dataset_info=tuple(ordered_refs),
                )
            )
    rng.shuffle(examples)
    return examples


def dataset_config(args: argparse.Namespace) -> Dict[str, Any]:
    train_lengths = parse_int_tokens(args.train_lengths)
    length_ood_lengths = parse_int_tokens(args.length_ood_lengths)
    if bool(args.smoke_test):
        per_count = {
            "train_per_count": min(int(args.train_per_count), 1),
            "val_per_count": min(int(args.val_per_count), 1),
            "iid_test_per_count": min(int(args.iid_test_per_count), 1),
            "length_ood_per_count": min(int(args.length_ood_per_count), 1),
            "composition_ood_per_count": min(int(args.composition_ood_per_count), 1),
        }
    else:
        per_count = {
            "train_per_count": int(args.train_per_count),
            "val_per_count": int(args.val_per_count),
            "iid_test_per_count": int(args.iid_test_per_count),
            "length_ood_per_count": int(args.length_ood_per_count),
            "composition_ood_per_count": int(args.composition_ood_per_count),
        }
    if any(value <= 0 for value in per_count.values()):
        raise ValueError(f"Per-count sizes must be positive: {per_count}")
    return {
        "dataset_seed": int(args.dataset_seed),
        "source_dataset_root": os.fspath(Path(args.source_dataset_root).resolve()),
        "fallback_source_dataset_root": os.fspath(Path(args.fallback_source_dataset_root).resolve())
        if args.fallback_source_dataset_root is not None
        else None,
        "source_split": str(args.source_split),
        "train_lengths": train_lengths,
        "length_ood_lengths": length_ood_lengths,
        "heldout_compositions": int(args.heldout_compositions),
        "splits": {
            TRAIN_SPLIT: {
                "lengths": train_lengths,
                "examples_per_count": per_count["train_per_count"],
                "source_partition": "train",
                "composition": "train_allowed",
            },
            VAL_SPLIT: {
                "lengths": train_lengths,
                "examples_per_count": per_count["val_per_count"],
                "source_partition": "val",
                "composition": "train_allowed",
            },
            IID_TEST_SPLIT: {
                "lengths": train_lengths,
                "examples_per_count": per_count["iid_test_per_count"],
                "source_partition": "test",
                "composition": "train_allowed",
            },
            LENGTH_OOD_SPLIT: {
                "lengths": length_ood_lengths,
                "examples_per_count": per_count["length_ood_per_count"],
                "source_partition": "test",
                "composition": "train_allowed",
            },
            COMPOSITION_OOD_SPLIT: {
                "lengths": train_lengths,
                "examples_per_count": per_count["composition_ood_per_count"],
                "source_partition": "test",
                "composition": "heldout_only",
            },
        },
        "evidence_positions_randomized": True,
        "hard_semantic_distractors": str(args.filler_kind) == "distractor",
        "filler_kind": str(args.filler_kind),
        "neutral_rule": "queried character absent and queried room empty",
        "distractor_rule": "queried character elsewhere or queried room occupied by other characters",
        "balanced_counts_per_length": True,
        **per_count,
    }


def ensure_dataset(
    args: argparse.Namespace,
    dataset_base: Path,
) -> Tuple[Path, Dict[str, List[FrameMemoryExample]], Dict[str, Any]]:
    config = dataset_config(args)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    dataset_dir = dataset_base / digest
    manifest_path = dataset_dir / "dataset_manifest.json"
    split_paths = {split: dataset_dir / f"{split}.jsonl" for split in config["splits"]}
    regenerate = bool(args.force_regenerate_dataset) or not manifest_path.is_file() or not all(
        path.is_file() for path in split_paths.values()
    )
    if regenerate:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        all_lengths = sorted(set(config["train_lengths"]) | set(config["length_ood_lengths"]))
        pools, source_manifest = scan_source_frame_pools(
            Path(args.source_dataset_root),
            Path(args.fallback_source_dataset_root) if args.fallback_source_dataset_root is not None else None,
            str(args.source_split),
            int(args.dataset_seed),
            all_lengths,
        )
        filler_kind = str(config.get("filler_kind", "neutral"))
        synthesized_lengths = fill_empty_output_length_pools(pools, all_lengths, filler_kind)
        heldout_pairs = choose_holdout_compositions(
            pools,
            config["train_lengths"],
            config["length_ood_lengths"],
            int(args.heldout_compositions),
            int(args.dataset_seed),
            filler_kind,
        )
        train_pairs = sorted(valid_pairs_for_lengths(pools, all_lengths, "train", filler_kind) - set(heldout_pairs))
        if not train_pairs:
            train_pairs = sorted(
                valid_pairs_for_lengths(pools, config["train_lengths"], "train", filler_kind) - set(heldout_pairs)
            )
        split_seed_offsets = {
            TRAIN_SPLIT: 11,
            VAL_SPLIT: 23,
            IID_TEST_SPLIT: 59,
            LENGTH_OOD_SPLIT: 83,
            COMPOSITION_OOD_SPLIT: 107,
        }
        generated: Dict[str, List[FrameMemoryExample]] = {}
        for split, split_cfg in config["splits"].items():
            allowed_pairs = heldout_pairs if split_cfg["composition"] == "heldout_only" else train_pairs
            generated[split] = generate_split(
                split=split,
                lengths=split_cfg["lengths"],
                examples_per_count=int(split_cfg["examples_per_count"]),
                templates=visual_base.TRAIN_TEMPLATES,
                pools=pools,
                source_partition_name=str(split_cfg["source_partition"]),
                allowed_pairs=allowed_pairs,
                seed=int(args.dataset_seed) + split_seed_offsets[split],
                filler_kind=filler_kind,
            )
            write_jsonl(split_paths[split], [example_to_json(example) for example in generated[split]])
        manifest = {
            "dataset_hash": digest,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config,
            "source_manifest": source_manifest,
            "synthesized_output_lengths_from_merged_frame_pools": synthesized_lengths,
            "heldout_compositions": [f"{char}|{room}" for char, room in heldout_pairs],
            "train_compositions": [f"{char}|{room}" for char, room in train_pairs],
            "splits": {
                split: {
                    "path": os.fspath(path),
                    "n": len(generated[split]),
                    "length_count_histogram": {
                        f"{length}|{count}": sum(
                            int(ex.num_frames) == int(length) and int(ex.gold_count) == int(count)
                            for ex in generated[split]
                        )
                        for length in config["splits"][split]["lengths"]
                        for count in range(int(length) + 1)
                    },
                }
                for split, path in split_paths.items()
            },
        }
        write_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = {
        split: [example_from_json(row) for row in read_jsonl(path)]
        for split, path in split_paths.items()
    }
    assert_dataset(config, examples)
    return dataset_dir, examples, manifest


def assert_dataset(config: Dict[str, Any], examples: Dict[str, List[FrameMemoryExample]]) -> None:
    all_ids: set[str] = set()
    for split, rows in examples.items():
        if split not in config["splits"]:
            raise RuntimeError(f"Unexpected split {split}")
        split_cfg = config["splits"][split]
        expected_lengths = set(int(x) for x in split_cfg["lengths"])
        seen_ids = {row.example_id for row in rows}
        if len(seen_ids) != len(rows):
            raise RuntimeError(f"{split}: duplicate generated sample IDs")
        if all_ids & seen_ids:
            raise RuntimeError(f"{split}: sample IDs overlap another split")
        all_ids |= seen_ids
        for row in rows:
            if int(row.num_frames) not in expected_lengths:
                raise RuntimeError(f"{split}: unexpected length {row.num_frames}")
            if not (0 <= int(row.gold_count) <= int(row.num_frames)):
                raise RuntimeError(f"{split}: invalid count {row.gold_count} for length {row.num_frames}")
            if len(row.frame_paths) != int(row.num_frames):
                raise RuntimeError(f"{split}: frame path count mismatch for {row.example_id}")
            if len(set(row.evidence_frame_indices)) != int(row.gold_count):
                raise RuntimeError(f"{split}: evidence index/count mismatch for {row.example_id}")
            for frame_path in row.frame_paths:
                resolved = resolve_frame_path(frame_path)
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
        for length in expected_lengths:
            missing = [
                count
                for count in range(int(length) + 1)
                if not any(int(row.num_frames) == int(length) and int(row.gold_count) == count for row in rows)
            ]
            if missing:
                raise RuntimeError(f"{split}: missing counts for length={length}: {missing}")


def resolve_frame_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def token_positions_for_prompt_span(
    tokenizer: Any,
    prompt_text: str,
    prompt_start: int,
    char_start: int,
    char_end: int,
) -> List[int]:
    return visual_base.token_positions_for_prompt_span(
        tokenizer,
        prompt_text,
        int(prompt_start),
        int(char_start),
        int(char_end),
    )


def move_inputs_to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


def prepare_batch(
    *,
    examples: Sequence[FrameMemoryExample],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
    answer_ids: Optional[Dict[int, Tuple[int, ...]]] = None,
    answer_count_override: Optional[int] = None,
) -> FrameMemoryBatch:
    if len(examples) != 1:
        raise ValueError("Visual Qwen batches are kept at batch_size=1 for image/token alignment")
    example = examples[0]
    expected_frames = int(example.num_frames)
    frames: List[Image.Image] = []
    try:
        for path_text in example.frame_paths:
            with Image.open(resolve_frame_path(path_text)) as image:
                frames.append(image.convert("RGB"))
        if len(frames) != expected_frames:
            raise AssertionError(f"{example.example_id}: loaded {len(frames)} frames, expected {expected_frames}")
        prompt_text = build_prompt(example.question, example.num_frames)
        messages = [{
            "role": "user",
            "content": (
                [{"type": "image", "image": frame} for frame in frames]
                + [{"type": "text", "text": prompt_text}]
            ),
        }]
        raw_inputs = dict(
            processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )
    finally:
        for frame in frames:
            frame.close()

    visual_keys = [key for key in VISUAL_INPUT_KEYS if key in raw_inputs and torch.is_tensor(raw_inputs[key])]
    if not any(key.startswith("pixel_values") for key in visual_keys):
        raise AssertionError(f"{example.example_id}: no visual tensor found in processor output")
    input_ids = raw_inputs["input_ids"]
    attention_mask = raw_inputs["attention_mask"]
    if int(input_ids.shape[0]) != 1:
        raise AssertionError("Expected one visual example per batch")
    prompt_last = int(attention_mask[0].nonzero(as_tuple=True)[0][-1].item())
    tokenizer = processor.tokenizer
    prompt_ids = [int(x) for x in tokenizer(prompt_text, add_special_tokens=False)["input_ids"]]
    full_ids = [int(x) for x in input_ids[0].tolist()]
    prompt_start = find_subsequence(full_ids, prompt_ids)
    if prompt_start is None:
        raise RuntimeError(f"{example.example_id}: textual prompt not found after visual tokens")
    question_start = prompt_text.index(example.question)
    character_start = question_start + example.question.index(example.queried_character)
    room_start = question_start + example.question.index(example.queried_room)
    character_positions = token_positions_for_prompt_span(
        tokenizer,
        prompt_text,
        int(prompt_start),
        character_start,
        character_start + len(example.queried_character),
    )
    room_positions = token_positions_for_prompt_span(
        tokenizer,
        prompt_text,
        int(prompt_start),
        room_start,
        room_start + len(example.queried_room),
    )
    carrier_pairs = [("character", pos) for pos in character_positions] + [("room", pos) for pos in room_positions]
    carriers = sorted({int(pos) for _kind, pos in carrier_pairs if 0 <= int(pos) < prompt_last})
    carrier_identities = [
        ("character" if int(pos) in set(character_positions) else "room")
        for pos in carriers
    ]
    groups = image_token_groups(input_ids[0].detach().cpu(), expected_frames, processor=processor)
    groups = [[int(position) for position in group] for group in groups]
    flattened = [position for group in groups for position in group]
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(tokenizer, "image_token_id", None)
    if image_token_id is None and hasattr(tokenizer, "convert_tokens_to_ids"):
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    errors: List[str] = []
    if not carriers:
        errors.append("room/character carrier token positions not found")
    if len(groups) != expected_frames or any(not group for group in groups):
        errors.append(f"expected {expected_frames} non-empty visual token spans, found {[len(g) for g in groups]}")
    if image_token_id is None or any(int(input_ids[0, position].item()) != int(image_token_id) for position in flattened):
        errors.append("frame groups include non-image-pad tokens")
    if prompt_last in carriers:
        errors.append("final prompt token selected as carrier")
    for left, right in zip(groups, groups[1:]):
        if set(left) & set(right):
            errors.append("frame spans overlap unexpectedly")
            break
    if errors:
        raise AssertionError(f"{example.example_id}: {'; '.join(errors)}")

    loss_positions: Optional[torch.Tensor] = None
    loss_targets: Optional[torch.Tensor] = None
    if answer_ids is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise RuntimeError("Tokenizer must expose eos_token_id for answer-token training")
        answer_count = int(example.gold_count if answer_count_override is None else answer_count_override)
        targets = [*answer_ids[int(answer_count)], int(eos_token_id)]
        positions = list(range(prompt_last, prompt_last + len(targets)))
        suffix = torch.tensor([targets], dtype=input_ids.dtype)
        raw_inputs["input_ids"] = torch.cat([input_ids, suffix], dim=1)
        raw_inputs["attention_mask"] = torch.cat(
            [attention_mask, torch.ones((1, len(targets)), dtype=attention_mask.dtype)],
            dim=1,
        )
        loss_positions = torch.tensor([positions], dtype=torch.long)
        loss_targets = torch.tensor([targets], dtype=torch.long)

    frame_valid_mask = torch.ones((1, expected_frames), dtype=torch.bool, device=device)
    return FrameMemoryBatch(
        inputs=move_inputs_to_device(raw_inputs, device),
        prompt_last_indices=torch.tensor([prompt_last], device=device, dtype=torch.long),
        gold_counts=torch.tensor([example.gold_count], device=device, dtype=torch.long),
        loss_positions=loss_positions.to(device) if loss_positions is not None else None,
        loss_targets=loss_targets.to(device) if loss_targets is not None else None,
        carrier_positions=[carriers],
        carrier_identities=[carrier_identities],
        frame_groups=[groups],
        frame_valid_mask=frame_valid_mask,
        evidence_frame_indices=[list(example.evidence_frame_indices)],
        sample_ids=[example.example_id],
        sample_indices=[int(sample_indices[0])],
        visual_input_keys=visual_keys,
    )


class LoRALinearWrapper(nn.Module):
    def __init__(self, base_layer: nn.Module, *, rank: int, alpha: float, dropout: float, gated: bool) -> None:
        super().__init__()
        object.__setattr__(self, "base_layer", base_layer)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(1, int(rank))
        self.dropout = float(dropout)
        self.gated = bool(gated)
        weight = getattr(base_layer, "weight", None)
        in_features = getattr(base_layer, "in_features", None)
        out_features = getattr(base_layer, "out_features", None)
        if in_features is None and torch.is_tensor(weight):
            in_features = int(weight.shape[1])
        if out_features is None and torch.is_tensor(weight):
            out_features = int(weight.shape[0])
        if in_features is None or out_features is None:
            raise ValueError(f"Cannot infer LoRA dims for {type(base_layer).__name__}")
        device = weight.device if torch.is_tensor(weight) else torch.device("cpu")
        self.lora_A = nn.Parameter(torch.empty((self.rank, int(in_features)), device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty((int(out_features), self.rank), device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_B)
        self._carrier_positions: Optional[List[List[int]]] = None
        self.last_delta_norm: List[float] = []
        self.last_noncarrier_delta_max: float = 0.0

    def set_context(self, carrier_positions: Sequence[Sequence[int]]) -> None:
        self._carrier_positions = [[int(pos) for pos in row] for row in carrier_positions]
        self.last_delta_norm = []
        self.last_noncarrier_delta_max = 0.0

    def clear_context(self) -> None:
        self._carrier_positions = None

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        if self.dropout > 0.0:
            x_float = F.dropout(x_float, p=float(self.dropout), training=self.training)
        return F.linear(F.linear(x_float, self.lora_A), self.lora_B) * float(self.scaling)

    def _gate_mask(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.gated:
            return None
        if x.dim() < 3 or self._carrier_positions is None:
            return torch.zeros((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
        batch, seq_len = int(x.shape[0]), int(x.shape[1])
        mask = torch.zeros((batch, seq_len, 1), device=x.device, dtype=torch.float32)
        for batch_idx, positions in enumerate(self._carrier_positions[:batch]):
            valid = [int(pos) for pos in positions if 0 <= int(pos) < seq_len]
            if valid:
                mask[batch_idx, torch.tensor(valid, device=x.device, dtype=torch.long), 0] = 1.0
        return mask

    def forward_with_gate(self, x: torch.Tensor, gate_mask: Optional[torch.Tensor]) -> torch.Tensor:
        base_out = self.base_layer(x)
        delta = self._delta(x)
        if gate_mask is not None:
            delta = delta * gate_mask.to(device=delta.device, dtype=delta.dtype)
        if delta.dim() >= 3:
            self.last_delta_norm = [float(v) for v in delta.detach().float().norm(dim=-1).mean(dim=-1).cpu().tolist()]
        elif delta.dim() == 2:
            self.last_delta_norm = [float(v) for v in delta.detach().float().norm(dim=-1).cpu().tolist()]
        if gate_mask is not None and delta.dim() >= 3:
            noncarrier = delta.detach().float() * (1.0 - gate_mask.float())
            self.last_noncarrier_delta_max = float(noncarrier.abs().max().cpu().item()) if noncarrier.numel() else 0.0
        return base_out + delta.to(dtype=base_out.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_gate(x, self._gate_mask(x))


class AttentionLoRAAdapter(nn.Module):
    def __init__(
        self,
        *,
        inject_layers: Sequence[int],
        rank: int,
        alpha: float,
        dropout: float,
        target_modules: Sequence[str],
        gated: bool,
    ) -> None:
        super().__init__()
        self.inject_layers = [int(layer) for layer in inject_layers]
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.target_modules = [str(name) for name in target_modules]
        self.gated = bool(gated)
        self.wrappers = nn.ModuleList()
        self._wrapped: List[Tuple[Any, str, nn.Module, LoRALinearWrapper, int]] = []

    def attach(self, model: Any) -> None:
        if self._wrapped:
            return
        layers = get_layers(model)
        wrapper_idx = 0
        for layer_idx in self.inject_layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"LoRA layer={layer_idx} outside [0, {len(layers) - 1}]")
            attn = getattr(layers[int(layer_idx)], "self_attn", None)
            if attn is None:
                raise RuntimeError(f"layer={layer_idx} has no self_attn")
            for name in self.target_modules:
                base_layer = getattr(attn, name, None)
                if base_layer is None:
                    raise RuntimeError(f"layer={layer_idx}.self_attn has no {name}")
                if wrapper_idx < len(self.wrappers):
                    wrapper = self.wrappers[wrapper_idx]
                    original = getattr(wrapper, "base_layer", None)
                    if original is None:
                        raise RuntimeError("Existing LoRA wrapper lost its base_layer reference")
                else:
                    if isinstance(base_layer, LoRALinearWrapper):
                        original = base_layer.base_layer
                    else:
                        original = base_layer
                    wrapper = LoRALinearWrapper(
                        original,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                        gated=self.gated,
                    )
                    self.wrappers.append(wrapper)
                setattr(attn, name, wrapper)
                self._wrapped.append((attn, name, original, wrapper, int(layer_idx)))
                wrapper_idx += 1

    def detach(self) -> None:
        for parent, name, original, _wrapper, _layer in reversed(self._wrapped):
            setattr(parent, name, original)
        self._wrapped = []

    def set_context(self, batch: FrameMemoryBatch) -> None:
        for wrapper in self.wrappers:
            wrapper.set_context(batch.carrier_positions)

    def clear_context(self) -> None:
        for wrapper in self.wrappers:
            wrapper.clear_context()

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        layer_values: Dict[str, List[float]] = defaultdict(list)
        noncarrier_max = 0.0
        for _parent, name, _original, wrapper, layer_idx in self._wrapped:
            del name
            if row < len(wrapper.last_delta_norm):
                layer_values[str(int(layer_idx))].append(float(wrapper.last_delta_norm[row]))
            noncarrier_max = max(noncarrier_max, float(wrapper.last_noncarrier_delta_max))
        by_layer = {layer: finite_mean(values, default=0.0) for layer, values in layer_values.items()}
        return {
            "lora_delta_norm_by_layer": by_layer,
            "lora_delta_norm": finite_mean(by_layer.values(), default=0.0),
            "noncarrier_lora_update_max": noncarrier_max,
        }


def _repeat_kv(states: torch.Tensor, num_heads: int) -> torch.Tensor:
    if int(states.shape[1]) == int(num_heads):
        return states
    repeats = int(num_heads) // int(states.shape[1])
    return states.repeat_interleave(repeats, dim=1)


def _apply_o_proj(attn: Any, x: torch.Tensor, gate_all: bool) -> torch.Tensor:
    o_proj = attn.o_proj
    if gate_all and isinstance(o_proj, LoRALinearWrapper):
        return o_proj.forward_with_gate(x, torch.ones((*x.shape[:-1], 1), device=x.device, dtype=torch.float32))
    return o_proj(x)


class LayerwiseFrameMessageMemory(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        hidden_size: int,
        memory_dim: int,
        layers: Sequence[int],
        gamma_init: float,
        projection_sharing: str,
        memory_projection_sharing: str,
        message_mode: str,
        reconstruction_tol: float,
        fail_on_reconstruction_error: bool,
    ) -> None:
        super().__init__()
        self.variant = str(variant)
        self.hidden_size = int(hidden_size)
        self.memory_dim = int(memory_dim)
        self.layers = [int(layer) for layer in layers]
        self.layer_to_pos = {int(layer): pos for pos, layer in enumerate(self.layers)}
        self.projection_sharing = str(projection_sharing)
        self.memory_projection_sharing = str(memory_projection_sharing)
        self.message_mode = str(message_mode)
        self.reconstruction_tol = float(reconstruction_tol)
        self.fail_on_reconstruction_error = bool(fail_on_reconstruction_error)
        self.enabled = True
        self.ablation_mode = "normal"
        self.ablation_seed = 0

        n_write = 1 if self.projection_sharing == "shared" else len(self.layers)
        n_mem = 1 if self.memory_projection_sharing == "shared" else len(self.layers)
        self.message_norm = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(n_write)])
        self.slot_norm = nn.ModuleList([nn.LayerNorm(self.memory_dim) for _ in range(n_write)])
        self.carrier_norm = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(n_mem)])
        self.message_to_slot = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.memory_dim, bias=False) for _ in range(n_write)]
        )
        self.w_sum = nn.ModuleList([nn.Linear(self.memory_dim, self.memory_dim, bias=False) for _ in range(n_mem)])
        self.w_q = nn.ModuleList([nn.Linear(self.hidden_size, self.memory_dim, bias=False) for _ in range(n_mem)])
        self.w_k = nn.ModuleList([nn.Linear(self.memory_dim, self.memory_dim, bias=False) for _ in range(n_mem)])
        self.w_v = nn.ModuleList([nn.Linear(self.memory_dim, self.memory_dim, bias=False) for _ in range(n_mem)])
        self.w_out = nn.ModuleList([nn.Linear(self.memory_dim, self.hidden_size, bias=False) for _ in range(n_mem)])
        self.w_out_pna = (
            nn.ModuleList([nn.Linear(12 * self.memory_dim, self.hidden_size, bias=False) for _ in range(n_mem)])
            if self.variant == CARRIER_PNA
            else None
        )
        self.gamma = nn.Parameter(torch.full((len(self.layers),), float(gamma_init), dtype=torch.float32))
        for module in [*self.message_to_slot, *self.w_sum, *self.w_q, *self.w_k, *self.w_v]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)
        for module in (list(self.w_out) + (list(self.w_out_pna) if self.w_out_pna is not None else [])):
            nn.init.zeros_(module.weight)

        self._carrier_positions: Optional[List[List[int]]] = None
        self._carrier_identities: Optional[List[List[str]]] = None
        self._frame_groups: Optional[List[List[List[int]]]] = None
        self._frame_valid_mask: Optional[torch.Tensor] = None
        self._evidence_frame_indices: Optional[List[List[int]]] = None
        self._sample_ids: Optional[List[str]] = None
        self._slots: Optional[torch.Tensor] = None
        self._original_forwards: List[Tuple[Any, Any]] = []
        self._last_stats: Dict[str, Dict[str, Any]] = {}
        self._last_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        self.reconstruction_errors: Dict[int, List[float]] = defaultdict(list)
        self.hook_fire_counts: Dict[int, int] = defaultdict(int)
        self.exact_failure_counts: Dict[str, int] = defaultdict(int)
        self.exact_failure_examples: List[str] = []

    def write_pos(self, layer_idx: int) -> int:
        return 0 if self.projection_sharing == "shared" else self.layer_to_pos[int(layer_idx)]

    def mem_pos(self, layer_idx: int) -> int:
        return 0 if self.memory_projection_sharing == "shared" else self.layer_to_pos[int(layer_idx)]

    def set_context(self, batch: FrameMemoryBatch) -> None:
        self._carrier_positions = [[int(pos) for pos in row] for row in batch.carrier_positions]
        self._carrier_identities = [[str(value) for value in row] for row in batch.carrier_identities]
        self._frame_groups = [[[int(pos) for pos in group] for group in row] for row in batch.frame_groups]
        self._frame_valid_mask = batch.frame_valid_mask
        self._evidence_frame_indices = [[int(x) for x in row] for row in batch.evidence_frame_indices]
        self._sample_ids = list(batch.sample_ids)
        self._slots = None
        self._last_stats = {
            "raw_message_norm_by_layer": {},
            "slot_norm_by_layer": {},
            "read_norm_by_layer": {},
            "injection_norm_by_layer": {},
            "carrier_state_norm_by_layer": {},
            "injection_to_carrier_ratio_by_layer": {},
            "gamma_by_layer": {},
            "effective_rank_by_layer": {},
            "slot_cosine_by_layer": {},
            "reconstruction_error_by_layer": {},
            "message_mode_by_layer": {},
            "tensor_shapes_by_layer": {},
        }
        self._last_tensors = {}

    def clear_context(self) -> None:
        self._carrier_positions = None
        self._carrier_identities = None
        self._frame_groups = None
        self._frame_valid_mask = None
        self._evidence_frame_indices = None
        self._sample_ids = None
        self._slots = None

    def attach(self, model: Any) -> None:
        self.detach()
        layers = get_layers(model)
        for layer_idx in self.layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"memory layer={layer_idx} outside [0, {len(layers) - 1}]")
            layer_module = layers[int(layer_idx)]
            original_forward = layer_module.forward
            adapter = self

            def wrapped_forward(
                module_self: Any,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[Any] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False,
                cache_position: Optional[torch.Tensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                _layer_idx: int = int(layer_idx),
                **kwargs: Any,
            ) -> Any:
                layer_kwargs = dict(kwargs)
                layer_kwargs.update(
                    {
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "past_key_values": past_key_values,
                        "output_attentions": output_attentions,
                        "use_cache": use_cache,
                        "cache_position": cache_position,
                        "position_embeddings": position_embeddings,
                    }
                )
                return adapter.forward_layer(
                    module_self,
                    hidden_states,
                    int(_layer_idx),
                    layer_kwargs,
                )

            layer_module.forward = MethodType(wrapped_forward, layer_module)
            self._original_forwards.append((layer_module, original_forward))

    def detach(self) -> None:
        for layer_module, original_forward in reversed(self._original_forwards):
            layer_module.forward = original_forward
        self._original_forwards = []

    def _record_exact_failure(self, reason: str) -> None:
        key = str(reason).split(":", 1)[0][:80]
        self.exact_failure_counts[key] += 1
        if len(self.exact_failure_examples) < 20:
            self.exact_failure_examples.append(str(reason)[:500])

    def _proxy_messages(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        assert self._carrier_positions is not None and self._frame_groups is not None
        batch, seq_len, hidden = hidden_states.shape
        max_carriers = max(len(row) for row in self._carrier_positions)
        max_frames = max(len(row) for row in self._frame_groups)
        messages = hidden_states.new_zeros((batch, max_carriers, max_frames, hidden), dtype=torch.float32)
        carrier_states = hidden_states.new_zeros((batch, max_carriers, hidden), dtype=torch.float32)
        valid = torch.zeros((batch, max_carriers, max_frames), device=hidden_states.device, dtype=torch.bool)
        source = hidden_states.float()
        for b in range(batch):
            for c, pos in enumerate(self._carrier_positions[b]):
                if 0 <= int(pos) < seq_len:
                    carrier_states[b, c] = source[b, int(pos)]
                for f, group in enumerate(self._frame_groups[b]):
                    frame_positions = [int(x) for x in group if 0 <= int(x) < seq_len]
                    if frame_positions:
                        idx = torch.tensor(frame_positions, device=hidden_states.device, dtype=torch.long)
                        messages[b, c, f] = source[b, idx].mean(dim=0)
                        valid[b, c, f] = True
        return messages, valid, carrier_states, "proxy"

    def _extract_messages(
        self,
        module: Any,
        pre_hidden: torch.Tensor,
        attn_hidden: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, float]:
        if self.message_mode == "proxy":
            messages, valid, carriers, mode = self._proxy_messages(pre_hidden)
            return messages, valid, carriers, mode, 0.0
        try:
            return self._exact_messages(module, pre_hidden, attn_hidden, layer_idx, kwargs)
        except Exception as exc:
            self._record_exact_failure(f"layer {layer_idx}: {type(exc).__name__}: {exc}")
            if self.message_mode == "exact":
                raise
            messages, valid, carriers, mode = self._proxy_messages(pre_hidden)
            return messages, valid, carriers, mode, math.nan

    def _exact_messages(
        self,
        module: Any,
        pre_hidden: torch.Tensor,
        attn_hidden: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, float]:
        del layer_idx
        if self._carrier_positions is None or self._frame_groups is None:
            raise RuntimeError("message context has not been set")
        if not hasattr(module, "input_layernorm") or not hasattr(module, "self_attn"):
            raise RuntimeError("decoder layer does not expose input_layernorm/self_attn")
        attn = module.self_attn
        hs = module.input_layernorm(pre_hidden)
        batch, seq_len, hidden = hs.shape
        q = attn.q_proj(hs)
        k = attn.k_proj(hs)
        v = attn.v_proj(hs)
        head_dim = int(getattr(attn, "head_dim", q.shape[-1] // int(getattr(attn, "num_heads", 1))))
        num_heads = int(getattr(attn, "num_heads", q.shape[-1] // head_dim))
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, -1, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, -1, head_dim).transpose(1, 2)
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and apply_multimodal_rotary_pos_emb is not None and hasattr(attn, "rope_scaling"):
            q, k = apply_multimodal_rotary_pos_emb(
                q,
                k,
                position_embeddings[0],
                position_embeddings[1],
                attn.rope_scaling["mrope_section"],
            )
        elif position_embeddings is not None and apply_multimodal_rotary_pos_emb is not None and hasattr(attn, "rotary_emb"):
            pass
        elif "qwen" in type(module).__name__.lower() and apply_multimodal_rotary_pos_emb is not None:
            raise RuntimeError("exact unavailable: Qwen layer kwargs did not include multimodal position embeddings")
        k = _repeat_kv(k, num_heads)
        v = _repeat_kv(v, num_heads)
        scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
        attention_mask = kwargs.get("attention_mask")
        max_carriers = max(len(row) for row in self._carrier_positions)
        max_frames = max(len(row) for row in self._frame_groups)
        messages = hs.new_zeros((batch, max_carriers, max_frames, hidden), dtype=torch.float32)
        carrier_states = hs.new_zeros((batch, max_carriers, hidden), dtype=torch.float32)
        valid = torch.zeros((batch, max_carriers, max_frames), device=hs.device, dtype=torch.bool)
        arange = torch.arange(seq_len, device=hs.device)
        errors: List[torch.Tensor] = []
        for b in range(batch):
            carriers = [int(pos) for pos in self._carrier_positions[b] if 0 <= int(pos) < seq_len]
            if not carriers:
                continue
            c_idx = torch.tensor(carriers, device=hs.device, dtype=torch.long)
            carrier_states[b, : len(carriers)] = (pre_hidden[b, c_idx] + attn_hidden[b, c_idx]).float()
            scores = torch.einsum("hcd,hsd->hcs", q[b, :, c_idx, :].float(), k[b].float()) * scaling
            causal_allowed = arange.unsqueeze(0) <= c_idx.unsqueeze(1)
            sliding_window = getattr(attn, "sliding_window", None)
            if sliding_window is not None:
                causal_allowed &= arange.unsqueeze(0) >= (c_idx.unsqueeze(1) - int(sliding_window))
            scores = scores.masked_fill(~causal_allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
            if torch.is_tensor(attention_mask):
                if attention_mask.dim() == 4:
                    selected_mask = attention_mask[b : b + 1, :, c_idx, :].float()
                    scores = scores + selected_mask.squeeze(0)
                elif attention_mask.dim() == 2:
                    src_valid = attention_mask[b].bool()
                    scores = scores.masked_fill(~src_valid.view(1, 1, -1), torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            context_all = torch.einsum("hcs,hsd->hcd", probs, v[b].float())
            context_flat = context_all.permute(1, 0, 2).reshape(len(carriers), num_heads * head_dim)
            reconstructed = _apply_o_proj(attn, context_flat.to(dtype=hs.dtype), gate_all=True).float()
            target = attn_hidden[b, c_idx, :].float()
            denom = target.norm(dim=-1).clamp_min(1e-6)
            candidate_errors = [((reconstructed - target).norm(dim=-1) / denom).detach()]
            # SDPA kernels can differ slightly from the explicit eager decomposition in BF16.
            # Compare the guard against a backend-consistent carrier reconstruction when possible.
            attn_impl = str(getattr(getattr(attn, "config", None), "_attn_implementation", ""))
            dropout_p = float(getattr(attn, "attention_dropout", 0.0)) if bool(getattr(attn, "training", False)) else 0.0
            if attn_impl == "sdpa" and dropout_p == 0.0:
                try:
                    selected_mask: Optional[torch.Tensor] = None
                    if torch.is_tensor(attention_mask):
                        if attention_mask.dim() == 4:
                            selected_mask = attention_mask[b : b + 1].index_select(2, c_idx)[..., :seq_len]
                        elif attention_mask.dim() == 2:
                            src_valid = attention_mask[b, :seq_len].bool()
                            allowed = causal_allowed & src_valid.view(1, -1)
                            selected_mask = torch.zeros(
                                (1, 1, len(carriers), seq_len),
                                device=hs.device,
                                dtype=q.dtype,
                            ).masked_fill(~allowed.view(1, 1, len(carriers), seq_len), torch.finfo(q.dtype).min)
                    else:
                        selected_mask = torch.zeros(
                            (1, 1, len(carriers), seq_len),
                            device=hs.device,
                            dtype=q.dtype,
                        ).masked_fill(~causal_allowed.view(1, 1, len(carriers), seq_len), torch.finfo(q.dtype).min)
                    backend_context = F.scaled_dot_product_attention(
                        q[b : b + 1].index_select(2, c_idx),
                        k[b : b + 1],
                        v[b : b + 1],
                        attn_mask=selected_mask,
                        dropout_p=0.0,
                        scale=scaling,
                        is_causal=False,
                    )
                    backend_flat = backend_context.squeeze(0).transpose(0, 1).reshape(len(carriers), num_heads * head_dim)
                    backend_reconstructed = _apply_o_proj(attn, backend_flat.to(dtype=hs.dtype), gate_all=True).float()
                    candidate_errors.append(((backend_reconstructed - target).norm(dim=-1) / denom).detach())
                except Exception as exc:
                    self._record_exact_failure(f"layer {layer_idx}: backend reconstruction skipped: {type(exc).__name__}: {exc}")
            errors.append(torch.stack(candidate_errors, dim=0).min(dim=0).values)
            for local_c, _carrier_pos in enumerate(carriers):
                for f, group in enumerate(self._frame_groups[b]):
                    frame_positions = [int(x) for x in group if 0 <= int(x) < seq_len]
                    if not frame_positions:
                        continue
                    f_idx = torch.tensor(frame_positions, device=hs.device, dtype=torch.long)
                    contrib = torch.einsum("hf,hfd->hd", probs[:, local_c, f_idx], v[b, :, f_idx, :].float())
                    contrib_flat = contrib.reshape(1, num_heads * head_dim)
                    projected = _apply_o_proj(attn, contrib_flat.to(dtype=hs.dtype), gate_all=True).float()[0]
                    messages[b, local_c, f] = projected
                    valid[b, local_c, f] = True
        error = float(torch.cat(errors).max().cpu().item()) if errors else 0.0
        return messages, valid, carrier_states, "exact", error

    def _call_self_attn(self, module: Any, normed: torch.Tensor, kwargs: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[Any], bool]:
        attn = module.self_attn
        try:
            result = attn(
                hidden_states=normed,
                attention_mask=kwargs.get("attention_mask"),
                position_ids=kwargs.get("position_ids"),
                past_key_values=kwargs.get("past_key_values"),
                output_attentions=kwargs.get("output_attentions", False),
                use_cache=kwargs.get("use_cache", False),
                cache_position=kwargs.get("cache_position"),
                position_embeddings=kwargs.get("position_embeddings"),
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key
                    not in {
                        "attention_mask",
                        "position_ids",
                        "past_key_values",
                        "output_attentions",
                        "use_cache",
                        "cache_position",
                        "position_embeddings",
                    }
                    and value is not None
                },
            )
        except TypeError:
            result = attn(normed, attention_mask=kwargs.get("attention_mask"))
        if torch.is_tensor(result):
            return result, None, False
        if isinstance(result, (tuple, list)) and result and torch.is_tensor(result[0]):
            weights = result[1] if len(result) > 1 else None
            return result[0], weights, True
        raise RuntimeError(f"Unsupported self-attention output type: {type(result).__name__}")

    def forward_layer(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> Any:
        residual = hidden_states
        normed = module.input_layernorm(hidden_states)
        attn_out, attn_weights, tuple_output = self._call_self_attn(module, normed, kwargs)
        h_attn = residual + attn_out
        h_injected = h_attn
        if self.enabled and self._carrier_positions is not None:
            h_injected = self.inject_after_attention(module, hidden_states, attn_out, h_attn, layer_idx, kwargs)
        residual = h_injected
        mlp_hidden = module.post_attention_layernorm(h_injected)
        mlp_out = module.mlp(mlp_hidden)
        output_hidden = residual + mlp_out
        if tuple_output:
            outputs: Tuple[Any, ...] = (output_hidden,)
            if kwargs.get("output_attentions", False):
                outputs += (attn_weights,)
            return outputs
        return output_hidden

    def _init_slots(self, messages: torch.Tensor) -> torch.Tensor:
        batch, carriers, frames, _hidden = messages.shape
        return messages.new_zeros((batch, carriers, frames, self.memory_dim), dtype=torch.float32)

    def _apply_ablation(self, slots: torch.Tensor, valid: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = str(self.ablation_mode)
        if mode == "normal":
            return slots, valid
        slots_eff = slots.clone()
        valid_eff = valid.clone()
        if mode == "zero_memory":
            return torch.zeros_like(slots_eff), valid_eff
        if mode == "shuffle_memory_between_samples":
            if slots_eff.shape[0] > 1:
                return slots_eff.roll(shifts=1, dims=0), valid_eff.roll(shifts=1, dims=0)
            return slots_eff.flip(dims=(2,)), valid_eff.flip(dims=(2,))
        if mode == "replace_slots_with_norm_matched_noise":
            generator = torch.Generator(device=slots.device)
            seed_text = f"{self.ablation_seed}:{layer_idx}:{self._sample_ids}"
            generator.manual_seed(int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:8], 16))
            noise = torch.randn(slots_eff.shape, device=slots_eff.device, dtype=slots_eff.dtype, generator=generator)
            noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-6) * slots_eff.norm(dim=-1, keepdim=True)
            return torch.where(valid_eff.unsqueeze(-1), noise, torch.zeros_like(noise)), valid_eff
        if self._evidence_frame_indices is None:
            return slots_eff, valid_eff
        for b in range(slots_eff.shape[0]):
            evidence = sorted({int(x) for x in self._evidence_frame_indices[b] if 0 <= int(x) < slots_eff.shape[2]})
            neutral = [idx for idx in range(slots_eff.shape[2]) if idx not in set(evidence)]
            target: Optional[int] = None
            if mode in {"remove_one_evidence_slot", "duplicate_one_evidence_slot"} and evidence:
                target = evidence[0]
            elif mode == "remove_one_neutral_slot" and neutral:
                target = neutral[0]
            if target is None:
                continue
            if mode.startswith("remove"):
                slots_eff[b, :, target, :] = 0.0
                valid_eff[b, :, target] = False
            elif mode == "duplicate_one_evidence_slot":
                slots_eff[b, :, target, :] = slots_eff[b, :, target, :] * 2.0
        return slots_eff, valid_eff

    def inject_after_attention(
        self,
        module: Any,
        pre_hidden: torch.Tensor,
        attn_hidden: torch.Tensor,
        h_attn: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        if int(layer_idx) not in self.layer_to_pos:
            return h_attn
        self.hook_fire_counts[int(layer_idx)] += 1
        messages, valid, carrier_states, mode, recon_error = self._extract_messages(
            module,
            pre_hidden,
            attn_hidden,
            int(layer_idx),
            kwargs,
        )
        if self._slots is None:
            self._slots = self._init_slots(messages)
        if self._slots.shape[:3] != messages.shape[:3]:
            raise RuntimeError(
                "Memory slot shape changed within a forward pass; this would append frames instead of updating them"
            )
        wpos = self.write_pos(int(layer_idx))
        mpos = self.mem_pos(int(layer_idx))
        z = self.message_to_slot[wpos](self.message_norm[wpos](messages.float()))
        candidate = self.slot_norm[wpos](self._slots + z.float())
        self._slots = torch.where(valid.unsqueeze(-1), candidate, self._slots)
        slots_for_read, valid_for_read = self._apply_ablation(self._slots, valid, int(layer_idx))
        batch, max_carriers, max_frames, _d = slots_for_read.shape
        if self.variant == CARRIER_DIRECT_SUM:
            read = (self.w_sum[mpos](slots_for_read) * valid_for_read.unsqueeze(-1).float()).sum(dim=2)
            matrix_shape = [batch, max_carriers, self.memory_dim]
        elif self.variant == CARRIER_PNA:
            # PNA readout over frames: [sum, mean, max, std] x [identity, amplify, attenuate].
            # Degree = number of valid frame-slots per carrier. Generalizes the sum read.
            s = self.w_sum[mpos](slots_for_read).float()
            mask = valid_for_read.unsqueeze(-1).float()
            s = s * mask
            n = valid_for_read.float().sum(dim=2, keepdim=True).clamp_min(1.0)  # [b, c, 1]
            agg_sum = s.sum(dim=2)
            agg_mean = agg_sum / n
            s_for_max = s.masked_fill(~valid_for_read.unsqueeze(-1), float("-inf"))
            agg_max = torch.nan_to_num(s_for_max.amax(dim=2), neginf=0.0)
            var = (((s - agg_mean.unsqueeze(2)) ** 2) * mask).sum(dim=2) / n
            agg_std = var.clamp_min(0.0).sqrt()
            aggs = [agg_sum, agg_mean, agg_max, agg_std]
            amplify = torch.log(n + 1.0)  # [b, c, 1]
            attenuate = 1.0 / amplify.clamp_min(1e-6)
            streams = [agg * scaler for scaler in (torch.ones_like(amplify), amplify, attenuate) for agg in aggs]
            read = torch.cat(streams, dim=-1)  # [b, c, 12*memory_dim]
            matrix_shape = [batch, max_carriers, 12 * self.memory_dim]
        elif self.variant == CARRIER_GLSTM_SOFTMAX:
            # Softmax-normalized read control: identical q/k/v machinery to the
            # associative read, but the per-frame contributions are normalized to a
            # convex combination (weighted mean) instead of an unnormalized sum.
            k = self.w_k[mpos](slots_for_read).float()
            v = self.w_v[mpos](slots_for_read).float()
            q = self.w_q[mpos](self.carrier_norm[mpos](carrier_states.float())).float()
            logits = torch.einsum("bcfe,bce->bcf", k, q) / math.sqrt(float(self.memory_dim))
            logits = logits.masked_fill(~valid_for_read, float("-inf"))
            weights = torch.softmax(logits, dim=2)
            weights = torch.nan_to_num(weights, nan=0.0)
            read = torch.einsum("bcf,bcfd->bcd", weights, v)
            matrix_shape = [batch, max_carriers, max_frames]
        else:
            k = self.w_k[mpos](slots_for_read).float()
            v = self.w_v[mpos](slots_for_read).float()
            q = self.w_q[mpos](self.carrier_norm[mpos](carrier_states.float())).float()
            matrix = torch.einsum("bcfd,bcfe,bcf->bcde", v, k, valid_for_read.float())
            read = torch.einsum("bcde,bce->bcd", matrix, q)
            matrix_shape = list(matrix.shape)
        should_inject = self.variant != CARRIER_GLSTM_FINAL_ONLY or int(layer_idx) == max(self.layers)
        if self.variant == CARRIER_PNA:
            injection = self.w_out_pna[mpos](read).float()
        else:
            injection = self.w_out[mpos](read).float()
        injection = self.gamma[self.layer_to_pos[int(layer_idx)]].float() * injection
        out = h_attn.clone()
        batch_size, seq_len, _hidden = h_attn.shape
        carrier_norms: List[float] = []
        injection_norms: List[float] = []
        ratios: List[float] = []
        if should_inject:
            assert self._carrier_positions is not None
            for b, positions in enumerate(self._carrier_positions):
                for c, pos in enumerate(positions):
                    if c >= injection.shape[1] or not (0 <= int(pos) < seq_len):
                        continue
                    update = injection[b, c].to(dtype=h_attn.dtype)
                    before = out[b, int(pos), :].float()
                    out[b, int(pos), :] = out[b, int(pos), :] + update
                    inj_norm = float(update.detach().float().norm().cpu().item())
                    state_norm = float(before.detach().float().norm().cpu().item())
                    injection_norms.append(inj_norm)
                    carrier_norms.append(state_norm)
                    ratios.append(inj_norm / max(state_norm, 1e-6))
        else:
            for b in range(batch_size):
                for c in range(max_carriers):
                    inj_norm = float(injection[b, c].detach().float().norm().cpu().item())
                    state_norm = float(carrier_states[b, c].detach().float().norm().cpu().item())
                    injection_norms.append(0.0 * inj_norm)
                    carrier_norms.append(state_norm)
                    ratios.append(0.0)
        layer_key = str(int(layer_idx))
        self.reconstruction_errors[int(layer_idx)].append(float(recon_error))
        if bool(torch.isfinite(torch.tensor(recon_error))) and float(recon_error) > self.reconstruction_tol and self.message_mode == "exact":
            message = f"Attention message reconstruction error {float(recon_error):.6g} exceeds tolerance {self.reconstruction_tol}"
            self._record_exact_failure(f"layer {layer_idx}: {message}")
            if self.fail_on_reconstruction_error:
                raise RuntimeError(message)
        slot_float = self._slots.detach().float()
        slot_norm = slot_float.norm(dim=-1)
        flat_slots = slot_float.reshape(-1, slot_float.shape[-1])
        nonzero_slots = flat_slots[flat_slots.norm(dim=-1) > 1e-8]
        if int(nonzero_slots.shape[0]) >= 2:
            normed = F.normalize(nonzero_slots, dim=-1)
            cosine = float((normed @ normed.T).triu(diagonal=1).mean().cpu().item())
            singular = torch.linalg.svdvals(nonzero_slots)
            effective_rank = float((singular > singular.max().clamp_min(1e-8) * 1e-3).sum().cpu().item())
        else:
            cosine = 0.0
            effective_rank = float(int(nonzero_slots.shape[0]))
        self._last_stats["raw_message_norm_by_layer"][layer_key] = [float(messages.detach().float().norm(dim=-1).mean().cpu().item())]
        self._last_stats["slot_norm_by_layer"][layer_key] = [float(slot_norm.mean().cpu().item())]
        self._last_stats["read_norm_by_layer"][layer_key] = [float(read.detach().float().norm(dim=-1).mean().cpu().item())]
        self._last_stats["injection_norm_by_layer"][layer_key] = [finite_mean(injection_norms, default=0.0)]
        self._last_stats["carrier_state_norm_by_layer"][layer_key] = [finite_mean(carrier_norms, default=0.0)]
        self._last_stats["injection_to_carrier_ratio_by_layer"][layer_key] = [finite_mean(ratios, default=0.0)]
        self._last_stats["gamma_by_layer"][layer_key] = [float(self.gamma[self.layer_to_pos[int(layer_idx)]].detach().cpu().item())]
        self._last_stats["effective_rank_by_layer"][layer_key] = [effective_rank]
        self._last_stats["slot_cosine_by_layer"][layer_key] = [cosine]
        self._last_stats["reconstruction_error_by_layer"][layer_key] = [float(recon_error)]
        self._last_stats["message_mode_by_layer"][layer_key] = [mode]
        self._last_stats["tensor_shapes_by_layer"][layer_key] = [{
            "raw_message": list(messages.shape),
            "slot": list(self._slots.shape),
            "matrix_memory": matrix_shape,
            "read": list(read.shape),
            "injection": list(injection.shape),
        }]
        self._last_tensors[layer_key] = {
            "raw_messages": messages.detach().float().cpu(),
            "slots": self._slots.detach().float().cpu(),
            "read": read.detach().float().cpu(),
            "injection": injection.detach().float().cpu(),
            "valid": valid.detach().cpu(),
        }
        return out

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        del row
        out: Dict[str, Any] = {}
        for key, by_layer in self._last_stats.items():
            payload: Dict[str, Any] = {}
            for layer, values in by_layer.items():
                if isinstance(values, list) and values:
                    payload[str(layer)] = values[0]
                else:
                    payload[str(layer)] = values
            out[key] = payload
        out["memory_gamma"] = {str(layer): float(self.gamma[pos].detach().cpu().item()) for layer, pos in self.layer_to_pos.items()}
        out["memory_reconstruction_error"] = finite_mean(
            [value for values in out.get("reconstruction_error_by_layer", {}).values() for value in ([values] if not isinstance(values, list) else values)],
            default=math.nan,
        )
        return out

    def diagnostic_tensors(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return self._last_tensors


class ExperimentAdapter(nn.Module):
    def __init__(self, lora: Optional[AttentionLoRAAdapter], memory: Optional[LayerwiseFrameMessageMemory]) -> None:
        super().__init__()
        self.lora = lora
        self.memory = memory

    def attach(self, model: Any) -> None:
        if self.lora is not None:
            self.lora.attach(model)
        if self.memory is not None:
            self.memory.attach(model)

    def detach(self) -> None:
        if self.memory is not None:
            self.memory.detach()
        if self.lora is not None:
            self.lora.detach()

    def set_context(self, batch: FrameMemoryBatch) -> None:
        if self.lora is not None:
            self.lora.set_context(batch)
        if self.memory is not None:
            self.memory.set_context(batch)

    def clear_context(self) -> None:
        if self.memory is not None:
            self.memory.clear_context()
        if self.lora is not None:
            self.lora.clear_context()

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if self.lora is not None:
            stats.update(self.lora.stats_for_row(row))
        if self.memory is not None:
            stats.update(self.memory.stats_for_row(row))
        return stats

    def set_memory_enabled(self, enabled: bool) -> None:
        if self.memory is not None:
            self.memory.enabled = bool(enabled)

    def set_ablation(self, mode: str, seed: int = 0) -> None:
        if self.memory is not None:
            self.memory.ablation_mode = str(mode)
            self.memory.ablation_seed = int(seed)


def make_adapter(args: argparse.Namespace, variant: str, hidden_size: int, layers: Sequence[int]) -> ExperimentAdapter:
    if bool(args.frame_kv_lora):
        raise NotImplementedError("--frame-kv-lora is reserved for follow-up sweeps and is not part of this matrix")
    if variant == GLOBAL_LORA:
        lora = AttentionLoRAAdapter(
            inject_layers=layers,
            rank=int(args.lora_rank),
            alpha=float(args.lora_alpha),
            dropout=float(args.lora_dropout),
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            gated=False,
        )
        return ExperimentAdapter(lora=lora, memory=None)
    lora = AttentionLoRAAdapter(
        inject_layers=layers,
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        dropout=float(args.lora_dropout),
        target_modules=("q_proj", "o_proj"),
        gated=True,
    )
    memory = None
    if variant in MEMORY_VARIANTS:
        memory = LayerwiseFrameMessageMemory(
            variant=variant,
            hidden_size=int(hidden_size),
            memory_dim=int(args.memory_dim),
            layers=layers,
            gamma_init=float(args.gamma_init),
            projection_sharing=str(args.projection_sharing),
            memory_projection_sharing=str(args.memory_projection_sharing),
            message_mode=str(args.message_mode),
            reconstruction_tol=float(args.reconstruction_tol),
            fail_on_reconstruction_error=bool(args.fail_on_reconstruction_error),
        )
    if bool(getattr(args, "no_carrier_lora", False)) and memory is not None:
        lora = None  # memory-only control: isolate the memory adapter's contribution
    return ExperimentAdapter(lora=lora, memory=memory)


def trainable_parameter_summary(model: Any, adapter: ExperimentAdapter) -> Dict[str, Any]:
    total_model = sum(int(param.numel()) for param in model.parameters())
    trainable_model_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_A" not in name and "lora_B" not in name
    ]
    if trainable_model_names:
        raise RuntimeError(f"Unexpected trainable Qwen base parameters: {trainable_model_names[:20]}")
    trainable: List[Dict[str, Any]] = []
    grouped = {"lora": 0, "memory": 0, "other": 0}
    for name, param in adapter.named_parameters():
        if not param.requires_grad:
            continue
        count = int(param.numel())
        trainable.append({"name": name, "shape": list(param.shape), "numel": count})
        if "lora_" in name or ".lora" in name or "wrappers" in name:
            grouped["lora"] += count
        elif any(key in name for key in ("message_to_slot", "w_", "gamma", "norm")):
            grouped["memory"] += count
        else:
            grouped["other"] += count
    total_parameters = int(total_model + grouped["memory"] + grouped["other"])
    return {
        "total_parameter_count": total_parameters,
        "total_model_parameters_including_attached_lora": int(total_model),
        "trainable_model_parameter_tensors": trainable_model_names,
        "trainable_adapter_parameters": int(sum(row["numel"] for row in trainable)),
        "trainable_parameter_names": [row["name"] for row in trainable],
        "trainable_parameters": trainable,
        "groups": grouped,
    }


def unexpected_frozen_model_grads(model: Any) -> List[str]:
    bad: List[str] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "lora_A" in name or "lora_B" in name:
            continue
        if param.requires_grad:
            bad.append(name)
            continue
        bad.append(name)
    return bad


def answer_sequence_cross_entropy(logits: torch.Tensor, batch: FrameMemoryBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    if batch.loss_positions is None or batch.loss_targets is None:
        raise RuntimeError("Answer-token loss requested without targets")
    positions = batch.loss_positions.clamp_min(0)
    batch_idx = torch.arange(int(logits.shape[0]), device=logits.device).unsqueeze(1)
    selected = logits[batch_idx, positions, :].float()
    targets = batch.loss_targets
    losses = F.cross_entropy(
        selected.reshape(-1, selected.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)
    valid = targets.ne(-100)
    row_losses = (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    return row_losses.mean(), row_losses


def candidate_score_from_logits(logits: torch.Tensor, batch: FrameMemoryBatch) -> float:
    if batch.loss_positions is None or batch.loss_targets is None:
        raise RuntimeError("Candidate scoring requires loss targets")
    positions = batch.loss_positions.clamp_min(0)
    batch_idx = torch.arange(int(logits.shape[0]), device=logits.device).unsqueeze(1)
    selected = logits[batch_idx, positions, :].float()
    log_probs = selected.log_softmax(dim=-1)
    gathered = log_probs.gather(-1, batch.loss_targets.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum(dim=1)[0].detach().cpu().item())


@torch.no_grad()
def predict_count(
    *,
    model: Any,
    processor: Any,
    adapter: Optional[ExperimentAdapter],
    example: FrameMemoryExample,
    sample_idx: int,
    answer_ids: Dict[int, Tuple[int, ...]],
    count_values: Sequence[int],
    device: str,
) -> Tuple[int, Dict[str, float]]:
    scores: Dict[str, float] = {}
    for candidate in count_values:
        batch = prepare_batch(
            examples=[example],
            sample_indices=[sample_idx],
            processor=processor,
            device=device,
            answer_ids=answer_ids,
            answer_count_override=int(candidate),
        )
        if adapter is not None:
            adapter.set_context(batch)
        outputs = model(**batch.inputs, use_cache=False)
        if adapter is not None:
            adapter.clear_context()
        scores[str(int(candidate))] = candidate_score_from_logits(outputs.logits, batch)
    pred = max((int(key) for key in scores), key=lambda value: scores[str(value)])
    return int(pred), scores


def limited_indices(examples: Sequence[Any], limit: int, seed: int) -> List[int]:
    return text_base.limited_indices(examples, int(limit), int(seed))


def batch_indices(indices: Sequence[int], batch_size: int, seed: int, shuffle: bool) -> List[List[int]]:
    if int(batch_size) != 1:
        raise ValueError("--batch-size must be 1 for visual frame alignment; use --grad-accum for effective batch size")
    return text_base.batch_indices(indices, 1, int(seed), bool(shuffle))


def train_adapter(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant: str,
    model: Any,
    processor: Any,
    adapter: ExperimentAdapter,
    examples: Dict[str, List[FrameMemoryExample]],
    answer_ids: Dict[int, Tuple[int, ...]],
    device: str,
) -> Tuple[List[Dict[str, Any]], Path]:
    train_indices = limited_indices(examples[TRAIN_SPLIT], int(args.max_train_examples), int(args.seed))
    val_indices = limited_indices(examples[VAL_SPLIT], int(args.max_eval_examples), int(args.seed) + 17)
    adapter.attach(model)
    params = [param for param in adapter.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable adapter parameters")
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "adapter_best.pt"
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_acc = -math.inf
    backward_diag: Dict[str, Any] = {}
    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            adapter.train()
            shuffled = list(train_indices)
            random.Random(int(args.seed) + epoch * 7919).shuffle(shuffled)
            optimizer.zero_grad(set_to_none=True)
            train_loss_total = 0.0
            train_steps = 0
            backward_steps = 0
            max_steps = 1 if bool(args.smoke_test) else math.inf
            for step, idxs in enumerate(batch_indices(shuffled, int(args.batch_size), int(args.seed) + epoch, True), start=1):
                idx = int(idxs[0])
                example = examples[TRAIN_SPLIT][idx]
                batch = prepare_batch(
                    examples=[example],
                    sample_indices=[idx],
                    processor=processor,
                    device=device,
                    answer_ids=answer_ids,
                )
                adapter.set_context(batch)
                outputs = model(**batch.inputs, use_cache=False)
                loss, _row_loss = answer_sequence_cross_entropy(outputs.logits, batch)
                adapter.clear_context()
                (loss / max(1, int(args.grad_accum))).backward()
                train_loss_total += float(loss.detach().cpu().item())
                train_steps += 1
                if not backward_diag:
                    bad_model_grads = unexpected_frozen_model_grads(model)
                    adapter_grad_tensors = sum(1 for param in adapter.parameters() if param.grad is not None)
                    backward_diag = {
                        "unexpected_model_grad_tensors": int(len(bad_model_grads)),
                        "adapter_grad_tensors": int(adapter_grad_tensors),
                        "frozen_parameters_no_grad": int(not bad_model_grads),
                    }
                    if bad_model_grads:
                        raise RuntimeError(f"Frozen Qwen parameters received gradients: {bad_model_grads[:20]}")
                if step % max(1, int(args.grad_accum)) == 0:
                    torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    backward_steps += 1
                if train_steps >= max_steps:
                    break
            if train_steps and train_steps % max(1, int(args.grad_accum)) != 0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                backward_steps += 1
            val_eval = evaluate_split(
                variant=variant,
                split_name=VAL_SPLIT,
                model=model,
                processor=processor,
                adapter=adapter,
                examples=examples[VAL_SPLIT],
                indices=val_indices[: max(1, min(len(val_indices), 4 if args.smoke_test else len(val_indices)))],
                answer_ids=answer_ids,
                count_values=list(range(int(args.candidate_min), int(args.candidate_max) + 1)),
                device=device,
                seed=int(args.seed),
                collect_diagnostics=False,
                manage_attachment=False,
            )
            val_metrics = summarize_prediction_rows(val_eval["rows"])
            row = {
                "variant": variant,
                "epoch": epoch,
                "train_loss": train_loss_total / max(1, train_steps),
                "train_steps": train_steps,
                "optimizer_steps": backward_steps,
                "val_accuracy": val_metrics["accuracy"],
                "val_mae": val_metrics["mae"],
                **backward_diag,
            }
            history.append(row)
            print(
                f"[{variant}] epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"val_acc={row['val_accuracy']:.3f} val_mae={row['val_mae']:.3f}"
            )
            if float(row["val_accuracy"]) >= best_val_acc:
                best_val_acc = float(row["val_accuracy"])
                best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
                torch.save(best_state, checkpoint_path)
    finally:
        adapter.detach()
    if best_state is not None:
        adapter.load_state_dict(best_state)
    else:
        torch.save(adapter.state_dict(), checkpoint_path)
    return history, checkpoint_path


@torch.no_grad()
def evaluate_split(
    *,
    variant: str,
    split_name: str,
    model: Any,
    processor: Any,
    adapter: Optional[ExperimentAdapter],
    examples: Sequence[FrameMemoryExample],
    indices: Sequence[int],
    answer_ids: Dict[int, Tuple[int, ...]],
    count_values: Sequence[int],
    device: str,
    seed: int,
    collect_diagnostics: bool,
    diagnostic_dir: Optional[Path] = None,
    manage_attachment: bool = True,
) -> Dict[str, Any]:
    model.eval()
    if adapter is not None:
        adapter.eval()
        if manage_attachment:
            adapter.attach(model)
    rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    try:
        for order, idx in enumerate(indices, start=1):
            example = examples[int(idx)]
            batch = prepare_batch(
                examples=[example],
                sample_indices=[int(idx)],
                processor=processor,
                device=device,
                answer_ids=answer_ids,
            )
            if adapter is not None:
                adapter.set_context(batch)
            outputs = model(**batch.inputs, use_cache=False)
            ce, ce_vec = answer_sequence_cross_entropy(outputs.logits, batch)
            stats = adapter.stats_for_row(0) if adapter is not None else {}
            diag_tensors = adapter.memory.diagnostic_tensors() if adapter is not None and adapter.memory is not None else {}
            if adapter is not None:
                adapter.clear_context()
            pred, candidate_scores = predict_count(
                model=model,
                processor=processor,
                adapter=adapter,
                example=example,
                sample_idx=int(idx),
                answer_ids=answer_ids,
                count_values=count_values,
                device=device,
            )
            density = float(example.gold_count) / max(1, int(example.num_frames))
            row: Dict[str, Any] = {
                "variant": variant,
                "split": split_name,
                "example_id": example.example_id,
                "sample_index": int(idx),
                "num_frames": int(example.num_frames),
                "gold_count": int(example.gold_count),
                "predicted_count": int(pred),
                "correct": int(int(pred) == int(example.gold_count)),
                "abs_error": abs(int(pred) - int(example.gold_count)),
                "evidence_density": density,
                "evidence_frame_indices": list(example.evidence_frame_indices),
                "composition_key": example.composition_key,
                "queried_character": example.queried_character,
                "queried_room": example.queried_room,
                "template_id": example.template_id,
                "carrier_token_positions": list(batch.carrier_positions[0]),
                "carrier_identities": list(batch.carrier_identities[0]),
                "visual_token_spans_per_frame": [list(group) for group in batch.frame_groups[0]],
                "visual_token_counts_per_frame": [len(group) for group in batch.frame_groups[0]],
                "ce": float(ce_vec[0].detach().cpu().item()),
                "candidate_scores": candidate_scores,
                **stats,
            }
            rows.append(row)
            if collect_diagnostics and diagnostic_dir is not None and len(diagnostic_rows) < 256:
                saved_layers: Dict[str, str] = {}
                sample_stem = safe_name(f"{split_name}_{order:04d}_{example.example_id}")
                for layer, tensors in diag_tensors.items():
                    npz_path = diagnostic_dir / f"{sample_stem}_layer{layer}.npz"
                    np.savez_compressed(
                        npz_path,
                        **{key: tensor.numpy() for key, tensor in tensors.items()},
                    )
                    saved_layers[str(layer)] = os.fspath(npz_path.relative_to(diagnostic_dir.parent))
                diagnostic_rows.append({
                    "variant": variant,
                    "split": split_name,
                    "example_id": example.example_id,
                    "gold_count": int(example.gold_count),
                    "num_frames": int(example.num_frames),
                    "evidence_frame_indices": list(example.evidence_frame_indices),
                    "carrier_identities": list(batch.carrier_identities[0]),
                    "layers": saved_layers,
                })
    finally:
        if adapter is not None and manage_attachment:
            adapter.detach()
    if diagnostic_rows and diagnostic_dir is not None:
        write_jsonl(diagnostic_dir / f"{split_name}_diagnostics_manifest.jsonl", diagnostic_rows)
    return {"rows": rows, "diagnostics": diagnostic_rows}


def summarize_prediction_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"accuracy": math.nan, "mae": math.nan, "n": 0}
    y_true = [int(row["gold_count"]) for row in rows]
    y_pred = [int(row["predicted_count"]) for row in rows]
    return {
        "n": len(rows),
        "accuracy": accuracy(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }


def metrics_from_rows(variant: str, rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    split_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    for split in sorted({str(row["split"]) for row in rows}):
        split_data = [row for row in rows if str(row["split"]) == split]
        summary = summarize_prediction_rows(split_data)
        split_rows.append({"variant": variant, "split": split, **summary})
        for count in sorted({int(row["gold_count"]) for row in split_data}):
            count_data = [row for row in split_data if int(row["gold_count"]) == count]
            count_summary = summarize_prediction_rows(count_data)
            group_rows.append({
                "variant": variant,
                "split": split,
                "group": "count",
                "value": count,
                "mean_predicted_count": finite_mean([row["predicted_count"] for row in count_data], default=math.nan),
                **count_summary,
            })
        for length in sorted({int(row["num_frames"]) for row in split_data}):
            length_data = [row for row in split_data if int(row["num_frames"]) == length]
            group_rows.append({
                "variant": variant,
                "split": split,
                "group": "sequence_length",
                "value": length,
                **summarize_prediction_rows(length_data),
            })
    iid = next((row for row in split_rows if row["split"] == IID_TEST_SPLIT), None)
    for row in split_rows:
        row["iid_to_split_accuracy_drop"] = (
            float(iid["accuracy"]) - float(row["accuracy"]) if iid and math.isfinite(float(iid["accuracy"])) else math.nan
        )
        row["iid_to_split_mae_increase"] = (
            float(row["mae"]) - float(iid["mae"]) if iid and math.isfinite(float(iid["mae"])) else math.nan
        )
    return split_rows, group_rows


def confusion_matrix(rows: Sequence[Dict[str, Any]], count_values: Sequence[int]) -> np.ndarray:
    idx = {int(value): pos for pos, value in enumerate(count_values)}
    mat = np.zeros((len(count_values), len(count_values)), dtype=np.int64)
    for row in rows:
        gold = int(row["gold_count"])
        pred = int(row["predicted_count"])
        if gold in idx and pred in idx:
            mat[idx[gold], idx[pred]] += 1
    return mat


def save_plot(path: Path, fig: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def make_run_plots(run_dir: Path, rows: Sequence[Dict[str, Any]], group_rows: Sequence[Dict[str, Any]], probe_rows: Sequence[Dict[str, Any]], causal_rows: Sequence[Dict[str, Any]], variant: str, count_values: Sequence[int]) -> None:
    plots = run_dir / "plots"
    for split in sorted({str(row["split"]) for row in rows}):
        split_group = [row for row in group_rows if row["split"] == split and row["group"] == "count"]
        if split_group:
            fig, ax = plt.subplots(figsize=(7, 4))
            xs = [int(row["value"]) for row in split_group]
            ax.plot(xs, [float(row["accuracy"]) for row in split_group], marker="o")
            ax.set_xlabel("Evidence count")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            ax.set_title(f"{variant}: accuracy by evidence count ({split})")
            save_plot(plots / f"accuracy_by_evidence_count_{safe_name(split)}", fig)
        split_len = [row for row in group_rows if row["split"] == split and row["group"] == "sequence_length"]
        if split_len:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([int(row["value"]) for row in split_len], [float(row["mae"]) for row in split_len], marker="o")
            ax.set_xlabel("Sequence length")
            ax.set_ylabel("MAE")
            ax.set_title(f"{variant}: MAE by sequence length ({split})")
            save_plot(plots / f"mae_by_sequence_length_{safe_name(split)}", fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    iid_rows = [row for row in rows if row["split"] == IID_TEST_SPLIT]
    mat = confusion_matrix(iid_rows or rows, count_values)
    ax.imshow(mat, cmap="Blues")
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("True count")
    ax.set_xticks(range(len(count_values)), count_values)
    ax.set_yticks(range(len(count_values)), count_values)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j]:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title(f"{variant}: confusion matrix")
    save_plot(plots / "confusion_matrix", fig)
    count_group = [row for row in group_rows if row["split"] == IID_TEST_SPLIT and row["group"] == "count"]
    if count_group:
        fig, ax = plt.subplots(figsize=(6, 5))
        xs = [int(row["value"]) for row in count_group]
        ax.plot(xs, [float(row["mean_predicted_count"]) for row in count_group], marker="o", label="mean predicted")
        ax.plot(xs, xs, linestyle="--", color="black", label="ideal")
        ax.set_xlabel("True count")
        ax.set_ylabel("Mean predicted count")
        ax.legend()
        ax.set_title(f"{variant}: mean prediction")
        save_plot(plots / "mean_predicted_count_vs_true_count", fig)
    split_acc = defaultdict(list)
    for row in rows:
        split_acc[str(row["split"])].append(int(row["correct"]))
    if split_acc:
        fig, ax = plt.subplots(figsize=(7, 4))
        keys = [IID_TEST_SPLIT, LENGTH_OOD_SPLIT, COMPOSITION_OOD_SPLIT]
        values = [finite_mean(split_acc.get(key, []), default=math.nan) for key in keys]
        ax.bar(keys, values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{variant}: IID vs OOD")
        ax.tick_params(axis="x", rotation=20)
        save_plot(plots / "iid_vs_length_ood_comparison", fig)
    if probe_rows:
        for feature_name, filename in [
            ("raw_message_sum", "raw_message_probe_accuracy_by_layer"),
            ("slot_sum", "slot_probe_accuracy_by_layer"),
            ("read", "memory_read_probe_accuracy_by_layer"),
        ]:
            data = [row for row in probe_rows if row.get("feature") == feature_name and row.get("task") == "count"]
            if data:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot([int(row["layer"]) for row in data], [float(row["accuracy"]) for row in data], marker="o")
                ax.set_xlabel("Layer")
                ax.set_ylabel("Count accuracy")
                ax.set_ylim(0, 1)
                ax.set_title(filename.replace("_", " "))
                save_plot(plots / filename, fig)
    memory_rows = [row for row in rows if isinstance(row.get("injection_norm_by_layer"), dict)]
    if memory_rows:
        layer_values: Dict[str, List[float]] = defaultdict(list)
        rank_values: Dict[str, List[float]] = defaultdict(list)
        gamma_values: Dict[str, List[float]] = defaultdict(list)
        slot_norm_values: Dict[str, List[float]] = defaultdict(list)
        for row in memory_rows:
            for layer, value in row.get("injection_norm_by_layer", {}).items():
                layer_values[str(layer)].append(float(value))
            for layer, value in row.get("effective_rank_by_layer", {}).items():
                rank_values[str(layer)].append(float(value))
            for layer, value in row.get("gamma_by_layer", {}).items():
                gamma_values[str(layer)].append(float(value))
            for layer, value in row.get("slot_norm_by_layer", {}).items():
                slot_norm_values[str(layer)].append(float(value))
        for values, ylabel, filename in [
            (layer_values, "Injection norm", "injection_norm_by_layer"),
            (gamma_values, "Gamma", "gamma_values_by_layer"),
            (rank_values, "Effective rank", "effective_slot_rank_by_layer"),
            (slot_norm_values, "Slot norm", "evidence_vs_neutral_slot_norms"),
        ]:
            if values:
                fig, ax = plt.subplots(figsize=(7, 4))
                layers = sorted(values, key=int)
                ax.plot([int(layer) for layer in layers], [finite_mean(values[layer], default=math.nan) for layer in layers], marker="o")
                ax.set_xlabel("Layer")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{variant}: {ylabel}")
                save_plot(plots / filename, fig)
    if causal_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        modes = sorted({str(row["ablation"]) for row in causal_rows})
        deltas = [
            finite_mean([row["delta_predicted_count"] for row in causal_rows if row["ablation"] == mode], default=math.nan)
            for mode in modes
        ]
        ax.bar(modes, deltas)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Mean delta predicted count")
        ax.set_title(f"{variant}: causal intervention effect")
        ax.tick_params(axis="x", rotation=25)
        save_plot(plots / "causal_intervention_effect_on_predicted_count", fig)


def collect_probe_records(
    diagnostic_manifests: Sequence[Dict[str, Any]],
    diagnostic_base: Path,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in diagnostic_manifests:
        evidence = set(int(x) for x in item["evidence_frame_indices"])
        for layer, rel_path in item.get("layers", {}).items():
            npz = np.load(diagnostic_base / rel_path)
            raw = np.asarray(npz["raw_messages"])[0]
            slots = np.asarray(npz["slots"])[0]
            read = np.asarray(npz["read"])[0]
            carriers = raw.shape[0]
            frames = raw.shape[1]
            for c in range(carriers):
                for f in range(frames):
                    records.append({
                        "task": "frame_evidence",
                        "feature": "raw_message",
                        "layer": int(layer),
                        "x": raw[c, f].astype(np.float32),
                        "y": int(f in evidence),
                    })
                    records.append({
                        "task": "frame_evidence",
                        "feature": "slot",
                        "layer": int(layer),
                        "x": slots[c, f].astype(np.float32),
                        "y": int(f in evidence),
                    })
            records.append({
                "task": "count",
                "feature": "raw_message_sum",
                "layer": int(layer),
                "x": raw.sum(axis=(0, 1)).astype(np.float32),
                "y": int(item["gold_count"]),
            })
            records.append({
                "task": "count",
                "feature": "slot_sum",
                "layer": int(layer),
                "x": slots.sum(axis=(0, 1)).astype(np.float32),
                "y": int(item["gold_count"]),
            })
            records.append({
                "task": "count",
                "feature": "read",
                "layer": int(layer),
                "x": read.mean(axis=0).astype(np.float32),
                "y": int(item["gold_count"]),
            })
    return records


def probe_metrics(records: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    if not records:
        return []
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
        from sklearn.model_selection import train_test_split
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    keys = sorted({(row["task"], row["feature"], int(row["layer"])) for row in records})
    for task, feature, layer in keys:
        data = [row for row in records if row["task"] == task and row["feature"] == feature and int(row["layer"]) == layer]
        if len(data) < 8 or len({int(row["y"]) for row in data}) < 2:
            continue
        x = np.stack([row["x"] for row in data])
        y = np.asarray([int(row["y"]) for row in data])
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=int(seed), stratify=y if len(set(y)) > 1 else None)
            if task == "frame_evidence":
                clf = LogisticRegression(max_iter=1000, random_state=int(seed)).fit(x_train, y_train)
                pred = clf.predict(x_test)
                prob = clf.predict_proba(x_test)[:, 1] if len(clf.classes_) == 2 else pred
                rows.append({
                    "task": task,
                    "feature": feature,
                    "layer": int(layer),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "f1": float(f1_score(y_test, pred, zero_division=0)),
                    "auroc": float(roc_auc_score(y_test, prob)) if len(set(y_test)) == 2 else math.nan,
                    "mae": math.nan,
                })
            else:
                clf = Ridge(alpha=1.0, random_state=int(seed)).fit(x_train, y_train)
                pred_float = clf.predict(x_test)
                pred = np.rint(pred_float).astype(int)
                rows.append({
                    "task": task,
                    "feature": feature,
                    "layer": int(layer),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "count_accuracy": float(accuracy_score(y_test, pred)),
                    "mae": float(mean_absolute_error(y_test, pred_float)),
                    "f1": math.nan,
                    "auroc": math.nan,
                    "mean_predicted_by_true": {
                        str(value): float(np.mean(pred_float[y_test == value]))
                        for value in sorted(set(int(v) for v in y_test))
                    },
                })
        except Exception:
            continue
    return rows


@torch.no_grad()
def evaluate_causal_ablations(
    *,
    variant: str,
    model: Any,
    processor: Any,
    adapter: ExperimentAdapter,
    examples: Sequence[FrameMemoryExample],
    indices: Sequence[int],
    answer_ids: Dict[int, Tuple[int, ...]],
    count_values: Sequence[int],
    device: str,
    seed: int,
) -> List[Dict[str, Any]]:
    if adapter.memory is None:
        return []
    modes = [
        "zero_memory",
        "shuffle_memory_between_samples",
        "remove_one_evidence_slot",
        "duplicate_one_evidence_slot",
        "remove_one_neutral_slot",
        "replace_slots_with_norm_matched_noise",
    ]
    rows: List[Dict[str, Any]] = []
    adapter.attach(model)
    try:
        for idx in indices:
            example = examples[int(idx)]
            adapter.set_ablation("normal", seed)
            normal_pred, normal_scores = predict_count(
                model=model,
                processor=processor,
                adapter=adapter,
                example=example,
                sample_idx=int(idx),
                answer_ids=answer_ids,
                count_values=count_values,
                device=device,
            )
            gold_score = normal_scores.get(str(int(example.gold_count)), math.nan)
            for mode in modes:
                if mode in {"remove_one_evidence_slot", "duplicate_one_evidence_slot"} and not example.evidence_frame_indices:
                    continue
                if mode == "remove_one_neutral_slot" and len(example.evidence_frame_indices) >= int(example.num_frames):
                    continue
                adapter.set_ablation(mode, seed)
                pred, scores = predict_count(
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    example=example,
                    sample_idx=int(idx),
                    answer_ids=answer_ids,
                    count_values=count_values,
                    device=device,
                )
                delta_pred = int(pred) - int(normal_pred)
                delta_gold_logit = float(scores.get(str(int(example.gold_count)), math.nan) - gold_score)
                if mode == "remove_one_evidence_slot":
                    expected = delta_pred < 0
                elif mode == "duplicate_one_evidence_slot":
                    expected = delta_pred > 0
                elif mode == "remove_one_neutral_slot":
                    expected = abs(delta_pred) <= 1
                else:
                    expected = False
                rows.append({
                    "variant": variant,
                    "split": IID_TEST_SPLIT,
                    "example_id": example.example_id,
                    "ablation": mode,
                    "gold_count": int(example.gold_count),
                    "normal_predicted_count": int(normal_pred),
                    "ablation_predicted_count": int(pred),
                    "delta_predicted_count": int(delta_pred),
                    "delta_gold_answer_score": delta_gold_logit,
                    "expected_direction": int(expected),
                    "original_label_evaluation": int(int(pred) == int(example.gold_count)),
                })
        adapter.set_ablation("normal", seed)
    finally:
        adapter.detach()
    return rows


def run_memory_disabled_eval(
    *,
    variant: str,
    model: Any,
    processor: Any,
    adapter: ExperimentAdapter,
    examples: Sequence[FrameMemoryExample],
    indices: Sequence[int],
    answer_ids: Dict[int, Tuple[int, ...]],
    count_values: Sequence[int],
    device: str,
    seed: int,
) -> Dict[str, Any]:
    if adapter.memory is None:
        return {"accuracy": math.nan, "mae": math.nan, "n": 0}
    adapter.set_memory_enabled(False)
    result = evaluate_split(
        variant=f"{variant}_memory_disabled",
        split_name=IID_TEST_SPLIT,
        model=model,
        processor=processor,
        adapter=adapter,
        examples=examples,
        indices=indices,
        answer_ids=answer_ids,
        count_values=count_values,
        device=device,
        seed=seed,
        collect_diagnostics=False,
    )
    adapter.set_memory_enabled(True)
    return summarize_prediction_rows(result["rows"])


def run_variant(
    *,
    args: argparse.Namespace,
    variant: str,
    model: Any,
    processor: Any,
    examples: Dict[str, List[FrameMemoryExample]],
    dataset_manifest: Dict[str, Any],
    parent_output_root: Path,
    answer_ids: Dict[int, Tuple[int, ...]],
    device: str,
    timestamp: str,
) -> Dict[str, Any]:
    layers = parse_int_tokens(args.layers)
    if bool(args.tiny_debug_model):
        max_layer = int(args.tiny_num_layers) - 1
        if any(layer > max_layer for layer in layers):
            raise ValueError(f"Tiny model has layers 0..{max_layer}, requested {layers}")
    run_prefix = f"{safe_name(args.run_prefix)}_" if str(args.run_prefix).strip() else ""
    smoke = "smoke_" if bool(args.smoke_test) else ""
    run_dir = parent_output_root / f"{timestamp}_{run_prefix}{smoke}{variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle, old_stdout, old_stderr = setup_logging(run_dir)
    adapter: Optional[ExperimentAdapter] = None
    try:
        print(f"Running {variant} into {run_dir}")
        write_json(run_dir / "config.json", {**vars(args), "variant": variant, "layers": layers})
        write_json(run_dir / "split_manifest.json", dataset_manifest)
        hidden_size = hidden_size_from_model(model)
        adapter = make_adapter(args, variant, hidden_size=hidden_size, layers=layers).to(device)
        adapter.attach(model)
        try:
            param_summary = trainable_parameter_summary(model, adapter)
        finally:
            adapter.detach()
        write_json(run_dir / "parameter_summary.json", param_summary)
        print(
            f"Trainable parameters: {param_summary['trainable_adapter_parameters']} "
            f"(LoRA={param_summary['groups']['lora']} memory={param_summary['groups']['memory']})"
        )
        history, checkpoint_path = train_adapter(
            args=args,
            run_dir=run_dir,
            variant=variant,
            model=model,
            processor=processor,
            adapter=adapter,
            examples=examples,
            answer_ids=answer_ids,
            device=device,
        )
        write_csv_dynamic(run_dir / "training_history.csv", history, leading=("variant", "epoch"))
        count_values = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
        all_rows: List[Dict[str, Any]] = []
        all_diag_manifest: List[Dict[str, Any]] = []
        diagnostics_dir = run_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        for split in EVAL_SPLITS:
            indices = limited_indices(examples[split], int(args.max_eval_examples), int(args.seed) + 101)
            if bool(args.smoke_test):
                indices = indices[: min(2, len(indices))]
            collect_diag = variant in MEMORY_VARIANTS and len(all_diag_manifest) < int(args.diagnostic_subset)
            if collect_diag:
                indices_for_diag = indices[: max(1, min(len(indices), int(args.diagnostic_subset) - len(all_diag_manifest)))]
            result = evaluate_split(
                variant=variant,
                split_name=split,
                model=model,
                processor=processor,
                adapter=adapter,
                examples=examples[split],
                indices=indices,
                answer_ids=answer_ids,
                count_values=count_values,
                device=device,
                seed=int(args.seed),
                collect_diagnostics=collect_diag,
                diagnostic_dir=diagnostics_dir,
            )
            all_rows.extend(result["rows"])
            all_diag_manifest.extend(result["diagnostics"])
        write_csv_dynamic(run_dir / "per_sample_predictions.csv", all_rows, leading=("variant", "split", "example_id"))
        split_metrics, group_metrics = metrics_from_rows(variant, all_rows)
        memory_disabled = run_memory_disabled_eval(
            variant=variant,
            model=model,
            processor=processor,
            adapter=adapter,
            examples=examples[IID_TEST_SPLIT],
            indices=limited_indices(examples[IID_TEST_SPLIT], int(args.max_eval_examples), int(args.seed) + 103)[: (2 if args.smoke_test else 999999)],
            answer_ids=answer_ids,
            count_values=count_values,
            device=device,
            seed=int(args.seed),
        ) if variant in MEMORY_VARIANTS else {"accuracy": math.nan, "mae": math.nan, "n": 0}
        for row in split_metrics:
            row["trainable_parameter_count"] = int(param_summary["trainable_adapter_parameters"])
            row["memory_disabled_accuracy"] = memory_disabled["accuracy"]
            row["memory_disabled_mae"] = memory_disabled["mae"]
        write_json(run_dir / "metrics.json", {"split_metrics": split_metrics, "group_metrics": group_metrics, "memory_disabled": memory_disabled})
        write_csv_dynamic(run_dir / "metrics.csv", split_metrics, leading=("variant", "split"))
        write_csv_dynamic(run_dir / "metrics_by_group.csv", group_metrics, leading=("variant", "split", "group", "value"))
        probe_rows: List[Dict[str, Any]] = []
        if variant in MEMORY_VARIANTS and not bool(args.no_probes):
            probe_records = collect_probe_records(all_diag_manifest, run_dir)
            probe_rows = probe_metrics(probe_records, int(args.seed))
            write_csv_dynamic(diagnostics_dir / "posthoc_probe_metrics.csv", probe_rows, leading=("task", "feature", "layer"))
            write_json(diagnostics_dir / "posthoc_probe_metrics.json", probe_rows)
        causal_rows: List[Dict[str, Any]] = []
        if variant in MEMORY_VARIANTS:
            ablation_indices = limited_indices(
                examples[IID_TEST_SPLIT],
                int(args.max_ablation_examples),
                int(args.seed) + 211,
            )
            if bool(args.smoke_test):
                ablation_indices = ablation_indices[:1]
            causal_rows = evaluate_causal_ablations(
                variant=variant,
                model=model,
                processor=processor,
                adapter=adapter,
                examples=examples[IID_TEST_SPLIT],
                indices=ablation_indices,
                answer_ids=answer_ids,
                count_values=count_values,
                device=device,
                seed=int(args.seed),
            )
            write_csv_dynamic(diagnostics_dir / "causal_ablation_metrics.csv", causal_rows, leading=("variant", "ablation", "example_id"))
            write_json(diagnostics_dir / "causal_ablation_metrics.json", causal_rows)
        if not bool(args.no_plots):
            make_run_plots(run_dir, all_rows, group_metrics, probe_rows, causal_rows, variant, count_values)
        write_report(run_dir, variant, split_metrics, probe_rows, causal_rows, checkpoint_path)
        if bool(args.smoke_test):
            print_smoke_summary(run_dir, all_rows, adapter)
        return {
            "variant": variant,
            "run_dir": os.fspath(run_dir),
            "checkpoint": os.fspath(checkpoint_path),
            "metrics": split_metrics,
            "parameter_summary": param_summary,
        }
    finally:
        if adapter is not None:
            adapter.detach()
        restore_logging(log_handle, old_stdout, old_stderr)


def print_smoke_summary(run_dir: Path, rows: Sequence[Dict[str, Any]], adapter: Optional[ExperimentAdapter]) -> None:
    first = rows[0] if rows else {}
    print("Smoke test diagnostics")
    print(f"detected frame spans: {first.get('visual_token_spans_per_frame')}")
    print(f"detected carrier positions: {first.get('carrier_token_positions')}")
    print(f"attention reconstruction error: {first.get('memory_reconstruction_error')}")
    shapes = first.get("tensor_shapes_by_layer", {})
    if isinstance(shapes, dict):
        for layer, payload in shapes.items():
            print(f"layer {layer} tensor shapes: {payload}")
    print(f"non-carrier LoRA update max: {first.get('noncarrier_lora_update_max')}")
    if adapter is not None and adapter.memory is not None:
        print(f"hook fire counts: {dict(adapter.memory.hook_fire_counts)}")
    print(f"output directory: {run_dir}")


def write_report(run_dir: Path, variant: str, split_metrics: Sequence[Dict[str, Any]], probe_rows: Sequence[Dict[str, Any]], causal_rows: Sequence[Dict[str, Any]], checkpoint_path: Path) -> None:
    lines = [
        f"# {variant} report",
        "",
        "## IID and OOD results",
        "",
        "| split | accuracy | MAE | memory disabled acc |",
        "|---|---:|---:|---:|",
    ]
    for row in split_metrics:
        lines.append(
            f"| {row['split']} | {float(row['accuracy']):.3f} | {float(row['mae']):.3f} | "
            f"{finite_float(row.get('memory_disabled_accuracy')) or math.nan:.3f} |"
        )
    def best_probe(feature: str) -> Optional[Dict[str, Any]]:
        candidates = [row for row in probe_rows if row.get("feature") == feature]
        if not candidates:
            return None
        return max(candidates, key=lambda row: finite_float(row.get("accuracy")) or -1)
    lines.extend(["", "## Diagnostic failure map", ""])
    for feature, question in [
        ("raw_message_sum", "useful evidence in raw messages"),
        ("slot_sum", "persistent slots preserve count information"),
        ("read", "associative read contains count information"),
    ]:
        row = best_probe(feature)
        if row is None:
            lines.append(f"- {question}: not enough probe data collected.")
        else:
            lines.append(f"- {question}: best layer {row['layer']} accuracy {float(row.get('accuracy', math.nan)):.3f}.")
    if causal_rows:
        expected = finite_mean([row["expected_direction"] for row in causal_rows], default=math.nan)
        lines.append(f"- causal-ablation expected-direction rate: {expected:.3f}.")
    lines.extend([
        "",
        "## Comparisons",
        "",
        "- Direct-sum versus gLSTM, layerwise versus final-only, and global versus carrier-only comparisons are finalized in the parent combined report after all variants finish.",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Plots: `{run_dir / 'plots'}`",
    ])
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_parent_outputs(parent: Path) -> Dict[str, Any]:
    metric_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(parent.glob("*/metrics.csv")):
        metric_rows.extend(read_csv_rows(metrics_path))
    for pred_path in sorted(parent.glob("*/per_sample_predictions.csv")):
        prediction_rows.extend(read_csv_rows(pred_path))
    if metric_rows:
        write_csv_dynamic(parent / "combined_results.csv", metric_rows, leading=("variant", "split"))
    summary: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_metric_rows": len(metric_rows),
        "variants": sorted({row.get("variant", "") for row in metric_rows}),
        "splits": sorted({row.get("split", "") for row in metric_rows}),
    }
    write_json(parent / "combined_summary.json", summary)
    if metric_rows:
        comparison_dir = parent / "comparison_plots"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = [f"{row.get('variant')}:{row.get('split')}" for row in metric_rows]
        values = [float(row.get("accuracy") or "nan") for row in metric_rows]
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model x split accuracy")
        save_plot(comparison_dir / "global_lora_vs_carrier_lora_comparison", fig)
        write_final_parent_report(parent, metric_rows)
    return summary


def metric_value(rows: Sequence[Dict[str, Any]], variant: str, split: str, key: str) -> Optional[float]:
    for row in rows:
        if row.get("variant") == variant and row.get("split") == split:
            return finite_float(row.get(key))
    return None


def write_final_parent_report(parent: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Layerwise frame-message gLSTM summary",
        "",
        "| variant | split | accuracy | MAE | trainable params | memory disabled accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant')} | {row.get('split')} | {float(row.get('accuracy') or 'nan'):.3f} | "
            f"{float(row.get('mae') or 'nan'):.3f} | {row.get('trainable_parameter_count', '')} | "
            f"{float(row.get('memory_disabled_accuracy') or 'nan'):.3f} |"
        )
    direct = metric_value(rows, CARRIER_DIRECT_SUM, IID_TEST_SPLIT, "accuracy")
    glstm = metric_value(rows, CARRIER_GLSTM_LAYERWISE, IID_TEST_SPLIT, "accuracy")
    final = metric_value(rows, CARRIER_GLSTM_FINAL_ONLY, IID_TEST_SPLIT, "accuracy")
    carrier = metric_value(rows, CARRIER_LORA, IID_TEST_SPLIT, "accuracy")
    global_lora = metric_value(rows, GLOBAL_LORA, IID_TEST_SPLIT, "accuracy")
    lines.extend([
        "",
        "## Inferred comparisons",
        "",
        f"- Direct-sum versus layerwise gLSTM IID accuracy: {direct} vs {glstm}.",
        f"- Layerwise versus final-only gLSTM IID accuracy: {glstm} vs {final}.",
        f"- Global versus carrier-only LoRA IID accuracy: {global_lora} vs {carrier}.",
        "- Diagnostic failure maps are in each run's `report.md` and `diagnostics/` directory.",
        "- Causal ablation summaries are in each memory run's `diagnostics/causal_ablation_metrics.csv`.",
        f"- Combined plots: `{parent / 'comparison_plots'}`",
    ])
    (parent / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    parent_output_root = Path(args.output_root).resolve()
    parent_output_root.mkdir(parents=True, exist_ok=True)
    if bool(args.aggregate_only):
        aggregate_parent_outputs(parent_output_root)
        return 0
    variants = parse_variants(args.variants)
    if bool(args.smoke_test):
        args.tiny_debug_model = True
        args.device = "cpu" if args.device == "cuda" else args.device
        args.epochs = min(int(args.epochs), 1)
        args.grad_accum = 1
        args.max_train_examples = 2 if int(args.max_train_examples) <= 0 else min(int(args.max_train_examples), 2)
        args.max_eval_examples = 2 if int(args.max_eval_examples) <= 0 else min(int(args.max_eval_examples), 2)
        args.max_ablation_examples = min(int(args.max_ablation_examples), 1)
        if args.variants == [",".join(VARIANTS)]:
            variants = [CARRIER_GLSTM_LAYERWISE]
    device = resolve_device(str(args.device))
    dtype = dtype_from_arg(str(args.dtype), device)
    dataset_dir, examples, dataset_manifest = ensure_dataset(args, parent_output_root / "cache")
    dataset_manifest = {**dataset_manifest, "dataset_dir": os.fspath(dataset_dir)}
    model, processor, load_in_4bit, load_mode = load_model_and_processor(args, device=device, dtype=dtype)
    tokenizer = processor.tokenizer
    tokenization_mode, answer_ids = text_base.answer_token_ids(tokenizer, int(args.candidate_min), int(args.candidate_max))
    print(
        "Reusing repository utilities: visual dataset/prompt/token detection from "
        "experiments.carrier_mixing.visual_fixed8_iid_carrier_slots_lora; LoRA/loss/model helpers from "
        "experiments.carrier_mixing.pnamix_clean_aggregation_lora; message extraction patterns from "
        "experiments.carrier_probes.run_message_memory_carrier_update_seq8_7b."
    )
    print(f"Model load mode={load_mode} load_in_4bit={load_in_4bit} answer_tokenization={tokenization_mode}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_summaries: List[Dict[str, Any]] = []
    for variant in variants:
        run_summaries.append(
            run_variant(
                args=args,
                variant=variant,
                model=model,
                processor=processor,
                examples=examples,
                dataset_manifest=dataset_manifest,
                parent_output_root=parent_output_root,
                answer_ids=answer_ids,
                device=device,
                timestamp=timestamp,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(parent_output_root / f"{timestamp}_run_summaries.json", run_summaries)
    aggregate_parent_outputs(parent_output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

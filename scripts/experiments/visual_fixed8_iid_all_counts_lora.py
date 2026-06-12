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
from types import SimpleNamespace
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
from scripts.experiments import pnamix_clean_aggregation_lora as text_base
from scripts.probes import run_message_memory_adapter_stage1_stage3_seq8 as mmred


EXPERIMENT_NAME = "visual_fixed8_iid_all_counts_lora"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_SOURCE_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images_park"

FROZEN_QWEN_BASELINE = "frozen_qwen_baseline"
LORA_BASELINE = "lora_baseline"
MAXMIX_CARRIER_14_17 = "maxmix_carrier_14_17"
PNAMIX_CARRIER_14_17 = "pnamix_carrier_14_17"
VARIANTS = (
    FROZEN_QWEN_BASELINE,
    LORA_BASELINE,
    MAXMIX_CARRIER_14_17,
    PNAMIX_CARRIER_14_17,
)
DEFAULT_VARIANTS = (
    LORA_BASELINE,
    MAXMIX_CARRIER_14_17,
    PNAMIX_CARRIER_14_17,
)
VARIANT_ALIASES = {
    "frozen": FROZEN_QWEN_BASELINE,
    FROZEN_QWEN_BASELINE: FROZEN_QWEN_BASELINE,
    "lora": LORA_BASELINE,
    LORA_BASELINE: LORA_BASELINE,
    "maxmix": MAXMIX_CARRIER_14_17,
    MAXMIX_CARRIER_14_17: MAXMIX_CARRIER_14_17,
    "pna": PNAMIX_CARRIER_14_17,
    "pnamix": PNAMIX_CARRIER_14_17,
    PNAMIX_CARRIER_14_17: PNAMIX_CARRIER_14_17,
}

TRAIN_TEMPLATES = ("train_how_many", "train_count_where", "train_number_containing")
HELDOUT_TEMPLATES = ("heldout_total_matching", "heldout_frequency")
TRAIN_SPLIT = "train_fixed8_all_counts"
VAL_SPLIT = "val_fixed8_all_counts"
TEST_SPLIT = "test_fixed8_all_counts"
PILOT_SPLITS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
EVAL_SPLITS = (VAL_SPLIT, TEST_SPLIT)
OPTIONAL_SPLITS: Tuple[str, ...] = ()
COUNT_VALUES = tuple(range(9))
VISUAL_INPUT_KEYS = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")
EPS = 1e-8
_RUNTIME_CHECK_LOGGED = False


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
class VisualExample:
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
    source_dataset_info: Tuple[Dict[str, Any], ...]


@dataclass
class VisualBatch:
    inputs: Dict[str, Any]
    prompt_last_indices: torch.Tensor
    gold_counts: torch.Tensor
    loss_positions: Optional[torch.Tensor]
    loss_targets: Optional[torch.Tensor]
    carrier_positions: List[List[int]]
    frame_groups: List[List[List[int]]]
    token_selection_ok: List[bool]
    token_selection_errors: List[str]
    sample_indices: List[int]
    visual_input_keys: List[str]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-frame fixed-8 IID all-count baseline with LoRA, MaxMix, and PNA-Mix."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--source-split", default="all_uniform")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--variants", nargs="+", default=[",".join(DEFAULT_VARIANTS)])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=24680)
    parser.add_argument("--train-per-count", "--train_per_count", dest="train_per_count", type=int, default=60)
    parser.add_argument("--val-per-count", "--val_per_count", dest="val_per_count", type=int, default=15)
    parser.add_argument(
        "--all-count-test-per-count",
        "--all_count_test_per_count",
        dest="all_count_test_per_count",
        type=int,
        default=20,
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

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-layers", nargs="+", default=["14,15,16,17,20,21,22,23,24,25,26,27"])
    parser.add_argument("--carrier-layers", nargs="+", default=["14,15,16,17"])
    parser.add_argument("--lora-targets", nargs="+", default=["q_proj,k_proj,v_proj,o_proj"])
    parser.add_argument("--carrier-gate-init", type=float, default=-2.0)
    parser.add_argument("--message-mode", choices=["auto", "exact", "approx"], default="auto")

    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
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

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--tiny-debug-model", action="store_true", default=False)
    parser.add_argument("--tiny-num-layers", type=int, default=28)
    parser.add_argument("--tiny-hidden-size", type=int, default=64)
    parser.add_argument("--tiny-num-heads", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--submit-mode", default="local")
    return parser.parse_args()


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
        if token not in VARIANT_ALIASES:
            raise ValueError(f"Unknown variant {token!r}; valid values are {sorted(VARIANT_ALIASES)}")
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


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv_dynamic(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(leading)
    for key in sorted({key for row in rows for key in row}):
        if key not in fields:
            fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def setup_logging(run_dir: Path) -> Tuple[Any, Any, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    handle = (run_dir / "run.log").open("a", encoding="utf-8", buffering=1)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, handle)
    sys.stderr = Tee(sys.stderr, handle)
    return handle, old_stdout, old_stderr


def restore_logging(handle: Any, old_stdout: Any, old_stderr: Any) -> None:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    handle.close()


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16], 16)


def source_partition(sample_id: str, seed: int) -> str:
    bucket = stable_hash_int(f"{int(seed)}:{sample_id}") % 10
    if bucket <= 6:
        return "train"
    if bucket == 7:
        return "val"
    return "test"


def question_for_template(template_id: str, character: str, room: str) -> str:
    templates = {
        "train_how_many": f"How many frames show {character} in the {room}?",
        "train_count_where": f"Count the frames where {character} is in the {room}.",
        "train_number_containing": f"What is the number of frames containing {character} in the {room}?",
        "heldout_total_matching": f"Give the total number of frames matching {character} in the {room}.",
        "heldout_frequency": f"In how many of the frames can {character} be seen inside the {room}?",
    }
    if template_id not in templates:
        raise ValueError(f"Unknown template_id={template_id!r}")
    return templates[template_id]


def build_prompt(question: str, num_frames: int) -> str:
    return (
        f"You will be shown {int(num_frames)} separate visual frames from a house.\n"
        f"Count frames matching the question. Respond with one integer from 0 to {int(num_frames)}. "
        "Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )


def example_to_json(example: VisualExample) -> Dict[str, Any]:
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
        "queried_entity": example.queried_character,
        "queried_room": example.queried_room,
        "queried_location": example.queried_room,
        "template_id": example.template_id,
        "source_dataset_info": list(example.source_dataset_info),
    }


def example_from_json(row: Dict[str, Any]) -> VisualExample:
    return VisualExample(
        example_id=str(row["id"]),
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
        source_dataset_info=tuple(dict(item) for item in row.get("source_dataset_info", [])),
    )


def scan_source_frame_pools(
    source_root: Path,
    source_split: str,
    dataset_seed: int,
) -> Tuple[Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]], Dict[str, Any]]:
    source_root = Path(source_root).resolve()
    split_root = source_root / "seq_len_8" / str(source_split)
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
        partition: defaultdict(lambda: {"evidence": [], "neutral": []})
        for partition in ("train", "val", "test")
    }
    for sample_dir, states in parsed_samples:
        partition = source_partition(sample_dir.name, int(dataset_seed))
        for frame_idx, state in enumerate(states):
            room_to_chars = eval_utils.rooms_to_room2chars(state.get("rooms", {}))
            present_chars = {char for values in room_to_chars.values() for char in values}
            frame_path = sample_dir / f"{int(frame_idx):03d}.png"
            if not frame_path.is_file():
                raise FileNotFoundError(frame_path)
            for character in sorted(characters):
                for room in sorted(rooms):
                    kind: Optional[str] = None
                    if character in room_to_chars.get(room, []):
                        kind = "evidence"
                    elif character not in present_chars and not room_to_chars.get(room, []):
                        kind = "neutral"
                    if kind is None:
                        continue
                    pools[partition][(character, room)][kind].append(
                        {
                            "frame_path": os.fspath(frame_path.relative_to(PROJECT_ROOT)),
                            "source_sample_id": sample_dir.name,
                            "source_frame_index": int(frame_idx),
                            "source_partition": partition,
                            "selection_type": kind,
                            "state": state,
                        }
                    )

    pool_counts: Dict[str, Any] = {}
    for partition, pair_pools in pools.items():
        pool_counts[partition] = {
            f"{character}|{room}": {
                "evidence": len(kind_pools["evidence"]),
                "neutral": len(kind_pools["neutral"]),
            }
            for (character, room), kind_pools in sorted(pair_pools.items())
        }
    manifest = {
        "source_dataset_root": os.fspath(Path(source_root).resolve()),
        "source_split": str(source_split),
        "source_sample_count": len(sample_dirs),
        "characters": sorted(characters),
        "rooms": sorted(rooms),
        "partition_rule": "sha256(dataset_seed:sample_id) modulo 10; train=0..6, val=7, test=8..9",
        "neutral_rule": "queried character absent from the frame and queried room empty",
        "evidence_rule": "queried character present in queried room",
        "pool_counts": pool_counts,
    }
    return pools, manifest


def choose_refs(rng: random.Random, pool: Sequence[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if int(n) <= 0:
        return []
    if not pool:
        raise RuntimeError(f"Cannot select {n} frame references from an empty pool")
    if len(pool) >= int(n):
        return [dict(item) for item in rng.sample(list(pool), int(n))]
    return [dict(rng.choice(list(pool))) for _ in range(int(n))]


def generate_split(
    *,
    split: str,
    num_frames: int,
    examples_per_count: int,
    counts: Sequence[int],
    templates: Sequence[str],
    pair_pools: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]],
    seed: int,
) -> List[VisualExample]:
    rng = random.Random(int(seed))
    valid_pairs = sorted(
        pair
        for pair, pools in pair_pools.items()
        if pools.get("evidence") and pools.get("neutral")
    )
    if not valid_pairs:
        raise RuntimeError(f"{split}: no character/room pair has both evidence and neutral frame pools")
    count_values = [int(value) for value in counts]
    if any(count < 0 or count > int(num_frames) for count in count_values):
        raise ValueError(f"{split}: counts {count_values} are invalid for num_frames={num_frames}")
    schedule = [
        count
        for count in count_values
        for _ in range(int(examples_per_count))
    ]
    rng.shuffle(schedule)
    examples: List[VisualExample] = []
    for sample_idx, gold_count in enumerate(schedule):
        character, room = rng.choice(valid_pairs)
        evidence_positions = tuple(sorted(rng.sample(range(int(num_frames)), int(gold_count))))
        evidence_set = set(evidence_positions)
        evidence_refs = choose_refs(rng, pair_pools[(character, room)]["evidence"], int(gold_count))
        neutral_refs = choose_refs(
            rng,
            pair_pools[(character, room)]["neutral"],
            int(num_frames) - int(gold_count),
        )
        rng.shuffle(evidence_refs)
        rng.shuffle(neutral_refs)
        ordered_refs: List[Dict[str, Any]] = []
        evidence_cursor = 0
        neutral_cursor = 0
        for frame_idx in range(int(num_frames)):
            if frame_idx in evidence_set:
                ref = evidence_refs[evidence_cursor]
                evidence_cursor += 1
            else:
                ref = neutral_refs[neutral_cursor]
                neutral_cursor += 1
            ref["output_frame_index"] = int(frame_idx)
            ordered_refs.append(ref)
        template_id = rng.choice(tuple(templates))
        question = question_for_template(template_id, character, room)
        examples.append(
            VisualExample(
                example_id=f"{split}_{sample_idx:06d}",
                split=split,
                frame_paths=tuple(str(ref["frame_path"]) for ref in ordered_refs),
                num_frames=int(num_frames),
                gold_count=int(gold_count),
                evidence_frame_indices=evidence_positions,
                question=question,
                answer=str(int(gold_count)),
                queried_character=character,
                queried_room=room,
                template_id=template_id,
                source_dataset_info=tuple(ordered_refs),
            )
        )
    rng.shuffle(examples)
    return examples


def dataset_config(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(args.debug):
        train_per_count = min(int(args.train_per_count), 2)
        val_per_count = min(int(args.val_per_count), 1)
        all_count_test_per_count = min(int(args.all_count_test_per_count), 1)
    else:
        train_per_count = int(args.train_per_count)
        val_per_count = int(args.val_per_count)
        all_count_test_per_count = int(args.all_count_test_per_count)
    per_count_values = {
        "train_per_count": train_per_count,
        "val_per_count": val_per_count,
        "all_count_test_per_count": all_count_test_per_count,
    }
    if any(value <= 0 for value in per_count_values.values()):
        raise ValueError(f"All per-count sizes must be positive: {per_count_values}")
    splits: Dict[str, Dict[str, Any]] = {
        TRAIN_SPLIT: {
            "num_frames": 8,
            "counts": list(range(9)),
            "examples_per_count": train_per_count,
            "templates": list(TRAIN_TEMPLATES),
            "source_partition": "train",
        },
        VAL_SPLIT: {
            "num_frames": 8,
            "counts": list(range(9)),
            "examples_per_count": val_per_count,
            "templates": list(TRAIN_TEMPLATES),
            "source_partition": "val",
        },
        TEST_SPLIT: {
            "num_frames": 8,
            "counts": list(range(9)),
            "examples_per_count": all_count_test_per_count,
            "templates": list(TRAIN_TEMPLATES),
            "source_partition": "test",
        },
    }
    return {
        "dataset_seed": int(args.dataset_seed),
        "source_dataset_root": os.fspath(Path(args.source_dataset_root).resolve()),
        "source_split": str(args.source_split),
        **per_count_values,
        "counts": list(range(9)),
        "count_range": "0..8",
        "fixed_num_frames": 8,
        "splits": splits,
        "evidence_positions_randomized": True,
        "hard_semantic_distractors": False,
        "neutral_rule": "queried character absent and queried room empty",
        "ood_count_split": False,
        "real_visual_frames": True,
        "variable_num_frames": False,
    }


def ensure_dataset(
    args: argparse.Namespace,
    dataset_base: Path,
) -> Tuple[Path, Dict[str, List[VisualExample]], Dict[str, Any]]:
    config = dataset_config(args)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    dataset_dir = dataset_base / digest
    manifest_path = dataset_dir / "dataset_manifest.json"
    split_names = list(config["splits"])
    split_paths = {split: dataset_dir / f"{split}.jsonl" for split in split_names}
    regenerate = bool(args.force_regenerate_dataset) or not manifest_path.is_file() or not all(
        path.is_file() for path in split_paths.values()
    )
    if regenerate:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        pools, source_manifest = scan_source_frame_pools(
            Path(args.source_dataset_root),
            str(args.source_split),
            int(args.dataset_seed),
        )
        split_seed_offsets = {
            TRAIN_SPLIT: 11,
            VAL_SPLIT: 23,
            TEST_SPLIT: 59,
        }
        generated: Dict[str, List[VisualExample]] = {}
        for split in split_names:
            split_cfg = config["splits"][split]
            partition = str(split_cfg["source_partition"])
            generated[split] = generate_split(
                split=split,
                num_frames=int(split_cfg["num_frames"]),
                examples_per_count=int(split_cfg["examples_per_count"]),
                counts=split_cfg["counts"],
                templates=split_cfg["templates"],
                pair_pools=pools[partition],
                seed=int(args.dataset_seed) + split_seed_offsets[split],
            )
            write_jsonl(split_paths[split], [example_to_json(example) for example in generated[split]])
        manifest = {
            "dataset_hash": digest,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config,
            "source_manifest": source_manifest,
            "splits": {
                split: {
                    "path": os.fspath(path),
                    "n": len(generated[split]),
                    "count_histogram": {
                        str(count): sum(example.gold_count == count for example in generated[split])
                        for count in COUNT_VALUES
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
    for split, rows in examples.items():
        expected_frames = int(config["splits"][split]["num_frames"])
        if any(
            example.num_frames != expected_frames or len(example.frame_paths) != expected_frames
            for example in rows
        ):
            raise RuntimeError(
                f"{split}: visual frame-count validation failed; expected {expected_frames}"
            )
        for example in rows:
            for frame_path in example.frame_paths:
                path = Path(frame_path)
                resolved = path if path.is_absolute() else PROJECT_ROOT / path
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
    assert_iid_all_counts_dataset(config, examples)
    return dataset_dir, examples, manifest


def assert_iid_all_counts_dataset(
    config: Dict[str, Any],
    examples: Dict[str, List[VisualExample]],
) -> None:
    expected_splits = set(PILOT_SPLITS)
    actual_splits = set(examples)
    if actual_splits != expected_splits:
        raise RuntimeError(f"Expected only IID all-count splits {sorted(expected_splits)}, got {sorted(actual_splits)}")
    for split, split_cfg in config["splits"].items():
        counts = [int(value) for value in split_cfg.get("counts", [])]
        if counts == list(range(6)):
            raise RuntimeError(f"{split}: old counts=list(range(6)) split is forbidden")
        if counts != list(range(9)):
            raise RuntimeError(f"{split}: expected counts 0..8, got {counts}")
        if int(split_cfg.get("num_frames", -1)) != 8:
            raise RuntimeError(f"{split}: expected fixed_num_frames=8, got {split_cfg.get('num_frames')}")
    if any("high_count" in split for split in actual_splits):
        raise RuntimeError("High-count OOD split is forbidden in this IID baseline")
    if bool(config.get("ood_count_split")):
        raise RuntimeError("ood_count_split must be false for this IID baseline")
    for split, rows in examples.items():
        if any(len(example.frame_paths) != 8 or int(example.num_frames) != 8 for example in rows):
            raise RuntimeError(f"{split}: every example must have exactly 8 frame paths")
        histogram = {count: 0 for count in COUNT_VALUES}
        for example in rows:
            histogram[int(example.gold_count)] += 1
        missing = [count for count, n in histogram.items() if n <= 0]
        if missing:
            raise RuntimeError(f"{split}: missing labels {missing}; histogram={histogram}")


class TinyVisualTokenizer(text_base.TinyTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.vision_start_token_id = 268
        self.image_token_id = 269
        self.vision_end_token_id = 270
        self.vocab_size = 271

    def convert_tokens_to_ids(self, token: str) -> Optional[int]:
        return {
            "<|vision_start|>": self.vision_start_token_id,
            "<|image_pad|>": self.image_token_id,
            "<|vision_end|>": self.vision_end_token_id,
        }.get(str(token))


class TinyVisualProcessor:
    def __init__(self) -> None:
        self.tokenizer = TinyVisualTokenizer()
        self.image_token_id = self.tokenizer.image_token_id

    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool,
        return_tensors: str,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        if not messages or isinstance(messages[0], list):
            raise ValueError("TinyVisualProcessor supports one conversation at a time")
        images: List[Image.Image] = []
        text_parts: List[str] = []
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                text_parts.append(content)
                continue
            for item in content:
                if item.get("type") == "image":
                    images.append(item["image"])
                elif item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
        ids = self.tokenizer.encode("User: ", add_special_tokens=False)
        for _image in images:
            ids.extend(
                [
                    self.tokenizer.vision_start_token_id,
                    *([self.tokenizer.image_token_id] * 4),
                    self.tokenizer.vision_end_token_id,
                ]
            )
        ids.extend(self.tokenizer.encode("".join(text_parts), add_special_tokens=False))
        if add_generation_prompt:
            ids.extend(self.tokenizer.encode("\nAssistant: ", add_special_tokens=False))
        means = []
        for image in images:
            array = np.asarray(image.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
            means.append(array.mean(axis=(0, 1)))
        pixel_values = torch.tensor(np.stack(means), dtype=torch.float32)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 2, 2] for _ in images], dtype=torch.long),
        }


class TinyVisualQwen(text_base.TinyQwenLikeForCausalLM):
    def __init__(self, *, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, image_token_id: int) -> None:
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.image_token_id = int(image_token_id)
        self.image_proj = nn.Linear(3, hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **_: Any,
    ) -> Any:
        del use_cache
        hidden_states = self.embed_tokens(input_ids)
        if pixel_values is not None:
            visual = self.image_proj(pixel_values.float()).to(dtype=hidden_states.dtype)
            positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
            expected = int(visual.shape[0]) * 4
            if int(positions.numel()) != expected:
                raise RuntimeError(f"tiny visual token count {positions.numel()} != expected {expected}")
            hidden_states[0, positions, :] = hidden_states[0, positions, :] + visual.repeat_interleave(4, dim=0)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


def is_oom(exc: BaseException) -> bool:
    return text_base.is_oom(exc)


def load_model_and_processor(
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
) -> Tuple[Any, Any, bool, str]:
    if bool(args.tiny_debug_model):
        processor = TinyVisualProcessor()
        model = TinyVisualQwen(
            vocab_size=processor.tokenizer.vocab_size,
            hidden_size=int(args.tiny_hidden_size),
            num_layers=int(args.tiny_num_layers),
            num_heads=int(args.tiny_num_heads),
            image_token_id=processor.image_token_id,
        ).to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, processor, False, "tiny_visual_debug_model"

    if text_base.AutoProcessor is None or text_base.AutoModelForImageTextToText is None:
        raise RuntimeError("transformers multimodal classes are unavailable")

    def load_once(load_in_4bit: bool) -> Tuple[Any, Any]:
        processor = text_base.AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        if args.max_pixels is not None and hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = int(args.max_pixels)
        if args.min_pixels is not None and hasattr(processor, "image_processor"):
            processor.image_processor.min_pixels = int(args.min_pixels)
        if getattr(processor.tokenizer, "pad_token", None) is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"
        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": str(args.attn_implementation),
        }
        if load_in_4bit:
            if text_base.BitsAndBytesConfig is None:
                raise RuntimeError("BitsAndBytesConfig unavailable")
            kwargs["quantization_config"] = text_base.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = {"": device}
        else:
            kwargs["torch_dtype"] = dtype
        model = text_base.AutoModelForImageTextToText.from_pretrained(args.model_name, **kwargs)
        if not load_in_4bit:
            model.to(device)
        model.eval()
        model.config.use_cache = False
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, processor

    try:
        model, processor = load_once(bool(args.load_in_4bit))
        return model, processor, bool(args.load_in_4bit), "requested_4bit" if args.load_in_4bit else "bf16_or_requested_dtype"
    except RuntimeError as exc:
        if args.load_in_4bit or not args.auto_qlora_fallback or not is_oom(exc):
            raise
        print(f"BF16 model load hit OOM; retrying with 4-bit QLoRA: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, processor = load_once(True)
        return model, processor, True, "auto_qlora_fallback_after_bf16_oom"


def move_inputs_to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


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
    prompt_ids = [int(x) for x in tokenizer(prompt_text, add_special_tokens=False)["input_ids"]]
    local = text_base.token_positions_for_span(
        tokenizer,
        prompt_text,
        int(char_start),
        int(char_end),
        prompt_ids,
    )
    return [int(prompt_start) + int(position) for position in local]


def prepare_batch(
    *,
    examples: Sequence[VisualExample],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
    answer_ids: Optional[Dict[int, Tuple[int, ...]]] = None,
) -> VisualBatch:
    global _RUNTIME_CHECK_LOGGED
    if len(examples) != 1:
        raise ValueError("Visual Qwen batches are intentionally batch_size=1 to preserve image/token alignment")
    example = examples[0]
    expected_frames = int(example.num_frames)
    if expected_frames <= 0 or len(example.frame_paths) != expected_frames:
        raise AssertionError(
            f"{example.example_id}: metadata says {expected_frames} frames but received "
            f"{len(example.frame_paths)} frame paths"
        )
    if expected_frames != 8:
        raise AssertionError(f"{example.example_id}: IID fixed8 baseline received {expected_frames} frames")
    frames: List[Image.Image] = []
    try:
        for path_text in example.frame_paths:
            with Image.open(resolve_frame_path(path_text)) as image:
                frames.append(image.convert("RGB"))
        if len(frames) != expected_frames or not all(isinstance(frame, Image.Image) for frame in frames):
            raise AssertionError(f"{example.example_id}: actual PIL visual frames were not loaded")
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
        raise AssertionError(f"{example.example_id}: model input has no pixel_values visual tensor; keys={sorted(raw_inputs)}")
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
        raise RuntimeError(f"{example.example_id}: could not locate textual prompt after visual tokens")
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
    carriers = sorted(
        {
            int(position)
            for position in [*character_positions, *room_positions]
            if 0 <= int(position) < prompt_last
        }
    )
    groups = image_token_groups(input_ids[0].detach().cpu(), expected_frames, processor=processor)
    groups = [[int(position) for position in group] for group in groups]
    errors: List[str] = []
    if not carriers:
        errors.append("room/character carrier token positions not found")
    if len(groups) != expected_frames or any(not group for group in groups):
        errors.append(
            f"expected {expected_frames} non-empty visual token spans, "
            f"found {[len(group) for group in groups]}"
        )
    flattened = [position for group in groups for position in group]
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(tokenizer, "image_token_id", None)
    if image_token_id is None and hasattr(tokenizer, "convert_tokens_to_ids"):
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None or any(int(input_ids[0, position].item()) != int(image_token_id) for position in flattened):
        errors.append("mixer source groups are not exclusively visual image-pad tokens")
    if prompt_last in carriers:
        errors.append("final prompt token incorrectly selected as a carrier")
    if errors:
        raise AssertionError(f"{example.example_id}: {'; '.join(errors)}")

    loss_positions: Optional[torch.Tensor] = None
    loss_targets: Optional[torch.Tensor] = None
    if answer_ids is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise RuntimeError("Tokenizer must expose eos_token_id for answer-token training")
        targets = [*answer_ids[int(example.gold_count)], int(eos_token_id)]
        positions = list(range(prompt_last, prompt_last + len(targets)))
        suffix = torch.tensor([targets], dtype=input_ids.dtype)
        raw_inputs["input_ids"] = torch.cat([input_ids, suffix], dim=1)
        raw_inputs["attention_mask"] = torch.cat(
            [attention_mask, torch.ones((1, len(targets)), dtype=attention_mask.dtype)],
            dim=1,
        )
        loss_positions = torch.tensor([positions], dtype=torch.long)
        loss_targets = torch.tensor([targets], dtype=torch.long)

    if not _RUNTIME_CHECK_LOGGED:
        print(
            "Visual runtime assertions passed: "
            f"actual_frames={len(example.frame_paths)} visual_input_keys={visual_keys} "
            f"visual_spans={len(groups)} visual_tokens_per_frame={[len(group) for group in groups]} "
            f"carrier_positions={carriers} source_tokens_are_visual=True final_token_modified=False"
        )
        _RUNTIME_CHECK_LOGGED = True
    return VisualBatch(
        inputs=move_inputs_to_device(raw_inputs, device),
        prompt_last_indices=torch.tensor([prompt_last], device=device, dtype=torch.long),
        gold_counts=torch.tensor([example.gold_count], device=device, dtype=torch.long),
        loss_positions=loss_positions.to(device) if loss_positions is not None else None,
        loss_targets=loss_targets.to(device) if loss_targets is not None else None,
        carrier_positions=[carriers],
        frame_groups=[groups],
        token_selection_ok=[True],
        token_selection_errors=[""],
        sample_indices=[int(sample_indices[0])],
        visual_input_keys=visual_keys,
    )


class VisualCarrierMixingAdapter(nn.Module):
    def __init__(
        self,
        *,
        method: str,
        hidden_size: int,
        inject_layers: Sequence[int],
        gate_init: float,
        message_mode: str,
    ) -> None:
        super().__init__()
        if method not in {"maxmix", "pnamix"}:
            raise ValueError(method)
        self.method = str(method)
        self.hidden_size = int(hidden_size)
        self.inject_layers = [int(layer) for layer in inject_layers]
        self.layer_to_pos = {layer: pos for pos, layer in enumerate(self.inject_layers)}
        self.message_mode = str(message_mode)
        in_dim = self.hidden_size if method == "maxmix" else self.hidden_size * 5 + 1
        self.feature_norms = nn.ModuleList([nn.LayerNorm(in_dim) for _ in self.inject_layers])
        self.projections = nn.ModuleList([nn.Linear(in_dim, self.hidden_size, bias=False) for _ in self.inject_layers])
        self.gate_logits = nn.Parameter(torch.full((len(self.inject_layers),), float(gate_init), dtype=torch.float32))
        for projection in self.projections:
            nn.init.normal_(projection.weight, mean=0.0, std=0.002)
        self._carrier_positions: Optional[List[List[int]]] = None
        self._frame_groups: Optional[List[List[List[int]]]] = None
        self._handles: List[Any] = []
        self._last_stats: Dict[str, Dict[str, Any]] = {}
        self.hook_fire_counts: Dict[int, int] = defaultdict(int)
        self.message_mode_counts: Dict[str, int] = defaultdict(int)
        self.exact_failure_counts: Dict[str, int] = defaultdict(int)
        self.exact_failure_examples: List[str] = []

    def set_context(self, batch: VisualBatch) -> None:
        self._carrier_positions = [[int(position) for position in row] for row in batch.carrier_positions]
        self._frame_groups = [
            [[int(position) for position in group] for group in row]
            for row in batch.frame_groups
        ]
        self._last_stats = {
            "gate_values_by_layer": {},
            "residual_norm_by_layer": {},
            "hidden_norm_by_layer": {},
            "residual_to_hidden_ratio_by_layer": {},
            "visual_attention_mass_by_layer": {},
            "visual_message_norm_by_layer": {},
            "per_frame_message_norms_by_layer": {},
            "message_mode_by_layer": {},
        }

    def clear_context(self) -> None:
        self._carrier_positions = None
        self._frame_groups = None

    @staticmethod
    def _hidden_from_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        if args and torch.is_tensor(args[0]):
            return args[0]
        hidden = kwargs.get("hidden_states")
        return hidden if torch.is_tensor(hidden) else None

    @staticmethod
    def _replace_hidden(
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        hidden_states: torch.Tensor,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        if args and torch.is_tensor(args[0]):
            return (hidden_states,) + tuple(args[1:]), kwargs
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = hidden_states
        return args, new_kwargs

    @staticmethod
    def _repeat_kv(states: torch.Tensor, num_heads: int) -> torch.Tensor:
        if int(states.shape[1]) == int(num_heads):
            return states
        return states.repeat_interleave(int(num_heads) // int(states.shape[1]), dim=1)

    def _record_failure(self, message: str) -> None:
        key = str(message).split(":", 1)[0][:100]
        self.exact_failure_counts[key] += 1
        if len(self.exact_failure_examples) < 20:
            self.exact_failure_examples.append(str(message)[:500])

    def _frame_messages(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        kwargs: Dict[str, Any],
        *,
        require_exact: bool,
        force_approx: bool = False,
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], str]:
        attn = module.self_attn
        normalized = module.input_layernorm(hidden_states) if hasattr(module, "input_layernorm") else hidden_states
        batch, seq_len, _hidden = normalized.shape
        q = attn.q_proj(normalized)
        k = attn.k_proj(normalized)
        v = attn.v_proj(normalized)
        head_dim = int(getattr(attn, "head_dim", q.shape[-1] // int(getattr(attn, "num_heads", 1))))
        num_heads = int(getattr(attn, "num_heads", q.shape[-1] // head_dim))
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, -1, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, -1, head_dim).transpose(1, 2)
        mode = "approx"
        if not force_approx and (require_exact or self.message_mode in {"auto", "exact"}):
            position_embeddings = kwargs.get("position_embeddings")
            if (
                text_base.apply_multimodal_rotary_pos_emb is None
                or position_embeddings is None
                or not hasattr(attn, "rope_scaling")
            ):
                if require_exact or self.message_mode == "exact":
                    raise RuntimeError("exact multimodal alpha*v unavailable")
            else:
                q, k = text_base.apply_multimodal_rotary_pos_emb(
                    q,
                    k,
                    position_embeddings[0],
                    position_embeddings[1],
                    attn.rope_scaling["mrope_section"],
                )
                mode = "exact"
        k = self._repeat_kv(k, num_heads)
        v = self._repeat_kv(v, num_heads)
        scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
        attention_mask = kwargs.get("attention_mask")
        arange = torch.arange(seq_len, device=hidden_states.device)
        assert self._carrier_positions is not None and self._frame_groups is not None
        z_by_batch: List[List[torch.Tensor]] = []
        mass_by_batch: List[List[torch.Tensor]] = []
        for batch_idx in range(batch):
            carriers = [position for position in self._carrier_positions[batch_idx] if 0 <= position < seq_len]
            carrier_messages: List[torch.Tensor] = []
            carrier_masses: List[torch.Tensor] = []
            if not carriers:
                z_by_batch.append(carrier_messages)
                mass_by_batch.append(carrier_masses)
                continue
            c_idx = torch.tensor(carriers, device=hidden_states.device, dtype=torch.long)
            scores = torch.einsum(
                "hcd,hsd->hcs",
                q[batch_idx, :, c_idx, :].float(),
                k[batch_idx].float(),
            ) * scaling
            causal_allowed = arange.unsqueeze(0) <= c_idx.unsqueeze(1)
            sliding_window = getattr(attn, "sliding_window", None)
            if sliding_window is not None:
                causal_allowed &= arange.unsqueeze(0) >= c_idx.unsqueeze(1) - int(sliding_window)
            scores = scores.masked_fill(~causal_allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
            if torch.is_tensor(attention_mask):
                if attention_mask.dim() == 4:
                    scores = scores + attention_mask[batch_idx : batch_idx + 1, :, c_idx, :].float().squeeze(0)
                elif attention_mask.dim() == 2:
                    scores = scores.masked_fill(
                        ~attention_mask[batch_idx].bool().view(1, 1, -1),
                        torch.finfo(scores.dtype).min,
                    )
            probs = torch.softmax(scores, dim=-1)
            for carrier_row in range(len(carriers)):
                frame_messages: List[torch.Tensor] = []
                frame_masses: List[torch.Tensor] = []
                for group in self._frame_groups[batch_idx]:
                    if not group:
                        raise AssertionError("Visual mixer received an empty visual frame token group")
                    frame_idx = torch.tensor(group, device=hidden_states.device, dtype=torch.long)
                    alpha = probs[:, carrier_row, frame_idx]
                    values = v[batch_idx, :, frame_idx, :].float()
                    message = torch.einsum("ht,htd->hd", alpha, values).reshape(num_heads * head_dim)
                    frame_messages.append(message)
                    frame_masses.append(alpha.sum(dim=-1).mean())
                carrier_messages.append(torch.stack(frame_messages, dim=0))
                carrier_masses.append(torch.stack(frame_masses, dim=0))
            z_by_batch.append(carrier_messages)
            mass_by_batch.append(carrier_masses)
        return z_by_batch, mass_by_batch, mode

    def _messages_with_fallback(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], str]:
        try:
            return self._frame_messages(
                module,
                hidden_states,
                kwargs,
                require_exact=self.message_mode == "exact",
            )
        except Exception as exc:
            self._record_failure(f"layer {layer_idx}: {type(exc).__name__}: {exc}")
            if self.message_mode == "exact":
                raise
            return self._frame_messages(
                module,
                hidden_states,
                kwargs,
                require_exact=False,
                force_approx=True,
            )

    def inject_before_layer(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        if self._carrier_positions is None or self._frame_groups is None:
            return hidden_states
        layer_pos = self.layer_to_pos[int(layer_idx)]
        self.hook_fire_counts[int(layer_idx)] += 1
        z_by_batch, mass_by_batch, mode = self._messages_with_fallback(
            module,
            hidden_states,
            int(layer_idx),
            kwargs,
        )
        self.message_mode_counts[f"{int(layer_idx)}:{mode}"] += int(hidden_states.shape[0])
        out = hidden_states.clone()
        gate = torch.sigmoid(self.gate_logits[layer_pos].float())
        residual_norms: List[float] = []
        hidden_norms: List[float] = []
        ratios: List[float] = []
        attention_masses: List[float] = []
        visual_message_norms: List[float] = []
        per_frame_norm_rows: List[List[float]] = []
        for batch_idx in range(int(hidden_states.shape[0])):
            carriers = self._carrier_positions[batch_idx]
            sample_residuals: List[float] = []
            sample_hidden: List[float] = []
            sample_ratios: List[float] = []
            sample_masses: List[float] = []
            sample_messages: List[float] = []
            sample_per_frame: List[List[float]] = []
            for carrier_row, carrier_pos in enumerate(carriers):
                z = z_by_batch[batch_idx][carrier_row].float()
                masses = mass_by_batch[batch_idx][carrier_row].float()
                expected_frames = len(self._frame_groups[batch_idx])
                if int(z.shape[0]) != expected_frames:
                    raise AssertionError(
                        f"Expected {expected_frames} visual frame messages, found {z.shape[0]}"
                    )
                if self.method == "maxmix":
                    features = z.max(dim=0).values
                else:
                    std = z.std(dim=0, unbiased=False)
                    features = torch.cat(
                        [
                            z.sum(dim=0),
                            z.mean(dim=0),
                            z.max(dim=0).values,
                            z.min(dim=0).values,
                            std,
                            masses.sum().reshape(1),
                        ],
                        dim=-1,
                    )
                projected = self.projections[layer_pos](
                    self.feature_norms[layer_pos](features.float())
                ).float()
                residual = gate * projected
                before = hidden_states[batch_idx, carrier_pos, :].float()
                out[batch_idx, carrier_pos, :] = (
                    out[batch_idx, carrier_pos, :] + residual.to(dtype=out.dtype)
                )
                residual_norm = residual.norm()
                hidden_norm = before.norm()
                sample_residuals.append(float(residual_norm.detach().cpu().item()))
                sample_hidden.append(float(hidden_norm.detach().cpu().item()))
                sample_ratios.append(float((residual_norm / hidden_norm.clamp_min(EPS)).detach().cpu().item()))
                sample_masses.append(float(masses.sum().detach().cpu().item()))
                sample_messages.append(float(z.sum(dim=0).norm().detach().cpu().item()))
                sample_per_frame.append([float(value) for value in z.norm(dim=-1).detach().cpu().tolist()])
            residual_norms.append(finite_mean(sample_residuals, default=0.0))
            hidden_norms.append(finite_mean(sample_hidden, default=0.0))
            ratios.append(finite_mean(sample_ratios, default=0.0))
            attention_masses.append(finite_mean(sample_masses, default=0.0))
            visual_message_norms.append(finite_mean(sample_messages, default=0.0))
            if sample_per_frame:
                per_frame_norm_rows.append(
                    [
                        finite_mean([row[frame_idx] for row in sample_per_frame], default=0.0)
                        for frame_idx in range(len(sample_per_frame[0]))
                    ]
                )
            else:
                per_frame_norm_rows.append([0.0] * len(self._frame_groups[batch_idx]))
        key = str(int(layer_idx))
        self._last_stats["gate_values_by_layer"][key] = [float(gate.detach().cpu().item())] * len(residual_norms)
        self._last_stats["residual_norm_by_layer"][key] = residual_norms
        self._last_stats["hidden_norm_by_layer"][key] = hidden_norms
        self._last_stats["residual_to_hidden_ratio_by_layer"][key] = ratios
        self._last_stats["visual_attention_mass_by_layer"][key] = attention_masses
        self._last_stats["visual_message_norm_by_layer"][key] = visual_message_norms
        self._last_stats["per_frame_message_norms_by_layer"][key] = per_frame_norm_rows
        self._last_stats["message_mode_by_layer"][key] = [mode] * len(residual_norms)
        return out

    def attach(self, model: Any) -> None:
        self.detach()
        layers = get_layers(model)
        for layer_idx in self.inject_layers:
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(f"carrier layer={layer_idx} outside [0, {len(layers) - 1}]")

            def hook(
                module: Any,
                args: Tuple[Any, ...],
                kwargs: Dict[str, Any],
                *,
                layer: int = int(layer_idx),
            ) -> Any:
                hidden = self._hidden_from_args(args, kwargs)
                if hidden is None:
                    return args, kwargs
                return self._replace_hidden(
                    args,
                    kwargs,
                    self.inject_before_layer(module, hidden, layer, kwargs),
                )

            self._handles.append(layers[layer_idx].register_forward_pre_hook(hook, with_kwargs=True))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, by_layer in self._last_stats.items():
            row_payload: Dict[str, Any] = {}
            for layer, values in by_layer.items():
                row_payload[layer] = values[row] if isinstance(values, list) and row < len(values) else values
            out[name] = row_payload
        for prefix in (
            "residual_norm",
            "hidden_norm",
            "residual_to_hidden_ratio",
            "visual_attention_mass",
            "visual_message_norm",
            "gate_values",
        ):
            out[prefix] = finite_mean(out.get(f"{prefix}_by_layer", {}).values(), default=0.0)
        return out


class VariantAdapter(nn.Module):
    def __init__(
        self,
        lora: Optional[text_base.MinimalAttentionLoRAAdapter],
        mixer: Optional[VisualCarrierMixingAdapter],
    ) -> None:
        super().__init__()
        self.lora = lora
        self.mixer = mixer

    def attach(self, model: Any) -> None:
        if self.lora is not None:
            self.lora.attach(model)
        if self.mixer is not None:
            self.mixer.attach(model)

    def detach(self) -> None:
        if self.mixer is not None:
            self.mixer.detach()
        if self.lora is not None:
            self.lora.detach()

    def set_context(self, batch: VisualBatch) -> None:
        if self.lora is not None:
            self.lora.set_context(batch)
        if self.mixer is not None:
            self.mixer.set_context(batch)

    def clear_context(self) -> None:
        if self.mixer is not None:
            self.mixer.clear_context()
        if self.lora is not None:
            self.lora.clear_context()

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if self.lora is not None:
            stats.update(self.lora.stats_for_row(row))
        if self.mixer is not None:
            stats.update(self.mixer.stats_for_row(row))
        return stats


def make_variant_adapter(
    *,
    variant: str,
    args: argparse.Namespace,
    hidden_size: int,
    lora_layers: Sequence[int],
    carrier_layers: Sequence[int],
    lora_targets: Sequence[str],
) -> Optional[VariantAdapter]:
    if variant == FROZEN_QWEN_BASELINE:
        return None
    lora = text_base.MinimalAttentionLoRAAdapter(
        inject_layers=lora_layers,
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        dropout=float(args.lora_dropout),
        target_modules=lora_targets,
    )
    mixer: Optional[VisualCarrierMixingAdapter] = None
    if variant in {MAXMIX_CARRIER_14_17, PNAMIX_CARRIER_14_17}:
        mixer = VisualCarrierMixingAdapter(
            method="maxmix" if variant == MAXMIX_CARRIER_14_17 else "pnamix",
            hidden_size=int(hidden_size),
            inject_layers=carrier_layers,
            gate_init=float(args.carrier_gate_init),
            message_mode=str(args.message_mode),
        )
    return VariantAdapter(lora, mixer)


def count_trainable_parameters(module: Optional[nn.Module]) -> int:
    return 0 if module is None else int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def single_token_count_ids(answer_ids: Dict[int, Tuple[int, ...]]) -> Dict[int, int]:
    token_ids: Dict[int, int] = {}
    for count in COUNT_VALUES:
        encoded = answer_ids.get(int(count))
        if encoded is None or len(encoded) != 1:
            raise RuntimeError(f"Count {count} must be one token for forced-choice evaluation; got {encoded}")
        token_ids[int(count)] = int(encoded[0])
    if len(set(token_ids.values())) != len(token_ids):
        raise RuntimeError(f"Count answer token IDs are not unique: {token_ids}")
    return token_ids


def answer_sequence_cross_entropy(logits: torch.Tensor, batch: VisualBatch) -> Tuple[torch.Tensor, torch.Tensor]:
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


def limited_indices(examples: Sequence[VisualExample], limit: int, seed: int) -> List[int]:
    return text_base.limited_indices(examples, int(limit), int(seed))


def batch_indices(indices: Sequence[int], batch_size: int, seed: int, shuffle: bool) -> List[List[int]]:
    if int(batch_size) != 1:
        raise ValueError("--batch-size must be 1 for visual frame alignment; use --grad-accum for effective batch size")
    return text_base.batch_indices(indices, 1, int(seed), bool(shuffle))


@torch.no_grad()
def evaluate_split(
    *,
    variant: str,
    split_name: str,
    model: Any,
    processor: Any,
    adapter: Optional[VariantAdapter],
    examples: Sequence[VisualExample],
    indices: Sequence[int],
    answer_ids: Dict[int, Tuple[int, ...]],
    digit_token_ids: Dict[int, int],
    device: str,
    seed: int,
) -> Dict[str, Any]:
    model.eval()
    if adapter is not None:
        adapter.eval()
    rows: List[Dict[str, Any]] = []
    ce_total = 0.0
    count_values = sorted(digit_token_ids)
    for step, idx in enumerate(indices, start=1):
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
        count_logits = text_base.select_count_logits(outputs.logits, batch.prompt_last_indices, digit_token_ids)
        _ce, ce_vec = answer_sequence_cross_entropy(outputs.logits, batch)
        pred = int(count_values[int(count_logits[0].argmax().detach().cpu().item())])
        stats = adapter.stats_for_row(0) if adapter is not None else {}
        if adapter is not None:
            adapter.clear_context()
        ce_total += float(ce_vec[0].detach().cpu().item())
        row: Dict[str, Any] = {
            "variant": variant,
            "example_id": example.example_id,
            "id": example.example_id,
            "split": split_name,
            "frame_paths": list(example.frame_paths),
            "source_dataset_info": list(example.source_dataset_info),
            "num_frames": int(example.num_frames),
            "gold_count": int(example.gold_count),
            "predicted_count": pred,
            "pred_count": pred,
            "correct": int(pred == example.gold_count),
            "outcome": "correct" if pred == example.gold_count else "incorrect",
            "abs_error": abs(pred - example.gold_count),
            "undercount": int(pred < example.gold_count),
            "evidence_frame_indices": list(example.evidence_frame_indices),
            "carrier_token_positions": list(batch.carrier_positions[0]),
            "visual_token_spans_per_frame": [list(group) for group in batch.frame_groups[0]],
            "visual_token_counts_per_frame": [len(group) for group in batch.frame_groups[0]],
            "visual_input_keys": list(batch.visual_input_keys),
            "source_tokens_are_visual": True,
            "template_id": example.template_id,
            "queried_character": example.queried_character,
            "queried_room": example.queried_room,
            "question": example.question,
            "answer": example.answer,
            "ce": float(ce_vec[0].detach().cpu().item()),
            "candidate_logits": text_base.candidate_logits_payload(count_logits[0], count_values),
        }
        for key, value in stats.items():
            row[f"{key}_json" if isinstance(value, dict) else key] = json_compact(value) if isinstance(value, dict) else value
        rows.append(row)
        if step == 1 or step % 25 == 0:
            print(f"  eval {variant} {split_name}: {step}/{len(indices)}")
    return {
        "rows": rows,
        "ce": ce_total / max(1, len(rows)),
        "accuracy": accuracy([row["gold_count"] for row in rows], [row["predicted_count"] for row in rows]),
        "mae": mae([row["gold_count"] for row in rows], [row["predicted_count"] for row in rows]),
    }


def train_adapter(
    *,
    variant: str,
    args: argparse.Namespace,
    run_dir: Path,
    model: Any,
    processor: Any,
    adapter: VariantAdapter,
    train_examples: Sequence[VisualExample],
    val_examples: Sequence[VisualExample],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    answer_ids: Dict[int, Tuple[int, ...]],
    digit_token_ids: Dict[int, int],
    device: str,
) -> Tuple[List[Dict[str, Any]], Path]:
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"{variant}: no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_adapter.pt"
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_accuracy = -math.inf
    best_val_ce = math.inf
    history: List[Dict[str, Any]] = []
    count_values = sorted(digit_token_ids)
    for epoch in range(1, int(args.epochs) + 1):
        model.eval()
        adapter.train()
        order = batch_indices(train_indices, 1, int(args.seed) + epoch * 1009, True)
        optimizer.zero_grad(set_to_none=True)
        total_ce = 0.0
        correct = 0
        n = 0
        for step, idxs in enumerate(order, start=1):
            idx = idxs[0]
            example = train_examples[int(idx)]
            batch = prepare_batch(
                examples=[example],
                sample_indices=[idx],
                processor=processor,
                device=device,
                answer_ids=answer_ids,
            )
            adapter.set_context(batch)
            outputs = model(**batch.inputs, use_cache=False)
            count_logits = text_base.select_count_logits(outputs.logits, batch.prompt_last_indices, digit_token_ids)
            loss, _ = answer_sequence_cross_entropy(outputs.logits, batch)
            torch.autograd.backward(loss / max(1, int(args.grad_accum)))
            pred = int(count_values[int(count_logits[0].argmax().detach().cpu().item())])
            correct += int(pred == example.gold_count)
            n += 1
            total_ce += float(loss.detach().cpu().item())
            adapter.clear_context()
            if step % max(1, int(args.grad_accum)) == 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 25 == 0:
                print(
                    f"  {variant} epoch={epoch} step={step}/{len(order)} "
                    f"train_ce={total_ce / max(1, n):.4f} train_acc={correct / max(1, n):.4f}"
                )
        if len(order) % max(1, int(args.grad_accum)):
            torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        val_result = evaluate_split(
            variant=variant,
            split_name=VAL_SPLIT,
            model=model,
            processor=processor,
            adapter=adapter,
            examples=val_examples,
            indices=val_indices,
            answer_ids=answer_ids,
            digit_token_ids=digit_token_ids,
            device=device,
            seed=int(args.seed) + 5000 + epoch,
        )
        history_row = {
            "variant": variant,
            "epoch": epoch,
            "train_loss": total_ce / max(1, n),
            "train_ce": total_ce / max(1, n),
            "train_accuracy": correct / max(1, n),
            "train_steps": len(order),
            "val_ce": float(val_result["ce"]),
            "val_accuracy": float(val_result["accuracy"]),
            "val_mae": float(val_result["mae"]),
            "trainable_parameters": count_trainable_parameters(adapter),
        }
        history.append(history_row)
        print(
            f"  {variant} epoch={epoch} train_ce={history_row['train_ce']:.4f} "
            f"train_acc={history_row['train_accuracy']:.4f} val_ce={history_row['val_ce']:.4f} "
            f"val_acc={history_row['val_accuracy']:.4f}"
        )
        improved = history_row["val_accuracy"] > best_val_accuracy + 1e-9 or (
            abs(history_row["val_accuracy"] - best_val_accuracy) <= 1e-9
            and history_row["val_ce"] < best_val_ce
        )
        if improved:
            best_val_accuracy = float(history_row["val_accuracy"])
            best_val_ce = float(history_row["val_ce"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
            torch.save(
                {
                    "variant": variant,
                    "adapter_state_dict": best_state,
                    "history": history,
                    "note": "Adapter-only checkpoint; frozen Qwen weights are not stored.",
                },
                checkpoint_path,
            )
    if best_state is not None:
        adapter.load_state_dict(best_state)
    return history, checkpoint_path


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gold = [int(row["gold_count"]) for row in rows]
    pred = [int(row["predicted_count"]) for row in rows]
    high = [row for row in rows if int(row["gold_count"]) in {5, 6, 7, 8}]
    high_accuracy = accuracy(
        [int(row["gold_count"]) for row in high],
        [int(row["predicted_count"]) for row in high],
    )
    return {
        "n": len(rows),
        "accuracy": accuracy(gold, pred),
        "mae": mae(gold, pred),
        "high_count_accuracy": high_accuracy,
        "high_count_5_8_accuracy": high_accuracy,
        "mean_predicted_count": finite_mean(pred),
        "mean_gold_count": finite_mean(gold),
        "undercount_rate": finite_mean([int(p < g) for p, g in zip(pred, gold)]),
    }


def build_metrics(
    rows: Sequence[Dict[str, Any]],
    variant: str,
    split_names: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    split_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []
    for split in split_names:
        selected = [row for row in rows if row["split"] == split]
        split_rows.append({"variant": variant, "split": split, **summarize_rows(selected)})
        for count in COUNT_VALUES:
            count_selected = [row for row in selected if int(row["gold_count"]) == count]
            count_rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "gold_count": count,
                    **summarize_rows(count_selected),
                }
            )
        for gold in COUNT_VALUES:
            for pred in COUNT_VALUES:
                confusion_rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "gold_count": gold,
                        "predicted_count": pred,
                        "n": sum(
                            int(row["gold_count"]) == gold and int(row["predicted_count"]) == pred
                            for row in selected
                        ),
                    }
                )
    return split_rows, count_rows, confusion_rows


def plot_lines(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    value_key: str,
    ylabel: str,
    split_filter: Optional[Sequence[str]] = None,
    variant_lines: bool = False,
    ideal: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    filtered = [
        row
        for row in rows
        if split_filter is None or str(row.get("split")) in set(split_filter)
    ]
    line_key = "variant" if variant_lines else "split"
    for label in sorted({str(row[line_key]) for row in filtered}):
        selected = sorted(
            [row for row in filtered if str(row[line_key]) == label],
            key=lambda row: int(row["gold_count"]),
        )
        xs = [int(row["gold_count"]) for row in selected if finite_float(row.get(value_key)) is not None]
        ys = [float(row[value_key]) for row in selected if finite_float(row.get(value_key)) is not None]
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
    if ideal:
        ax.plot(COUNT_VALUES, COUNT_VALUES, linestyle="--", color="black", linewidth=1, label="ideal")
    ax.set_xlabel("Gold count")
    ax.set_ylabel(ylabel)
    if value_key in {"accuracy", "undercount_rate"}:
        ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    if filtered:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    selected = [row for row in rows if row["split"] == TEST_SPLIT]
    matrix = np.zeros((9, 9), dtype=float)
    for row in selected:
        matrix[int(row["gold_count"]), int(row["predicted_count"])] += int(row["n"])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("Gold count")
    for gold in range(9):
        for pred in range(9):
            if matrix[gold, pred]:
                ax.text(pred, gold, str(int(matrix[gold, pred])), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bar(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    label_key: str,
    value_key: str,
    ylabel: str,
) -> None:
    labels = [str(row[label_key]) for row in rows]
    values = [float(row[value_key]) if finite_float(row.get(value_key)) is not None else math.nan for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(range(len(labels)), values, color="#4c78a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if value_key in {"accuracy", "high_count_accuracy"}:
        ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_json_dict(row: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = row.get(key, {})
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def diagnostic_rows(
    rows: Sequence[Dict[str, Any]],
    json_key: str,
    *,
    by_count: bool,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in rows:
        payload = parse_json_dict(row, json_key)
        for layer, value in payload.items():
            if isinstance(value, list):
                numeric = [float(x) for x in value if finite_float(x) is not None]
                scalar = finite_mean(numeric)
            else:
                scalar = float(value) if finite_float(value) is not None else math.nan
            if finite_float(scalar) is None:
                continue
            key = str(int(row["gold_count"])) if by_count else str(layer)
            grouped[(key, int(row["gold_count"]) if by_count else 0)].append(float(scalar))
    output: List[Dict[str, Any]] = []
    if by_count:
        by_gold: Dict[int, List[float]] = defaultdict(list)
        for (_label, gold), values in grouped.items():
            by_gold[gold].extend(values)
        output = [{"gold_count": gold, "value": finite_mean(values)} for gold, values in sorted(by_gold.items())]
    else:
        by_layer: Dict[int, List[float]] = defaultdict(list)
        for (label, _), values in grouped.items():
            by_layer[int(label)].extend(values)
        output = [{"layer": layer, "value": finite_mean(values)} for layer, values in sorted(by_layer.items())]
    return output


def plot_diagnostic(path: Path, rows: Sequence[Dict[str, Any]], *, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if rows:
        key = "gold_count" if "gold_count" in rows[0] else "layer"
        ax.plot([row[key] for row in rows], [row["value"] for row in rows], marker="o")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No mixer diagnostics for this variant", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_run_artifacts(
    *,
    run_dir: Path,
    variant: str,
    rows: Sequence[Dict[str, Any]],
    history: Sequence[Dict[str, Any]],
    split_names: Sequence[str],
    no_plots: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    split_metrics, count_metrics, confusion = build_metrics(rows, variant, split_names)
    write_jsonl(run_dir / "predictions.jsonl", list(rows))
    for split in split_names:
        write_jsonl(run_dir / f"predictions_{split}.jsonl", [row for row in rows if row["split"] == split])
    write_csv_dynamic(run_dir / "metrics_by_split.csv", split_metrics, ["variant", "split"])
    write_json(run_dir / "metrics_by_split.json", {"rows": split_metrics})
    write_csv_dynamic(run_dir / "metrics_by_count.csv", count_metrics, ["variant", "split", "gold_count"])
    write_csv_dynamic(run_dir / "confusion_matrix.csv", confusion, ["variant", "split", "gold_count", "predicted_count"])
    write_csv_dynamic(run_dir / "train_history.csv", history, ["variant", "epoch"])
    write_json(run_dir / "train_history.json", {"rows": list(history)})
    if not no_plots:
        plot_lines(run_dir / "accuracy_by_count.png", count_metrics, value_key="accuracy", ylabel="Accuracy")
        plot_lines(run_dir / "mae_by_count.png", count_metrics, value_key="mae", ylabel="MAE")
        plot_lines(
            run_dir / "mean_predicted_count_by_gold_count.png",
            count_metrics,
            value_key="mean_predicted_count",
            ylabel="Mean predicted count",
            ideal=True,
        )
        plot_lines(
            run_dir / "undercount_rate_by_count.png",
            count_metrics,
            value_key="undercount_rate",
            ylabel="Undercount rate",
        )
        plot_confusion(run_dir / "confusion_matrix.png", confusion)
        plot_bar(
            run_dir / "accuracy_by_split.png",
            [row for row in split_metrics if row["split"] != TRAIN_SPLIT],
            label_key="split",
            value_key="accuracy",
            ylabel="Accuracy",
        )
        text_base.plot_train_val_curves(run_dir / "train_val_curves.png", history)
        diagnostic_specs = (
            ("gate_values_by_layer_json", "gate_values_by_layer.png", False, "Layer", "Gate value"),
            ("residual_norm_by_layer_json", "residual_norm_by_layer.png", False, "Layer", "Residual norm"),
            (
                "residual_to_hidden_ratio_by_layer_json",
                "residual_to_hidden_ratio_by_layer.png",
                False,
                "Layer",
                "Residual / hidden norm",
            ),
            ("residual_norm_by_layer_json", "residual_norm_by_count.png", True, "Gold count", "Residual norm"),
            (
                "visual_attention_mass_by_layer_json",
                "visual_attention_mass_by_count.png",
                True,
                "Gold count",
                "Visual attention mass",
            ),
            (
                "visual_message_norm_by_layer_json",
                "visual_message_norm_by_count.png",
                True,
                "Gold count",
                "Visual message norm",
            ),
        )
        for json_key, filename, by_count, xlabel, ylabel in diagnostic_specs:
            plot_diagnostic(
                run_dir / filename,
                diagnostic_rows(rows, json_key, by_count=by_count),
                xlabel=xlabel,
                ylabel=ylabel,
            )
    return split_metrics, count_metrics


def grouped_summary_bar(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    splits: Sequence[str],
    value_key: str,
    ylabel: str,
) -> None:
    variants = [variant for variant in VARIANTS if any(row["variant"] == variant for row in rows)]
    width = 0.8 / max(1, len(variants))
    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(11, 5))
    for pos, variant in enumerate(variants):
        values = []
        for split in splits:
            match = next((row for row in rows if row["variant"] == variant and row["split"] == split), None)
            values.append(float(match[value_key]) if match and finite_float(match.get(value_key)) is not None else math.nan)
        ax.bar(x + (pos - (len(variants) - 1) / 2) * width, values, width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if value_key in {"accuracy", "high_count_accuracy"}:
        ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_iid_baseline_summary(
    summary_dir: Path,
    split_rows: Sequence[Dict[str, Any]],
    count_rows: Sequence[Dict[str, Any]],
) -> None:
    rows: List[Dict[str, Any]] = []
    variants = sorted({str(row["variant"]) for row in split_rows})
    for variant in variants:
        val = next((row for row in split_rows if row["variant"] == variant and row["split"] == VAL_SPLIT), None)
        test = next((row for row in split_rows if row["variant"] == variant and row["split"] == TEST_SPLIT), None)
        output: Dict[str, Any] = {
            "variant": variant,
            "val_accuracy": val.get("accuracy", math.nan) if val else math.nan,
            "test_accuracy": test.get("accuracy", math.nan) if test else math.nan,
            "test_mae": test.get("mae", math.nan) if test else math.nan,
            "test_high_count_5_8_accuracy": test.get("high_count_5_8_accuracy", math.nan) if test else math.nan,
        }
        for count in (5, 6, 7, 8):
            match = next(
                (
                    row
                    for row in count_rows
                    if row["variant"] == variant
                    and row["split"] == TEST_SPLIT
                    and int(row["gold_count"]) == count
                ),
                None,
            )
            output[f"test_mean_predicted_count_gold_{count}"] = (
                match.get("mean_predicted_count", math.nan) if match else math.nan
            )
        rows.append(output)
    write_csv_dynamic(summary_dir / "iid_all_counts_summary.csv", rows, ["variant"])
    write_json(summary_dir / "iid_all_counts_summary.json", {"rows": rows})


def write_summary(
    summary_dir: Path,
    split_rows: Sequence[Dict[str, Any]],
    count_rows: Sequence[Dict[str, Any]],
    no_plots: bool,
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv_dynamic(summary_dir / "summary_metrics.csv", split_rows, ["variant", "split"])
    write_json(summary_dir / "summary_metrics.json", {"rows": list(split_rows)})
    write_csv_dynamic(summary_dir / "summary_metrics_by_count.csv", count_rows, ["variant", "split", "gold_count"])
    write_iid_baseline_summary(summary_dir, split_rows, count_rows)
    if no_plots:
        return
    eval_splits = [
        split
        for split in EVAL_SPLITS
        if any(row["split"] == split for row in split_rows)
    ]
    grouped_summary_bar(
        summary_dir / "accuracy_by_split.png",
        split_rows,
        splits=eval_splits,
        value_key="accuracy",
        ylabel="Accuracy",
    )
    grouped_summary_bar(
        summary_dir / "mae_by_split.png",
        split_rows,
        splits=eval_splits,
        value_key="mae",
        ylabel="MAE",
    )
    all_count_rows = [row for row in count_rows if row["split"] == TEST_SPLIT]
    plot_lines(
        summary_dir / "accuracy_by_count_comparison.png",
        all_count_rows,
        value_key="accuracy",
        ylabel="Accuracy",
        variant_lines=True,
    )
    plot_lines(
        summary_dir / "mae_by_count_comparison.png",
        all_count_rows,
        value_key="mae",
        ylabel="MAE",
        variant_lines=True,
    )
    plot_lines(
        summary_dir / "mean_predicted_count_by_gold_count_comparison.png",
        all_count_rows,
        value_key="mean_predicted_count",
        ylabel="Mean predicted count",
        variant_lines=True,
        ideal=True,
    )
    plot_lines(
        summary_dir / "undercount_rate_by_count_comparison.png",
        all_count_rows,
        value_key="undercount_rate",
        ylabel="Undercount rate",
        variant_lines=True,
    )
    high_rows = [row for row in split_rows if row["split"] == TEST_SPLIT]
    plot_bar(
        summary_dir / "high_count_5_8_accuracy_bar.png",
        high_rows,
        label_key="variant",
        value_key="high_count_5_8_accuracy",
        ylabel="Gold 5..8 accuracy",
    )


def run_dir_for_variant(output_root: Path, variant: str, prefix: str, debug: bool) -> Path:
    base = Path(output_root).resolve() / "debug" if debug else Path(output_root).resolve()
    prefix_part = f"{safe_name(prefix)}_" if str(prefix).strip() else ""
    return base / f"{time.strftime('%Y%m%d_%H%M%S')}_{prefix_part}{safe_name(variant)}"


def main() -> int:
    args = parse_args()
    if int(args.batch_size) != 1:
        raise ValueError("This visual experiment requires --batch-size 1")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    variants = parse_variants(args.variants)
    lora_layers = parse_int_tokens(args.lora_layers)
    carrier_layers = parse_int_tokens(args.carrier_layers)
    lora_targets = split_tokens(args.lora_targets)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir, examples_by_split, dataset_manifest = ensure_dataset(args, output_root / "datasets")
    split_names = list(examples_by_split)
    print(f"Visual dataset: {dataset_dir}")
    for split, examples in examples_by_split.items():
        print(f"  {split}: n={len(examples)} counts={sorted({example.gold_count for example in examples})}")

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu --tiny-debug-model for debug")
    dtype = text_base.dtype_from_arg(str(args.dtype))
    model, processor, qlora_used, load_note = load_model_and_processor(args, device, dtype)
    tokenizer = processor.tokenizer
    count_style, answer_ids = text_base.answer_token_ids(
        tokenizer,
        int(args.candidate_min),
        int(args.candidate_max),
    )
    digit_token_ids = single_token_count_ids(answer_ids)
    hidden_size = text_base.hidden_size_from_model(model)
    layers = get_layers(model)
    print(
        f"Loaded model={args.model_name} hidden_size={hidden_size} num_layers={len(layers)} "
        f"qlora_used={qlora_used} load_note={load_note}"
    )
    print(f"Count token style={count_style} ids={answer_ids}")

    train_indices = limited_indices(
        examples_by_split[TRAIN_SPLIT],
        int(args.max_train_examples),
        int(args.seed) + 101,
    )
    eval_indices: Dict[str, List[int]] = {}
    for pos, split in enumerate(split_names):
        limit = int(args.max_eval_examples)
        if split == TRAIN_SPLIT:
            values = train_indices if limit <= 0 else train_indices[:limit]
        else:
            values = limited_indices(examples_by_split[split], limit, int(args.seed) + 200 + pos)
        eval_indices[split] = values

    all_split_rows: List[Dict[str, Any]] = []
    all_count_rows: List[Dict[str, Any]] = []
    for variant in variants:
        run_dir = run_dir_for_variant(output_root, variant, str(args.run_prefix), bool(args.debug))
        handle, old_stdout, old_stderr = setup_logging(run_dir)
        adapter: Optional[VariantAdapter] = None
        try:
            print(f"Starting variant={variant} run_dir={run_dir}")
            random.seed(int(args.seed))
            np.random.seed(int(args.seed))
            torch.manual_seed(int(args.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed))
            adapter = make_variant_adapter(
                variant=variant,
                args=args,
                hidden_size=hidden_size,
                lora_layers=lora_layers,
                carrier_layers=carrier_layers,
                lora_targets=lora_targets,
            )
            if adapter is not None:
                adapter.attach(model)
                adapter.to(device)
            trainable_parameters = count_trainable_parameters(adapter)
            config = {
                "experiment": EXPERIMENT_NAME,
                "variant": variant,
                "model_name": str(args.model_name),
                "run_dir": os.fspath(run_dir),
                "dataset_dir": os.fspath(dataset_dir),
                "dataset_manifest": dataset_manifest,
                "seed": int(args.seed),
                "dataset_seed": int(args.dataset_seed),
                "device": device,
                "dtype": str(args.dtype),
                "qlora_used": qlora_used,
                "load_note": load_note,
                "real_visual_frames": True,
                "runtime_assertions": {
                    "pixel_values_required": True,
                    "fixed8_required_for_primary_splits": True,
                    "one_visual_token_span_per_frame_required": True,
                    "source_groups_must_be_image_pad_tokens": True,
                    "room_character_carriers_required": True,
                    "final_token_injection_forbidden": True,
                },
                "lora": {
                    "enabled": variant != FROZEN_QWEN_BASELINE,
                    "rank": int(args.lora_rank),
                    "alpha": float(args.lora_alpha),
                    "dropout": float(args.lora_dropout),
                    "layers": lora_layers,
                    "target_modules": lora_targets,
                },
                "carrier_mixing": {
                    "enabled": variant in {MAXMIX_CARRIER_14_17, PNAMIX_CARRIER_14_17},
                    "layers": carrier_layers,
                    "method": (
                        "maxmix"
                        if variant == MAXMIX_CARRIER_14_17
                        else "pnamix"
                        if variant == PNAMIX_CARRIER_14_17
                        else "none"
                    ),
                    "source": "one alpha*v message per real visual frame",
                    "target": "queried room and character carrier tokens only",
                    "uses_gold_count": False,
                    "uses_evidence_labels": False,
                    "message_mode": str(args.message_mode),
                },
                "training": {
                    "epochs": 0 if variant == FROZEN_QWEN_BASELINE else int(args.epochs),
                    "lr": float(args.lr),
                    "batch_size": 1,
                    "grad_accum": int(args.grad_accum),
                    "loss": "normal Qwen answer-token cross entropy through the LM head",
                    "trainable_parameters": trainable_parameters,
                },
                "submit_mode": str(args.submit_mode),
                "debug": bool(args.debug),
                "tiny_debug_model": bool(args.tiny_debug_model),
            }
            write_json(run_dir / "config.json", config)
            print(f"Trainable parameters: {trainable_parameters:,}")
            history: List[Dict[str, Any]] = []
            if adapter is not None:
                history, checkpoint_path = train_adapter(
                    variant=variant,
                    args=args,
                    run_dir=run_dir,
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    train_examples=examples_by_split[TRAIN_SPLIT],
                    val_examples=examples_by_split[VAL_SPLIT],
                    train_indices=train_indices,
                    val_indices=eval_indices[VAL_SPLIT],
                    answer_ids=answer_ids,
                    digit_token_ids=digit_token_ids,
                    device=device,
                )
                print(f"Best checkpoint: {checkpoint_path}")
            else:
                print("Frozen Qwen baseline: evaluation only.")
            prediction_rows: List[Dict[str, Any]] = []
            eval_split_names = [split for split in EVAL_SPLITS if split in examples_by_split]
            for split in eval_split_names:
                result = evaluate_split(
                    variant=variant,
                    split_name=split,
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    examples=examples_by_split[split],
                    indices=eval_indices[split],
                    answer_ids=answer_ids,
                    digit_token_ids=digit_token_ids,
                    device=device,
                    seed=int(args.seed) + 7000,
                )
                print(
                    f"  {split}: accuracy={result['accuracy']:.4f} "
                    f"mae={result['mae']:.4f} ce={result['ce']:.4f}"
                )
                prediction_rows.extend(result["rows"])
            split_rows, count_rows = write_run_artifacts(
                run_dir=run_dir,
                variant=variant,
                rows=prediction_rows,
                history=history,
                split_names=eval_split_names,
                no_plots=bool(args.no_plots),
            )
            all_split_rows.extend(split_rows)
            all_count_rows.extend(count_rows)
            if adapter is not None and adapter.mixer is not None:
                write_json(
                    run_dir / "carrier_message_diagnostics.json",
                    {
                        "hook_fire_counts": {str(key): value for key, value in sorted(adapter.mixer.hook_fire_counts.items())},
                        "message_mode_counts": dict(sorted(adapter.mixer.message_mode_counts.items())),
                        "exact_failure_counts": dict(sorted(adapter.mixer.exact_failure_counts.items())),
                        "exact_failure_examples": adapter.mixer.exact_failure_examples,
                        "learned_gate_values_by_layer": {
                            str(layer): float(torch.sigmoid(adapter.mixer.gate_logits[pos]).detach().cpu().item())
                            for pos, layer in enumerate(adapter.mixer.inject_layers)
                        },
                    },
                )
            print(f"Finished variant={variant}")
        finally:
            if adapter is not None:
                adapter.detach()
            restore_logging(handle, old_stdout, old_stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary_dir = output_root / "debug" / "summary" if bool(args.debug) else output_root / "summary"
    write_summary(summary_dir, all_split_rows, all_count_rows, bool(args.no_plots))
    print(f"Summary written to {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

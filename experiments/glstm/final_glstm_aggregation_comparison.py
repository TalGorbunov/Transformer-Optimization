#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
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

from experiments.glstm import layerwise_frame_message_glstm as base
from models.model import find_subsequence, image_token_groups


EXPERIMENT_NAME = "final_glstm_aggregation_comparison"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_LAYERS = (14, 15, 16, 17)
COUNT_VALUES = tuple(range(13))
DATASET_SEED = 24680
PNG_DPI = 160
MISSING = math.nan

TRAIN_SPLIT = "train"
VAL_SPLIT = "iid_val"
IID_TEST_SPLIT = "iid_test"
SEEN_COUNT_SPLIT = "length_extrapolation_seen_counts"
HIGH_AGG_SPLIT = "high_aggregation_extrapolation"
PAIRED_SPLIT = "paired_neutral_extension_test"
EVAL_SPLITS = (IID_TEST_SPLIT, SEEN_COUNT_SPLIT, HIGH_AGG_SPLIT, PAIRED_SPLIT)
MAIN_METRIC_SPLITS = (IID_TEST_SPLIT, SEEN_COUNT_SPLIT, HIGH_AGG_SPLIT)

LORA_BASELINE = "lora_baseline"
SUM_FINAL_ONLY_PERSISTENT = "sum_final_only_persistent"
GLSTM_FINAL_ONLY_PERSISTENT = "glstm_final_only_persistent"
VARIANTS = (LORA_BASELINE, SUM_FINAL_ONLY_PERSISTENT, GLSTM_FINAL_ONLY_PERSISTENT)
MEMORY_VARIANTS = (SUM_FINAL_ONLY_PERSISTENT, GLSTM_FINAL_ONLY_PERSISTENT)
VARIANT_ALIASES = {
    "all": "all",
    "lora": LORA_BASELINE,
    "lora_baseline": LORA_BASELINE,
    "sum": SUM_FINAL_ONLY_PERSISTENT,
    "sum_final": SUM_FINAL_ONLY_PERSISTENT,
    "sum_final_only_persistent": SUM_FINAL_ONLY_PERSISTENT,
    "glstm": GLSTM_FINAL_ONLY_PERSISTENT,
    "glstm_final": GLSTM_FINAL_ONLY_PERSISTENT,
    "glstm_final_only_persistent": GLSTM_FINAL_ONLY_PERSISTENT,
}

FrameMemoryExample = base.FrameMemoryExample
FrameMemoryBatch = base.FrameMemoryBatch
ExperimentAdapter = base.ExperimentAdapter


class BatchedTinyVisualQwen(base.visual_base.TinyVisualQwen):
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
            cursor = 0
            for batch_idx in range(int(input_ids.shape[0])):
                positions = (input_ids[batch_idx] == self.image_token_id).nonzero(as_tuple=True)[0]
                image_count = int(positions.numel()) // 4
                if int(positions.numel()) != image_count * 4:
                    raise RuntimeError("tiny visual token count is not divisible by four")
                if image_count:
                    chunk = visual[cursor : cursor + image_count]
                    hidden_states[batch_idx, positions, :] = (
                        hidden_states[batch_idx, positions, :] + chunk.repeat_interleave(4, dim=0)
                    )
                    cursor += image_count
            if cursor != int(visual.shape[0]):
                raise RuntimeError(f"unused tiny visual rows: used {cursor}, had {visual.shape[0]}")
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final-only gLSTM aggregation comparison.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--source-dataset-root", type=Path, default=base.DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--fallback-source-dataset-root", type=Path, default=base.DEFAULT_FALLBACK_SOURCE_DATASET_ROOT)
    parser.add_argument("--source-split", default="all_uniform")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=DATASET_SEED)
    parser.add_argument("--force-regenerate-dataset", action="store_true", default=False)

    parser.add_argument("--train-per-cell", type=int, default=20)
    parser.add_argument("--val-per-cell", type=int, default=5)
    parser.add_argument("--iid-test-per-cell", type=int, default=10)
    parser.add_argument("--seen-count-test-per-cell", type=int, default=10)
    parser.add_argument("--high-aggregation-per-cell", type=int, default=20)
    parser.add_argument("--paired-families", type=int, default=24)

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--min-train-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)

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
    parser.add_argument("--reconstruction-tol", type=float, default=5e-3)
    parser.add_argument("--fail-on-reconstruction-error", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=12)
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

    parser.add_argument("--prepare-dataset-only", action="store_true", default=False)
    parser.add_argument("--run-correctness-tests", action="store_true", default=False)
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--submit-slurm", action="store_true", default=False)
    parser.add_argument("--smoke-test", action="store_true", default=False)
    parser.add_argument("--real-qwen-smoke", action="store_true", default=False)
    parser.add_argument("--tiny-debug-model", action="store_true", default=False)
    parser.add_argument("--tiny-num-layers", type=int, default=18)
    parser.add_argument("--tiny-hidden-size", type=int, default=64)
    parser.add_argument("--tiny-num-heads", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--no-aggregate-after-run", action="store_true", default=False)
    return parser.parse_args()


def split_tokens(raw_values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for raw in raw_values:
        out.extend(part.strip() for part in re.split(r"[,\s:]+", str(raw)) if part.strip())
    return out


def parse_variants(args: argparse.Namespace) -> List[str]:
    raw_values: List[Any] = []
    if str(args.variant).strip():
        raw_values.append(str(args.variant).strip())
    raw_values.extend(args.variants)
    variants: List[str] = []
    for token in split_tokens(raw_values):
        key = token.lower()
        if key == "all":
            variants.extend(VARIANTS)
            continue
        if key not in VARIANT_ALIASES:
            raise ValueError(f"Unknown variant {token!r}; valid values are {sorted(VARIANT_ALIASES)}")
        variants.append(str(VARIANT_ALIASES[key]))
    out = list(dict.fromkeys(variants))
    if not out:
        raise ValueError("No variants selected")
    return out


def parse_layers(args: argparse.Namespace) -> List[int]:
    layers = base.parse_int_tokens(args.layers)
    if layers != list(DEFAULT_LAYERS):
        print(f"Warning: requested layers {layers}; experiment spec uses {list(DEFAULT_LAYERS)}")
    return layers


def ensure_candidate_range(args: argparse.Namespace) -> List[int]:
    if int(args.candidate_min) != 0 or int(args.candidate_max) != 12:
        raise ValueError("Candidate answers must be exactly 0..12")
    return list(COUNT_VALUES)


def safe_name(value: Any) -> str:
    return base.safe_name(value)


def finite_float(value: Any) -> Optional[float]:
    return base.finite_float(value)


def finite_mean(values: Iterable[Any], default: float = MISSING) -> float:
    return base.finite_mean(values, default=default)


def write_json(path: Path, payload: Any) -> None:
    base.write_json(path, payload)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    base.write_jsonl(path, rows)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return base.read_jsonl(path)


def write_csv_dynamic(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str] = ()) -> None:
    base.write_csv_dynamic(path, rows, leading=leading)


def resolve_frame_path(path_text: str) -> Path:
    return base.resolve_frame_path(path_text)


def frame_identity(ref: Dict[str, Any]) -> str:
    return f"{ref.get('source_sample_id')}:{int(ref.get('source_frame_index', -1))}:{ref.get('frame_path')}"


def choose_unique_refs(
    rng: random.Random,
    pool: Sequence[Dict[str, Any]],
    n: int,
    forbidden: Optional[set[str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    if int(n) <= 0:
        return [], 0
    forbidden = set() if forbidden is None else set(forbidden)
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for ref in pool:
        ident = frame_identity(ref)
        if ident in forbidden or ident in seen:
            continue
        seen.add(ident)
        unique.append(dict(ref))
    rng.shuffle(unique)
    if len(unique) >= int(n):
        return [dict(ref) for ref in unique[: int(n)]], 0
    refs = [dict(ref) for ref in unique]
    fallback = [dict(ref) for ref in pool]
    if not fallback:
        raise RuntimeError("Cannot choose refs from an empty pool")
    duplicate_fallbacks = 0
    while len(refs) < int(n):
        refs.append(dict(rng.choice(fallback)))
        duplicate_fallbacks += 1
    return refs, duplicate_fallbacks


def output_lengths() -> List[int]:
    return list(range(4, 13))


def source_length_available(root: Optional[Path], length: int, source_split: str) -> bool:
    if root is None:
        return False
    return (Path(root) / f"seq_len_{int(length)}" / str(source_split)).is_dir()


def scan_or_synthesize_source_pools(
    args: argparse.Namespace,
    lengths: Sequence[int],
) -> Tuple[Dict[int, Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]], Dict[str, Any], List[int]]:
    source_root = Path(args.source_dataset_root)
    fallback_root = Path(args.fallback_source_dataset_root) if args.fallback_source_dataset_root is not None else None
    available = [
        int(length)
        for length in sorted({int(x) for x in lengths})
        if source_length_available(source_root, int(length), str(args.source_split))
        or source_length_available(fallback_root, int(length), str(args.source_split))
    ]
    if not available:
        raise RuntimeError(f"No requested source lengths are available for {lengths}")
    pools, source_manifest = base.scan_source_frame_pools(
        source_root,
        fallback_root,
        str(args.source_split),
        int(args.dataset_seed),
        available,
    )
    synthesized: List[int] = []
    merged: Optional[Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]] = None
    for length in sorted(set(int(x) for x in lengths) - set(available)):
        if merged is None:
            merged = base.merged_frame_pools_for_partition(pools, sorted(pools))
        pools[int(length)] = merged
        synthesized.append(int(length))
    for length in sorted({int(x) for x in lengths}):
        required = {"test"}
        if 4 <= int(length) <= 8:
            required |= {"train", "val"}
        has_required = all(base.valid_pairs_for_lengths(pools, [int(length)], part) for part in required)
        if not has_required:
            if merged is None:
                merged = base.merged_frame_pools_for_partition(pools, sorted(pools))
            pools[int(length)] = merged
            if int(length) not in synthesized:
                synthesized.append(int(length))
    source_manifest["synthesized_output_lengths"] = sorted(synthesized)
    source_manifest["synthesized_length_rule"] = (
        "Missing or partition-empty output lengths are assembled from merged frame pools "
        "within each source partition; no frame crosses source train/val/test partitions."
    )
    return pools, source_manifest, sorted(synthesized)


def dataset_config(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(args.smoke_test) or bool(args.real_qwen_smoke):
        train_per = min(int(args.train_per_cell), 1)
        val_per = min(int(args.val_per_cell), 1)
        iid_per = min(int(args.iid_test_per_cell), 1)
        seen_per = min(int(args.seen_count_test_per_cell), 1)
        high_per = min(int(args.high_aggregation_per_cell), 1)
        families = min(int(args.paired_families), 9)
    else:
        train_per = int(args.train_per_cell)
        val_per = int(args.val_per_cell)
        iid_per = int(args.iid_test_per_cell)
        seen_per = int(args.seen_count_test_per_cell)
        high_per = int(args.high_aggregation_per_cell)
        families = int(args.paired_families)
    if min(train_per, val_per, iid_per, seen_per, high_per, families) <= 0:
        raise ValueError("Dataset sizes must be positive")
    return {
        "dataset_seed": int(args.dataset_seed),
        "source_dataset_root": os.fspath(Path(args.source_dataset_root).resolve()),
        "fallback_source_dataset_root": os.fspath(Path(args.fallback_source_dataset_root).resolve())
        if args.fallback_source_dataset_root is not None
        else None,
        "source_split": str(args.source_split),
        "candidate_counts": list(COUNT_VALUES),
        "neutral_rule": "queried character absent and queried room empty",
        "evidence_positions_randomized": True,
        "splits": {
            TRAIN_SPLIT: {
                "lengths": [4, 5, 6, 7, 8],
                "counts_by_length": {str(length): list(range(length + 1)) for length in range(4, 9)},
                "examples_per_cell": train_per,
                "source_partition": "train",
            },
            VAL_SPLIT: {
                "lengths": [4, 5, 6, 7, 8],
                "counts_by_length": {str(length): list(range(length + 1)) for length in range(4, 9)},
                "examples_per_cell": val_per,
                "source_partition": "val",
            },
            IID_TEST_SPLIT: {
                "lengths": [4, 5, 6, 7, 8],
                "counts_by_length": {str(length): list(range(length + 1)) for length in range(4, 9)},
                "examples_per_cell": iid_per,
                "source_partition": "test",
            },
            SEEN_COUNT_SPLIT: {
                "lengths": [9, 10, 11, 12],
                "counts_by_length": {str(length): list(range(9)) for length in range(9, 13)},
                "examples_per_cell": seen_per,
                "source_partition": "test",
            },
            HIGH_AGG_SPLIT: {
                "lengths": [9, 10, 11, 12],
                "counts_by_length": {str(length): list(range(9, length + 1)) for length in range(9, 13)},
                "examples_per_cell": high_per,
                "source_partition": "test",
            },
            PAIRED_SPLIT: {
                "lengths": [8, 12, 16],
                "families": families,
                "versions_per_family": 3,
                "source_partition": "test",
            },
        },
    }


def expected_split_size(split_cfg: Dict[str, Any]) -> int:
    if "families" in split_cfg:
        return int(split_cfg["families"]) * int(split_cfg["versions_per_family"])
    return sum(
        len(split_cfg["counts_by_length"][str(int(length))]) * int(split_cfg["examples_per_cell"])
        for length in split_cfg["lengths"]
    )


def generate_balanced_split(
    *,
    split: str,
    lengths: Sequence[int],
    counts_by_length: Dict[int, Sequence[int]],
    examples_per_cell: int,
    templates: Sequence[str],
    pools: Dict[int, Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]],
    source_partition_name: str,
    seed: int,
    duplicate_counter: Dict[str, int],
) -> List[FrameMemoryExample]:
    rng = random.Random(int(seed))
    examples: List[FrameMemoryExample] = []
    for length in [int(x) for x in lengths]:
        valid_pairs = sorted(
            pair
            for pair, pair_pool in pools[int(length)][str(source_partition_name)].items()
            if pair_pool.get("evidence") and pair_pool.get("neutral")
        )
        if not valid_pairs:
            raise RuntimeError(f"{split}: no valid pairs for length={length} partition={source_partition_name}")
        schedule = [
            int(count)
            for count in counts_by_length[int(length)]
            for _ in range(int(examples_per_cell))
        ]
        rng.shuffle(schedule)
        for local_idx, gold_count in enumerate(schedule):
            if not (0 <= int(gold_count) <= int(length)):
                raise RuntimeError(f"{split}: invalid count {gold_count} for length {length}")
            character, room = rng.choice(valid_pairs)
            pair_pool = pools[int(length)][str(source_partition_name)][(character, room)]
            evidence_positions = tuple(sorted(rng.sample(range(int(length)), int(gold_count))))
            evidence_set = set(evidence_positions)
            evidence_refs, dup_e = choose_unique_refs(rng, pair_pool["evidence"], int(gold_count))
            neutral_refs, dup_n = choose_unique_refs(rng, pair_pool["neutral"], int(length) - int(gold_count))
            duplicate_counter[split] += int(dup_e + dup_n)
            rng.shuffle(evidence_refs)
            rng.shuffle(neutral_refs)
            ordered_refs: List[Dict[str, Any]] = []
            e_cursor = 0
            n_cursor = 0
            for frame_idx in range(int(length)):
                if frame_idx in evidence_set:
                    ref = dict(evidence_refs[e_cursor])
                    e_cursor += 1
                    ref["mechanism_role"] = "evidence"
                else:
                    ref = dict(neutral_refs[n_cursor])
                    n_cursor += 1
                    ref["mechanism_role"] = "neutral"
                ref["output_frame_index"] = int(frame_idx)
                ordered_refs.append(ref)
            template_id = rng.choice(tuple(templates))
            question = base.question_for_template(template_id, character, room)
            sample_hash = hashlib.sha1(
                json.dumps(
                    {
                        "split": split,
                        "length": int(length),
                        "idx": int(local_idx),
                        "count": int(gold_count),
                        "refs": [(r["frame_path"], r["source_frame_index"]) for r in ordered_refs],
                        "pair": [character, room],
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:10]
            examples.append(
                FrameMemoryExample(
                    example_id=f"{split}_len{int(length)}_count{int(gold_count)}_{local_idx:05d}_{sample_hash}",
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


def _paired_metadata_from_example(example: FrameMemoryExample) -> Dict[str, Any]:
    if not example.source_dataset_info:
        return {}
    first = dict(example.source_dataset_info[0])
    return dict(first.get("paired_metadata", {}))


def example_to_json(example: FrameMemoryExample) -> Dict[str, Any]:
    row = base.example_to_json(example)
    paired_meta = _paired_metadata_from_example(example)
    if paired_meta:
        row["paired_metadata"] = paired_meta
    return row


def example_from_json(row: Dict[str, Any]) -> FrameMemoryExample:
    example = base.example_from_json(row)
    paired_meta = dict(row.get("paired_metadata", {}))
    if paired_meta:
        enriched = []
        for item in example.source_dataset_info:
            ref = dict(item)
            ref.setdefault("paired_metadata", paired_meta)
            enriched.append(ref)
        example = FrameMemoryExample(
            example_id=example.example_id,
            split=example.split,
            frame_paths=example.frame_paths,
            num_frames=example.num_frames,
            gold_count=example.gold_count,
            evidence_frame_indices=example.evidence_frame_indices,
            question=example.question,
            answer=example.answer,
            queried_character=example.queried_character,
            queried_room=example.queried_room,
            template_id=example.template_id,
            composition_key=example.composition_key,
            source_dataset_info=tuple(enriched),
        )
    return example


def build_paired_neutral_extension(
    *,
    templates: Sequence[str],
    pools: Dict[int, Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]],
    seed: int,
    num_families: int,
    duplicate_counter: Dict[str, int],
) -> List[FrameMemoryExample]:
    rng = random.Random(int(seed))
    base_length = 8
    version_lengths = (8, 12, 16)
    partition = "test"
    valid_pairs = sorted(
        pair
        for pair, pair_pool in pools[base_length][partition].items()
        if pair_pool.get("evidence") and pair_pool.get("neutral")
    )
    if not valid_pairs:
        raise RuntimeError("paired neutral extension: no valid test pairs at length 8")
    counts = list(range(9)) * max(1, math.ceil(int(num_families) / 9))
    rng.shuffle(counts)
    examples: List[FrameMemoryExample] = []
    for family_idx, gold_count in enumerate(counts[: int(num_families)]):
        character, room = rng.choice(valid_pairs)
        pair_pool = pools[base_length][partition][(character, room)]
        evidence_positions_base = tuple(sorted(rng.sample(range(base_length), int(gold_count))))
        evidence_set_base = set(evidence_positions_base)
        evidence_refs, dup_e = choose_unique_refs(rng, pair_pool["evidence"], int(gold_count))
        neutral_refs, dup_n = choose_unique_refs(rng, pair_pool["neutral"], base_length - int(gold_count))
        duplicate_counter[PAIRED_SPLIT] += int(dup_e + dup_n)
        rng.shuffle(evidence_refs)
        rng.shuffle(neutral_refs)
        base_order: List[Dict[str, Any]] = []
        e_cursor = 0
        n_cursor = 0
        for pos in range(base_length):
            if pos in evidence_set_base:
                ref = dict(evidence_refs[e_cursor])
                e_cursor += 1
                ref["mechanism_role"] = "evidence"
            else:
                ref = dict(neutral_refs[n_cursor])
                n_cursor += 1
                ref["mechanism_role"] = "base_neutral"
            ref["base_order_index"] = int(pos)
            base_order.append(ref)
        family_id = f"family_{family_idx:03d}"
        base_sample_id = f"paired_family_{family_idx:03d}_base_len8_count{int(gold_count)}"
        template_id = rng.choice(tuple(templates))
        question = base.question_for_template(template_id, character, room)
        evidence_identities = [frame_identity(ref) for ref in base_order if ref.get("mechanism_role") == "evidence"]
        used = {frame_identity(ref) for ref in base_order}
        for version_length in version_lengths:
            extra_count = int(version_length) - base_length
            added_refs, dup_added = choose_unique_refs(rng, pair_pool["neutral"], extra_count, forbidden=used)
            duplicate_counter[PAIRED_SPLIT] += int(dup_added)
            rng.shuffle(added_refs)
            insert_positions = set(rng.sample(range(int(version_length)), extra_count)) if extra_count else set()
            final_refs: List[Dict[str, Any]] = []
            base_cursor = 0
            extra_cursor = 0
            for out_pos in range(int(version_length)):
                if out_pos in insert_positions:
                    ref = dict(added_refs[extra_cursor])
                    extra_cursor += 1
                    ref["mechanism_role"] = "added_neutral"
                    ref["base_order_index"] = None
                else:
                    ref = dict(base_order[base_cursor])
                    base_cursor += 1
                ref["output_frame_index"] = int(out_pos)
                final_refs.append(ref)
            final_evidence_positions = tuple(
                int(i) for i, ref in enumerate(final_refs) if ref.get("mechanism_role") == "evidence"
            )
            new_neutral_ids = [frame_identity(ref) for ref in final_refs if ref.get("mechanism_role") == "added_neutral"]
            paired_meta = {
                "family_id": family_id,
                "base_sample_id": base_sample_id,
                "version_length": int(version_length),
                "original_evidence_frame_identities": list(evidence_identities),
                "new_neutral_frame_identities": list(new_neutral_ids),
                "final_evidence_positions": list(final_evidence_positions),
                "final_frame_ordering": [
                    {
                        "position": int(i),
                        "identity": frame_identity(ref),
                        "role": str(ref.get("mechanism_role")),
                        "base_order_index": ref.get("base_order_index"),
                    }
                    for i, ref in enumerate(final_refs)
                ],
            }
            enriched_refs: List[Dict[str, Any]] = []
            for ref in final_refs:
                ref = dict(ref)
                ref["paired_metadata"] = paired_meta
                enriched_refs.append(ref)
            examples.append(
                FrameMemoryExample(
                    example_id=f"{PAIRED_SPLIT}_{family_id}_len{int(version_length)}_count{int(gold_count)}",
                    split=PAIRED_SPLIT,
                    frame_paths=tuple(str(ref["frame_path"]) for ref in enriched_refs),
                    num_frames=int(version_length),
                    gold_count=int(gold_count),
                    evidence_frame_indices=final_evidence_positions,
                    question=question,
                    answer=str(int(gold_count)),
                    queried_character=character,
                    queried_room=room,
                    template_id=template_id,
                    composition_key=f"{character}|{room}",
                    source_dataset_info=tuple(enriched_refs),
                )
            )
    return examples


def assert_dataset(config: Dict[str, Any], examples: Dict[str, List[FrameMemoryExample]]) -> None:
    all_ids: set[str] = set()
    for split, split_cfg in config["splits"].items():
        rows = examples.get(split, [])
        expected_n = expected_split_size(split_cfg)
        if len(rows) != expected_n:
            raise RuntimeError(f"{split}: expected {expected_n} examples, found {len(rows)}")
        ids = {row.example_id for row in rows}
        if len(ids) != len(rows):
            raise RuntimeError(f"{split}: duplicate sample IDs")
        if ids & all_ids:
            raise RuntimeError(f"{split}: sample IDs overlap another split")
        all_ids |= ids
        expected_lengths = set(int(x) for x in split_cfg["lengths"])
        for row in rows:
            if int(row.num_frames) not in expected_lengths:
                raise RuntimeError(f"{split}: unexpected length {row.num_frames}")
            if not (0 <= int(row.gold_count) <= 12):
                raise RuntimeError(f"{split}: gold count outside 0..12: {row.gold_count}")
            if int(row.gold_count) > int(row.num_frames):
                raise RuntimeError(f"{split}: invalid count {row.gold_count} for length {row.num_frames}")
            if len(row.frame_paths) != int(row.num_frames):
                raise RuntimeError(f"{split}: frame path count mismatch for {row.example_id}")
            if len(set(row.evidence_frame_indices)) != int(row.gold_count):
                raise RuntimeError(f"{split}: evidence index/count mismatch for {row.example_id}")
            for frame_path in row.frame_paths:
                resolved = resolve_frame_path(frame_path)
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
        if split != PAIRED_SPLIT:
            for length in split_cfg["lengths"]:
                for count in split_cfg["counts_by_length"][str(int(length))]:
                    n = sum(int(row.num_frames) == int(length) and int(row.gold_count) == int(count) for row in rows)
                    if n != int(split_cfg["examples_per_cell"]):
                        raise RuntimeError(
                            f"{split}: expected {split_cfg['examples_per_cell']} examples for "
                            f"length={length} count={count}, found {n}"
                        )
    assert_paired_extension(examples[PAIRED_SPLIT], int(config["splits"][PAIRED_SPLIT]["families"]))


def assert_paired_extension(rows: Sequence[FrameMemoryExample], expected_families: int) -> None:
    by_family: Dict[str, List[FrameMemoryExample]] = defaultdict(list)
    for row in rows:
        meta = _paired_metadata_from_example(row)
        family_id = str(meta.get("family_id", ""))
        if not family_id:
            raise RuntimeError(f"{row.example_id}: missing paired family_id")
        by_family[family_id].append(row)
    if len(by_family) != int(expected_families):
        raise RuntimeError(f"paired set expected {expected_families} families, found {len(by_family)}")
    seen_counts = set()
    for family_id, family_rows in by_family.items():
        lengths = sorted(int(row.num_frames) for row in family_rows)
        if lengths != [8, 12, 16]:
            raise RuntimeError(f"{family_id}: expected lengths 8/12/16, found {lengths}")
        counts = {int(row.gold_count) for row in family_rows}
        if len(counts) != 1:
            raise RuntimeError(f"{family_id}: gold count changed across versions")
        seen_counts |= counts
        evidence_identities = None
        for row in family_rows:
            meta = _paired_metadata_from_example(row)
            current_ids = tuple(meta.get("original_evidence_frame_identities", []))
            if evidence_identities is None:
                evidence_identities = current_ids
            elif current_ids != evidence_identities:
                raise RuntimeError(f"{family_id}: evidence identity changed")
            if tuple(int(x) for x in meta.get("final_evidence_positions", [])) != tuple(row.evidence_frame_indices):
                raise RuntimeError(f"{family_id}: evidence position metadata mismatch")
    if int(expected_families) >= 9 and not set(range(9)).issubset(seen_counts):
        raise RuntimeError(f"paired set must cover counts 0..8, found {sorted(seen_counts)}")


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
        pools, source_manifest, synthesized_lengths = scan_or_synthesize_source_pools(args, output_lengths())
        duplicate_counter: Dict[str, int] = defaultdict(int)
        generated: Dict[str, List[FrameMemoryExample]] = {}
        seed_offsets = {
            TRAIN_SPLIT: 11,
            VAL_SPLIT: 23,
            IID_TEST_SPLIT: 59,
            SEEN_COUNT_SPLIT: 83,
            HIGH_AGG_SPLIT: 107,
            PAIRED_SPLIT: 131,
        }
        for split, split_cfg in config["splits"].items():
            if split == PAIRED_SPLIT:
                generated[split] = build_paired_neutral_extension(
                    templates=base.visual_base.TRAIN_TEMPLATES,
                    pools=pools,
                    seed=int(args.dataset_seed) + seed_offsets[split],
                    num_families=int(split_cfg["families"]),
                    duplicate_counter=duplicate_counter,
                )
            else:
                generated[split] = generate_balanced_split(
                    split=split,
                    lengths=[int(x) for x in split_cfg["lengths"]],
                    counts_by_length={int(k): [int(x) for x in v] for k, v in split_cfg["counts_by_length"].items()},
                    examples_per_cell=int(split_cfg["examples_per_cell"]),
                    templates=base.visual_base.TRAIN_TEMPLATES,
                    pools=pools,
                    source_partition_name=str(split_cfg["source_partition"]),
                    seed=int(args.dataset_seed) + seed_offsets[split],
                    duplicate_counter=duplicate_counter,
                )
            write_jsonl(split_paths[split], [example_to_json(example) for example in generated[split]])
        manifest = {
            "dataset_hash": digest,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config,
            "source_manifest": source_manifest,
            "synthesized_output_lengths_from_merged_frame_pools": synthesized_lengths,
            "duplicate_frame_fallbacks_by_split": dict(sorted(duplicate_counter.items())),
            "splits": {
                split: {
                    "path": os.fspath(path),
                    "n": len(generated[split]),
                    "expected_n": expected_split_size(config["splits"][split]),
                    "sample_ids": [example.example_id for example in generated[split]],
                    "length_count_histogram": {
                        f"{length}|{count}": sum(
                            int(ex.num_frames) == int(length) and int(ex.gold_count) == int(count)
                            for ex in generated[split]
                        )
                        for length in config["splits"][split].get("lengths", [])
                        for count in (
                            config["splits"][split].get("counts_by_length", {}).get(str(int(length)), [])
                            if split != PAIRED_SPLIT
                            else range(13)
                        )
                    },
                }
                for split, path in split_paths.items()
            },
        }
        write_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = {split: [example_from_json(row) for row in read_jsonl(path)] for split, path in split_paths.items()}
    assert_dataset(config, examples)
    return dataset_dir, examples, manifest


def _pad_tensor_rows(rows: Sequence[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(int(row.shape[1]) for row in rows)
    padded: List[torch.Tensor] = []
    for row in rows:
        pad = max_len - int(row.shape[1])
        if pad:
            fill = torch.full((1, pad), int(pad_value), dtype=row.dtype)
            row = torch.cat([row, fill], dim=1)
        padded.append(row)
    return torch.cat(padded, dim=0)


def _prepare_one(
    example: FrameMemoryExample,
    sample_index: int,
    processor: Any,
    answer_ids: Optional[Dict[int, Tuple[int, ...]]],
    answer_count_override: Optional[int],
) -> Dict[str, Any]:
    expected_frames = int(example.num_frames)
    frames: List[Image.Image] = []
    try:
        for path_text in example.frame_paths:
            with Image.open(resolve_frame_path(path_text)) as image:
                frames.append(image.convert("RGB"))
        if len(frames) != expected_frames:
            raise AssertionError(f"{example.example_id}: loaded {len(frames)} frames, expected {expected_frames}")
        prompt_text = base.build_prompt(example.question, example.num_frames)
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
    input_ids = raw_inputs["input_ids"]
    attention_mask = raw_inputs["attention_mask"]
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
    character_positions = base.token_positions_for_prompt_span(
        tokenizer,
        prompt_text,
        int(prompt_start),
        character_start,
        character_start + len(example.queried_character),
    )
    room_positions = base.token_positions_for_prompt_span(
        tokenizer,
        prompt_text,
        int(prompt_start),
        room_start,
        room_start + len(example.queried_room),
    )
    carriers = sorted({int(pos) for pos in [*character_positions, *room_positions] if 0 <= int(pos) < prompt_last})
    carrier_identities = ["character" if int(pos) in set(character_positions) else "room" for pos in carriers]
    groups = image_token_groups(input_ids[0].detach().cpu(), expected_frames, processor=processor)
    groups = [[int(position) for position in group] for group in groups]
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(tokenizer, "image_token_id", None)
    if image_token_id is None and hasattr(tokenizer, "convert_tokens_to_ids"):
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    flattened = [position for group in groups for position in group]
    errors: List[str] = []
    if not carriers:
        errors.append("room/character carrier token positions not found")
    if len(groups) != expected_frames or any(not group for group in groups):
        errors.append(f"expected {expected_frames} non-empty visual token spans, found {[len(g) for g in groups]}")
    if image_token_id is None or any(int(input_ids[0, pos].item()) != int(image_token_id) for pos in flattened):
        errors.append("frame groups include non-image-pad tokens")
    if prompt_last in carriers:
        errors.append("final prompt token selected as carrier")
    if errors:
        raise AssertionError(f"{example.example_id}: {'; '.join(errors)}")

    loss_positions: Optional[List[int]] = None
    loss_targets: Optional[List[int]] = None
    if answer_ids is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise RuntimeError("Tokenizer must expose eos_token_id")
        answer_count = int(example.gold_count if answer_count_override is None else answer_count_override)
        targets = [*answer_ids[int(answer_count)], int(eos_token_id)]
        positions = list(range(prompt_last, prompt_last + len(targets)))
        suffix = torch.tensor([targets], dtype=input_ids.dtype)
        raw_inputs["input_ids"] = torch.cat([input_ids, suffix], dim=1)
        raw_inputs["attention_mask"] = torch.cat(
            [attention_mask, torch.ones((1, len(targets)), dtype=attention_mask.dtype)],
            dim=1,
        )
        loss_positions = positions
        loss_targets = targets
    visual_keys = [key for key in base.VISUAL_INPUT_KEYS if key in raw_inputs and torch.is_tensor(raw_inputs[key])]
    if not any(key.startswith("pixel_values") for key in visual_keys):
        raise AssertionError(f"{example.example_id}: no visual tensor found")
    return {
        "raw_inputs": raw_inputs,
        "prompt_last": prompt_last,
        "loss_positions": loss_positions,
        "loss_targets": loss_targets,
        "carriers": carriers,
        "carrier_identities": carrier_identities,
        "groups": groups,
        "evidence_frame_indices": list(example.evidence_frame_indices),
        "sample_id": example.example_id,
        "sample_index": int(sample_index),
        "gold_count": int(example.gold_count),
        "num_frames": int(example.num_frames),
        "visual_keys": visual_keys,
    }


def prepare_batch(
    *,
    examples: Sequence[FrameMemoryExample],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
    answer_ids: Optional[Dict[int, Tuple[int, ...]]] = None,
    answer_count_override: Optional[int] = None,
) -> FrameMemoryBatch:
    if len(examples) != len(sample_indices):
        raise ValueError("examples and sample_indices length mismatch")
    prepared = [
        _prepare_one(example, int(sample_idx), processor, answer_ids, answer_count_override)
        for example, sample_idx in zip(examples, sample_indices)
    ]
    tokenizer = processor.tokenizer
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    input_ids = _pad_tensor_rows([item["raw_inputs"]["input_ids"] for item in prepared], int(pad_id))
    attention_mask = _pad_tensor_rows([item["raw_inputs"]["attention_mask"] for item in prepared], 0)
    inputs: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    for key in sorted({key for item in prepared for key in item["raw_inputs"].keys()}):
        if key in inputs or key in {"input_ids", "attention_mask"}:
            continue
        values = [item["raw_inputs"].get(key) for item in prepared if key in item["raw_inputs"]]
        if not values or not all(torch.is_tensor(value) for value in values):
            continue
        if key in base.VISUAL_INPUT_KEYS:
            inputs[key] = torch.cat(values, dim=0)
        elif all(tuple(value.shape) == tuple(values[0].shape) for value in values):
            inputs[key] = torch.cat(values, dim=0) if int(values[0].shape[0]) == 1 else torch.stack(values, dim=0)
    max_loss_len = max((len(item["loss_targets"] or []) for item in prepared), default=0)
    loss_positions: Optional[torch.Tensor] = None
    loss_targets: Optional[torch.Tensor] = None
    if answer_ids is not None:
        pos_rows: List[List[int]] = []
        target_rows: List[List[int]] = []
        for item in prepared:
            positions = list(item["loss_positions"] or [])
            targets = list(item["loss_targets"] or [])
            pad = max_loss_len - len(targets)
            pos_rows.append(positions + [-1] * pad)
            target_rows.append(targets + [-100] * pad)
        loss_positions = torch.tensor(pos_rows, dtype=torch.long)
        loss_targets = torch.tensor(target_rows, dtype=torch.long)
    max_frames = max(int(item["num_frames"]) for item in prepared)
    frame_valid_mask = torch.zeros((len(prepared), max_frames), dtype=torch.bool)
    for b, item in enumerate(prepared):
        frame_valid_mask[b, : int(item["num_frames"])] = True
    visual_keys = sorted({key for item in prepared for key in item["visual_keys"]})
    batch = FrameMemoryBatch(
        inputs=base.move_inputs_to_device(inputs, device),
        prompt_last_indices=torch.tensor([item["prompt_last"] for item in prepared], device=device, dtype=torch.long),
        gold_counts=torch.tensor([item["gold_count"] for item in prepared], device=device, dtype=torch.long),
        loss_positions=loss_positions.to(device) if loss_positions is not None else None,
        loss_targets=loss_targets.to(device) if loss_targets is not None else None,
        carrier_positions=[list(item["carriers"]) for item in prepared],
        carrier_identities=[list(item["carrier_identities"]) for item in prepared],
        frame_groups=[list(item["groups"]) for item in prepared],
        frame_valid_mask=frame_valid_mask.to(device),
        evidence_frame_indices=[list(item["evidence_frame_indices"]) for item in prepared],
        sample_ids=[str(item["sample_id"]) for item in prepared],
        sample_indices=[int(item["sample_index"]) for item in prepared],
        visual_input_keys=visual_keys,
    )
    if len(batch.sample_ids) != len(examples):
        raise RuntimeError("FrameMemoryBatch metadata does not match batch size")
    return batch


def load_model_and_processor(args: argparse.Namespace, device: str, dtype: torch.dtype) -> Tuple[Any, Any, bool, str]:
    if bool(args.tiny_debug_model):
        processor = base.visual_base.TinyVisualProcessor()
        model = BatchedTinyVisualQwen(
            vocab_size=processor.tokenizer.vocab_size,
            hidden_size=int(args.tiny_hidden_size),
            num_layers=int(args.tiny_num_layers),
            num_heads=int(args.tiny_num_heads),
            image_token_id=processor.image_token_id,
        ).to(device)
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
        return model, processor, False, "batched_tiny_visual_debug_model"
    return base.load_model_and_processor(args, device=device, dtype=dtype)


def answer_sequence_cross_entropy(logits: torch.Tensor, batch: FrameMemoryBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    return base.answer_sequence_cross_entropy(logits, batch)


def candidate_score_batch_from_logits(logits: torch.Tensor, batch: FrameMemoryBatch) -> torch.Tensor:
    if batch.loss_positions is None or batch.loss_targets is None:
        raise RuntimeError("Candidate scoring requires loss targets")
    positions = batch.loss_positions.clamp_min(0)
    batch_idx = torch.arange(int(logits.shape[0]), device=logits.device).unsqueeze(1)
    selected = logits[batch_idx, positions, :].float()
    log_probs = selected.log_softmax(dim=-1)
    targets = batch.loss_targets
    valid = targets.ne(-100)
    safe_targets = targets.clamp_min(0)
    gathered = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (gathered * valid.float()).sum(dim=1).detach().float().cpu()


class FinalAggregationMemory(base.LayerwiseFrameMessageMemory):
    def __init__(self, *, variant: str, **kwargs: Any) -> None:
        super().__init__(variant=variant, **kwargs)
        self.read_reconstruction_tol = 1e-4
        if self.variant == SUM_FINAL_ONLY_PERSISTENT:
            for module in [*self.w_q, *self.w_k, *self.w_v]:
                for param in module.parameters():
                    param.requires_grad_(False)
        elif self.variant == GLSTM_FINAL_ONLY_PERSISTENT:
            for module in self.w_sum:
                for param in module.parameters():
                    param.requires_grad_(False)
        else:
            raise ValueError(f"Unsupported memory variant {variant}")

    def set_context(self, batch: FrameMemoryBatch) -> None:
        super().set_context(batch)
        self._last_stats.update(
            {
                "read_reconstruction_error_by_layer": {},
                "previous_slot_norm_by_layer": {},
                "previous_slot_used_by_layer": {},
                "injected_by_layer": {},
            }
        )

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
            raise RuntimeError("Memory slot shape changed within a forward pass")
        wpos = self.write_pos(int(layer_idx))
        mpos = self.mem_pos(int(layer_idx))
        previous_slots = self._slots
        z = self.message_to_slot[wpos](self.message_norm[wpos](messages.float()))
        candidate = self.slot_norm[wpos](previous_slots + z.float())
        self._slots = torch.where(valid.unsqueeze(-1), candidate, previous_slots)
        slots_for_read, valid_for_read = self._apply_ablation(self._slots, valid, int(layer_idx))
        should_inject = int(layer_idx) == max(self.layers)
        layer_key = str(int(layer_idx))
        self.reconstruction_errors[int(layer_idx)].append(float(recon_error))
        if bool(torch.isfinite(torch.tensor(recon_error))) and float(recon_error) > self.reconstruction_tol and self.message_mode == "exact":
            message = f"Attention message reconstruction error {float(recon_error):.6g} exceeds tolerance {self.reconstruction_tol}"
            self._record_exact_failure(f"layer {layer_idx}: {message}")
            if self.fail_on_reconstruction_error:
                raise RuntimeError(message)
        previous_norm = float(previous_slots.detach().float()[valid_for_read.bool()].norm(dim=-1).mean().cpu().item()) if bool(valid_for_read.any()) else MISSING
        raw_norm = float(messages.detach().float()[valid.bool()].norm(dim=-1).mean().cpu().item()) if bool(valid.any()) else MISSING
        slot_float = self._slots.detach().float()
        slot_norm = float(slot_float[valid_for_read.bool()].norm(dim=-1).mean().cpu().item()) if bool(valid_for_read.any()) else MISSING
        self._last_stats["raw_message_norm_by_layer"][layer_key] = [raw_norm]
        self._last_stats["slot_norm_by_layer"][layer_key] = [slot_norm]
        self._last_stats["gamma_by_layer"][layer_key] = [float(self.gamma[self.layer_to_pos[int(layer_idx)]].detach().cpu().item())]
        self._last_stats["reconstruction_error_by_layer"][layer_key] = [float(recon_error)]
        self._last_stats["message_mode_by_layer"][layer_key] = [mode]
        self._last_stats["previous_slot_norm_by_layer"][layer_key] = [previous_norm]
        self._last_stats["previous_slot_used_by_layer"][layer_key] = [int(int(layer_idx) != min(self.layers))]
        self._last_stats["injected_by_layer"][layer_key] = [int(should_inject)]
        if not should_inject:
            self._last_stats["read_norm_by_layer"][layer_key] = [0.0]
            self._last_stats["injection_norm_by_layer"][layer_key] = [0.0]
            self._last_stats["carrier_state_norm_by_layer"][layer_key] = [
                float(carrier_states.detach().float().norm(dim=-1).mean().cpu().item())
            ]
            self._last_stats["injection_to_carrier_ratio_by_layer"][layer_key] = [0.0]
            self._last_stats["effective_rank_by_layer"][layer_key] = [0.0]
            self._last_stats["slot_cosine_by_layer"][layer_key] = [MISSING]
            self._last_stats["read_reconstruction_error_by_layer"][layer_key] = [0.0]
            self._last_stats["tensor_shapes_by_layer"][layer_key] = [{
                "raw_messages": list(messages.shape),
                "projected_writes_z": list(z.shape),
                "slots": list(self._slots.shape),
                "read_memory": [],
                "per_frame_read_contributions": [],
                "total_read": [],
                "injection": [],
            }]
            self._last_tensors[layer_key] = {
                "raw_messages": messages.detach().float().cpu(),
                "projected_writes_z": z.detach().float().cpu(),
                "slots": self._slots.detach().float().cpu(),
                "valid_frame_mask": valid_for_read.detach().cpu(),
            }
            return h_attn
        if self.variant == SUM_FINAL_ONLY_PERSISTENT:
            per_frame_read = self.w_sum[mpos](slots_for_read).float() * valid_for_read.unsqueeze(-1).float()
            read = per_frame_read.sum(dim=2)
            matrix_shape: List[int] = list(read.shape)
            keys = values = queries = compatibilities = None
        else:
            keys = self.w_k[mpos](slots_for_read).float()
            values = self.w_v[mpos](slots_for_read).float()
            queries = self.w_q[mpos](self.carrier_norm[mpos](carrier_states.float())).float()
            compatibilities = torch.einsum("bcfd,bcd->bcf", keys, queries)
            per_frame_read = values * compatibilities.unsqueeze(-1) * valid_for_read.unsqueeze(-1).float()
            read = per_frame_read.sum(dim=2)
            matrix_shape = [int(slots_for_read.shape[0]), int(slots_for_read.shape[1]), self.memory_dim, self.memory_dim]
        reconstructed_read_error = float((per_frame_read.sum(dim=2) - read).detach().float().norm().cpu().item())
        if reconstructed_read_error > self.read_reconstruction_tol:
            raise RuntimeError(
                f"Per-frame read contributions do not reconstruct total read at layer {layer_idx}: {reconstructed_read_error}"
            )
        injection = self.w_out[mpos](read).float()
        injection = self.gamma[self.layer_to_pos[int(layer_idx)]].float() * injection
        out = h_attn.clone()
        batch_size, seq_len, _hidden = h_attn.shape
        carrier_norms: List[float] = []
        injection_norms: List[float] = []
        ratios: List[float] = []
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
        flat = slot_float[valid_for_read.bool()]
        if int(flat.shape[0]) >= 2:
            norms = flat.norm(dim=-1)
            nonzero = flat[norms > 1e-8]
            if int(nonzero.shape[0]) >= 2:
                normed = F.normalize(nonzero, dim=-1)
                pairwise = normed @ normed.T
                denom = int(nonzero.shape[0]) * (int(nonzero.shape[0]) - 1) / 2
                cosine = float(pairwise.triu(diagonal=1).sum().cpu().item() / max(1.0, denom))
                singular = torch.linalg.svdvals(nonzero)
                rank = float((singular > singular.max().clamp_min(1e-8) * 1e-3).sum().cpu().item())
            else:
                cosine = MISSING
                rank = float(int(nonzero.shape[0]))
        else:
            cosine = MISSING
            rank = float(int(flat.shape[0]))
        self._last_stats["read_norm_by_layer"][layer_key] = [float(read.detach().float().norm(dim=-1).mean().cpu().item())]
        self._last_stats["injection_norm_by_layer"][layer_key] = [finite_mean(injection_norms, default=0.0)]
        self._last_stats["carrier_state_norm_by_layer"][layer_key] = [finite_mean(carrier_norms, default=0.0)]
        self._last_stats["injection_to_carrier_ratio_by_layer"][layer_key] = [finite_mean(ratios, default=0.0)]
        self._last_stats["effective_rank_by_layer"][layer_key] = [rank]
        self._last_stats["slot_cosine_by_layer"][layer_key] = [cosine]
        self._last_stats["read_reconstruction_error_by_layer"][layer_key] = [reconstructed_read_error]
        self._last_stats["tensor_shapes_by_layer"][layer_key] = [{
            "raw_messages": list(messages.shape),
            "projected_writes_z": list(z.shape),
            "slots": list(self._slots.shape),
            "read_memory": matrix_shape,
            "per_frame_read_contributions": list(per_frame_read.shape),
            "total_read": list(read.shape),
            "injection": list(injection.shape),
        }]
        tensor_payload: Dict[str, torch.Tensor] = {
            "raw_messages": messages.detach().float().cpu(),
            "projected_writes_z": z.detach().float().cpu(),
            "slots": self._slots.detach().float().cpu(),
            "per_frame_read_contributions": per_frame_read.detach().float().cpu(),
            "total_read": read.detach().float().cpu(),
            "injection": injection.detach().float().cpu(),
            "valid_frame_mask": valid_for_read.detach().cpu(),
        }
        if keys is not None:
            tensor_payload["keys"] = keys.detach().float().cpu()
        if values is not None:
            tensor_payload["values"] = values.detach().float().cpu()
        if queries is not None:
            tensor_payload["queries"] = queries.detach().float().cpu()
        if compatibilities is not None:
            tensor_payload["compatibilities"] = compatibilities.detach().float().cpu()
        self._last_tensors[layer_key] = tensor_payload
        return out


def make_adapter(args: argparse.Namespace, variant: str, hidden_size: int, layers: Sequence[int]) -> ExperimentAdapter:
    lora = base.AttentionLoRAAdapter(
        inject_layers=layers,
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        dropout=float(args.lora_dropout),
        target_modules=("q_proj", "o_proj"),
        gated=True,
    )
    memory = None
    if variant in MEMORY_VARIANTS:
        memory = FinalAggregationMemory(
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
    return ExperimentAdapter(lora=lora, memory=memory)


def trainable_parameter_summary(model: Any, adapter: ExperimentAdapter, variant: str) -> Dict[str, Any]:
    summary = base.trainable_parameter_summary(model, adapter)
    names = [str(name) for name in summary.get("trainable_parameter_names", [])]
    if variant == LORA_BASELINE:
        if adapter.memory is not None:
            raise RuntimeError("lora_baseline must not have a memory module")
        memory_names = [name for name in names if name.startswith("memory.")]
        if memory_names:
            raise RuntimeError(f"lora_baseline has memory parameters: {memory_names[:20]}")
    elif variant == SUM_FINAL_ONLY_PERSISTENT:
        forbidden = [name for name in names if ".w_q." in name or ".w_k." in name or ".w_v." in name]
        if forbidden:
            raise RuntimeError(f"Sum variant has trainable associative read parameters: {forbidden[:20]}")
    elif variant == GLSTM_FINAL_ONLY_PERSISTENT:
        forbidden = [name for name in names if ".w_sum." in name]
        if forbidden:
            raise RuntimeError(f"gLSTM variant has trainable direct-sum parameters: {forbidden[:20]}")
    return summary


def limited_indices(examples: Sequence[Any], limit: int, seed: int) -> List[int]:
    return base.limited_indices(examples, int(limit), int(seed))


def bucketed_batches(indices: Sequence[int], examples: Sequence[FrameMemoryExample], batch_size: int, seed: int, shuffle: bool) -> List[List[int]]:
    by_len: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        by_len[int(examples[int(idx)].num_frames)].append(int(idx))
    rng = random.Random(int(seed))
    batches: List[List[int]] = []
    for length in sorted(by_len):
        values = list(by_len[length])
        if shuffle:
            rng.shuffle(values)
        for start in range(0, len(values), max(1, int(batch_size))):
            batches.append(values[start : start + max(1, int(batch_size))])
    if shuffle:
        rng.shuffle(batches)
    return batches


def assert_batch_coverage(indices: Sequence[int], batches: Sequence[Sequence[int]]) -> None:
    flattened = [int(idx) for batch in batches for idx in batch]
    if sorted(flattened) != sorted(int(idx) for idx in indices):
        raise RuntimeError("Training batches did not cover exactly the selected indices")
    counts: Dict[int, int] = defaultdict(int)
    for idx in flattened:
        counts[int(idx)] += 1
    duplicated = [idx for idx, count in counts.items() if count != 1]
    if duplicated:
        raise RuntimeError(f"Training batch coverage duplicates/drops indices: {duplicated[:20]}")


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text or "cuda oom" in text


def clear_after_oom(optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def analyze_answer_ids(answer_ids: Dict[int, Tuple[int, ...]], count_values: Sequence[int]) -> Dict[str, Any]:
    lengths = {int(count): len(tuple(answer_ids[int(count)])) for count in count_values}
    return {
        "answer_token_ids": {str(count): list(answer_ids[int(count)]) for count in count_values},
        "answer_token_lengths": {str(k): int(v) for k, v in lengths.items()},
        "all_single_token": all(length == 1 for length in lengths.values()),
    }


@torch.no_grad()
def predict_count_batch(
    *,
    model: Any,
    processor: Any,
    adapter: Optional[ExperimentAdapter],
    examples: Sequence[FrameMemoryExample],
    sample_indices: Sequence[int],
    answer_ids: Dict[int, Tuple[int, ...]],
    count_values: Sequence[int],
    device: str,
    scorer_info: Dict[str, Any],
    counter: Dict[str, int],
) -> Tuple[List[int], List[Dict[str, float]]]:
    model.eval()
    if bool(scorer_info.get("all_single_token")):
        batch = prepare_batch(
            examples=examples,
            sample_indices=sample_indices,
            processor=processor,
            device=device,
            answer_ids=None,
        )
        if adapter is not None:
            adapter.set_context(batch)
        outputs = model(**batch.inputs, use_cache=False)
        counter["candidate_forwards"] += 1
        if adapter is not None:
            adapter.clear_context()
            if adapter.memory is not None and adapter.memory._slots is not None:
                raise RuntimeError("Memory slots were not cleared after candidate scoring")
        logits = outputs.logits.detach().float()
        log_probs = logits[
            torch.arange(int(logits.shape[0]), device=logits.device),
            batch.prompt_last_indices.to(logits.device),
            :,
        ].log_softmax(dim=-1)
        token_ids = torch.tensor([int(answer_ids[int(count)][0]) for count in count_values], device=logits.device, dtype=torch.long)
        score_tensor = log_probs.index_select(dim=-1, index=token_ids).detach().cpu()
        predictions: List[int] = []
        scores: List[Dict[str, float]] = []
        for row in range(int(score_tensor.shape[0])):
            row_scores = {str(int(count)): float(score_tensor[row, pos].item()) for pos, count in enumerate(count_values)}
            pred = max((int(count) for count in count_values), key=lambda value: row_scores[str(int(value))])
            predictions.append(int(pred))
            scores.append(row_scores)
        return predictions, scores
    score_rows = torch.empty((len(examples), len(count_values)), dtype=torch.float32)
    for cpos, candidate in enumerate(count_values):
        batch = prepare_batch(
            examples=examples,
            sample_indices=sample_indices,
            processor=processor,
            device=device,
            answer_ids=answer_ids,
            answer_count_override=int(candidate),
        )
        if adapter is not None:
            adapter.set_context(batch)
        outputs = model(**batch.inputs, use_cache=False)
        counter["candidate_forwards"] += 1
        if adapter is not None:
            adapter.clear_context()
            if adapter.memory is not None and adapter.memory._slots is not None:
                raise RuntimeError("Memory slots were not cleared after candidate scoring")
        score_rows[:, cpos] = candidate_score_batch_from_logits(outputs.logits, batch)
    predictions = []
    scores = []
    for row in range(int(score_rows.shape[0])):
        row_scores = {str(int(count)): float(score_rows[row, pos].item()) for pos, count in enumerate(count_values)}
        pred = max((int(count) for count in count_values), key=lambda value: row_scores[str(int(value))])
        predictions.append(int(pred))
        scores.append(row_scores)
    return predictions, scores


@torch.no_grad()
def legacy_predict_count_slow(
    *,
    model: Any,
    processor: Any,
    adapter: Optional[ExperimentAdapter],
    example: FrameMemoryExample,
    sample_idx: int,
    answer_ids: Dict[int, Tuple[int, ...]],
    count_values: Sequence[int],
    device: str,
    scorer_info: Dict[str, Any],
) -> Tuple[int, Dict[str, float]]:
    scores: Dict[str, float] = {}
    for candidate in count_values:
        if bool(scorer_info.get("all_single_token")):
            batch = prepare_batch(
                examples=[example],
                sample_indices=[sample_idx],
                processor=processor,
                device=device,
                answer_ids=None,
            )
            if adapter is not None:
                adapter.set_context(batch)
            outputs = model(**batch.inputs, use_cache=False)
            if adapter is not None:
                adapter.clear_context()
            token_id = int(answer_ids[int(candidate)][0])
            value = outputs.logits[0, int(batch.prompt_last_indices[0].item()), :].float().log_softmax(dim=-1)[token_id]
            scores[str(int(candidate))] = float(value.detach().cpu().item())
        else:
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
            scores[str(int(candidate))] = float(candidate_score_batch_from_logits(outputs.logits, batch)[0].item())
    pred = max((int(key) for key in scores), key=lambda value: scores[str(value)])
    return int(pred), scores


def run_gold_ce_batch(
    *,
    model: Any,
    processor: Any,
    adapter: ExperimentAdapter,
    examples: Sequence[FrameMemoryExample],
    sample_indices: Sequence[int],
    answer_ids: Dict[int, Tuple[int, ...]],
    device: str,
    counter: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, FrameMemoryBatch]:
    batch = prepare_batch(
        examples=examples,
        sample_indices=sample_indices,
        processor=processor,
        device=device,
        answer_ids=answer_ids,
    )
    adapter.set_context(batch)
    if adapter.memory is not None and adapter.memory._slots is not None:
        raise RuntimeError("Memory slots were not reset before forward")
    outputs = model(**batch.inputs, use_cache=False)
    counter["gold_forwards"] += 1
    loss, row_loss = answer_sequence_cross_entropy(outputs.logits, batch)
    adapter.clear_context()
    if adapter.memory is not None and adapter.memory._slots is not None:
        raise RuntimeError("Memory slots were not cleared after forward")
    return loss, row_loss.detach().float().cpu(), batch


@torch.no_grad()
def validation_gold_ce(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    adapter: ExperimentAdapter,
    examples: Sequence[FrameMemoryExample],
    indices: Sequence[int],
    answer_ids: Dict[int, Tuple[int, ...]],
    device: str,
    counter: Dict[str, int],
) -> float:
    model.eval()
    adapter.eval()
    losses: List[float] = []
    for idxs in bucketed_batches(indices, examples, int(args.eval_batch_size), int(args.seed) + 707, False):
        batch_examples = [examples[int(i)] for i in idxs]
        loss, row_loss, _batch = run_gold_ce_batch(
            model=model,
            processor=processor,
            adapter=adapter,
            examples=batch_examples,
            sample_indices=idxs,
            answer_ids=answer_ids,
            device=device,
            counter=counter,
        )
        del loss
        losses.extend(float(x) for x in row_loss.tolist())
    return finite_mean(losses, default=MISSING)


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
    metadata: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Path]:
    train_indices = limited_indices(examples[TRAIN_SPLIT], int(args.max_train_examples), int(args.seed))
    if bool(args.smoke_test) or bool(args.real_qwen_smoke):
        train_indices = train_indices[: min(6, len(train_indices))]
    val_indices = limited_indices(examples[VAL_SPLIT], int(args.max_eval_examples), int(args.seed) + 17)
    if bool(args.smoke_test) or bool(args.real_qwen_smoke):
        val_indices = val_indices[: min(4, len(val_indices))]
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
    best_val_ce = math.inf
    patience_left = int(args.early_stopping_patience)
    backward_checked = False
    train_counter: Dict[str, int] = defaultdict(int)
    started = time.time()
    fallback_occurred = False
    effective_batch_sizes: List[int] = []
    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            adapter.train()
            batches = bucketed_batches(
                train_indices,
                examples[TRAIN_SPLIT],
                int(args.batch_size),
                int(args.seed) + epoch * 1009,
                True,
            )
            assert_batch_coverage(train_indices, batches)
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_examples = 0
            optimizer_steps = 0
            micro_steps = 0

            def process_batch(idxs: Sequence[int]) -> None:
                nonlocal backward_checked, total_loss, total_examples, optimizer_steps, micro_steps
                batch_examples = [examples[TRAIN_SPLIT][int(i)] for i in idxs]
                batch = prepare_batch(
                    examples=batch_examples,
                    sample_indices=idxs,
                    processor=processor,
                    device=device,
                    answer_ids=answer_ids,
                )
                if len(batch.sample_ids) != len(idxs):
                    raise RuntimeError("FrameMemoryBatch metadata lost batch elements")
                adapter.set_context(batch)
                if adapter.memory is not None and adapter.memory._slots is not None:
                    raise RuntimeError("Memory slots were not reset between batches")
                outputs = model(**batch.inputs, use_cache=False)
                train_counter["train_forwards"] += 1
                loss, row_loss = answer_sequence_cross_entropy(outputs.logits, batch)
                adapter.clear_context()
                if adapter.memory is not None and adapter.memory._slots is not None:
                    raise RuntimeError("Memory slots were not cleared after batch")
                if not loss.requires_grad:
                    raise RuntimeError("Training loss has no gradient path; adapter hooks are not attached")
                (loss / max(1, int(args.grad_accum))).backward()
                total_loss += float(row_loss.detach().float().sum().cpu().item())
                total_examples += len(idxs)
                micro_steps += 1
                effective_batch_sizes.append(len(idxs))
                if not backward_checked:
                    bad_model_grads = base.unexpected_frozen_model_grads(model)
                    if bad_model_grads:
                        raise RuntimeError(f"Frozen Qwen parameters received gradients: {bad_model_grads[:20]}")
                    backward_checked = True
                if micro_steps % max(1, int(args.grad_accum)) == 0:
                    torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

            for idxs in batches:
                try:
                    process_batch(idxs)
                except Exception as exc:
                    if not is_oom(exc) or len(idxs) <= int(args.min_train_batch_size):
                        raise
                    fallback_occurred = True
                    clear_after_oom(optimizer)
                    for single in idxs:
                        process_batch([int(single)])
            if micro_steps and micro_steps % max(1, int(args.grad_accum)) != 0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            val_counter: Dict[str, int] = defaultdict(int)
            val_ce = validation_gold_ce(
                args=args,
                model=model,
                processor=processor,
                adapter=adapter,
                examples=examples[VAL_SPLIT],
                indices=val_indices,
                answer_ids=answer_ids,
                device=device,
                counter=val_counter,
            )
            row = {
                "variant": variant,
                "seed": int(args.seed),
                "epoch": int(epoch),
                "train_loss": total_loss / max(1, total_examples),
                "train_examples": int(total_examples),
                "optimizer_steps": int(optimizer_steps),
                "val_gold_ce": float(val_ce),
                "train_forwards": int(train_counter.get("train_forwards", 0)),
                "val_forwards": int(val_counter.get("gold_forwards", 0)),
                "oom_fallback_occurred": int(fallback_occurred),
            }
            history.append(row)
            print(
                f"[{variant} seed={args.seed}] epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"val_gold_ce={row['val_gold_ce']:.4f}"
            )
            if float(val_ce) < best_val_ce:
                best_val_ce = float(val_ce)
                patience_left = int(args.early_stopping_patience)
                best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
                torch.save(best_state, checkpoint_path)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping at epoch {epoch}")
                    break
    finally:
        adapter.detach()
    if best_state is not None:
        adapter.load_state_dict(best_state)
    else:
        torch.save(adapter.state_dict(), checkpoint_path)
    metadata["training_wall_seconds"] = time.time() - started
    metadata["train_model_forwards"] = int(train_counter.get("train_forwards", 0))
    metadata["oom_fallback_occurred"] = bool(fallback_occurred)
    metadata["effective_batch_sizes"] = sorted(set(int(x) for x in effective_batch_sizes))
    return history, checkpoint_path


def paired_row_fields(example: FrameMemoryExample) -> Dict[str, Any]:
    meta = _paired_metadata_from_example(example)
    return {
        "paired_family_id": meta.get("family_id"),
        "paired_base_sample_id": meta.get("base_sample_id"),
        "paired_version_length": meta.get("version_length"),
        "paired_original_evidence_frame_identities": meta.get("original_evidence_frame_identities"),
        "paired_new_neutral_frame_identities": meta.get("new_neutral_frame_identities"),
        "paired_final_evidence_positions": meta.get("final_evidence_positions"),
        "paired_final_frame_ordering": meta.get("final_frame_ordering"),
    }


def selected_diagnostic_ids(examples: Dict[str, List[FrameMemoryExample]]) -> set[str]:
    out: set[str] = set()
    for split in (IID_TEST_SPLIT, SEEN_COUNT_SPLIT, HIGH_AGG_SPLIT):
        rows = sorted(examples[split], key=lambda ex: (int(ex.num_frames), int(ex.gold_count), ex.example_id))
        if rows:
            step = max(1, len(rows) // 4)
            out.update(row.example_id for row in rows[::step][:4])
    return out


@torch.no_grad()
def evaluate_split(
    *,
    args: argparse.Namespace,
    variant: str,
    split_name: str,
    model: Any,
    processor: Any,
    adapter: ExperimentAdapter,
    examples: Sequence[FrameMemoryExample],
    indices: Sequence[int],
    answer_ids: Dict[int, Tuple[int, ...]],
    count_values: Sequence[int],
    device: str,
    scorer_info: Dict[str, Any],
    diagnostic_ids: set[str],
    diagnostic_dir: Path,
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    model.eval()
    adapter.eval()
    rows: List[Dict[str, Any]] = []
    eval_counter: Dict[str, int] = defaultdict(int)
    started = time.time()
    saved_diags = 0
    adapter.attach(model)
    try:
        batches = bucketed_batches(indices, examples, int(args.eval_batch_size), int(args.seed) + len(split_name), False)
        fallback_sizes = [int(args.eval_batch_size), 2, 1]
        fallback_sizes = [size for size in fallback_sizes if size <= int(args.eval_batch_size) and size >= 1]
        for idxs in batches:
            pending = [list(idxs)]
            while pending:
                current = pending.pop(0)
                batch_examples = [examples[int(i)] for i in current]
                try:
                    batch = prepare_batch(
                        examples=batch_examples,
                        sample_indices=current,
                        processor=processor,
                        device=device,
                        answer_ids=answer_ids,
                    )
                    adapter.set_context(batch)
                    outputs = model(**batch.inputs, use_cache=False)
                    eval_counter["gold_forwards"] += 1
                    _loss, ce_vec = answer_sequence_cross_entropy(outputs.logits, batch)
                    stats_by_row = [adapter.stats_for_row(row) for row in range(len(current))]
                    diag_tensors = adapter.memory.diagnostic_tensors() if adapter.memory is not None else {}
                    adapter.clear_context()
                    if adapter.memory is not None and adapter.memory._slots is not None:
                        raise RuntimeError("Memory slots were not cleared after eval forward")
                    preds, score_rows = predict_count_batch(
                        model=model,
                        processor=processor,
                        adapter=adapter,
                        examples=batch_examples,
                        sample_indices=current,
                        answer_ids=answer_ids,
                        count_values=count_values,
                        device=device,
                        scorer_info=scorer_info,
                        counter=eval_counter,
                    )
                except Exception as exc:
                    if not is_oom(exc) or len(current) == 1:
                        raise
                    clear_after_oom(None)
                    metadata["eval_oom_fallback_occurred"] = True
                    mid = max(1, len(current) // 2)
                    pending.insert(0, current[mid:])
                    pending.insert(0, current[:mid])
                    continue
                for local, (idx, example) in enumerate(zip(current, batch_examples)):
                    scores = score_rows[local]
                    pred = int(preds[local])
                    row: Dict[str, Any] = {
                        "variant": variant,
                        "seed": int(args.seed),
                        "split": split_name,
                        "example_id": example.example_id,
                        "sample_index": int(idx),
                        "num_frames": int(example.num_frames),
                        "gold_count": int(example.gold_count),
                        "predicted_count": int(pred),
                        "correct": int(pred == int(example.gold_count)),
                        "abs_error": abs(pred - int(example.gold_count)),
                        "signed_error": int(pred - int(example.gold_count)),
                        "undercount": int(pred < int(example.gold_count)),
                        "overcount": int(pred > int(example.gold_count)),
                        "mean_predicted_count": float(pred),
                        "evidence_density": float(example.gold_count) / max(1, int(example.num_frames)),
                        "evidence_frame_indices": list(example.evidence_frame_indices),
                        "composition_key": example.composition_key,
                        "queried_character": example.queried_character,
                        "queried_room": example.queried_room,
                        "template_id": example.template_id,
                        "carrier_token_positions": list(batch.carrier_positions[local]),
                        "carrier_identities": list(batch.carrier_identities[local]),
                        "visual_token_counts_per_frame": [len(group) for group in batch.frame_groups[local]],
                        "ce": float(ce_vec[local].detach().cpu().item()),
                        "candidate_scores": scores,
                        "gold_answer_score": float(scores.get(str(int(example.gold_count)), MISSING)),
                        **paired_row_fields(example),
                        **stats_by_row[local],
                    }
                    rows.append(row)
                    if (
                        variant in MEMORY_VARIANTS
                        and example.example_id in diagnostic_ids
                        and saved_diags < 12
                        and diag_tensors
                    ):
                        diagnostic_dir.mkdir(parents=True, exist_ok=True)
                        path = diagnostic_dir / f"{safe_name(split_name)}_{saved_diags:02d}_{safe_name(example.example_id)}.pt"
                        torch.save({layer: tensors for layer, tensors in diag_tensors.items()}, path)
                        saved_diags += 1
    finally:
        adapter.detach()
    metadata.setdefault("evaluation_wall_seconds_by_split", {})[split_name] = time.time() - started
    metadata.setdefault("eval_model_forwards_by_split", {})[split_name] = {
        key: int(value) for key, value in sorted(eval_counter.items())
    }
    return rows


def summarize_prediction_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "accuracy": MISSING,
            "mae": MISSING,
            "mean_signed_error": MISSING,
            "undercount_rate": MISSING,
            "overcount_rate": MISSING,
            "mean_predicted_count": MISSING,
            "mean_gold_answer_score": MISSING,
            "mean_cross_entropy": MISSING,
        }
    return {
        "n": len(rows),
        "accuracy": base.accuracy([int(row["gold_count"]) for row in rows], [int(row["predicted_count"]) for row in rows]),
        "mae": base.mae([int(row["gold_count"]) for row in rows], [int(row["predicted_count"]) for row in rows]),
        "mean_signed_error": finite_mean([row["signed_error"] for row in rows], default=MISSING),
        "undercount_rate": finite_mean([row["undercount"] for row in rows], default=MISSING),
        "overcount_rate": finite_mean([row["overcount"] for row in rows], default=MISSING),
        "mean_predicted_count": finite_mean([row["predicted_count"] for row in rows], default=MISSING),
        "mean_gold_answer_score": finite_mean([row.get("gold_answer_score") for row in rows], default=MISSING),
        "mean_cross_entropy": finite_mean([row.get("ce") for row in rows], default=MISSING),
    }


def metrics_from_rows(variant: str, seed: int, rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    split_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    length_rows: List[Dict[str, Any]] = []
    for split in sorted({str(row["split"]) for row in rows}):
        split_data = [row for row in rows if str(row["split"]) == split]
        split_rows.append({"variant": variant, "seed": int(seed), "split": split, **summarize_prediction_rows(split_data)})
        for count in sorted({int(row["gold_count"]) for row in split_data}):
            data = [row for row in split_data if int(row["gold_count"]) == count]
            count_rows.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "split": split,
                    "true_count": int(count),
                    **summarize_prediction_rows(data),
                }
            )
        for length in sorted({int(row["num_frames"]) for row in split_data}):
            data = [row for row in split_data if int(row["num_frames"]) == length]
            length_rows.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "split": split,
                    "sequence_length": int(length),
                    **summarize_prediction_rows(data),
                }
            )
    return split_rows, count_rows, length_rows


def paired_extension_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_family: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        family = str(row.get("paired_family_id"))
        by_family[family][int(row["num_frames"])] = row
    out: List[Dict[str, Any]] = []
    for family, versions in sorted(by_family.items()):
        if 8 not in versions:
            continue
        base_row = versions[8]
        base_pred = int(base_row["predicted_count"])
        base_gold_score = finite_float(base_row.get("gold_answer_score")) or MISSING
        for length, row in sorted(versions.items()):
            pred = int(row["predicted_count"])
            gold_score = finite_float(row.get("gold_answer_score")) or MISSING
            out.append(
                {
                    "variant": row["variant"],
                    "seed": int(row["seed"]),
                    "family_id": family,
                    "base_sample_id": row.get("paired_base_sample_id"),
                    "version_length": int(length),
                    "example_id": row["example_id"],
                    "gold_count": int(row["gold_count"]),
                    "base_predicted_count": int(base_pred),
                    "predicted_count": int(pred),
                    "prediction_unchanged_from_len8": int(pred == base_pred),
                    "correct": int(pred == int(row["gold_count"])),
                    "abs_prediction_drift": abs(int(pred - base_pred)),
                    "signed_prediction_drift": int(pred - base_pred),
                    "gold_answer_score": gold_score,
                    "gold_answer_score_change": (
                        float(gold_score - base_gold_score)
                        if math.isfinite(float(gold_score)) and math.isfinite(float(base_gold_score))
                        else MISSING
                    ),
                }
            )
    return out


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.fspath(PROJECT_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def assert_run_correctness(
    *,
    variant: str,
    adapter: ExperimentAdapter,
    prediction_rows: Sequence[Dict[str, Any]],
    dataset_manifest: Dict[str, Any],
    count_values: Sequence[int],
) -> None:
    if [int(x) for x in count_values] != list(COUNT_VALUES):
        raise RuntimeError("Candidate counts are not exactly 0..12")
    for split, payload in dataset_manifest["splits"].items():
        if int(payload["n"]) != int(payload["expected_n"]):
            raise RuntimeError(f"{split}: dataset manifest has n != expected_n")
    max_noncarrier = max([finite_float(row.get("noncarrier_lora_update_max")) or 0.0 for row in prediction_rows], default=0.0)
    if max_noncarrier > 1e-5:
        raise RuntimeError(f"Carrier-gated LoRA updated non-carrier positions: max={max_noncarrier}")
    if variant == LORA_BASELINE:
        if adapter.memory is not None:
            raise RuntimeError("lora_baseline unexpectedly has memory")
        allowed_layer_keys = {"lora_delta_norm_by_layer"}
        memory_keys = [
            key
            for row in prediction_rows
            for key in row
            if (key.endswith("_by_layer") and key not in allowed_layer_keys) or key.startswith("memory_")
        ]
        if memory_keys:
            raise RuntimeError(f"lora_baseline emitted memory diagnostics: {sorted(set(memory_keys))[:10]}")
        return
    if adapter.memory is None:
        raise RuntimeError("Memory variant is missing memory")
    injected: Dict[int, List[int]] = defaultdict(list)
    previous_used: Dict[int, List[int]] = defaultdict(list)
    for row in prediction_rows:
        for layer, value in (row.get("injected_by_layer", {}) or {}).items():
            injected[int(layer)].append(int(value))
        for layer, value in (row.get("previous_slot_used_by_layer", {}) or {}).items():
            previous_used[int(layer)].append(int(value))
        recon = row.get("read_reconstruction_error_by_layer", {})
        if isinstance(recon, dict):
            bad = {layer: value for layer, value in recon.items() if (finite_float(value) or 0.0) > 1e-4}
            if bad:
                raise RuntimeError(f"Read contribution reconstruction failed: {bad}")
    for layer in DEFAULT_LAYERS:
        expected = 1 if int(layer) == max(DEFAULT_LAYERS) else 0
        if injected.get(int(layer)) and any(int(v) != expected for v in injected[int(layer)]):
            raise RuntimeError(f"{variant}: final-only injection mismatch at layer {layer}")
    for layer in DEFAULT_LAYERS[1:]:
        if previous_used.get(int(layer)) and not all(int(v) == 1 for v in previous_used[int(layer)]):
            raise RuntimeError(f"{variant}: persistent slots not marked used at layer {layer}")


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
    count_values: Sequence[int],
    device: str,
    scorer_info: Dict[str, Any],
    timestamp: str,
) -> Dict[str, Any]:
    layers = parse_layers(args)
    run_prefix = f"{safe_name(args.run_prefix)}_" if str(args.run_prefix).strip() else ""
    smoke = "smoke_" if bool(args.smoke_test) or bool(args.real_qwen_smoke) else ""
    run_dir = parent_output_root / f"{timestamp}_{run_prefix}{smoke}{variant}_seed{int(args.seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle, old_stdout, old_stderr = base.setup_logging(run_dir)
    adapter: Optional[ExperimentAdapter] = None
    metadata: Dict[str, Any] = {
        "variant": variant,
        "seed": int(args.seed),
        "dataset_hash": dataset_manifest.get("dataset_hash"),
        "git_commit": git_commit_hash(),
        "oom_fallback_occurred": False,
        "eval_oom_fallback_occurred": False,
    }
    try:
        print(f"Running {variant} seed={args.seed} into {run_dir}")
        write_json(run_dir / "config.json", {**vars(args), "variant": variant, "layers": layers})
        write_json(run_dir / "dataset_manifest.json", dataset_manifest)
        hidden_size = base.hidden_size_from_model(model)
        adapter = make_adapter(args, variant, hidden_size=hidden_size, layers=layers).to(device)
        adapter.attach(model)
        try:
            param_summary = trainable_parameter_summary(model, adapter, variant)
        finally:
            adapter.detach()
        write_json(run_dir / "parameter_summary.json", param_summary)
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
            metadata=metadata,
        )
        write_csv_dynamic(run_dir / "training_history.csv", history, leading=("variant", "seed", "epoch"))
        all_rows: List[Dict[str, Any]] = []
        diagnostics_dir = run_dir / "diagnostics"
        diag_ids = selected_diagnostic_ids(examples)
        for split in EVAL_SPLITS:
            indices = limited_indices(examples[split], int(args.max_eval_examples), int(args.seed) + len(split) + 101)
            if bool(args.smoke_test) or bool(args.real_qwen_smoke):
                indices = indices[: min(4, len(indices))]
            rows = evaluate_split(
                args=args,
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
                scorer_info=scorer_info,
                diagnostic_ids=diag_ids,
                diagnostic_dir=diagnostics_dir,
                metadata=metadata,
            )
            all_rows.extend(rows)
        split_metrics, count_metrics, length_metrics = metrics_from_rows(variant, int(args.seed), all_rows)
        paired_rows = [row for row in all_rows if row["split"] == PAIRED_SPLIT]
        paired_summary_rows = paired_extension_summary(paired_rows)
        write_csv_dynamic(run_dir / "per_sample_predictions.csv", all_rows, leading=("variant", "seed", "split", "example_id"))
        write_csv_dynamic(run_dir / "metrics.csv", split_metrics, leading=("variant", "seed", "split"))
        write_csv_dynamic(run_dir / "metrics_by_count.csv", count_metrics, leading=("variant", "seed", "split", "true_count"))
        write_csv_dynamic(run_dir / "metrics_by_length.csv", length_metrics, leading=("variant", "seed", "split", "sequence_length"))
        write_csv_dynamic(run_dir / "paired_extension_summary.csv", paired_summary_rows, leading=("variant", "seed", "family_id", "version_length"))
        if not bool(args.no_plots):
            plot_run_metrics(
                run_dir=run_dir,
                variant=variant,
                history=history,
                split_metrics=split_metrics,
                count_metrics=count_metrics,
                length_metrics=length_metrics,
                paired_rows=paired_summary_rows,
            )
        metadata["evaluation_wall_seconds"] = sum(float(x) for x in metadata.get("evaluation_wall_seconds_by_split", {}).values())
        metadata["trainable_adapter_parameters"] = param_summary["trainable_adapter_parameters"]
        metadata["trainable_parameter_groups"] = param_summary["groups"]
        metadata["checkpoint"] = os.fspath(checkpoint_path)
        write_json(run_dir / "run_metadata.json", metadata)
        assert_run_correctness(
            variant=variant,
            adapter=adapter,
            prediction_rows=all_rows,
            dataset_manifest=dataset_manifest,
            count_values=count_values,
        )
        write_run_report(run_dir, variant, split_metrics, paired_summary_rows, checkpoint_path, metadata)
        return {
            "variant": variant,
            "seed": int(args.seed),
            "run_dir": os.fspath(run_dir),
            "checkpoint": os.fspath(checkpoint_path),
            "metrics": split_metrics,
            "metadata": metadata,
        }
    finally:
        if adapter is not None:
            adapter.detach()
        base.restore_logging(log_handle, old_stdout, old_stderr)


def write_run_report(
    run_dir: Path,
    variant: str,
    split_metrics: Sequence[Dict[str, Any]],
    paired_rows: Sequence[Dict[str, Any]],
    checkpoint_path: Path,
    metadata: Dict[str, Any],
) -> None:
    lines = [
        f"# {variant} seed {metadata.get('seed')} run report",
        "",
        f"Dataset hash: `{metadata.get('dataset_hash', '')}`",
        f"Checkpoint: `{checkpoint_path}`",
        "",
        "| split | accuracy | MAE | signed error | under | over | mean pred |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_metrics:
        lines.append(
            f"| {row['split']} | {float(row.get('accuracy', MISSING)):.3f} | "
            f"{float(row.get('mae', MISSING)):.3f} | {float(row.get('mean_signed_error', MISSING)):.3f} | "
            f"{float(row.get('undercount_rate', MISSING)):.3f} | {float(row.get('overcount_rate', MISSING)):.3f} | "
            f"{float(row.get('mean_predicted_count', MISSING)):.3f} |"
        )
    if paired_rows:
        lines.extend(["", "## Paired Neutral Extension", "", "| length | mean abs drift | retention | accuracy |", "|---:|---:|---:|---:|"])
        for length in (8, 12, 16):
            data = [row for row in paired_rows if int(row["version_length"]) == int(length)]
            lines.append(
                f"| {length} | {finite_mean([row['abs_prediction_drift'] for row in data], default=MISSING):.3f} | "
                f"{finite_mean([row['prediction_unchanged_from_len8'] for row in data], default=MISSING):.3f} | "
                f"{finite_mean([row['correct'] for row in data], default=MISSING):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Training wall seconds: `{float(metadata.get('training_wall_seconds', 0.0)):.1f}`",
            f"- Evaluation wall seconds: `{float(metadata.get('evaluation_wall_seconds', 0.0)):.1f}`",
            f"- OOM fallback occurred: `{bool(metadata.get('oom_fallback_occurred') or metadata.get('eval_oom_fallback_occurred'))}`",
        ]
    )
    (run_dir / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_dirs_for_aggregation(parent: Path) -> Dict[Tuple[str, int], Path]:
    found: Dict[Tuple[str, int], List[Tuple[float, Path]]] = defaultdict(list)
    for config_path in sorted(Path(parent).glob("*/config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if bool(config.get("smoke_test")) or bool(config.get("real_qwen_smoke")):
            continue
        variant = str(config.get("variant", ""))
        seed = int(config.get("seed", -1))
        if variant in VARIANTS and seed in {0, 1, 2} and (config_path.parent / "metrics.csv").is_file():
            found[(variant, seed)].append((config_path.parent.stat().st_mtime, config_path.parent))
    return {key: sorted(entries)[-1][1] for key, entries in found.items() if entries}


def assert_shared_dataset_across_runs(run_dirs: Dict[Tuple[str, int], Path]) -> Dict[str, Any]:
    hashes: set[str] = set()
    sample_ids_by_split: Optional[Dict[str, List[str]]] = None
    for key, run_dir in run_dirs.items():
        manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        dataset_hash = str(manifest.get("dataset_hash"))
        hashes.add(dataset_hash)
        ids = {split: list(payload.get("sample_ids", [])) for split, payload in manifest.get("splits", {}).items()}
        if sample_ids_by_split is None:
            sample_ids_by_split = ids
        elif ids != sample_ids_by_split:
            raise RuntimeError(f"{key}: sample IDs differ from the shared dataset")
    if len(hashes) != 1:
        raise RuntimeError(f"Variant jobs did not use the same dataset hash: {sorted(hashes)}")
    return {"dataset_hash": next(iter(hashes)) if hashes else "", "sample_ids_by_split": sample_ids_by_split or {}}


def mean_std(values: Sequence[Any]) -> Tuple[float, float]:
    nums = [float(v) for v in values if finite_float(v) is not None]
    if not nums:
        return MISSING, MISSING
    return float(np.mean(nums)), float(np.std(nums, ddof=0))


def save_plot(path: Path, fig: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)


def plot_run_metrics(
    *,
    run_dir: Path,
    variant: str,
    history: Sequence[Dict[str, Any]],
    split_metrics: Sequence[Dict[str, Any]],
    count_metrics: Sequence[Dict[str, Any]],
    length_metrics: Sequence[Dict[str, Any]],
    paired_rows: Sequence[Dict[str, Any]],
) -> None:
    plot_dir = Path(run_dir) / "plots"

    if history:
        epochs = [int(row["epoch"]) for row in history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, [float(row["train_loss"]) for row in history], marker="o", label="train loss")
        ax.plot(epochs, [float(row["val_gold_ce"]) for row in history], marker="o", label="validation CE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(variant)
        ax.legend()
        save_plot(plot_dir / "training_curve.png", fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for split in MAIN_METRIC_SPLITS:
        rows = sorted(
            (row for row in count_metrics if row.get("split") == split),
            key=lambda row: int(row["true_count"]),
        )
        if rows:
            ax.plot(
                [int(row["true_count"]) for row in rows],
                [float(row.get("accuracy", MISSING)) for row in rows],
                marker="o",
                label=split,
            )
    ax.set_xlabel("Evidence count")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(variant)
    ax.legend(fontsize=8)
    save_plot(plot_dir / "accuracy_by_evidence_count.png", fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    rows = sorted(count_metrics, key=lambda row: (str(row.get("split")), int(row["true_count"])))
    for split in MAIN_METRIC_SPLITS:
        split_rows = [row for row in rows if row.get("split") == split]
        if split_rows:
            ax.plot(
                [int(row["true_count"]) for row in split_rows],
                [float(row.get("mean_predicted_count", MISSING)) for row in split_rows],
                marker="o",
                label=split,
            )
    ax.plot(list(COUNT_VALUES), list(COUNT_VALUES), linestyle="--", color="black", label="ideal")
    ax.set_xlabel("True count")
    ax.set_ylabel("Mean predicted count")
    ax.set_title(variant)
    ax.legend(fontsize=8)
    save_plot(plot_dir / "mean_predicted_vs_true_count.png", fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for split in MAIN_METRIC_SPLITS:
        split_rows = sorted(
            (row for row in length_metrics if row.get("split") == split),
            key=lambda row: int(row["sequence_length"]),
        )
        if split_rows:
            ax.plot(
                [int(row["sequence_length"]) for row in split_rows],
                [float(row.get("accuracy", MISSING)) for row in split_rows],
                marker="o",
                label=split,
            )
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(variant)
    ax.legend(fontsize=8)
    save_plot(plot_dir / "accuracy_by_sequence_length.png", fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [str(row["split"]) for row in split_metrics]
    values = [float(row.get("accuracy", MISSING)) for row in split_metrics]
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(variant)
    save_plot(plot_dir / "accuracy_by_split.png", fig)

    if paired_rows:
        by_length: Dict[int, List[float]] = defaultdict(list)
        for row in paired_rows:
            by_length[int(row["version_length"])].append(float(row.get("abs_prediction_drift", MISSING)))
        lengths = sorted(by_length)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(lengths, [finite_mean(by_length[length]) for length in lengths])
        ax.set_xlabel("Version length")
        ax.set_ylabel("Mean absolute prediction drift")
        ax.set_title(variant)
        save_plot(plot_dir / "paired_neutral_extension_drift.png", fig)


def plot_combined(parent: Path, metric_rows: Sequence[Dict[str, Any]], count_rows: Sequence[Dict[str, Any]], length_rows: Sequence[Dict[str, Any]], paired_rows: Sequence[Dict[str, Any]]) -> None:
    plot_dir = parent / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    high_count_rows = [row for row in count_rows if row.get("split") in {HIGH_AGG_SPLIT, SEEN_COUNT_SPLIT}]
    fig, ax = plt.subplots(figsize=(8, 4))
    for variant in VARIANTS:
        by_count: Dict[int, List[float]] = defaultdict(list)
        for row in high_count_rows:
            if row.get("variant") == variant:
                by_count[int(row["true_count"])].append(float(row.get("accuracy", MISSING)))
        xs = sorted(by_count)
        if xs:
            ax.plot(xs, [mean_std(by_count[x])[0] for x in xs], marker="o", label=variant)
    ax.set_xlabel("Evidence count")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    save_plot(plot_dir / "accuracy_by_evidence_count.png", fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in VARIANTS:
        by_count = defaultdict(list)
        for row in count_rows:
            if row.get("variant") == variant:
                by_count[int(row["true_count"])].append(float(row.get("mean_predicted_count", MISSING)))
        xs = sorted(by_count)
        if xs:
            ax.plot(xs, [mean_std(by_count[x])[0] for x in xs], marker="o", label=variant)
    ax.plot(list(COUNT_VALUES), list(COUNT_VALUES), linestyle="--", color="black", label="ideal")
    ax.set_xlabel("True count")
    ax.set_ylabel("Mean predicted count")
    ax.legend(fontsize=8)
    save_plot(plot_dir / "mean_predicted_vs_true_count.png", fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for variant in VARIANTS:
        by_length = defaultdict(list)
        for row in length_rows:
            if row.get("variant") == variant:
                by_length[int(row["sequence_length"])].append(float(row.get("accuracy", MISSING)))
        xs = sorted(x for x in by_length if 4 <= x <= 12)
        if xs:
            ax.plot(xs, [mean_std(by_length[x])[0] for x in xs], marker="o", label=variant)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    save_plot(plot_dir / "accuracy_by_sequence_length.png", fig)

    splits = [IID_TEST_SPLIT, SEEN_COUNT_SPLIT, HIGH_AGG_SPLIT]
    x = np.arange(len(splits))
    width = 0.8 / len(VARIANTS)
    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, variant in enumerate(VARIANTS):
        means = []
        stds = []
        for split in splits:
            vals = [float(row.get("accuracy", MISSING)) for row in metric_rows if row.get("variant") == variant and row.get("split") == split]
            mean, std = mean_std(vals)
            means.append(mean)
            stds.append(std)
        ax.bar(x + offset * width, means, yerr=stds, width=width, capsize=3, label=variant)
    ax.set_xticks(x + width, splits, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    save_plot(plot_dir / "accuracy_by_split.png", fig)

    x_labels = [12, 16]
    x = np.arange(len(x_labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, variant in enumerate(VARIANTS):
        means = []
        stds = []
        for length in x_labels:
            vals = [
                float(row.get("abs_prediction_drift", MISSING))
                for row in paired_rows
                if row.get("variant") == variant and int(row.get("version_length", -1)) == int(length)
            ]
            mean, std = mean_std(vals)
            means.append(mean)
            stds.append(std)
        ax.bar(x + offset * width, means, yerr=stds, width=width, capsize=3, label=variant)
    ax.set_xticks(x + width, [str(v) for v in x_labels])
    ax.set_xlabel("Version length")
    ax.set_ylabel("Mean absolute prediction drift")
    ax.legend(fontsize=8)
    save_plot(plot_dir / "paired_neutral_extension_drift.png", fig)


def metric_lookup(rows: Sequence[Dict[str, Any]], variant: str, split: str, metric: str) -> float:
    vals = [float(row.get(metric, MISSING)) for row in rows if row.get("variant") == variant and row.get("split") == split]
    return mean_std(vals)[0]


def write_final_report(parent: Path, metric_rows: Sequence[Dict[str, Any]], count_rows: Sequence[Dict[str, Any]], paired_rows: Sequence[Dict[str, Any]], shared: Dict[str, Any]) -> None:
    lora = metric_lookup(metric_rows, LORA_BASELINE, HIGH_AGG_SPLIT, "accuracy")
    sum_acc = metric_lookup(metric_rows, SUM_FINAL_ONLY_PERSISTENT, HIGH_AGG_SPLIT, "accuracy")
    glstm = metric_lookup(metric_rows, GLSTM_FINAL_ONLY_PERSISTENT, HIGH_AGG_SPLIT, "accuracy")
    high_adv = []
    for count in range(9, 13):
        g = mean_std([
            row.get("accuracy")
            for row in count_rows
            if row.get("variant") == GLSTM_FINAL_ONLY_PERSISTENT and row.get("split") == HIGH_AGG_SPLIT and int(row.get("true_count", -1)) == count
        ])[0]
        s = mean_std([
            row.get("accuracy")
            for row in count_rows
            if row.get("variant") == SUM_FINAL_ONLY_PERSISTENT and row.get("split") == HIGH_AGG_SPLIT and int(row.get("true_count", -1)) == count
        ])[0]
        if math.isfinite(g) and math.isfinite(s):
            high_adv.append(g - s)
    l16_drift = {
        variant: mean_std([
            row.get("abs_prediction_drift")
            for row in paired_rows
            if row.get("variant") == variant and int(row.get("version_length", -1)) == 16
        ])[0]
        for variant in VARIANTS
    }
    lines = [
        "# Final gLSTM aggregation comparison",
        "",
        f"Dataset hash: `{shared.get('dataset_hash', '')}`",
        "",
        "## Main Results",
        "",
        "| variant | split | accuracy mean | accuracy std | MAE mean |",
        "|---|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for split in MAIN_METRIC_SPLITS:
            acc_mean, acc_std = mean_std([row.get("accuracy") for row in metric_rows if row.get("variant") == variant and row.get("split") == split])
            mae_mean, _ = mean_std([row.get("mae") for row in metric_rows if row.get("variant") == variant and row.get("split") == split])
            lines.append(f"| {variant} | {split} | {acc_mean:.3f} | {acc_std:.3f} | {mae_mean:.3f} |")
    lines.extend(
        [
            "",
            "## Questions",
            "",
            f"- Sum beats LoRA on high aggregation: {bool(sum_acc > lora) if math.isfinite(sum_acc) and math.isfinite(lora) else 'unknown'} ({sum_acc:.3f} vs {lora:.3f}).",
            f"- gLSTM beats LoRA on high aggregation: {bool(glstm > lora) if math.isfinite(glstm) and math.isfinite(lora) else 'unknown'} ({glstm:.3f} vs {lora:.3f}).",
            f"- gLSTM beats matched sum on high aggregation: {bool(glstm > sum_acc) if math.isfinite(glstm) and math.isfinite(sum_acc) else 'unknown'} ({glstm:.3f} vs {sum_acc:.3f}).",
            f"- gLSTM advantage over sum by high counts 9..12: {json.dumps([round(x, 4) for x in high_adv])}.",
            f"- Length-16 paired absolute drift by variant: {json.dumps(l16_drift, sort_keys=True)}.",
            f"- gLSTM is more stable than sum under neutral extension: {bool(l16_drift.get(GLSTM_FINAL_ONLY_PERSISTENT, math.inf) < l16_drift.get(SUM_FINAL_ONLY_PERSISTENT, -math.inf))}.",
            "",
            "Plots: `comparison_plots/`",
        ]
    )
    (parent / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_parent_outputs(parent: Path) -> Dict[str, Any]:
    parent = Path(parent)
    run_dirs = run_dirs_for_aggregation(parent)
    required = {(variant, seed) for variant in VARIANTS for seed in (0, 1, 2)}
    missing = sorted(required - set(run_dirs))
    if missing:
        raise RuntimeError(f"Cannot aggregate; missing completed variant/seed runs: {missing}")
    shared = assert_shared_dataset_across_runs(run_dirs)
    metric_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    length_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    for key, run_dir in sorted(run_dirs.items()):
        required_files = [
            "config.json",
            "dataset_manifest.json",
            "parameter_summary.json",
            "training_history.csv",
            "per_sample_predictions.csv",
            "metrics.csv",
            "metrics_by_count.csv",
            "metrics_by_length.csv",
            "checkpoint/adapter_best.pt",
            "run_report.md",
        ]
        missing_files = [name for name in required_files if not (run_dir / name).is_file()]
        if missing_files:
            raise RuntimeError(f"{key}: missing required files {missing_files}")
        metric_rows.extend(read_csv_rows(run_dir / "metrics.csv"))
        count_rows.extend(read_csv_rows(run_dir / "metrics_by_count.csv"))
        length_rows.extend(read_csv_rows(run_dir / "metrics_by_length.csv"))
        paired_rows.extend(read_csv_rows(run_dir / "paired_extension_summary.csv"))
    write_csv_dynamic(parent / "combined_results.csv", metric_rows, leading=("variant", "seed", "split"))
    write_csv_dynamic(parent / "combined_results_by_count.csv", count_rows, leading=("variant", "seed", "split", "true_count"))
    write_csv_dynamic(parent / "combined_results_by_length.csv", length_rows, leading=("variant", "seed", "split", "sequence_length"))
    write_csv_dynamic(parent / "paired_extension_combined.csv", paired_rows, leading=("variant", "seed", "family_id", "version_length"))
    if not (parent / ".no_plots").exists():
        plot_combined(parent, metric_rows, count_rows, length_rows, paired_rows)
    write_final_report(parent, metric_rows, count_rows, paired_rows, shared)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_hash": shared.get("dataset_hash"),
        "run_dirs": {f"{variant}:{seed}": os.fspath(path) for (variant, seed), path in sorted(run_dirs.items())},
        "num_metric_rows": len(metric_rows),
    }
    write_json(parent / "combined_summary.json", summary)
    return summary


def run_scorer_equivalence(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    examples: Dict[str, List[FrameMemoryExample]],
    answer_ids: Dict[int, Tuple[int, ...]],
    count_values: Sequence[int],
    device: str,
    scorer_info: Dict[str, Any],
) -> None:
    layers = parse_layers(args)
    hidden_size = base.hidden_size_from_model(model)
    low_precision_cuda = str(device).startswith("cuda") and str(args.dtype).lower() in {
        "float16",
        "fp16",
        "bfloat16",
        "bf16",
    }
    score_tolerance = 0.5 if low_precision_cuda else 5e-4
    sample_groups: List[Tuple[str, List[int]]] = []
    if bool(args.smoke_test) or bool(args.real_qwen_smoke):
        for split in (IID_TEST_SPLIT, SEEN_COUNT_SPLIT, HIGH_AGG_SPLIT):
            by_shape: Dict[Tuple[int, int], List[int]] = defaultdict(list)
            for idx, example in enumerate(examples[split][:120]):
                item = _prepare_one(example, idx, processor, None, None)
                seq_len = int(item["raw_inputs"]["input_ids"].shape[1])
                by_shape[(int(example.num_frames), seq_len)].append(int(idx))
            candidates = [idxs for _shape, idxs in sorted(by_shape.items()) if len(idxs) >= 2]
            if not candidates:
                raise RuntimeError(f"No same-token-length scorer-equivalence pair found for {split}")
            sample_groups.append((split, candidates[0][:2]))
    else:
        sample_pairs: List[Tuple[str, int]] = []
        for split in (IID_TEST_SPLIT, SEEN_COUNT_SPLIT, HIGH_AGG_SPLIT):
            rows = list(range(len(examples[split])))
            sample_pairs.extend((split, idx) for idx in rows[:8])
        sample_pairs = sample_pairs[:20]
        grouped_pairs: Dict[str, List[int]] = defaultdict(list)
        for split, idx in sample_pairs:
            grouped_pairs[split].append(idx)
        sample_groups = [(split, idxs) for split, idxs in grouped_pairs.items()]
    for variant in parse_variants(args):
        adapter = make_adapter(args, variant, hidden_size=hidden_size, layers=layers).to(device)
        model.eval()
        adapter.eval()
        adapter.attach(model)
        adapter.eval()
        try:
            for split, idxs in sample_groups:
                batch_examples = [examples[split][idx] for idx in idxs[: min(4, len(idxs))]]
                batch_indices = idxs[: min(4, len(idxs))]
                counter: Dict[str, int] = defaultdict(int)
                preds, scores = predict_count_batch(
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    examples=batch_examples,
                    sample_indices=batch_indices,
                    answer_ids=answer_ids,
                    count_values=count_values,
                    device=device,
                    scorer_info=scorer_info,
                    counter=counter,
                )
                for local, (example, idx) in enumerate(zip(batch_examples, batch_indices)):
                    legacy_pred, legacy_scores = legacy_predict_count_slow(
                        model=model,
                        processor=processor,
                        adapter=adapter,
                        example=example,
                        sample_idx=idx,
                        answer_ids=answer_ids,
                        count_values=count_values,
                        device=device,
                        scorer_info=scorer_info,
                    )
                    max_diff = max(abs(float(scores[local][str(c)]) - float(legacy_scores[str(c)])) for c in count_values)
                    if int(preds[local]) != int(legacy_pred) or not math.isfinite(max_diff) or max_diff > score_tolerance:
                        raise RuntimeError(
                            f"Optimized scorer mismatch variant={variant} sample={example.example_id}: "
                            f"pred {preds[local]} vs {legacy_pred}, max diff {max_diff}, "
                            f"tolerance {score_tolerance}"
                        )
                    if max_diff > 5e-4:
                        print(
                            f"Scorer batch/singleton numerical drift variant={variant} "
                            f"sample={example.example_id}: max diff {max_diff:.6f}"
                        )
        finally:
            adapter.detach()


def run_reference_equivalence(
    *,
    args: argparse.Namespace,
    processor: Any,
    examples: Dict[str, List[FrameMemoryExample]],
    answer_ids: Dict[int, Tuple[int, ...]],
    device: str,
) -> None:
    from experiments.glstm import layerwise_glstm_mechanism_ablation as ref

    layers = parse_layers(args)
    model_a, _processor_a, _load4_a, _mode_a = load_model_and_processor(args, device=device, dtype=torch.float32)
    model_b, _processor_b, _load4_b, _mode_b = load_model_and_processor(args, device=device, dtype=torch.float32)
    model_b.load_state_dict(model_a.state_dict())
    hidden_size = base.hidden_size_from_model(model_a)
    reference = ref.MechanismFrameMessageMemory(
        variant=ref.GLSTM_FINAL_ONLY_PERSISTENT,
        hidden_size=hidden_size,
        memory_dim=int(args.memory_dim),
        layers=layers,
        gamma_init=float(args.gamma_init),
        projection_sharing=str(args.projection_sharing),
        memory_projection_sharing=str(args.memory_projection_sharing),
        message_mode=str(args.message_mode),
        reconstruction_tol=float(args.reconstruction_tol),
        fail_on_reconstruction_error=bool(args.fail_on_reconstruction_error),
    ).to(device)
    optimized = FinalAggregationMemory(
        variant=GLSTM_FINAL_ONLY_PERSISTENT,
        hidden_size=hidden_size,
        memory_dim=int(args.memory_dim),
        layers=layers,
        gamma_init=float(args.gamma_init),
        projection_sharing=str(args.projection_sharing),
        memory_projection_sharing=str(args.memory_projection_sharing),
        message_mode=str(args.message_mode),
        reconstruction_tol=float(args.reconstruction_tol),
        fail_on_reconstruction_error=bool(args.fail_on_reconstruction_error),
    ).to(device)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    batch_examples = [examples[VAL_SPLIT][i] for i in range(min(2, len(examples[VAL_SPLIT])))]
    batch_indices = list(range(len(batch_examples)))
    batch_a = prepare_batch(
        examples=batch_examples,
        sample_indices=batch_indices,
        processor=processor,
        device=device,
        answer_ids=answer_ids,
    )
    batch_b = prepare_batch(
        examples=batch_examples,
        sample_indices=batch_indices,
        processor=processor,
        device=device,
        answer_ids=answer_ids,
    )
    reference.attach(model_a)
    optimized.attach(model_b)
    try:
        reference.set_context(batch_a)
        optimized.set_context(batch_b)
        out_a = model_a(**batch_a.inputs, use_cache=False).logits.detach().float().cpu()
        out_b = model_b(**batch_b.inputs, use_cache=False).logits.detach().float().cpu()
        tensors_a = reference.diagnostic_tensors()
        tensors_b = optimized.diagnostic_tensors()
        max_diff = float((out_a - out_b).abs().max().item())
        if max_diff > 5e-4:
            raise RuntimeError(f"Optimized final-only gLSTM logits differ from reference by {max_diff}")
        for layer in map(str, layers):
            for key in ("slots",):
                diff = float((tensors_a[layer][key] - tensors_b[layer][key]).abs().max().item())
                if diff > 5e-4:
                    raise RuntimeError(f"Reference equivalence failed for layer {layer} {key}: {diff}")
        for key in ("total_read", "injection"):
            diff = float((tensors_a[str(max(layers))][key] - tensors_b[str(max(layers))][key]).abs().max().item())
            if diff > 5e-4:
                raise RuntimeError(f"Reference equivalence failed for layer17 {key}: {diff}")
    finally:
        reference.detach()
        optimized.detach()
        reference.clear_context()
        optimized.clear_context()


def run_correctness_tests(
    *,
    args: argparse.Namespace,
    examples: Dict[str, List[FrameMemoryExample]],
    dataset_manifest: Dict[str, Any],
    answer_ids: Dict[int, Tuple[int, ...]],
    count_values: Sequence[int],
    device: str,
    scorer_info: Dict[str, Any],
) -> Dict[str, Any]:
    args.tiny_debug_model = True
    args.device = "cpu" if str(args.device) == "cuda" else str(args.device)
    device = base.resolve_device(str(args.device))
    model, processor, _load4, _mode = load_model_and_processor(args, device=device, dtype=torch.float32)
    layers = parse_layers(args)
    hidden_size = base.hidden_size_from_model(model)
    assert_dataset(dataset_manifest["config"], examples)
    if [int(x) for x in count_values] != list(COUNT_VALUES):
        raise RuntimeError("Candidate counts are not exactly 0..12")
    batches = bucketed_batches(list(range(len(examples[TRAIN_SPLIT]))), examples[TRAIN_SPLIT], 2, 123, True)
    assert_batch_coverage(list(range(len(examples[TRAIN_SPLIT]))), batches)
    same_len = next(batch for batch in batches if len(batch) == 2)
    for variant in VARIANTS:
        adapter = make_adapter(args, variant, hidden_size=hidden_size, layers=layers).to(device)
        adapter.attach(model)
        try:
            trainable_parameter_summary(model, adapter, variant)
            batch_examples = [examples[TRAIN_SPLIT][idx] for idx in same_len]
            counter: Dict[str, int] = defaultdict(int)
            loss_b, row_loss_b, batch = run_gold_ce_batch(
                model=model,
                processor=processor,
                adapter=adapter,
                examples=batch_examples,
                sample_indices=same_len,
                answer_ids=answer_ids,
                device=device,
                counter=counter,
            )
            single_losses = []
            for idx in same_len:
                _loss_s, row_loss_s, _batch_s = run_gold_ce_batch(
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    examples=[examples[TRAIN_SPLIT][idx]],
                    sample_indices=[idx],
                    answer_ids=answer_ids,
                    device=device,
                    counter=counter,
                )
                single_losses.append(float(row_loss_s[0].item()))
            if abs(float(loss_b.detach().cpu().item()) - finite_mean(single_losses)) > 1e-4:
                raise RuntimeError(f"{variant}: batch-size-1 and batch-size-2 losses differ")
            preds_b, _scores_b = predict_count_batch(
                model=model,
                processor=processor,
                adapter=adapter,
                examples=batch_examples,
                sample_indices=same_len,
                answer_ids=answer_ids,
                count_values=count_values,
                device=device,
                scorer_info=scorer_info,
                counter=counter,
            )
            preds_s = []
            for idx in same_len:
                pred_s, _score_s = legacy_predict_count_slow(
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    example=examples[TRAIN_SPLIT][idx],
                    sample_idx=idx,
                    answer_ids=answer_ids,
                    count_values=count_values,
                    device=device,
                    scorer_info=scorer_info,
                )
                preds_s.append(pred_s)
            if [int(x) for x in preds_b] != [int(x) for x in preds_s]:
                raise RuntimeError(f"{variant}: batched predictions differ from batch-size-1")
            if variant == LORA_BASELINE and adapter.memory is not None:
                raise RuntimeError("lora_baseline has memory")
            if variant in MEMORY_VARIANTS:
                stats = adapter.memory.stats_for_row(0) if adapter.memory is not None else {}
                injected = stats.get("injected_by_layer", {})
                for layer in DEFAULT_LAYERS:
                    expected = 1 if int(layer) == max(DEFAULT_LAYERS) else 0
                    if int(injected.get(str(layer), expected)) != expected:
                        raise RuntimeError(f"{variant}: injection layer check failed at {layer}")
        finally:
            adapter.detach()
    run_scorer_equivalence(
        args=args,
        model=model,
        processor=processor,
        examples=examples,
        answer_ids=answer_ids,
        count_values=count_values,
        device=device,
        scorer_info=scorer_info,
    )
    run_reference_equivalence(
        args=args,
        processor=processor,
        examples=examples,
        answer_ids=answer_ids,
        device=device,
    )
    return {"ok": True, "checks": 20, "device": device}


def run_checked(cmd: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(str(x) for x in cmd))
    return subprocess.run(
        [str(x) for x in cmd],
        cwd=os.fspath(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def submit_slurm_jobs(args: argparse.Namespace, parent_output_root: Path) -> Dict[str, Any]:
    parent_output_root.mkdir(parents=True, exist_ok=True)
    (parent_output_root / "slurm").mkdir(parents=True, exist_ok=True)
    ensure_candidate_range(args)
    dataset_dir, _examples, dataset_manifest = ensure_dataset(args, parent_output_root / "cache")
    slurm_script = PROJECT_ROOT / "scripts" / "slurm" / "final_glstm_aggregation_comparison.sbatch"
    aggregate_script = PROJECT_ROOT / "scripts" / "slurm" / "final_glstm_aggregation_comparison_aggregate.sbatch"
    seed_jobs: Dict[str, Dict[str, Any]] = {}
    previous_job_id: Optional[str] = None
    for seed, task_range in (("0", "0-2"), ("1", "3-5"), ("2", "6-8")):
        cmd = ["sbatch", "--parsable", "--array", task_range]
        dependency: Optional[str] = None
        if previous_job_id is not None:
            dependency = f"afterok:{previous_job_id}"
            cmd.extend(["--dependency", dependency])
        cmd.append(os.fspath(slurm_script))
        result = run_checked(cmd, cwd=PROJECT_ROOT)
        job_id = result.stdout.strip().splitlines()[-1].split(";")[0]
        seed_jobs[seed] = {
            "job_id": job_id,
            "array": task_range,
            "dependency": dependency,
            "variants": list(VARIANTS),
            "slurm_stdout": os.fspath(parent_output_root / "slurm" / f"fgac_array-{job_id}_%a.out"),
            "slurm_stderr": os.fspath(parent_output_root / "slurm" / f"fgac_array-{job_id}_%a.err"),
        }
        previous_job_id = job_id
    if previous_job_id is None:
        raise RuntimeError("No seed jobs were submitted")
    aggregate_result = run_checked(
        [
            "sbatch",
            "--parsable",
            "--dependency",
            f"afterok:{previous_job_id}",
            os.fspath(aggregate_script),
        ],
        cwd=PROJECT_ROOT,
    )
    aggregate_job_id = aggregate_result.stdout.strip().splitlines()[-1].split(";")[0]
    submitted = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": os.fspath(parent_output_root),
        "dataset_dir": os.fspath(dataset_dir),
        "dataset_hash": dataset_manifest.get("dataset_hash"),
        "seed_jobs": seed_jobs,
        "aggregate_job": {
            "job_id": aggregate_job_id,
            "dependency": f"afterok:{previous_job_id}",
            "slurm_stdout": os.fspath(parent_output_root / "slurm" / f"fgac_aggregate-{aggregate_job_id}.out"),
            "slurm_stderr": os.fspath(parent_output_root / "slurm" / f"fgac_aggregate-{aggregate_job_id}.err"),
        },
    }
    write_json(parent_output_root / "submitted_jobs.json", submitted)
    return submitted


def main() -> int:
    args = parse_args()
    parent_output_root = Path(args.output_root).resolve()
    parent_output_root.mkdir(parents=True, exist_ok=True)
    count_values = ensure_candidate_range(args)
    if bool(args.aggregate_only):
        summary = aggregate_parent_outputs(parent_output_root)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if bool(args.submit_slurm):
        submitted = submit_slurm_jobs(args, parent_output_root)
        print(json.dumps(submitted, indent=2, sort_keys=True))
        return 0
    if bool(args.smoke_test):
        args.tiny_debug_model = True
        if str(args.device) == "cuda":
            args.device = "cpu"
        args.epochs = min(int(args.epochs), 2)
        args.batch_size = min(int(args.batch_size), 2)
        args.eval_batch_size = min(int(args.eval_batch_size), 2)
        args.grad_accum = 1
        args.max_train_examples = 6 if int(args.max_train_examples) <= 0 else min(int(args.max_train_examples), 6)
        args.max_eval_examples = 4 if int(args.max_eval_examples) <= 0 else min(int(args.max_eval_examples), 4)
    if bool(args.real_qwen_smoke):
        args.epochs = min(int(args.epochs), 2)
        args.max_train_examples = 6 if int(args.max_train_examples) <= 0 else min(int(args.max_train_examples), 6)
        args.max_eval_examples = 4 if int(args.max_eval_examples) <= 0 else min(int(args.max_eval_examples), 4)
    dataset_dir, examples, dataset_manifest = ensure_dataset(args, parent_output_root / "cache")
    dataset_manifest = {**dataset_manifest, "dataset_dir": os.fspath(dataset_dir)}
    if bool(args.prepare_dataset_only):
        print(f"Prepared dataset {dataset_manifest['dataset_hash']} at {dataset_dir}")
        print(json.dumps({split: payload["n"] for split, payload in dataset_manifest["splits"].items()}, indent=2, sort_keys=True))
        return 0
    device = base.resolve_device(str(args.device))
    dtype = base.dtype_from_arg(str(args.dtype), device)
    model, processor, load_in_4bit, load_mode = load_model_and_processor(args, device=device, dtype=dtype)
    tokenizer = processor.tokenizer
    tokenization_mode, answer_ids = base.text_base.answer_token_ids(tokenizer, int(args.candidate_min), int(args.candidate_max))
    scorer_info = analyze_answer_ids(answer_ids, count_values)
    print(f"Model load mode={load_mode} load_in_4bit={load_in_4bit} answer_tokenization={tokenization_mode}")
    print(json.dumps(scorer_info, indent=2, sort_keys=True))
    if bool(args.run_correctness_tests):
        result = run_correctness_tests(
            args=args,
            examples=examples,
            dataset_manifest=dataset_manifest,
            answer_ids=answer_ids,
            count_values=count_values,
            device=device,
            scorer_info=scorer_info,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if bool(args.real_qwen_smoke):
        run_scorer_equivalence(
            args=args,
            model=model,
            processor=processor,
            examples=examples,
            answer_ids=answer_ids,
            count_values=count_values,
            device=device,
            scorer_info=scorer_info,
        )
        print("Real-Qwen smoke scorer equivalence passed")
    variants = parse_variants(args)
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
                count_values=count_values,
                device=device,
                scorer_info=scorer_info,
                timestamp=timestamp,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(parent_output_root / f"{timestamp}_run_summaries.json", run_summaries)
    if bool(args.real_qwen_smoke):
        marker = parent_output_root / "SMOKE_SUCCESS"
        marker.write_text(json.dumps({"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": run_summaries}, indent=2) + "\n", encoding="utf-8")
    if not bool(args.no_aggregate_after_run) and not bool(args.smoke_test) and not bool(args.real_qwen_smoke):
        aggregate_parent_outputs(parent_output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

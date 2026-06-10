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
from torch import nn

from experiments import layerwise_frame_message_glstm as base


EXPERIMENT_NAME = "layerwise_glstm_mechanism_ablation"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_LAYERS = (14, 15, 16, 17)

TRAIN_SPLIT = "train"
VAL_SPLIT = "iid_val"
IID_TEST_SPLIT = "iid_test"
LENGTH_INTERPOLATION_SPLIT = "length_interpolation_test"
LENGTH_EXTRAPOLATION_SPLIT = "length_extrapolation_test"
PAIRED_EXTENSION_SPLIT = "paired_neutral_extension_test"
MAIN_EVAL_SPLITS = (
    TRAIN_SPLIT,
    VAL_SPLIT,
    IID_TEST_SPLIT,
    LENGTH_INTERPOLATION_SPLIT,
    LENGTH_EXTRAPOLATION_SPLIT,
)

GLSTM_LAYERWISE_PERSISTENT = "glstm_layerwise_persistent"
GLSTM_LAYERWISE_FRESH = "glstm_layerwise_fresh"
GLSTM_FINAL_ONLY_PERSISTENT = "glstm_final_only_persistent"
DIRECT_SUM_LAYERWISE = "direct_sum_layerwise"
VARIANTS = (
    GLSTM_LAYERWISE_PERSISTENT,
    GLSTM_LAYERWISE_FRESH,
    GLSTM_FINAL_ONLY_PERSISTENT,
    DIRECT_SUM_LAYERWISE,
)
ASSOCIATIVE_VARIANTS = {
    GLSTM_LAYERWISE_PERSISTENT,
    GLSTM_LAYERWISE_FRESH,
    GLSTM_FINAL_ONLY_PERSISTENT,
}
VARIANT_ALIASES = {
    "all": "all",
    "persistent": GLSTM_LAYERWISE_PERSISTENT,
    "layerwise_persistent": GLSTM_LAYERWISE_PERSISTENT,
    "glstm_layerwise_persistent": GLSTM_LAYERWISE_PERSISTENT,
    "fresh": GLSTM_LAYERWISE_FRESH,
    "layerwise_fresh": GLSTM_LAYERWISE_FRESH,
    "glstm_layerwise_fresh": GLSTM_LAYERWISE_FRESH,
    "final": GLSTM_FINAL_ONLY_PERSISTENT,
    "final_only": GLSTM_FINAL_ONLY_PERSISTENT,
    "glstm_final_only_persistent": GLSTM_FINAL_ONLY_PERSISTENT,
    "direct": DIRECT_SUM_LAYERWISE,
    "direct_sum": DIRECT_SUM_LAYERWISE,
    "direct_sum_layerwise": DIRECT_SUM_LAYERWISE,
}

MISSING = math.nan
PNG_DPI = 160


FrameMemoryExample = base.FrameMemoryExample
FrameMemoryBatch = base.FrameMemoryBatch
ExperimentAdapter = base.ExperimentAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused mechanism ablation for layerwise gLSTM frame-message memory."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--source-dataset-root", type=Path, default=base.DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--fallback-source-dataset-root", type=Path, default=base.DEFAULT_FALLBACK_SOURCE_DATASET_ROOT)
    parser.add_argument("--source-split", default="all_uniform")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--variants", nargs="+", default=[",".join(VARIANTS)])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=24680)
    parser.add_argument("--force-regenerate-dataset", action="store_true", default=False)
    parser.add_argument("--train-per-count", type=int, default=20)
    parser.add_argument("--val-per-count", type=int, default=20)
    parser.add_argument("--iid-test-per-count", type=int, default=20)
    parser.add_argument("--interpolation-test-per-count", type=int, default=20)
    parser.add_argument("--extrapolation-test-per-count", type=int, default=20)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
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
    parser.add_argument("--reconstruction-tol", type=float, default=5e-3)
    parser.add_argument("--fail-on-reconstruction-error", action=argparse.BooleanOptionalAction, default=False)

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

    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--smoke-test", action="store_true", default=False)
    parser.add_argument("--tiny-debug-model", action="store_true", default=False)
    parser.add_argument("--tiny-num-layers", type=int, default=18)
    parser.add_argument("--tiny-hidden-size", type=int, default=64)
    parser.add_argument("--tiny-num-heads", type=int, default=4)
    parser.add_argument("--prepare-dataset-only", action="store_true", default=False)
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--no-aggregate-after-run", action="store_true", default=False)
    parser.add_argument("--submit-slurm", action="store_true", default=False)
    parser.add_argument("--skip-submit-smoke", action="store_true", default=False)
    return parser.parse_args()


def parse_variants(raw_values: Sequence[Any]) -> List[str]:
    variants: List[str] = []
    for raw in raw_values:
        for token in re.split(r"[,:]", str(raw)):
            token = token.strip()
            if not token:
                continue
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


def safe_name(value: Any) -> str:
    return base.safe_name(value)


def finite_mean(values: Iterable[Any], default: float = math.nan) -> float:
    return base.finite_mean(values, default=default)


def finite_float(value: Any) -> Optional[float]:
    return base.finite_float(value)


def write_json(path: Path, payload: Any) -> None:
    base.write_json(path, payload)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    base.write_jsonl(path, rows)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return base.read_jsonl(path)


def write_csv_dynamic(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str] = ()) -> None:
    base.write_csv_dynamic(path, rows, leading=leading)


def save_plot(path: Path, fig: Any) -> None:
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)


def ensure_candidate_range(args: argparse.Namespace) -> List[int]:
    if int(args.candidate_min) != 0 or int(args.candidate_max) != 8:
        raise ValueError("This experiment must use candidate counts exactly 0..8")
    return list(range(0, 9))


def frame_identity(ref: Dict[str, Any]) -> str:
    return f"{ref.get('source_sample_id')}:{int(ref.get('source_frame_index', -1))}:{ref.get('frame_path')}"


def choose_unique_refs(
    rng: random.Random,
    pool: Sequence[Dict[str, Any]],
    n: int,
    forbidden: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    if int(n) <= 0:
        return []
    forbidden = set() if forbidden is None else set(forbidden)
    unique = [dict(ref) for ref in pool if frame_identity(ref) not in forbidden]
    rng.shuffle(unique)
    if len(unique) >= int(n):
        return [dict(ref) for ref in unique[: int(n)]]
    refs = [dict(ref) for ref in unique]
    fallback = [dict(ref) for ref in pool]
    if not fallback:
        raise RuntimeError("Cannot choose refs from an empty pool")
    while len(refs) < int(n):
        refs.append(dict(rng.choice(fallback)))
    return refs


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
) -> List[FrameMemoryExample]:
    rng = random.Random(int(seed))
    examples: List[FrameMemoryExample] = []
    for length in [int(x) for x in lengths]:
        schedule: List[int] = [
            int(count)
            for count in counts_by_length[int(length)]
            for _ in range(int(examples_per_cell))
        ]
        rng.shuffle(schedule)
        valid_pairs = sorted(
            pair
            for pair, pair_pool in pools[int(length)][str(source_partition_name)].items()
            if pair_pool.get("evidence") and pair_pool.get("neutral")
        )
        if not valid_pairs:
            raise RuntimeError(f"{split}: no valid character-room pairs for length={length} partition={source_partition_name}")
        for local_idx, gold_count in enumerate(schedule):
            if not (0 <= int(gold_count) <= int(length)):
                raise RuntimeError(f"{split}: invalid count {gold_count} for length {length}")
            character, room = rng.choice(valid_pairs)
            pair_pool = pools[int(length)][str(source_partition_name)][(character, room)]
            evidence_positions = tuple(sorted(rng.sample(range(int(length)), int(gold_count))))
            evidence_set = set(evidence_positions)
            evidence_refs = choose_unique_refs(rng, pair_pool["evidence"], int(gold_count))
            neutral_refs = choose_unique_refs(rng, pair_pool["neutral"], int(length) - int(gold_count))
            rng.shuffle(evidence_refs)
            rng.shuffle(neutral_refs)
            ordered_refs: List[Dict[str, Any]] = []
            evidence_cursor = 0
            neutral_cursor = 0
            for frame_idx in range(int(length)):
                if frame_idx in evidence_set:
                    ref = dict(evidence_refs[evidence_cursor])
                    evidence_cursor += 1
                    ref["mechanism_role"] = "evidence"
                else:
                    ref = dict(neutral_refs[neutral_cursor])
                    neutral_cursor += 1
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
        enriched: List[Dict[str, Any]] = []
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
    num_families: int = 24,
) -> List[FrameMemoryExample]:
    rng = random.Random(int(seed))
    base_length = 8
    version_lengths = (8, 10, 12, 16)
    partition = "test"
    valid_pairs = sorted(
        pair
        for pair, pair_pool in pools[base_length][partition].items()
        if pair_pool.get("evidence") and pair_pool.get("neutral")
    )
    if not valid_pairs:
        raise RuntimeError("paired neutral extension: no valid test character-room pairs at length 8")
    counts = list(range(9)) * (num_families // 9)
    counts.extend(list(range(num_families - len(counts))))
    rng.shuffle(counts)
    examples: List[FrameMemoryExample] = []
    for family_idx, gold_count in enumerate(counts[:num_families]):
        character, room = rng.choice(valid_pairs)
        pair_pool = pools[base_length][partition][(character, room)]
        evidence_positions_base = tuple(sorted(rng.sample(range(base_length), int(gold_count))))
        evidence_set_base = set(evidence_positions_base)
        evidence_refs = choose_unique_refs(rng, pair_pool["evidence"], int(gold_count))
        neutral_refs = choose_unique_refs(rng, pair_pool["neutral"], base_length - int(gold_count))
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
        evidence_identities = [frame_identity(ref) for ref in base_order if ref["mechanism_role"] == "evidence"]
        base_sample_id = f"paired_family_{family_idx:03d}_base_len8_count{int(gold_count)}"
        family_id = f"family_{family_idx:03d}"
        template_id = rng.choice(tuple(templates))
        question = base.question_for_template(template_id, character, room)
        used = {frame_identity(ref) for ref in base_order}
        for version_length in version_lengths:
            extra_count = int(version_length) - base_length
            added_refs = choose_unique_refs(rng, pair_pool["neutral"], extra_count, forbidden=used)
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
            final_ordering = [
                {
                    "position": int(i),
                    "identity": frame_identity(ref),
                    "role": str(ref.get("mechanism_role")),
                    "base_order_index": ref.get("base_order_index"),
                }
                for i, ref in enumerate(final_refs)
            ]
            paired_meta = {
                "family_id": family_id,
                "base_sample_id": base_sample_id,
                "version_length": int(version_length),
                "original_evidence_frame_identities": list(evidence_identities),
                "new_neutral_frame_identities": list(new_neutral_ids),
                "final_evidence_positions": list(final_evidence_positions),
                "final_frame_ordering": final_ordering,
            }
            enriched_refs = []
            for ref in final_refs:
                ref = dict(ref)
                ref["paired_metadata"] = paired_meta
                enriched_refs.append(ref)
            examples.append(
                FrameMemoryExample(
                    example_id=f"{PAIRED_EXTENSION_SPLIT}_{family_id}_len{int(version_length)}_count{int(gold_count)}",
                    split=PAIRED_EXTENSION_SPLIT,
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


def requested_output_lengths() -> List[int]:
    return [4, 5, 6, 7, 8, 10, 12]


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
    missing = sorted(set(int(x) for x in lengths) - set(available))
    if not available:
        raise RuntimeError(f"No requested source lengths are available for {lengths}")
    pools, source_manifest = base.scan_source_frame_pools(
        source_root,
        fallback_root,
        str(args.source_split),
        int(args.dataset_seed),
        available,
    )
    merged: Optional[Dict[str, Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]]] = None
    if missing:
        merged = base.merged_frame_pools_for_partition(pools, sorted(pools))
        for length in missing:
            pools[int(length)] = merged
        source_manifest["synthesized_missing_output_lengths"] = missing
        source_manifest["synthesized_length_rule"] = (
            "Missing requested output lengths are assembled from merged frame pools "
            "within the requested source partition; no frame crosses source train/val/test partitions."
        )
    synthesized_for_empty_partitions: List[int] = []
    for length in sorted({int(x) for x in lengths}):
        required_partitions = {"test"}
        if int(length) in {4, 6, 8}:
            required_partitions |= {"train", "val"}
        has_required = all(base.valid_pairs_for_lengths(pools, [int(length)], partition) for partition in required_partitions)
        if not has_required:
            if merged is None:
                source_lengths = sorted(k for k in pools if int(k) != int(length))
                merged = base.merged_frame_pools_for_partition(pools, source_lengths or sorted(pools))
            pools[int(length)] = merged
            synthesized_for_empty_partitions.append(int(length))
    if synthesized_for_empty_partitions:
        source_manifest["synthesized_empty_partition_output_lengths"] = synthesized_for_empty_partitions
    return pools, source_manifest, sorted(set(missing) | set(synthesized_for_empty_partitions))


def dataset_config(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(getattr(args, "smoke_test", False)):
        train_per_count = min(int(args.train_per_count), 1)
        val_per_count = min(int(args.val_per_count), 1)
        iid_test_per_count = min(int(args.iid_test_per_count), 1)
        interpolation_test_per_count = min(int(args.interpolation_test_per_count), 1)
        extrapolation_test_per_count = min(int(args.extrapolation_test_per_count), 1)
    else:
        train_per_count = int(args.train_per_count)
        val_per_count = int(args.val_per_count)
        iid_test_per_count = int(args.iid_test_per_count)
        interpolation_test_per_count = int(args.interpolation_test_per_count)
        extrapolation_test_per_count = int(args.extrapolation_test_per_count)
    per_count_values = {
        "train_per_count": train_per_count,
        "val_per_count": val_per_count,
        "iid_test_per_count": iid_test_per_count,
        "interpolation_test_per_count": interpolation_test_per_count,
        "extrapolation_test_per_count": extrapolation_test_per_count,
    }
    if any(value <= 0 for value in per_count_values.values()):
        raise ValueError(f"Per-count sizes must be positive: {per_count_values}")
    return {
        "dataset_seed": int(args.dataset_seed),
        "source_dataset_root": os.fspath(Path(args.source_dataset_root).resolve()),
        "fallback_source_dataset_root": os.fspath(Path(args.fallback_source_dataset_root).resolve())
        if args.fallback_source_dataset_root is not None
        else None,
        "source_split": str(args.source_split),
        "splits": {
            TRAIN_SPLIT: {
                "lengths": [4, 6, 8],
                "counts_by_length": {"4": list(range(5)), "6": list(range(7)), "8": list(range(9))},
                "examples_per_cell": train_per_count,
                "source_partition": "train",
            },
            VAL_SPLIT: {
                "lengths": [4, 6, 8],
                "counts_by_length": {"4": list(range(5)), "6": list(range(7)), "8": list(range(9))},
                "examples_per_cell": val_per_count,
                "source_partition": "val",
            },
            IID_TEST_SPLIT: {
                "lengths": [4, 6, 8],
                "counts_by_length": {"4": list(range(5)), "6": list(range(7)), "8": list(range(9))},
                "examples_per_cell": iid_test_per_count,
                "source_partition": "test",
            },
            LENGTH_INTERPOLATION_SPLIT: {
                "lengths": [5, 7],
                "counts_by_length": {"5": list(range(6)), "7": list(range(8))},
                "examples_per_cell": interpolation_test_per_count,
                "source_partition": "test",
            },
            LENGTH_EXTRAPOLATION_SPLIT: {
                "lengths": [10, 12],
                "counts_by_length": {"10": list(range(9)), "12": list(range(9))},
                "examples_per_cell": extrapolation_test_per_count,
                "source_partition": "test",
            },
            PAIRED_EXTENSION_SPLIT: {
                "lengths": [8, 10, 12, 16],
                "families": 24,
                "versions_per_family": 4,
                "source_partition": "test",
            },
        },
        "neutral_rule": "queried character absent and queried room empty",
        "evidence_positions_randomized": True,
        "hard_semantic_distractors": False,
        "composition_ood_removed": True,
        "heldout_compositions": [],
        "candidate_counts": list(range(9)),
        **per_count_values,
    }


def expected_split_size(split_cfg: Dict[str, Any]) -> int:
    if "families" in split_cfg:
        return int(split_cfg["families"]) * int(split_cfg["versions_per_family"])
    total = 0
    for length in split_cfg["lengths"]:
        total += len(split_cfg["counts_by_length"][str(int(length))]) * int(split_cfg["examples_per_cell"])
    return total


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
        pools, source_manifest, synthesized_lengths = scan_or_synthesize_source_pools(args, requested_output_lengths())
        generated: Dict[str, List[FrameMemoryExample]] = {}
        seed_offsets = {
            TRAIN_SPLIT: 11,
            VAL_SPLIT: 23,
            IID_TEST_SPLIT: 59,
            LENGTH_INTERPOLATION_SPLIT: 83,
            LENGTH_EXTRAPOLATION_SPLIT: 107,
            PAIRED_EXTENSION_SPLIT: 131,
        }
        for split, split_cfg in config["splits"].items():
            if split == PAIRED_EXTENSION_SPLIT:
                generated[split] = build_paired_neutral_extension(
                    templates=base.visual_base.TRAIN_TEMPLATES,
                    pools=pools,
                    seed=int(args.dataset_seed) + seed_offsets[split],
                    num_families=int(split_cfg["families"]),
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
                )
            write_jsonl(split_paths[split], [example_to_json(example) for example in generated[split]])
        manifest = {
            "dataset_hash": digest,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config,
            "source_manifest": source_manifest,
            "synthesized_output_lengths_from_merged_frame_pools": synthesized_lengths,
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
                            if split != PAIRED_EXTENSION_SPLIT
                            else range(9)
                        )
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
    for split, split_cfg in config["splits"].items():
        rows = examples.get(split, [])
        expected_n = expected_split_size(split_cfg)
        if len(rows) != expected_n:
            raise RuntimeError(f"{split}: expected {expected_n} examples, found {len(rows)}")
        seen = {row.example_id for row in rows}
        if len(seen) != len(rows):
            raise RuntimeError(f"{split}: duplicate sample IDs")
        if all_ids & seen:
            raise RuntimeError(f"{split}: sample IDs overlap another split")
        all_ids |= seen
        expected_lengths = set(int(x) for x in split_cfg["lengths"])
        for row in rows:
            if int(row.num_frames) not in expected_lengths:
                raise RuntimeError(f"{split}: unexpected length {row.num_frames}")
            if not (0 <= int(row.gold_count) <= 8):
                raise RuntimeError(f"{split}: gold count outside candidate/training label range: {row.gold_count}")
            if split != LENGTH_EXTRAPOLATION_SPLIT and split != PAIRED_EXTENSION_SPLIT:
                if int(row.gold_count) > int(row.num_frames):
                    raise RuntimeError(f"{split}: invalid count {row.gold_count} for length {row.num_frames}")
            if len(row.frame_paths) != int(row.num_frames):
                raise RuntimeError(f"{split}: frame path count mismatch for {row.example_id}")
            if len(set(row.evidence_frame_indices)) != int(row.gold_count):
                raise RuntimeError(f"{split}: evidence index/count mismatch for {row.example_id}")
            for frame_path in row.frame_paths:
                resolved = base.resolve_frame_path(frame_path)
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
        if split != PAIRED_EXTENSION_SPLIT:
            for length in split_cfg["lengths"]:
                counts = [int(x) for x in split_cfg["counts_by_length"][str(int(length))]]
                for count in counts:
                    n = sum(int(row.num_frames) == int(length) and int(row.gold_count) == int(count) for row in rows)
                    if n != int(split_cfg["examples_per_cell"]):
                        raise RuntimeError(
                            f"{split}: expected {split_cfg['examples_per_cell']} examples for length={length} count={count}, found {n}"
                        )
    assert_paired_extension(examples[PAIRED_EXTENSION_SPLIT])


def assert_paired_extension(rows: Sequence[FrameMemoryExample]) -> None:
    by_family: Dict[str, List[FrameMemoryExample]] = defaultdict(list)
    for row in rows:
        meta = _paired_metadata_from_example(row)
        family_id = str(meta.get("family_id", ""))
        if not family_id:
            raise RuntimeError(f"{row.example_id}: missing paired family_id")
        by_family[family_id].append(row)
    if len(by_family) != 24:
        raise RuntimeError(f"paired set expected 24 families, found {len(by_family)}")
    seen_counts = set()
    for family_id, family_rows in by_family.items():
        lengths = sorted(int(row.num_frames) for row in family_rows)
        if lengths != [8, 10, 12, 16]:
            raise RuntimeError(f"{family_id}: expected lengths 8/10/12/16, found {lengths}")
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
                raise RuntimeError(f"{family_id}: final evidence positions metadata mismatch")
    if seen_counts != set(range(9)):
        raise RuntimeError(f"paired set must cover counts 0..8, found {sorted(seen_counts)}")


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    keep = np.isfinite(scores)
    labels = labels[keep]
    scores = scores[keep]
    positives = int(labels.sum())
    negatives = int(labels.shape[0] - positives)
    if positives == 0 or negatives == 0:
        return MISSING
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    label_ranks = ranks[inv]
    sum_pos_ranks = float(label_ranks[labels == 1].sum())
    return float((sum_pos_ranks - positives * (positives + 1) / 2.0) / (positives * negatives))


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if not bool(mask.any()):
        return MISSING
    return float(values.detach().float()[mask].mean().cpu().item())


def masked_norm_total(contrib: torch.Tensor, mask: torch.Tensor) -> float:
    mask_f = mask.unsqueeze(-1).float()
    if not bool(mask.any()):
        return MISSING
    total = (contrib.detach().float() * mask_f).sum(dim=2).norm(dim=-1)
    active_carriers = mask.any(dim=2)
    return float(total[active_carriers].mean().cpu().item()) if bool(active_carriers.any()) else MISSING


def slot_rank_and_cosine(slots: torch.Tensor, valid: torch.Tensor) -> Tuple[float, float]:
    flat = slots.detach().float()[valid.bool()]
    if int(flat.shape[0]) == 0:
        return 0.0, MISSING
    norms = flat.norm(dim=-1)
    nonzero = flat[norms > 1e-8]
    if int(nonzero.shape[0]) < 2:
        return float(int(nonzero.shape[0])), MISSING
    normed = F.normalize(nonzero, dim=-1)
    pairwise = normed @ normed.T
    upper = pairwise.triu(diagonal=1)
    denom = int(nonzero.shape[0]) * (int(nonzero.shape[0]) - 1) / 2
    cosine = float(upper.sum().cpu().item() / max(1.0, denom))
    singular = torch.linalg.svdvals(nonzero)
    rank = float((singular > singular.max().clamp_min(1e-8) * 1e-3).sum().cpu().item())
    return rank, cosine


class MechanismFrameMessageMemory(base.LayerwiseFrameMessageMemory):
    def __init__(self, *, variant: str, **kwargs: Any) -> None:
        super().__init__(variant=variant, **kwargs)
        self.disabled_injection_layers: set[int] = set()
        self.read_reconstruction_tol = 1e-4
        if self.variant == DIRECT_SUM_LAYERWISE:
            for module in [*self.w_q, *self.w_k, *self.w_v]:
                for param in module.parameters():
                    param.requires_grad_(False)
        else:
            for module in self.w_sum:
                for param in module.parameters():
                    param.requires_grad_(False)

    def set_context(self, batch: FrameMemoryBatch) -> None:
        super().set_context(batch)
        self._last_stats.update(
            {
                "mean_evidence_compatibility_by_layer": {},
                "mean_neutral_compatibility_by_layer": {},
                "compatibility_auroc_by_layer": {},
                "mean_evidence_contribution_norm_by_layer": {},
                "mean_neutral_contribution_norm_by_layer": {},
                "contribution_norm_auroc_by_layer": {},
                "total_evidence_contribution_norm_by_layer": {},
                "total_neutral_contribution_norm_by_layer": {},
                "evidence_to_neutral_contribution_ratio_by_layer": {},
                "read_reconstruction_error_by_layer": {},
                "previous_slot_norm_by_layer": {},
                "previous_slot_used_by_layer": {},
                "injected_by_layer": {},
            }
        )

    def _evidence_mask(self, valid: torch.Tensor) -> torch.Tensor:
        evidence = torch.zeros_like(valid, dtype=torch.bool)
        if self._evidence_frame_indices is None:
            return evidence
        for b, indices in enumerate(self._evidence_frame_indices):
            for frame_idx in indices:
                if 0 <= int(frame_idx) < valid.shape[2]:
                    evidence[b, :, int(frame_idx)] = True
        return evidence & valid.bool()

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
        if self.variant == GLSTM_LAYERWISE_FRESH:
            candidate = self.slot_norm[wpos](z.float())
            previous_used = False
        else:
            candidate = self.slot_norm[wpos](previous_slots + z.float())
            previous_used = True
        self._slots = torch.where(valid.unsqueeze(-1), candidate, previous_slots)
        slots_for_read, valid_for_read = self._apply_ablation(self._slots, valid, int(layer_idx))
        evidence_mask = self._evidence_mask(valid_for_read)
        neutral_mask = valid_for_read.bool() & ~evidence_mask
        compatibilities: Optional[torch.Tensor] = None
        keys: Optional[torch.Tensor] = None
        values: Optional[torch.Tensor] = None
        queries: Optional[torch.Tensor] = None
        if self.variant == DIRECT_SUM_LAYERWISE:
            per_frame_read = self.w_sum[mpos](slots_for_read).float() * valid_for_read.unsqueeze(-1).float()
            read = per_frame_read.sum(dim=2)
            matrix_shape: List[int] = list(read.shape)
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
        default_should_inject = self.variant != GLSTM_FINAL_ONLY_PERSISTENT or int(layer_idx) == max(self.layers)
        should_inject = default_should_inject and int(layer_idx) not in self.disabled_injection_layers
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
                for c in range(injection.shape[1]):
                    state_norm = float(carrier_states[b, c].detach().float().norm().cpu().item())
                    carrier_norms.append(state_norm)
                    injection_norms.append(0.0)
                    ratios.append(0.0)
        layer_key = str(int(layer_idx))
        self.reconstruction_errors[int(layer_idx)].append(float(recon_error))
        if bool(torch.isfinite(torch.tensor(recon_error))) and float(recon_error) > self.reconstruction_tol and self.message_mode == "exact":
            message = f"Attention message reconstruction error {float(recon_error):.6g} exceeds tolerance {self.reconstruction_tol}"
            self._record_exact_failure(f"layer {layer_idx}: {message}")
            if self.fail_on_reconstruction_error:
                raise RuntimeError(message)
        slot_float = self._slots.detach().float()
        rank, cosine = slot_rank_and_cosine(slot_float, valid_for_read)
        read_norm = read.detach().float().norm(dim=-1)
        contribution_norm = per_frame_read.detach().float().norm(dim=-1)
        raw_message_norm = messages.detach().float().norm(dim=-1)
        slot_norm = slot_float.norm(dim=-1)
        evidence_total = masked_norm_total(per_frame_read, evidence_mask)
        neutral_total = masked_norm_total(per_frame_read, neutral_mask)
        ratio = (
            float(evidence_total / max(neutral_total, 1e-8))
            if math.isfinite(evidence_total) and math.isfinite(neutral_total)
            else MISSING
        )
        contrib_labels = torch.cat(
            [torch.ones(int(evidence_mask.sum().item())), torch.zeros(int(neutral_mask.sum().item()))]
        ).numpy()
        contrib_scores = torch.cat(
            [contribution_norm[evidence_mask].cpu(), contribution_norm[neutral_mask].cpu()]
        ).numpy() if (bool(evidence_mask.any()) or bool(neutral_mask.any())) else np.asarray([])
        contribution_auc = roc_auc_binary(contrib_labels, contrib_scores) if contrib_scores.size else MISSING
        compatibility_auc = MISSING
        evidence_compat = MISSING
        neutral_compat = MISSING
        if compatibilities is not None:
            evidence_compat = masked_mean(compatibilities, evidence_mask)
            neutral_compat = masked_mean(compatibilities, neutral_mask)
            comp_scores = torch.cat(
                [compatibilities[evidence_mask].detach().float().cpu(), compatibilities[neutral_mask].detach().float().cpu()]
            ).numpy() if (bool(evidence_mask.any()) or bool(neutral_mask.any())) else np.asarray([])
            compatibility_auc = roc_auc_binary(contrib_labels, comp_scores) if comp_scores.size else MISSING
        previous_norm = masked_mean(previous_slots.detach().float().norm(dim=-1), valid_for_read)
        stats_updates = {
            "raw_message_norm_by_layer": masked_mean(raw_message_norm, valid_for_read),
            "slot_norm_by_layer": masked_mean(slot_norm, valid_for_read),
            "read_norm_by_layer": float(read_norm.mean().cpu().item()),
            "injection_norm_by_layer": finite_mean(injection_norms, default=0.0),
            "carrier_state_norm_by_layer": finite_mean(carrier_norms, default=0.0),
            "injection_to_carrier_ratio_by_layer": finite_mean(ratios, default=0.0),
            "gamma_by_layer": float(self.gamma[self.layer_to_pos[int(layer_idx)]].detach().cpu().item()),
            "effective_rank_by_layer": rank,
            "slot_cosine_by_layer": cosine,
            "reconstruction_error_by_layer": float(recon_error),
            "message_mode_by_layer": mode,
            "mean_evidence_compatibility_by_layer": evidence_compat,
            "mean_neutral_compatibility_by_layer": neutral_compat,
            "compatibility_auroc_by_layer": compatibility_auc,
            "mean_evidence_contribution_norm_by_layer": masked_mean(contribution_norm, evidence_mask),
            "mean_neutral_contribution_norm_by_layer": masked_mean(contribution_norm, neutral_mask),
            "contribution_norm_auroc_by_layer": contribution_auc,
            "total_evidence_contribution_norm_by_layer": evidence_total,
            "total_neutral_contribution_norm_by_layer": neutral_total,
            "evidence_to_neutral_contribution_ratio_by_layer": ratio,
            "read_reconstruction_error_by_layer": reconstructed_read_error,
            "previous_slot_norm_by_layer": previous_norm,
            "previous_slot_used_by_layer": int(previous_used),
            "injected_by_layer": int(should_inject),
        }
        for key, value in stats_updates.items():
            self._last_stats[key][layer_key] = [value]
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
            "evidence_frame_mask": evidence_mask.detach().cpu(),
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
    if bool(args.frame_kv_lora):
        raise NotImplementedError("--frame-kv-lora is not part of this focused ablation")
    lora = base.AttentionLoRAAdapter(
        inject_layers=layers,
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        dropout=float(args.lora_dropout),
        target_modules=("q_proj", "o_proj"),
        gated=True,
    )
    memory = MechanismFrameMessageMemory(
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


def unexpected_frozen_model_grads(model: Any) -> List[str]:
    return base.unexpected_frozen_model_grads(model)


def answer_sequence_cross_entropy(logits: torch.Tensor, batch: FrameMemoryBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    return base.answer_sequence_cross_entropy(logits, batch)


def candidate_score_from_logits(logits: torch.Tensor, batch: FrameMemoryBatch) -> float:
    return base.candidate_score_from_logits(logits, batch)


def limited_indices(examples: Sequence[Any], limit: int, seed: int) -> List[int]:
    return base.limited_indices(examples, int(limit), int(seed))


def prepare_batch(
    *,
    examples: Sequence[FrameMemoryExample],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
    answer_ids: Optional[Dict[int, Tuple[int, ...]]] = None,
    answer_count_override: Optional[int] = None,
) -> FrameMemoryBatch:
    return base.prepare_batch(
        examples=examples,
        sample_indices=sample_indices,
        processor=processor,
        device=device,
        answer_ids=answer_ids,
        answer_count_override=answer_count_override,
    )


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
    if [int(x) for x in count_values] != list(range(9)):
        raise RuntimeError("Candidate counts must be exactly 0..8")
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
            if adapter.memory is not None and adapter.memory._slots is not None:
                raise RuntimeError("Memory slots were not reset before candidate scoring forward")
        outputs = model(**batch.inputs, use_cache=False)
        if adapter is not None:
            adapter.clear_context()
            if adapter.memory is not None and adapter.memory._slots is not None:
                raise RuntimeError("Memory slots were not cleared after candidate scoring forward")
        scores[str(int(candidate))] = candidate_score_from_logits(outputs.logits, batch)
    pred = max((int(key) for key in scores), key=lambda value: scores[str(value)])
    return int(pred), scores


def trainable_parameter_summary(model: Any, adapter: ExperimentAdapter, variant: str) -> Dict[str, Any]:
    summary = base.trainable_parameter_summary(model, adapter)
    names = [str(name) for name in summary.get("trainable_parameter_names", [])]
    if variant == DIRECT_SUM_LAYERWISE:
        forbidden = [name for name in names if ".w_q." in name or ".w_k." in name or ".w_v." in name]
        if forbidden:
            raise RuntimeError(f"Direct-sum variant has trainable associative read parameters: {forbidden[:20]}")
    else:
        forbidden = [name for name in names if ".w_sum." in name]
        if forbidden:
            raise RuntimeError(f"Associative variant has trainable direct-sum parameters: {forbidden[:20]}")
    return summary


def batch_indices(indices: Sequence[int], batch_size: int, seed: int, shuffle: bool) -> List[List[int]]:
    return base.batch_indices(indices, int(batch_size), int(seed), bool(shuffle))


def split_eval_indices(args: argparse.Namespace, rows: Sequence[FrameMemoryExample], split: str) -> List[int]:
    indices = limited_indices(rows, int(args.max_eval_examples), int(args.seed) + 101 + len(split))
    if bool(args.smoke_test):
        return indices[: min(2, len(indices))]
    return indices


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
    count_values: Sequence[int],
    device: str,
) -> Tuple[List[Dict[str, Any]], Path]:
    train_indices = limited_indices(examples[TRAIN_SPLIT], int(args.max_train_examples), int(args.seed))
    val_indices = split_eval_indices(args, examples[VAL_SPLIT], VAL_SPLIT)
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
                if adapter.memory is not None and adapter.memory._slots is not None:
                    raise RuntimeError("Memory slots were not reset before training forward")
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
                indices=val_indices,
                answer_ids=answer_ids,
                count_values=count_values,
                device=device,
                collect_npz=False,
                diagnostic_dir=None,
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


def gold_score_from_candidate_scores(row: Dict[str, Any]) -> float:
    scores = row.get("candidate_scores", {})
    if not isinstance(scores, dict):
        return MISSING
    value = scores.get(str(int(row.get("gold_count", 0))))
    return float(value) if value is not None else MISSING


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


def diagnostic_scalar_rows(prediction_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scalar_keys = [
        "mean_evidence_compatibility_by_layer",
        "mean_neutral_compatibility_by_layer",
        "compatibility_auroc_by_layer",
        "mean_evidence_contribution_norm_by_layer",
        "mean_neutral_contribution_norm_by_layer",
        "contribution_norm_auroc_by_layer",
        "total_evidence_contribution_norm_by_layer",
        "total_neutral_contribution_norm_by_layer",
        "evidence_to_neutral_contribution_ratio_by_layer",
        "raw_message_norm_by_layer",
        "slot_norm_by_layer",
        "read_norm_by_layer",
        "injection_norm_by_layer",
        "carrier_state_norm_by_layer",
        "injection_to_carrier_ratio_by_layer",
        "effective_rank_by_layer",
        "slot_cosine_by_layer",
        "gamma_by_layer",
        "reconstruction_error_by_layer",
        "read_reconstruction_error_by_layer",
        "previous_slot_norm_by_layer",
        "previous_slot_used_by_layer",
        "injected_by_layer",
    ]
    rows: List[Dict[str, Any]] = []
    for row in prediction_rows:
        layers = sorted(
            {
                str(layer)
                for key in scalar_keys
                if isinstance(row.get(key), dict)
                for layer in row.get(key, {})
            },
            key=int,
        )
        for layer in layers:
            out = {
                "variant": row.get("variant"),
                "split": row.get("split"),
                "example_id": row.get("example_id"),
                "num_frames": row.get("num_frames"),
                "gold_count": row.get("gold_count"),
                "predicted_count": row.get("predicted_count"),
                "layer": int(layer),
            }
            for key in scalar_keys:
                payload = row.get(key, {})
                out[key.replace("_by_layer", "")] = payload.get(layer) if isinstance(payload, dict) else MISSING
            rows.append(out)
    return rows


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
    collect_npz: bool,
    diagnostic_dir: Optional[Path],
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
                if adapter.memory is not None and adapter.memory._slots is not None:
                    raise RuntimeError("Memory slots were not reset between examples")
            outputs = model(**batch.inputs, use_cache=False)
            ce, ce_vec = answer_sequence_cross_entropy(outputs.logits, batch)
            stats = adapter.stats_for_row(0) if adapter is not None else {}
            diag_tensors = adapter.memory.diagnostic_tensors() if adapter is not None and adapter.memory is not None else {}
            if adapter is not None:
                adapter.clear_context()
                if adapter.memory is not None and adapter.memory._slots is not None:
                    raise RuntimeError("Memory slots were not cleared after example forward")
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
                "signed_error": int(pred) - int(example.gold_count),
                "undercount": int(int(pred) < int(example.gold_count)),
                "overcount": int(int(pred) > int(example.gold_count)),
                "evidence_density": float(example.gold_count) / max(1, int(example.num_frames)),
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
                **paired_row_fields(example),
                **stats,
            }
            row["gold_answer_score"] = gold_score_from_candidate_scores(row)
            rows.append(row)
            if collect_npz and diagnostic_dir is not None:
                saved_layers: Dict[str, str] = {}
                sample_stem = safe_name(f"{split_name}_{order:04d}_{example.example_id}")
                for layer, tensors in diag_tensors.items():
                    npz_path = diagnostic_dir / f"{sample_stem}_layer{layer}.npz"
                    npz_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        npz_path,
                        **{key: tensor.numpy() for key, tensor in tensors.items()},
                    )
                    saved_layers[str(layer)] = os.fspath(npz_path.relative_to(diagnostic_dir.parent))
                diagnostic_rows.append(
                    {
                        "variant": variant,
                        "split": split_name,
                        "example_id": example.example_id,
                        "gold_count": int(example.gold_count),
                        "num_frames": int(example.num_frames),
                        "evidence_frame_indices": list(example.evidence_frame_indices),
                        "carrier_identities": list(batch.carrier_identities[0]),
                        "layers": saved_layers,
                        **paired_row_fields(example),
                    }
                )
    finally:
        if adapter is not None and manage_attachment:
            adapter.detach()
    if diagnostic_rows and diagnostic_dir is not None:
        write_jsonl(diagnostic_dir / f"{split_name}_diagnostics_manifest.jsonl", diagnostic_rows)
    return {"rows": rows, "diagnostics": diagnostic_rows}


def summarize_prediction_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "accuracy": MISSING,
            "mae": MISSING,
            "undercount_rate": MISSING,
            "overcount_rate": MISSING,
            "mean_signed_error": MISSING,
            "mean_gold_answer_score": MISSING,
            "mean_cross_entropy": MISSING,
        }
    return {
        "n": len(rows),
        "accuracy": base.accuracy([int(row["gold_count"]) for row in rows], [int(row["predicted_count"]) for row in rows]),
        "mae": base.mae([int(row["gold_count"]) for row in rows], [int(row["predicted_count"]) for row in rows]),
        "undercount_rate": finite_mean([row["undercount"] for row in rows], default=MISSING),
        "overcount_rate": finite_mean([row["overcount"] for row in rows], default=MISSING),
        "mean_signed_error": finite_mean([row["signed_error"] for row in rows], default=MISSING),
        "mean_gold_answer_score": finite_mean([row.get("gold_answer_score") for row in rows], default=MISSING),
        "mean_cross_entropy": finite_mean([row.get("ce") for row in rows], default=MISSING),
    }


def metrics_from_rows(
    variant: str,
    rows: Sequence[Dict[str, Any]],
    count_values: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    split_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    length_rows: List[Dict[str, Any]] = []
    confusion: Dict[str, Any] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        split_data = [row for row in rows if str(row["split"]) == split]
        split_rows.append({"variant": variant, "split": split, **summarize_prediction_rows(split_data)})
        confusion[split] = confusion_matrix(split_data, count_values).tolist()
        for count in sorted({int(row["gold_count"]) for row in split_data}):
            data = [row for row in split_data if int(row["gold_count"]) == count]
            count_rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "true_count": int(count),
                    "mean_predicted_count": finite_mean([row["predicted_count"] for row in data], default=MISSING),
                    **summarize_prediction_rows(data),
                }
            )
        for length in sorted({int(row["num_frames"]) for row in split_data}):
            data = [row for row in split_data if int(row["num_frames"]) == length]
            length_rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "sequence_length": int(length),
                    "mean_predicted_count": finite_mean([row["predicted_count"] for row in data], default=MISSING),
                    **summarize_prediction_rows(data),
                }
            )
    return split_rows, count_rows, length_rows, confusion


def confusion_matrix(rows: Sequence[Dict[str, Any]], count_values: Sequence[int]) -> np.ndarray:
    return base.confusion_matrix(rows, count_values)


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
        base_ce = finite_float(base_row.get("ce")) or MISSING
        for length, row in sorted(versions.items()):
            pred = int(row["predicted_count"])
            gold_score = finite_float(row.get("gold_answer_score")) or MISSING
            ce = finite_float(row.get("ce")) or MISSING
            out.append(
                {
                    "variant": row["variant"],
                    "family_id": family,
                    "base_sample_id": row.get("paired_base_sample_id"),
                    "version_length": int(length),
                    "example_id": row["example_id"],
                    "gold_count": int(row["gold_count"]),
                    "base_predicted_count": int(base_pred),
                    "predicted_count": int(pred),
                    "prediction_unchanged_from_len8": int(pred == base_pred),
                    "correct": int(pred == int(row["gold_count"])),
                    "delta_prediction_from_len8": int(pred - base_pred),
                    "abs_prediction_drift": abs(int(pred - base_pred)),
                    "signed_prediction_drift": int(pred - base_pred),
                    "undercount": int(pred < int(row["gold_count"])),
                    "overcount": int(pred > int(row["gold_count"])),
                    "gold_answer_score": gold_score,
                    "delta_gold_answer_score_from_len8": (
                        float(gold_score - base_gold_score)
                        if math.isfinite(float(gold_score)) and math.isfinite(float(base_gold_score))
                        else MISSING
                    ),
                    "cross_entropy": ce,
                    "delta_cross_entropy_from_len8": (
                        float(ce - base_ce)
                        if math.isfinite(float(ce)) and math.isfinite(float(base_ce))
                        else MISSING
                    ),
                    "read_norm": layer_mean(row.get("read_norm_by_layer")),
                    "injection_norm": layer_mean(row.get("injection_norm_by_layer")),
                    "injection_to_carrier_ratio": layer_mean(row.get("injection_to_carrier_ratio_by_layer")),
                }
            )
    return out


def paired_extension_by_length(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for length in sorted({int(row["version_length"]) for row in rows}):
        data = [row for row in rows if int(row["version_length"]) == length]
        out.append(
            {
                "variant": data[0]["variant"] if data else "",
                "version_length": int(length),
                "n": len(data),
                "accuracy": finite_mean([row["correct"] for row in data], default=MISSING),
                "prediction_retention": finite_mean([row["prediction_unchanged_from_len8"] for row in data], default=MISSING),
                "percentage_still_correct": finite_mean([row["correct"] for row in data], default=MISSING),
                "mean_absolute_prediction_drift": finite_mean([row["abs_prediction_drift"] for row in data], default=MISSING),
                "mean_signed_prediction_drift": finite_mean([row["signed_prediction_drift"] for row in data], default=MISSING),
                "undercount_frequency": finite_mean([row["undercount"] for row in data], default=MISSING),
                "overcount_frequency": finite_mean([row["overcount"] for row in data], default=MISSING),
                "gold_answer_score_change": finite_mean([row["delta_gold_answer_score_from_len8"] for row in data], default=MISSING),
                "cross_entropy_change": finite_mean([row["delta_cross_entropy_from_len8"] for row in data], default=MISSING),
                "memory_read_norm": finite_mean([row["read_norm"] for row in data], default=MISSING),
                "memory_injection_norm": finite_mean([row["injection_norm"] for row in data], default=MISSING),
                "injection_to_carrier_state_ratio": finite_mean(
                    [row["injection_to_carrier_ratio"] for row in data], default=MISSING
                ),
            }
        )
    return out


def layer_mean(payload: Any) -> float:
    if not isinstance(payload, dict):
        return MISSING
    return finite_mean(payload.values(), default=MISSING)


@torch.no_grad()
def forward_gold_ce_and_stats(
    *,
    model: Any,
    processor: Any,
    adapter: ExperimentAdapter,
    example: FrameMemoryExample,
    sample_idx: int,
    answer_ids: Dict[int, Tuple[int, ...]],
    device: str,
) -> Tuple[float, Dict[str, Any]]:
    batch = prepare_batch(
        examples=[example],
        sample_indices=[sample_idx],
        processor=processor,
        device=device,
        answer_ids=answer_ids,
    )
    adapter.set_context(batch)
    outputs = model(**batch.inputs, use_cache=False)
    _ce, ce_vec = answer_sequence_cross_entropy(outputs.logits, batch)
    stats = adapter.stats_for_row(0)
    adapter.clear_context()
    return float(ce_vec[0].detach().cpu().item()), stats


@torch.no_grad()
def evaluate_layer_injection_intervention(
    *,
    variant: str,
    model: Any,
    processor: Any,
    adapter: ExperimentAdapter,
    examples: Sequence[FrameMemoryExample],
    answer_ids: Dict[int, Tuple[int, ...]],
    count_values: Sequence[int],
    device: str,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if variant != GLSTM_LAYERWISE_PERSISTENT or adapter.memory is None:
        return [], []
    length16_indices = [
        idx
        for idx, example in enumerate(examples)
        if int(example.num_frames) == 16
    ]
    modes = [("normal", None)] + [(f"disable_layer_{layer}", int(layer)) for layer in DEFAULT_LAYERS]
    per_example: List[Dict[str, Any]] = []
    adapter.attach(model)
    try:
        normal_cache: Dict[str, Dict[str, Any]] = {}
        for idx in length16_indices:
            example = examples[int(idx)]
            adapter.memory.disabled_injection_layers = set()
            normal_ce, _normal_stats = forward_gold_ce_and_stats(
                model=model,
                processor=processor,
                adapter=adapter,
                example=example,
                sample_idx=int(idx),
                answer_ids=answer_ids,
                device=device,
            )
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
            normal_cache[example.example_id] = {
                "pred": int(normal_pred),
                "scores": normal_scores,
                "ce": normal_ce,
                "gold_score": float(normal_scores.get(str(int(example.gold_count)), MISSING)),
            }
            for mode, disabled_layer in modes:
                adapter.memory.disabled_injection_layers = set() if disabled_layer is None else {int(disabled_layer)}
                if mode == "normal":
                    pred = int(normal_pred)
                    scores = dict(normal_scores)
                    ce = normal_ce
                else:
                    ce, _stats = forward_gold_ce_and_stats(
                        model=model,
                        processor=processor,
                        adapter=adapter,
                        example=example,
                        sample_idx=int(idx),
                        answer_ids=answer_ids,
                        device=device,
                    )
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
                    pred = int(pred)
                gold_score = float(scores.get(str(int(example.gold_count)), MISSING))
                per_example.append(
                    {
                        "variant": variant,
                        "intervention": mode,
                        "disabled_layer": disabled_layer if disabled_layer is not None else "",
                        "example_id": example.example_id,
                        "family_id": _paired_metadata_from_example(example).get("family_id"),
                        "gold_count": int(example.gold_count),
                        "normal_predicted_count": int(normal_cache[example.example_id]["pred"]),
                        "predicted_count": int(pred),
                        "correct": int(pred == int(example.gold_count)),
                        "abs_error": abs(pred - int(example.gold_count)),
                        "prediction_delta_vs_normal": int(pred - int(normal_cache[example.example_id]["pred"])),
                        "abs_prediction_delta_vs_normal": abs(int(pred - int(normal_cache[example.example_id]["pred"]))),
                        "changed_prediction": int(pred != int(normal_cache[example.example_id]["pred"])),
                        "gold_answer_score": gold_score,
                        "gold_answer_score_delta_vs_normal": gold_score - float(normal_cache[example.example_id]["gold_score"]),
                        "cross_entropy": ce,
                        "cross_entropy_delta_vs_normal": ce - float(normal_cache[example.example_id]["ce"]),
                    }
                )
        adapter.memory.disabled_injection_layers = set()
    finally:
        adapter.detach()
    summary: List[Dict[str, Any]] = []
    for mode, disabled_layer in modes:
        data = [row for row in per_example if row["intervention"] == mode]
        summary.append(
            {
                "variant": variant,
                "intervention": mode,
                "disabled_layer": disabled_layer if disabled_layer is not None else "",
                "n": len(data),
                "accuracy": finite_mean([row["correct"] for row in data], default=MISSING),
                "mae": finite_mean([row["abs_error"] for row in data], default=MISSING),
                "mean_prediction_delta_vs_normal": finite_mean(
                    [row["prediction_delta_vs_normal"] for row in data], default=MISSING
                ),
                "mean_abs_prediction_delta_vs_normal": finite_mean(
                    [row["abs_prediction_delta_vs_normal"] for row in data], default=MISSING
                ),
                "changed_predictions": int(sum(int(row["changed_prediction"]) for row in data)),
                "mean_gold_answer_score_delta_vs_normal": finite_mean(
                    [row["gold_answer_score_delta_vs_normal"] for row in data], default=MISSING
                ),
                "mean_cross_entropy_delta_vs_normal": finite_mean(
                    [row["cross_entropy_delta_vs_normal"] for row in data], default=MISSING
                ),
                "per_count_accuracy": {
                    str(count): finite_mean(
                        [row["correct"] for row in data if int(row["gold_count"]) == int(count)],
                        default=MISSING,
                    )
                    for count in range(9)
                },
            }
        )
    return per_example, summary


def plot_metric_by_count(plots: Path, count_rows: Sequence[Dict[str, Any]], split: str, metric: str, filename: str, ylabel: str) -> None:
    data = [row for row in count_rows if row["split"] == split]
    if not data:
        return
    data = sorted(data, key=lambda row: int(row["true_count"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([int(row["true_count"]) for row in data], [float(row.get(metric, MISSING)) for row in data], marker="o")
    ax.set_xlabel("True count")
    ax.set_ylabel(ylabel)
    if metric == "accuracy":
        ax.set_ylim(0, 1)
    ax.set_title(f"{ylabel} by count: {split}")
    save_plot(plots / filename, fig)


def plot_mean_predicted_vs_true(plots: Path, count_rows: Sequence[Dict[str, Any]], split: str) -> None:
    data = [row for row in count_rows if row["split"] == split]
    if not data:
        return
    data = sorted(data, key=lambda row: int(row["true_count"]))
    xs = [int(row["true_count"]) for row in data]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(xs, [float(row.get("mean_predicted_count", MISSING)) for row in data], marker="o", label="mean predicted")
    ax.plot(xs, xs, linestyle="--", color="black", label="ideal")
    ax.set_xlabel("True count")
    ax.set_ylabel("Mean predicted count")
    ax.legend()
    ax.set_title(f"Mean predicted vs true: {split}")
    save_plot(plots / f"mean_predicted_vs_true_{safe_name(split)}", fig)


def plot_confusion(plots: Path, rows: Sequence[Dict[str, Any]], split: str, count_values: Sequence[int]) -> None:
    split_rows = [row for row in rows if row["split"] == split]
    if not split_rows:
        return
    mat = confusion_matrix(split_rows, count_values)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(mat, cmap="Blues")
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("True count")
    ax.set_xticks(range(len(count_values)), count_values)
    ax.set_yticks(range(len(count_values)), count_values)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j]:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title(f"Confusion matrix: {split}")
    save_plot(plots / f"confusion_matrix_{safe_name(split)}", fig)


def plot_metric_by_length(plots: Path, length_rows: Sequence[Dict[str, Any]], metric: str, filename: str, ylabel: str) -> None:
    if not length_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for split in sorted({row["split"] for row in length_rows}):
        data = sorted([row for row in length_rows if row["split"] == split], key=lambda row: int(row["sequence_length"]))
        ax.plot(
            [int(row["sequence_length"]) for row in data],
            [float(row.get(metric, MISSING)) for row in data],
            marker="o",
            label=split,
        )
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    if metric == "accuracy":
        ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(ylabel)
    save_plot(plots / filename, fig)


def plot_undercount_overcount_by_length(plots: Path, length_rows: Sequence[Dict[str, Any]]) -> None:
    if not length_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for metric, linestyle in [("undercount_rate", "-"), ("overcount_rate", "--")]:
        data_by_length: Dict[int, List[float]] = defaultdict(list)
        for row in length_rows:
            data_by_length[int(row["sequence_length"])].append(float(row.get(metric, MISSING)))
        xs = sorted(data_by_length)
        ax.plot(xs, [finite_mean(data_by_length[x], default=MISSING) for x in xs], marker="o", linestyle=linestyle, label=metric)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Undercount and overcount by sequence length")
    save_plot(plots / "undercount_overcount_by_sequence_length", fig)


def plot_training_history(plots: Path, history: Sequence[Dict[str, Any]]) -> None:
    if not history:
        return
    fig, ax1 = plt.subplots(figsize=(7, 4))
    epochs = [int(row["epoch"]) for row in history]
    ax1.plot(epochs, [float(row.get("train_loss", MISSING)) for row in history], marker="o", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [float(row.get("val_accuracy", MISSING)) for row in history], marker="s", color="tab:green", label="val acc")
    ax2.set_ylabel("Validation accuracy")
    ax2.set_ylim(0, 1)
    ax1.set_title("Training history")
    save_plot(plots / "training_history", fig)


def plot_paired_by_length(plots: Path, paired_by_length: Sequence[Dict[str, Any]]) -> None:
    mapping = [
        ("accuracy", "paired_accuracy_by_length", "Accuracy"),
        ("prediction_retention", "paired_prediction_retention_by_length", "Prediction retention"),
        ("mean_absolute_prediction_drift", "paired_absolute_prediction_drift_by_length", "Absolute prediction drift"),
        ("mean_signed_prediction_drift", "paired_signed_prediction_drift_by_length", "Signed prediction drift"),
        ("gold_answer_score_change", "paired_gold_score_change_by_length", "Gold score change"),
        ("cross_entropy_change", "paired_cross_entropy_change_by_length", "Cross-entropy change"),
        ("memory_read_norm", "paired_read_norm_by_length", "Read norm"),
        ("memory_injection_norm", "paired_injection_norm_by_length", "Injection norm"),
        ("injection_to_carrier_state_ratio", "paired_injection_ratio_by_length", "Injection/state ratio"),
    ]
    for metric, filename, ylabel in mapping:
        if not paired_by_length:
            continue
        data = sorted(paired_by_length, key=lambda row: int(row["version_length"]))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([int(row["version_length"]) for row in data], [float(row.get(metric, MISSING)) for row in data], marker="o")
        ax.set_xlabel("Version length")
        ax.set_ylabel(ylabel)
        if metric in {"accuracy", "prediction_retention"}:
            ax.set_ylim(0, 1)
        ax.set_title(ylabel)
        save_plot(plots / filename, fig)


def diagnostic_layer_values(rows: Sequence[Dict[str, Any]], key: str) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        payload = row.get(key)
        if not isinstance(payload, dict):
            continue
        for layer, value in payload.items():
            fv = finite_float(value)
            if fv is not None:
                out[int(layer)].append(float(fv))
    return out


def plot_diagnostic_line(plots: Path, values: Dict[int, List[float]], filename: str, ylabel: str) -> None:
    if not values:
        return
    xs = sorted(values)
    ys = [finite_mean(values[x], default=MISSING) for x in xs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    save_plot(plots / filename, fig)


def plot_diagnostic_pairs(plots: Path, rows: Sequence[Dict[str, Any]]) -> None:
    pair_specs = [
        (
            "mean_evidence_compatibility_by_layer",
            "mean_neutral_compatibility_by_layer",
            "evidence_vs_neutral_compatibility_by_layer",
            "Compatibility",
        ),
        (
            "mean_evidence_contribution_norm_by_layer",
            "mean_neutral_contribution_norm_by_layer",
            "evidence_vs_neutral_contribution_norm_by_layer",
            "Contribution norm",
        ),
    ]
    for evidence_key, neutral_key, filename, ylabel in pair_specs:
        evidence = diagnostic_layer_values(rows, evidence_key)
        neutral = diagnostic_layer_values(rows, neutral_key)
        layers = sorted(set(evidence) | set(neutral))
        if not layers:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(layers, [finite_mean(evidence.get(layer, []), default=MISSING) for layer in layers], marker="o", label="evidence")
        ax.plot(layers, [finite_mean(neutral.get(layer, []), default=MISSING) for layer in layers], marker="s", label="neutral")
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_title(ylabel)
        save_plot(plots / filename, fig)


def plot_by_layer_and_length(plots: Path, rows: Sequence[Dict[str, Any]], key: str, filename: str, ylabel: str) -> None:
    values: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        payload = row.get(key)
        if not isinstance(payload, dict):
            continue
        length = int(row["num_frames"])
        for layer, value in payload.items():
            fv = finite_float(value)
            if fv is not None:
                values[int(length)][int(layer)].append(float(fv))
    if not values:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for length in sorted(values):
        layers = sorted(values[length])
        ax.plot(layers, [finite_mean(values[length][layer], default=MISSING) for layer in layers], marker="o", label=f"L={length}")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(ylabel)
    save_plot(plots / filename, fig)


def plot_memory_diagnostics(plots: Path, rows: Sequence[Dict[str, Any]]) -> None:
    plot_diagnostic_pairs(plots, rows)
    single_specs = [
        ("compatibility_auroc_by_layer", "compatibility_auroc_by_layer", "Compatibility AUROC"),
        ("contribution_norm_auroc_by_layer", "contribution_auroc_by_layer", "Contribution norm AUROC"),
        ("effective_rank_by_layer", "effective_slot_rank_by_layer", "Effective slot rank"),
        ("slot_cosine_by_layer", "slot_cosine_by_layer", "Mean slot cosine"),
        ("gamma_by_layer", "gamma_by_layer", "Gamma"),
        ("reconstruction_error_by_layer", "reconstruction_error_by_layer", "Attention reconstruction error"),
    ]
    for key, filename, ylabel in single_specs:
        plot_diagnostic_line(plots, diagnostic_layer_values(rows, key), filename, ylabel)
    length_specs = [
        ("total_evidence_contribution_norm_by_layer", "total_evidence_contribution_by_layer_and_length", "Total evidence contribution norm"),
        ("total_neutral_contribution_norm_by_layer", "total_neutral_contribution_by_layer_and_length", "Total neutral contribution norm"),
        ("evidence_to_neutral_contribution_ratio_by_layer", "evidence_to_neutral_ratio_by_layer_and_length", "Evidence/neutral contribution ratio"),
        ("slot_norm_by_layer", "slot_norm_by_layer_and_length", "Slot norm"),
        ("read_norm_by_layer", "read_norm_by_layer_and_length", "Read norm"),
        ("injection_norm_by_layer", "injection_norm_by_layer_and_length", "Injection norm"),
        ("injection_to_carrier_ratio_by_layer", "injection_ratio_by_layer_and_length", "Injection/state ratio"),
    ]
    for key, filename, ylabel in length_specs:
        plot_by_layer_and_length(plots, rows, key, filename, ylabel)


def plot_layer_intervention(plots: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    specs = [
        ("accuracy", "layer_injection_ablation_accuracy", "Accuracy"),
        ("mae", "layer_injection_ablation_mae", "MAE"),
        ("mean_prediction_delta_vs_normal", "layer_injection_ablation_prediction_delta", "Prediction delta"),
        ("mean_gold_answer_score_delta_vs_normal", "layer_injection_ablation_gold_score_delta", "Gold score delta"),
    ]
    for metric, filename, ylabel in specs:
        if not summary_rows:
            continue
        labels = [str(row["intervention"]).replace("disable_layer_", "no L") for row in summary_rows]
        values = [float(row.get(metric, MISSING)) for row in summary_rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        if metric == "accuracy":
            ax.set_ylim(0, 1)
        if "delta" in metric:
            ax.axhline(0, color="black", linewidth=1)
        ax.set_title(ylabel)
        save_plot(plots / filename, fig)


def make_run_plots(
    *,
    run_dir: Path,
    prediction_rows: Sequence[Dict[str, Any]],
    count_rows: Sequence[Dict[str, Any]],
    length_rows: Sequence[Dict[str, Any]],
    paired_by_length_rows: Sequence[Dict[str, Any]],
    intervention_summary_rows: Sequence[Dict[str, Any]],
    history: Sequence[Dict[str, Any]],
    count_values: Sequence[int],
) -> None:
    plots = run_dir / "plots"
    for split in sorted({row["split"] for row in prediction_rows}):
        plot_metric_by_count(plots, count_rows, split, "accuracy", f"accuracy_by_count_{safe_name(split)}", "Accuracy")
        plot_metric_by_count(plots, count_rows, split, "mae", f"mae_by_count_{safe_name(split)}", "MAE")
        plot_mean_predicted_vs_true(plots, count_rows, split)
        plot_confusion(plots, prediction_rows, split, count_values)
    plot_metric_by_length(plots, length_rows, "accuracy", "accuracy_by_sequence_length", "Accuracy by sequence length")
    plot_metric_by_length(plots, length_rows, "mae", "mae_by_sequence_length", "MAE by sequence length")
    plot_metric_by_length(plots, length_rows, "mean_signed_error", "signed_error_by_sequence_length", "Signed error by sequence length")
    plot_undercount_overcount_by_length(plots, length_rows)
    plot_training_history(plots, history)
    plot_paired_by_length(plots, paired_by_length_rows)
    plot_memory_diagnostics(plots, prediction_rows)
    plot_layer_intervention(plots, intervention_summary_rows)


def write_report(
    run_dir: Path,
    variant: str,
    split_metrics: Sequence[Dict[str, Any]],
    paired_by_length_rows: Sequence[Dict[str, Any]],
    intervention_summary: Sequence[Dict[str, Any]],
    checkpoint_path: Path,
) -> None:
    lines = [
        f"# {variant} report",
        "",
        "## Main Splits",
        "",
        "| split | accuracy | MAE | signed error | under | over |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in split_metrics:
        lines.append(
            f"| {row['split']} | {float(row.get('accuracy', MISSING)):.3f} | "
            f"{float(row.get('mae', MISSING)):.3f} | {float(row.get('mean_signed_error', MISSING)):.3f} | "
            f"{float(row.get('undercount_rate', MISSING)):.3f} | {float(row.get('overcount_rate', MISSING)):.3f} |"
        )
    if paired_by_length_rows:
        lines.extend(["", "## Paired Neutral Extension", "", "| length | accuracy | retention | abs drift | score change |", "|---:|---:|---:|---:|---:|"])
        for row in paired_by_length_rows:
            lines.append(
                f"| {row['version_length']} | {float(row.get('accuracy', MISSING)):.3f} | "
                f"{float(row.get('prediction_retention', MISSING)):.3f} | "
                f"{float(row.get('mean_absolute_prediction_drift', MISSING)):.3f} | "
                f"{float(row.get('gold_answer_score_change', MISSING)):.3f} |"
            )
    if intervention_summary:
        lines.extend(["", "## Layer Injection Intervention", "", "| intervention | accuracy | MAE | changed |", "|---|---:|---:|---:|"])
        for row in intervention_summary:
            lines.append(
                f"| {row['intervention']} | {float(row.get('accuracy', MISSING)):.3f} | "
                f"{float(row.get('mae', MISSING)):.3f} | {row.get('changed_predictions')} |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Checkpoint: `{checkpoint_path}`",
            f"- Diagnostics: `{run_dir / 'diagnostics'}`",
            f"- Plots: `{run_dir / 'plots'}`",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_smoke_summary(run_dir: Path, rows: Sequence[Dict[str, Any]], adapter: Optional[ExperimentAdapter]) -> None:
    first = rows[0] if rows else {}
    print("Smoke test diagnostics")
    print(f"detected frame spans: {first.get('visual_token_spans_per_frame')}")
    print(f"detected carrier positions: {first.get('carrier_token_positions')}")
    print(f"attention reconstruction error: {first.get('memory_reconstruction_error')}")
    print(f"non-carrier LoRA update max: {first.get('noncarrier_lora_update_max')}")
    if adapter is not None and adapter.memory is not None:
        print(f"hook fire counts: {dict(adapter.memory.hook_fire_counts)}")
    print(f"output directory: {run_dir}")


def assert_run_correctness(
    *,
    variant: str,
    adapter: ExperimentAdapter,
    prediction_rows: Sequence[Dict[str, Any]],
    dataset_manifest: Dict[str, Any],
    count_values: Sequence[int],
) -> None:
    if [int(x) for x in count_values] != list(range(9)):
        raise RuntimeError("Candidate counts are not exactly 0..8")
    for split, payload in dataset_manifest["splits"].items():
        if int(payload["n"]) != int(payload["expected_n"]):
            raise RuntimeError(f"{split}: dataset manifest has n != expected_n")
    max_noncarrier = max(
        [finite_float(row.get("noncarrier_lora_update_max")) or 0.0 for row in prediction_rows],
        default=0.0,
    )
    if max_noncarrier > 1e-5:
        raise RuntimeError(f"Carrier-gated LoRA updated non-carrier positions: max={max_noncarrier}")
    if adapter.memory is None:
        raise RuntimeError("All variants in this experiment must have memory")
    for row in prediction_rows:
        recon = row.get("read_reconstruction_error_by_layer", {})
        if isinstance(recon, dict):
            bad = {layer: value for layer, value in recon.items() if (finite_float(value) or 0.0) > 1e-4}
            if bad:
                raise RuntimeError(f"Read contribution reconstruction failed: {bad}")
        hooks = row.get("tensor_shapes_by_layer", {})
        if isinstance(hooks, dict) and row["split"] in MAIN_EVAL_SPLITS:
            missing = sorted(set(str(layer) for layer in DEFAULT_LAYERS) - set(hooks))
            if missing:
                raise RuntimeError(f"{row['example_id']}: missing memory hook layers {missing}")
    if variant == GLSTM_LAYERWISE_FRESH:
        used_values = [
            int(value)
            for row in prediction_rows
            for value in (row.get("previous_slot_used_by_layer", {}) or {}).values()
            if str(value).strip() != ""
        ]
        if any(used_values):
            raise RuntimeError("Fresh variant used previous-layer slots")
    if variant == GLSTM_LAYERWISE_PERSISTENT:
        used_values = [
            int(value)
            for row in prediction_rows
            for layer, value in (row.get("previous_slot_used_by_layer", {}) or {}).items()
            if int(layer) in DEFAULT_LAYERS[1:]
        ]
        if used_values and not all(used_values):
            raise RuntimeError("Persistent variant did not use previous-layer slots after the first layer")
    if variant == GLSTM_FINAL_ONLY_PERSISTENT:
        injected = defaultdict(list)
        for row in prediction_rows:
            payload = row.get("injected_by_layer", {})
            if isinstance(payload, dict):
                for layer, value in payload.items():
                    injected[int(layer)].append(int(value))
        for layer in DEFAULT_LAYERS:
            expected = 1 if int(layer) == max(DEFAULT_LAYERS) else 0
            if injected.get(int(layer)) and any(int(v) != expected for v in injected[int(layer)]):
                raise RuntimeError(f"Final-only injection mismatch at layer {layer}")
    if variant == DIRECT_SUM_LAYERWISE:
        trainable = [name for name, param in adapter.named_parameters() if param.requires_grad]
        forbidden = [name for name in trainable if ".w_q." in name or ".w_k." in name or ".w_v." in name]
        if forbidden:
            raise RuntimeError(f"Direct sum has trainable associative read params: {forbidden[:20]}")


def verify_no_pdfs(root: Path) -> None:
    pdfs = sorted(Path(root).rglob("*.pdf"))
    if pdfs:
        raise RuntimeError(f"PDF files are forbidden in this experiment; found {pdfs[:10]}")


def validate_smoke_slot_and_evidence_behavior(
    *,
    model: Any,
    processor: Any,
    adapter: ExperimentAdapter,
    examples: Dict[str, List[FrameMemoryExample]],
    answer_ids: Dict[int, Tuple[int, ...]],
    device: str,
) -> None:
    if adapter.memory is None:
        raise RuntimeError("Smoke validation requires memory")
    adapter.attach(model)
    try:
        example = examples[VAL_SPLIT][0]
        batch = prepare_batch(
            examples=[example],
            sample_indices=[0],
            processor=processor,
            device=device,
            answer_ids=answer_ids,
        )
        adapter.set_context(batch)
        logits_a = model(**batch.inputs, use_cache=False).logits.detach().float().cpu()
        stats_a = adapter.stats_for_row(0)
        adapter.clear_context()
        if adapter.memory._slots is not None:
            raise RuntimeError("Slots did not reset after clear_context")
        no_evidence_example = FrameMemoryExample(
            example_id=example.example_id + "_diag_no_evidence",
            split=example.split,
            frame_paths=example.frame_paths,
            num_frames=example.num_frames,
            gold_count=example.gold_count,
            evidence_frame_indices=tuple(),
            question=example.question,
            answer=example.answer,
            queried_character=example.queried_character,
            queried_room=example.queried_room,
            template_id=example.template_id,
            composition_key=example.composition_key,
            source_dataset_info=example.source_dataset_info,
        )
        batch_b = prepare_batch(
            examples=[no_evidence_example],
            sample_indices=[0],
            processor=processor,
            device=device,
            answer_ids=answer_ids,
        )
        adapter.set_context(batch_b)
        logits_b = model(**batch_b.inputs, use_cache=False).logits.detach().float().cpu()
        adapter.clear_context()
        if not torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Evidence masks affected normal forward computation")
        hook_layers = set(str(layer) for layer in DEFAULT_LAYERS)
        if set(stats_a.get("tensor_shapes_by_layer", {})) != hook_layers:
            raise RuntimeError("Memory hooks did not fire exactly once per selected layer in smoke forward")
    finally:
        adapter.detach()


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
    timestamp: str,
) -> Dict[str, Any]:
    layers = parse_layers(args)
    if bool(args.tiny_debug_model):
        max_layer = int(args.tiny_num_layers) - 1
        if any(layer > max_layer for layer in layers):
            raise ValueError(f"Tiny model has layers 0..{max_layer}, requested {layers}")
    run_prefix = f"{safe_name(args.run_prefix)}_" if str(args.run_prefix).strip() else ""
    smoke = "smoke_" if bool(args.smoke_test) else ""
    run_dir = parent_output_root / f"{timestamp}_{run_prefix}{smoke}{variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle, old_stdout, old_stderr = base.setup_logging(run_dir)
    adapter: Optional[ExperimentAdapter] = None
    try:
        print(f"Running {variant} into {run_dir}")
        write_json(run_dir / "config.json", {**vars(args), "variant": variant, "layers": layers})
        write_json(run_dir / "split_manifest.json", dataset_manifest)
        hidden_size = base.hidden_size_from_model(model)
        adapter = make_adapter(args, variant, hidden_size=hidden_size, layers=layers).to(device)
        adapter.attach(model)
        try:
            param_summary = trainable_parameter_summary(model, adapter, variant)
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
            count_values=count_values,
            device=device,
        )
        write_csv_dynamic(run_dir / "training_history.csv", history, leading=("variant", "epoch"))
        all_rows: List[Dict[str, Any]] = []
        diagnostics_dir = run_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        for split in MAIN_EVAL_SPLITS:
            indices = split_eval_indices(args, examples[split], split)
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
                collect_npz=False,
                diagnostic_dir=diagnostics_dir,
            )
            all_rows.extend(result["rows"])
        paired_indices = split_eval_indices(args, examples[PAIRED_EXTENSION_SPLIT], PAIRED_EXTENSION_SPLIT)
        paired_eval = evaluate_split(
            variant=variant,
            split_name=PAIRED_EXTENSION_SPLIT,
            model=model,
            processor=processor,
            adapter=adapter,
            examples=examples[PAIRED_EXTENSION_SPLIT],
            indices=paired_indices,
            answer_ids=answer_ids,
            count_values=count_values,
            device=device,
            collect_npz=True,
            diagnostic_dir=diagnostics_dir,
        )
        paired_rows = paired_eval["rows"]
        write_csv_dynamic(run_dir / "per_sample_predictions.csv", all_rows, leading=("variant", "split", "example_id"))
        write_csv_dynamic(run_dir / "paired_extension_predictions.csv", paired_rows, leading=("variant", "paired_family_id", "num_frames", "example_id"))
        split_metrics, count_metrics, length_metrics, confusion = metrics_from_rows(variant, all_rows, count_values)
        paired_summary_rows = paired_extension_summary(paired_rows)
        paired_by_length_rows = paired_extension_by_length(paired_summary_rows)
        scalar_rows = diagnostic_scalar_rows([*all_rows, *paired_rows])
        write_json(run_dir / "metrics.json", {
            "split_metrics": split_metrics,
            "confusion_matrices": confusion,
            "paired_extension_by_length": paired_by_length_rows,
        })
        write_csv_dynamic(run_dir / "metrics.csv", split_metrics, leading=("variant", "split"))
        write_csv_dynamic(run_dir / "metrics_by_count.csv", count_metrics, leading=("variant", "split", "true_count"))
        write_csv_dynamic(run_dir / "metrics_by_length.csv", length_metrics, leading=("variant", "split", "sequence_length"))
        write_csv_dynamic(diagnostics_dir / "scalar_diagnostics.csv", scalar_rows, leading=("variant", "split", "example_id", "layer"))
        write_json(diagnostics_dir / "scalar_diagnostics.json", scalar_rows)
        write_csv_dynamic(diagnostics_dir / "paired_extension_summary_by_example.csv", paired_summary_rows, leading=("variant", "family_id", "version_length"))
        write_csv_dynamic(diagnostics_dir / "paired_extension_summary_by_length.csv", paired_by_length_rows, leading=("variant", "version_length"))
        intervention_rows: List[Dict[str, Any]] = []
        intervention_summary: List[Dict[str, Any]] = []
        if variant == GLSTM_LAYERWISE_PERSISTENT and not bool(args.smoke_test):
            intervention_rows, intervention_summary = evaluate_layer_injection_intervention(
                variant=variant,
                model=model,
                processor=processor,
                adapter=adapter,
                examples=examples[PAIRED_EXTENSION_SPLIT],
                answer_ids=answer_ids,
                count_values=count_values,
                device=device,
                seed=int(args.seed),
            )
        elif variant == GLSTM_LAYERWISE_PERSISTENT and bool(args.smoke_test):
            intervention_rows, intervention_summary = evaluate_layer_injection_intervention(
                variant=variant,
                model=model,
                processor=processor,
                adapter=adapter,
                examples=examples[PAIRED_EXTENSION_SPLIT][:4],
                answer_ids=answer_ids,
                count_values=count_values,
                device=device,
                seed=int(args.seed),
            )
        write_csv_dynamic(diagnostics_dir / "layer_injection_ablation_per_example.csv", intervention_rows, leading=("variant", "intervention", "example_id"))
        write_csv_dynamic(diagnostics_dir / "layer_injection_ablation_summary.csv", intervention_summary, leading=("variant", "intervention"))
        write_json(diagnostics_dir / "layer_injection_ablation_summary.json", intervention_summary)
        assert_run_correctness(
            variant=variant,
            adapter=adapter,
            prediction_rows=[*all_rows, *paired_rows],
            dataset_manifest=dataset_manifest,
            count_values=count_values,
        )
        if bool(args.smoke_test):
            validate_smoke_slot_and_evidence_behavior(
                model=model,
                processor=processor,
                adapter=adapter,
                examples=examples,
                answer_ids=answer_ids,
                device=device,
            )
        if not bool(args.no_plots):
            make_run_plots(
                run_dir=run_dir,
                prediction_rows=all_rows,
                count_rows=count_metrics,
                length_rows=length_metrics,
                paired_by_length_rows=paired_by_length_rows,
                intervention_summary_rows=intervention_summary,
                history=history,
                count_values=count_values,
            )
        write_report(run_dir, variant, split_metrics, paired_by_length_rows, intervention_summary, checkpoint_path)
        if bool(args.smoke_test):
            print_smoke_summary(run_dir, [*all_rows, *paired_rows], adapter)
        verify_no_pdfs(parent_output_root)
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
        base.restore_logging(log_handle, old_stdout, old_stderr)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def variant_run_dirs(parent: Path) -> Dict[str, Path]:
    found: Dict[str, List[Tuple[float, Path]]] = defaultdict(list)
    for config_path in sorted(Path(parent).glob("*/config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if bool(config.get("smoke_test")):
            continue
        variant = str(config.get("variant", ""))
        if variant in VARIANTS and (config_path.parent / "metrics.csv").is_file():
            found[variant].append((config_path.parent.stat().st_mtime, config_path.parent))
    return {variant: sorted(entries)[-1][1] for variant, entries in found.items() if entries}


def assert_shared_dataset_across_runs(run_dirs: Dict[str, Path]) -> Dict[str, Any]:
    manifests: Dict[str, Any] = {}
    hashes: set[str] = set()
    sample_ids_by_split: Optional[Dict[str, List[str]]] = None
    for variant, run_dir in run_dirs.items():
        manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
        manifests[variant] = manifest
        dataset_hash = str(manifest.get("dataset_hash"))
        hashes.add(dataset_hash)
        ids = {
            split: list(payload.get("sample_ids", []))
            for split, payload in manifest.get("splits", {}).items()
        }
        if sample_ids_by_split is None:
            sample_ids_by_split = ids
        elif ids != sample_ids_by_split:
            raise RuntimeError(f"{variant}: sample IDs differ from the shared dataset")
    if len(hashes) != 1:
        raise RuntimeError(f"Variant jobs did not use the same dataset hash: {sorted(hashes)}")
    return {
        "dataset_hash": next(iter(hashes)) if hashes else "",
        "sample_ids_by_split": sample_ids_by_split or {},
        "manifests": manifests,
    }


def combined_bar_plot(
    comparison_dir: Path,
    rows: Sequence[Dict[str, Any]],
    metric: str,
    filename: str,
    ylabel: str,
) -> None:
    if not rows:
        return
    variants = [variant for variant in VARIANTS if any(row.get("variant") == variant for row in rows)]
    splits = sorted({row.get("split") for row in rows})
    x = np.arange(len(splits))
    width = 0.8 / max(1, len(variants))
    fig, ax = plt.subplots(figsize=(10, 4))
    for offset, variant in enumerate(variants):
        values = []
        for split in splits:
            match = next((row for row in rows if row.get("variant") == variant and row.get("split") == split), None)
            values.append(float(match.get(metric, MISSING)) if match else MISSING)
        ax.bar(x + offset * width, values, width=width, label=variant)
    ax.set_xticks(x + width * (len(variants) - 1) / 2, splits, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if metric == "accuracy":
        ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(ylabel)
    save_plot(comparison_dir / filename, fig)


def comparison_by_length_plot(
    comparison_dir: Path,
    length_rows: Sequence[Dict[str, Any]],
    variants: Sequence[str],
    filename: str,
    title: str,
) -> None:
    data = [
        row
        for row in length_rows
        if row.get("variant") in variants and row.get("split") in {LENGTH_INTERPOLATION_SPLIT, LENGTH_EXTRAPOLATION_SPLIT, IID_TEST_SPLIT}
    ]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for variant in variants:
        by_length: Dict[int, List[float]] = defaultdict(list)
        for row in data:
            if row.get("variant") == variant:
                by_length[int(row["sequence_length"])].append(float(row.get("accuracy", MISSING)))
        xs = sorted(by_length)
        ax.plot(xs, [finite_mean(by_length[x], default=MISSING) for x in xs], marker="o", label=variant)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(title)
    save_plot(comparison_dir / filename, fig)


def paired_combined_plot(
    comparison_dir: Path,
    paired_rows: Sequence[Dict[str, Any]],
    metric: str,
    filename: str,
    ylabel: str,
    only_length: Optional[int] = None,
) -> None:
    data = [row for row in paired_rows if only_length is None or int(row.get("version_length", -1)) == int(only_length)]
    if not data:
        return
    variants = [variant for variant in VARIANTS if any(row.get("variant") == variant for row in data)]
    values = []
    for variant in variants:
        values.append(finite_mean([row.get(metric) for row in data if row.get("variant") == variant], default=MISSING))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(variants)), values)
    ax.set_xticks(range(len(variants)), variants, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if metric in {"accuracy", "prediction_retention"}:
        ax.set_ylim(0, 1)
    ax.set_title(ylabel)
    save_plot(comparison_dir / filename, fig)


def metric_lookup(rows: Sequence[Dict[str, Any]], variant: str, split: str, metric: str) -> float:
    row = next((item for item in rows if item.get("variant") == variant and item.get("split") == split), None)
    return float(row.get(metric, MISSING)) if row else MISSING


def write_final_report(
    parent: Path,
    metric_rows: Sequence[Dict[str, Any]],
    paired_rows: Sequence[Dict[str, Any]],
    shared: Dict[str, Any],
) -> None:
    lines = [
        "# Layerwise gLSTM mechanism ablation",
        "",
        f"Dataset hash: `{shared.get('dataset_hash', '')}`",
        "",
        "## Main Results",
        "",
        "| variant | split | accuracy | MAE | signed error |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row.get('variant')} | {row.get('split')} | {float(row.get('accuracy', MISSING)):.3f} | "
            f"{float(row.get('mae', MISSING)):.3f} | {float(row.get('mean_signed_error', MISSING)):.3f} |"
        )
    split_for_components = LENGTH_EXTRAPOLATION_SPLIT
    persistent = metric_lookup(metric_rows, GLSTM_LAYERWISE_PERSISTENT, split_for_components, "accuracy")
    fresh = metric_lookup(metric_rows, GLSTM_LAYERWISE_FRESH, split_for_components, "accuracy")
    final_only = metric_lookup(metric_rows, GLSTM_FINAL_ONLY_PERSISTENT, split_for_components, "accuracy")
    direct = metric_lookup(metric_rows, DIRECT_SUM_LAYERWISE, split_for_components, "accuracy")
    paired_l16 = [
        row for row in paired_rows if int(row.get("version_length", -1)) == 16
    ]
    drift_by_variant = {
        variant: finite_mean(
            [row.get("abs_prediction_drift") for row in paired_l16 if row.get("variant") == variant],
            default=MISSING,
        )
        for variant in VARIANTS
    }
    lines.extend(
        [
            "",
            "## Direct Comparisons",
            "",
            f"- Persistent vs fresh on length extrapolation accuracy: {persistent:.3f} vs {fresh:.3f}.",
            f"- Layerwise vs final-only on length extrapolation accuracy: {persistent:.3f} vs {final_only:.3f}.",
            f"- Associative gLSTM vs direct sum on length extrapolation accuracy: {persistent:.3f} vs {direct:.3f}.",
            f"- Length-16 paired absolute drift by variant: {json.dumps(drift_by_variant, sort_keys=True)}.",
            "",
            "## Mechanism Questions",
            "",
            "1. Persistent cross-layer slot accumulation helps if persistent exceeds fresh, especially on longer lengths.",
            "2. Repeated injection helps if layerwise persistent exceeds final-only persistent.",
            "3. Associative reading helps if layerwise persistent exceeds direct sum under matched writes.",
            "4. The component with the largest extrapolation drop relative to persistent is the strongest candidate driver.",
            "",
            f"Combined plots: `{parent / 'comparison_plots'}`",
        ]
    )
    (parent / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_parent_outputs(parent: Path) -> Dict[str, Any]:
    parent = Path(parent)
    run_dirs = variant_run_dirs(parent)
    missing = [variant for variant in VARIANTS if variant not in run_dirs]
    if missing:
        raise RuntimeError(f"Cannot aggregate; missing completed variant runs: {missing}")
    shared = assert_shared_dataset_across_runs(run_dirs)
    metric_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    length_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    for variant, run_dir in sorted(run_dirs.items()):
        metric_rows.extend(read_csv_rows(run_dir / "metrics.csv"))
        count_rows.extend(read_csv_rows(run_dir / "metrics_by_count.csv"))
        length_rows.extend(read_csv_rows(run_dir / "metrics_by_length.csv"))
        paired_rows.extend(read_csv_rows(run_dir / "diagnostics" / "paired_extension_summary_by_example.csv"))
    write_csv_dynamic(parent / "combined_results.csv", metric_rows, leading=("variant", "split"))
    write_csv_dynamic(parent / "combined_results_by_count.csv", count_rows, leading=("variant", "split", "true_count"))
    write_csv_dynamic(parent / "combined_results_by_length.csv", length_rows, leading=("variant", "split", "sequence_length"))
    write_csv_dynamic(parent / "paired_extension_combined.csv", paired_rows, leading=("variant", "family_id", "version_length"))
    comparison_dir = parent / "comparison_plots"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    combined_bar_plot(comparison_dir, metric_rows, "accuracy", "variant_accuracy_by_split", "Accuracy by split")
    combined_bar_plot(comparison_dir, metric_rows, "mae", "variant_mae_by_split", "MAE by split")
    comparison_by_length_plot(
        comparison_dir,
        length_rows,
        [GLSTM_LAYERWISE_PERSISTENT, GLSTM_LAYERWISE_FRESH],
        "persistent_vs_fresh_by_length",
        "Persistent vs fresh by length",
    )
    comparison_by_length_plot(
        comparison_dir,
        length_rows,
        [GLSTM_LAYERWISE_PERSISTENT, GLSTM_FINAL_ONLY_PERSISTENT],
        "layerwise_vs_final_only_by_length",
        "Layerwise vs final-only by length",
    )
    comparison_by_length_plot(
        comparison_dir,
        length_rows,
        [GLSTM_LAYERWISE_PERSISTENT, DIRECT_SUM_LAYERWISE],
        "glstm_vs_direct_sum_by_length",
        "Associative gLSTM vs direct sum by length",
    )
    paired_combined_plot(
        comparison_dir,
        paired_rows,
        "abs_prediction_drift",
        "paired_prediction_drift_by_variant",
        "Paired prediction drift",
    )
    paired_combined_plot(
        comparison_dir,
        paired_rows,
        "correct",
        "paired_length16_accuracy_by_variant",
        "Length-16 paired accuracy",
        only_length=16,
    )
    paired_combined_plot(
        comparison_dir,
        paired_rows,
        "prediction_unchanged_from_len8",
        "paired_length16_retention_by_variant",
        "Length-16 paired retention",
        only_length=16,
    )
    write_final_report(parent, metric_rows, paired_rows, shared)
    verify_no_pdfs(parent)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variants": sorted(run_dirs),
        "run_dirs": {variant: os.fspath(path) for variant, path in run_dirs.items()},
        "dataset_hash": shared.get("dataset_hash"),
        "num_metric_rows": len(metric_rows),
    }
    write_json(parent / "combined_summary.json", summary)
    return summary


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
    syntax = run_checked([sys.executable, "-m", "py_compile", __file__], cwd=PROJECT_ROOT)
    if syntax.stdout:
        print(syntax.stdout)
    if syntax.stderr:
        print(syntax.stderr)
    if not bool(args.skip_submit_smoke):
        smoke_cmd = [
            sys.executable,
            __file__,
            "--smoke-test",
            "--variants",
            ",".join(VARIANTS),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--epochs",
            "1",
            "--max-train-examples",
            "2",
            "--max-eval-examples",
            "2",
            "--output-root",
            os.fspath(parent_output_root),
        ]
        smoke = run_checked(smoke_cmd, cwd=PROJECT_ROOT)
        print(smoke.stdout)
        if smoke.stderr:
            print(smoke.stderr)
    slurm_script = PROJECT_ROOT / "scripts" / "slurm" / "layerwise_glstm_mechanism_ablation.sbatch"
    aggregate_script = PROJECT_ROOT / "scripts" / "slurm" / "layerwise_glstm_mechanism_ablation_aggregate.sbatch"
    variant_jobs: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        job_name = f"lgma_{variant}"
        result = run_checked(
            [
                "sbatch",
                "--parsable",
                "--job-name",
                job_name,
                os.fspath(slurm_script),
                variant,
            ],
            cwd=PROJECT_ROOT,
        )
        job_id = result.stdout.strip().splitlines()[-1].split(";")[0]
        variant_jobs[variant] = {
            "job_id": job_id,
            "job_name": job_name,
            "slurm_stdout": os.fspath(parent_output_root / "slurm" / f"{job_name}-{job_id}.out"),
            "slurm_stderr": os.fspath(parent_output_root / "slurm" / f"{job_name}-{job_id}.err"),
        }
        print(f"Submitted {variant}: {job_id}")
    dependency = "afterok:" + ":".join(job["job_id"] for job in variant_jobs.values())
    aggregate_result = run_checked(
        [
            "sbatch",
            "--parsable",
            "--dependency",
            dependency,
            "--job-name",
            "lgma_aggregate",
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
        "variant_jobs": variant_jobs,
        "aggregate_job": {
            "job_id": aggregate_job_id,
            "job_name": "lgma_aggregate",
            "dependency": dependency,
            "slurm_stdout": os.fspath(parent_output_root / "slurm" / f"lgma_aggregate-{aggregate_job_id}.out"),
            "slurm_stderr": os.fspath(parent_output_root / "slurm" / f"lgma_aggregate-{aggregate_job_id}.err"),
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
    dataset_dir, examples, dataset_manifest = ensure_dataset(args, parent_output_root / "cache")
    dataset_manifest = {**dataset_manifest, "dataset_dir": os.fspath(dataset_dir)}
    if bool(args.prepare_dataset_only):
        print(f"Prepared dataset {dataset_manifest['dataset_hash']} at {dataset_dir}")
        return 0
    variants = parse_variants(args.variants)
    if bool(args.smoke_test):
        args.tiny_debug_model = True
        if str(args.device) == "cuda":
            args.device = "cpu"
        args.epochs = min(int(args.epochs), 1)
        args.grad_accum = 1
        args.max_train_examples = 2 if int(args.max_train_examples) <= 0 else min(int(args.max_train_examples), 2)
        args.max_eval_examples = 2 if int(args.max_eval_examples) <= 0 else min(int(args.max_eval_examples), 2)
    device = base.resolve_device(str(args.device))
    dtype = base.dtype_from_arg(str(args.dtype), device)
    model, processor, load_in_4bit, load_mode = base.load_model_and_processor(args, device=device, dtype=dtype)
    tokenizer = processor.tokenizer
    tokenization_mode, answer_ids = base.text_base.answer_token_ids(tokenizer, int(args.candidate_min), int(args.candidate_max))
    print(
        "Reusing exact frame-message extraction, carrier detection, carrier-gated LoRA, "
        "candidate scoring, diagnostics utilities, and frozen-Qwen checks from layerwise_frame_message_glstm."
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
                count_values=count_values,
                device=device,
                timestamp=timestamp,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(parent_output_root / f"{timestamp}_run_summaries.json", run_summaries)
    if not bool(args.no_aggregate_after_run) and not bool(args.smoke_test):
        aggregate_parent_outputs(parent_output_root)
    verify_no_pdfs(parent_output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

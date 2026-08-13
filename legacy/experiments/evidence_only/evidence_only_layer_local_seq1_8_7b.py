#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
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
from PIL import Image

from evaluations.helpers import utils as eval_utils
from experiments.oracle_bounds import translator_ablation_gold_count_seq8_7b as trans
from experiments.carrier_probes import run_message_memory_adapter_stage1_stage3_seq8 as prev
from experiments.carrier_probes import run_message_memory_carrier_update_seq8_7b as carrier


EXPERIMENT_NAME = "evidence_only_layer_local_seq1_8_7b"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images_park_evidence_only_seq1_8"
DEFAULT_SOURCE_DATASET_ROOT = PROJECT_ROOT / "data" / "mmred_images_park"

BASELINE = "baseline"
LAYER_LOCAL = "layer_local"
SIGMOID_GATE_READOUT = "sigmoid_gate"
RAW_MATRIX_READOUT = "raw_matrix"
COUNT_VALUES = list(range(9))
DEFAULT_TOKEN_GROUP = carrier.ALL_QUESTION_TOKEN_GROUP
TOKEN_GROUP_CHOICES = [carrier.ROOM_CHAR_TOKEN_GROUP, carrier.ALL_QUESTION_TOKEN_GROUP]


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
            "Evidence-only MMReD/Qwen2.5-VL-7B counting experiment for seq_len 1..8. "
            "Every frame is evidence, so gold_count=evidence_count=seq_len."
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
    parser.add_argument("--readout-mode", default=RAW_MATRIX_READOUT, choices=[SIGMOID_GATE_READOUT, RAW_MATRIX_READOUT])
    parser.add_argument("--token-group", default=DEFAULT_TOKEN_GROUP, choices=TOKEN_GROUP_CHOICES)

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


def split_int_tokens(raw_values: Sequence[Any]) -> List[int]:
    values: List[int] = []
    for raw in raw_values:
        for part in str(raw).replace(",", " ").split():
            if part.strip():
                values.append(int(part))
    return sorted(dict.fromkeys(values))


def safe_name(text: Any) -> str:
    safe = str(text)
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


def default_output_dir(args: argparse.Namespace) -> Path:
    token_group = carrier.canonical_token_group(getattr(args, "token_group", DEFAULT_TOKEN_GROUP))
    if str(args.run_name).strip():
        name = str(args.run_name).strip()
    elif str(getattr(args, "readout_mode", SIGMOID_GATE_READOUT)) == RAW_MATRIX_READOUT:
        name = f"{safe_name(token_group)}_messages_raw_matrix_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        name = f"{safe_name(token_group)}_messages_{time.strftime('%Y%m%d_%H%M%S')}"
    return Path(args.output_root).resolve() / safe_name(name)


def sample_dirs(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir() and (p / "qa.txt").is_file()], key=lambda p: p.name)


def metadata_for_sample(sample_dir: Path) -> Dict[str, Any]:
    path = sample_dir / "metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sample_dir(sample_dir: Path, seq_len: int) -> Tuple[bool, Dict[str, Any]]:
    meta = metadata_for_sample(sample_dir)
    question, states, gold_count = prev.parse_qa_file(sample_dir)
    evidence_indices = eval_utils.collect_evidence_frame_indices(question, states)
    frame_paths = [sample_dir / f"{idx:03d}.png" for idx in range(int(seq_len))]
    parsed = eval_utils.parse_target_character_room(question)
    target_ok = False
    if parsed is not None:
        character, room = parsed
        target_ok = all(
            character in eval_utils.rooms_to_room2chars(state.get("rooms", {})).get(room, [])
            for state in states
        )
    checks = {
        "sample_id": sample_dir.name,
        "metadata_evidence_count": meta.get("evidence_count"),
        "metadata_answer": meta.get("answer"),
        "gold_count": int(gold_count),
        "derived_evidence_count": len(evidence_indices),
        "states_len": len(states),
        "frame_paths_len": len(meta.get("frame_paths", [])) if isinstance(meta.get("frame_paths"), list) else None,
        "legacy_frame_paths_len": len(meta.get("legacy_frame_paths", [])) if isinstance(meta.get("legacy_frame_paths"), list) else None,
        "actual_legacy_frames_exist": all(path.is_file() for path in frame_paths),
        "target_character_in_target_room_every_frame": bool(target_ok),
    }
    ok = (
        int(gold_count) == int(seq_len)
        and len(evidence_indices) == int(seq_len)
        and len(states) == int(seq_len)
        and checks["actual_legacy_frames_exist"]
        and bool(target_ok)
        and int(meta.get("evidence_count", seq_len)) == int(seq_len)
        and int(meta.get("answer", seq_len)) == int(seq_len)
        and len(meta.get("frame_paths", list(range(seq_len)))) == int(seq_len)
    )
    return bool(ok), checks


def validate_evidence_only_dataset(
    dataset_root: Path,
    seq_lens: Sequence[int],
    requested_samples_per_seq_len: int,
) -> Tuple[bool, Dict[str, Any]]:
    per_seq: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    total = 0
    for seq_len in seq_lens:
        split_root = Path(dataset_root) / f"seq_len_{int(seq_len)}" / "all_uniform"
        dirs = sample_dirs(split_root)
        seq_checks: List[Dict[str, Any]] = []
        seq_ok = bool(dirs)
        for sample_dir in dirs:
            try:
                ok, checks = verify_sample_dir(sample_dir, int(seq_len))
            except Exception as exc:
                ok, checks = False, {"sample_id": sample_dir.name, "error": f"{type(exc).__name__}: {exc}"}
            seq_ok = seq_ok and ok
            if len(seq_checks) < 20 or not ok:
                seq_checks.append(checks)
        expected_floor = min(100, int(requested_samples_per_seq_len))
        if len(dirs) < expected_floor:
            seq_ok = False
        total += len(dirs)
        per_seq[str(int(seq_len))] = {
            "n": len(dirs),
            "ok": bool(seq_ok),
            "examples": seq_checks[:50],
        }
        all_ok = all_ok and seq_ok
    payload = {
        "dataset_root": os.fspath(Path(dataset_root).resolve()),
        "seq_lens": [int(x) for x in seq_lens],
        "requested_samples_per_seq_len": int(requested_samples_per_seq_len),
        "samples_per_seq_len": {seq: row["n"] for seq, row in per_seq.items()},
        "total_samples": int(total),
        "verification": {
            "all_checks_passed": bool(all_ok),
            "evidence_count_equals_seq_len_for_every_sample": bool(all_ok),
            "gold_count_equals_seq_len_for_every_sample": bool(all_ok),
            "frame_paths_len_equals_seq_len_for_every_sample": bool(all_ok),
            "target_character_in_target_room_every_frame": bool(all_ok),
        },
        "per_seq_len": per_seq,
    }
    return bool(all_ok), payload


def copy_sample_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def generate_evidence_only_dataset(
    *,
    dataset_root: Path,
    source_dataset_root: Path,
    seq_lens: Sequence[int],
    samples_per_seq_len: int,
    force: bool,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / "dataset_manifest.json"
    if manifest_path.is_file() and not force:
        ok, manifest = validate_evidence_only_dataset(dataset_root, seq_lens, int(samples_per_seq_len))
        if ok:
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            existing_seq_lens = {int(x) for x in existing.get("seq_lens", [])}
            requested_seq_lens = {int(x) for x in seq_lens}
            if not existing_seq_lens or requested_seq_lens.issuperset(existing_seq_lens):
                write_json(manifest_path, manifest)
                return manifest
            print(f"Dataset already verified; skipping generation: {dataset_root}")
            return existing
        raise RuntimeError(
            f"Existing dataset at {dataset_root} did not pass evidence-only validation. "
            "Pass --force-generate to replace it."
        )
    if dataset_root.exists() and force:
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    actual_counts: Dict[str, int] = {}
    for seq_len in seq_lens:
        src_root = (
            Path(source_dataset_root)
            / f"seq_len_{int(seq_len)}"
            / "by_evidence_count"
            / f"exact_{int(seq_len)}"
            / "all_uniform"
        )
        candidates = sample_dirs(src_root)
        if len(candidates) < min(100, int(samples_per_seq_len)):
            raise RuntimeError(
                f"Need at least {min(100, int(samples_per_seq_len))} evidence-only samples for "
                f"seq_len={seq_len}, found {len(candidates)} under {src_root}"
            )
        selected = candidates[: min(int(samples_per_seq_len), len(candidates))]
        actual_counts[str(int(seq_len))] = len(selected)
        dst_all = dataset_root / f"seq_len_{int(seq_len)}" / "all_uniform"
        dst_exact = (
            dataset_root
            / f"seq_len_{int(seq_len)}"
            / "by_evidence_count"
            / f"exact_{int(seq_len)}"
            / "all_uniform"
        )
        dst_all.mkdir(parents=True, exist_ok=True)
        dst_exact.mkdir(parents=True, exist_ok=True)
        for sample_dir in selected:
            copy_sample_tree(sample_dir, dst_all / sample_dir.name)
            copy_sample_tree(sample_dir, dst_exact / sample_dir.name)
        print(f"Generated seq_len={seq_len}: copied {len(selected)} evidence-only samples")

    ok, manifest = validate_evidence_only_dataset(dataset_root, seq_lens, int(samples_per_seq_len))
    manifest.update(
        {
            "source_dataset_root": os.fspath(Path(source_dataset_root).resolve()),
            "generation_mode": "copy_existing_exact_evidence_park_samples",
            "actual_samples_per_seq_len": actual_counts,
        }
    )
    write_json(manifest_path, manifest)
    if not ok:
        raise RuntimeError(f"Generated dataset failed validation: {manifest_path}")
    return manifest


def load_records_for_seq_len(dataset_root: Path, seq_len: int, limit: int = 0) -> List[prev.SampleRecord]:
    split_root = Path(dataset_root) / f"seq_len_{int(seq_len)}" / "all_uniform"
    dirs = sample_dirs(split_root)
    if int(limit) > 0:
        dirs = dirs[: int(limit)]
    records: List[prev.SampleRecord] = []
    for sample_dir in dirs:
        question, states, gold_count = prev.parse_qa_file(sample_dir)
        evidence_count = len(eval_utils.collect_evidence_frame_indices(question, states))
        frame_paths = tuple(sample_dir / f"{idx:03d}.png" for idx in range(len(states)))
        records.append(
            prev.SampleRecord(
                sample_id=sample_dir.name,
                sample_dir=sample_dir,
                frame_paths=frame_paths,
                question=question,
                states=tuple(states),
                gold_count=int(gold_count),
                evidence_count=int(evidence_count),
            )
        )
    if not records:
        raise RuntimeError(f"No samples found under {split_root}")
    return records


def load_all_records(dataset_root: Path, seq_lens: Sequence[int]) -> Tuple[List[prev.SampleRecord], Dict[int, List[int]]]:
    records: List[prev.SampleRecord] = []
    by_seq: Dict[int, List[int]] = {}
    for seq_len in seq_lens:
        seq_records = load_records_for_seq_len(dataset_root, int(seq_len))
        start = len(records)
        records.extend(seq_records)
        by_seq[int(seq_len)] = list(range(start, start + len(seq_records)))
    return records, by_seq


def split_indices_for_seq(
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    seed: int,
) -> Dict[str, List[int]]:
    ordered = sorted(
        [int(idx) for idx in indices],
        key=lambda idx: prev.stable_hash_int(f"{int(seed)}:{records[idx].sample_id}"),
    )
    n = len(ordered)
    if n == 1:
        return {"train": ordered, "val": [], "test": []}
    if n == 2:
        return {"train": ordered[:1], "val": ordered[1:], "test": []}
    n_val = max(1, int(round(0.15 * n)))
    n_test = max(1, int(round(0.15 * n)))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def limit_indices(indices: Sequence[int], records: Sequence[prev.SampleRecord], limit: int, seed: int) -> List[int]:
    values = [int(idx) for idx in indices]
    if int(limit) <= 0 or len(values) <= int(limit):
        return values
    values = sorted(values, key=lambda idx: prev.stable_hash_int(f"{int(seed)}:{records[idx].sample_id}"))
    return sorted(values[: int(limit)], key=lambda idx: records[idx].sample_id)


def make_splits(
    records: Sequence[prev.SampleRecord],
    by_seq: Dict[int, List[int]],
    *,
    seed: int,
    max_train_per_seq: int,
    max_eval_per_seq: int,
) -> Dict[str, List[int]]:
    splits = {"train": [], "val": [], "test": []}
    for seq_len in sorted(by_seq):
        seq_split = split_indices_for_seq(records, by_seq[seq_len], int(seed) + seq_len * 1009)
        splits["train"].extend(
            limit_indices(seq_split["train"], records, int(max_train_per_seq), int(seed) + seq_len * 11)
        )
        splits["val"].extend(
            limit_indices(seq_split["val"], records, int(max_eval_per_seq), int(seed) + seq_len * 17)
        )
        splits["test"].extend(
            limit_indices(seq_split["test"], records, int(max_eval_per_seq), int(seed) + seq_len * 23)
        )
    for split in splits:
        splits[split] = sorted(splits[split], key=lambda idx: (len(records[idx].frame_paths), records[idx].sample_id))
    return splits


def homogeneous_batches(
    indices: Sequence[int],
    records: Sequence[prev.SampleRecord],
    batch_size: int,
    *,
    seed: int,
    shuffle_batches: bool,
) -> List[List[int]]:
    by_seq: Dict[int, List[int]] = defaultdict(list)
    rng = random.Random(int(seed))
    for idx in indices:
        by_seq[len(records[int(idx)].frame_paths)].append(int(idx))
    batches: List[List[int]] = []
    for seq_len in sorted(by_seq):
        values = list(by_seq[seq_len])
        values.sort(key=lambda idx: records[idx].sample_id)
        if shuffle_batches:
            rng.shuffle(values)
        for start in range(0, len(values), max(1, int(batch_size))):
            batches.append(values[start : start + max(1, int(batch_size))])
    if shuffle_batches:
        rng.shuffle(batches)
    return batches


def parse_json_field(row: Dict[str, Any], key: str, default: Any) -> Any:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return json.loads(str(value))
    except Exception:
        return default


def flatten_numeric(value: Any) -> List[float]:
    if isinstance(value, dict):
        out: List[float] = []
        for item in value.values():
            out.extend(flatten_numeric(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_numeric(item))
        return out
    if finite_float(value) is not None:
        return [float(value)]
    return []


def blank_diagnostics(layers: Sequence[int], seq_len: int) -> Dict[str, Any]:
    return {
        "gate_values_by_layer": {str(layer): [] for layer in layers},
        "gate_sum_by_layer": {str(layer): 0.0 for layer in layers},
        "matrix_scores_by_layer": {str(layer): [] for layer in layers},
        "matrix_score_sum_by_layer": {str(layer): 0.0 for layer in layers},
        "matrix_score_abs_sum_by_layer": {str(layer): 0.0 for layer in layers},
        "matrix_score_mean_by_layer": {str(layer): 0.0 for layer in layers},
        "matrix_score_abs_mean_by_layer": {str(layer): 0.0 for layer in layers},
        "update_norm_by_layer": {str(layer): 0.0 for layer in layers},
        "memory_norm_by_layer": {str(layer): [] for layer in layers},
        "message_norm_by_layer": {str(layer): [] for layer in layers},
        "raw_message_norm_by_layer": {str(layer): [] for layer in layers},
        "message_mode_by_layer": {str(layer): "none" for layer in layers},
        "seq_len": int(seq_len),
    }


def mean_layer_frame_value(layer_json: Dict[str, Any]) -> float:
    values: List[float] = []
    for payload in layer_json.values():
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, list):
                    values.extend(float(x) for x in item if finite_float(x) is not None)
                elif finite_float(item) is not None:
                    values.append(float(item))
        elif finite_float(payload) is not None:
            values.append(float(payload))
    return float(np.mean(values)) if values else 0.0


def evidence_frame_mask(record: prev.SampleRecord) -> List[int]:
    try:
        indices = eval_utils.collect_evidence_frame_indices(record.question, list(record.states))
    except Exception:
        return []
    clean = {int(idx) for idx in indices if 0 <= int(idx) < len(record.frame_paths)}
    return [1 if idx in clean else 0 for idx in range(len(record.frame_paths))]


def load_frames(record: prev.SampleRecord) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for frame_path in record.frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))
    return frames


def prepare_batch(
    *,
    records: Sequence[prev.SampleRecord],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
    token_group: str,
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
        token_group=token_group,
    )


def select_count_logits(outputs: Any, prompt_last_indices: torch.Tensor, count_token_ids: Dict[int, int]) -> torch.Tensor:
    return prev.select_count_logits(outputs.logits, prompt_last_indices, count_token_ids)


@torch.no_grad()
def evaluate_model(
    *,
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
    token_group: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    ce_total = 0.0
    n = 0
    count_min = min(COUNT_VALUES)
    resolved_token_group = carrier.canonical_token_group(token_group)

    if adapter is not None:
        adapter.to(device)
        adapter.eval()
        adapter.enabled = True
        adapter.register_hooks(model)

    try:
        batches = homogeneous_batches(indices, records, int(batch_size), seed=int(seed), shuffle_batches=False)
        for batch_num, batch_indices in enumerate(batches, start=1):
            batch_records = [records[int(idx)] for idx in batch_indices]
            seq_len = len(batch_records[0].frame_paths)
            carrier.NUM_FRAMES = int(seq_len)
            batch = prepare_batch(
                records=batch_records,
                sample_indices=batch_indices,
                processor=processor,
                device=device,
                token_group=resolved_token_group,
            )
            if adapter is not None:
                adapter.set_context(
                    target_positions=batch.target_positions,
                    message_target_positions=batch.message_target_positions,
                    query_positions=batch.query_positions,
                    inject_positions=batch.inject_positions,
                    frame_groups=batch.frame_groups,
                )
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
                diag = adapter.stats_for_row(row_idx) if adapter is not None else blank_diagnostics(inject_layers, seq_len)
                readout_mode = str(getattr(adapter, "readout_mode", "none")) if adapter is not None else "none"
                gate_sum_by_layer = diag.get("gate_sum_by_layer", {})
                matrix_scores_by_layer = diag.get("matrix_scores_by_layer", {})
                matrix_score_sum_by_layer = diag.get("matrix_score_sum_by_layer", {})
                matrix_score_abs_sum_by_layer = diag.get("matrix_score_abs_sum_by_layer", {})
                matrix_score_mean_by_layer = diag.get("matrix_score_mean_by_layer", {})
                matrix_score_abs_mean_by_layer = diag.get("matrix_score_abs_mean_by_layer", {})
                update_norm_by_layer = diag.get("update_norm_by_layer", {})
                message_norm_by_layer = diag.get("message_norm_by_layer", {})
                memory_norm_by_layer = diag.get("memory_norm_by_layer", {})
                update_norm = finite_mean(update_norm_by_layer.values(), default=0.0)
                message_norm = mean_layer_frame_value(message_norm_by_layer)
                memory_norm = mean_layer_frame_value(memory_norm_by_layer)
                pred_offset = pred - int(count_min)
                rows.append(
                    {
                        "method": str(method),
                        "sample_id": record.sample_id,
                        "sample_index": int(sample_idx),
                        "token_group": resolved_token_group,
                        "readout_mode": readout_mode,
                        "seq_len": int(seq_len),
                        "gold_count": int(gold),
                        "evidence_count": int(record.evidence_count),
                        "pred_count": int(pred),
                        "correct": int(pred == gold),
                        "margin": float(margins[row_idx].detach().cpu().item()),
                        "gold_logit": float(gold_logits[row_idx].detach().cpu().item()),
                        "pred_logit": logits_list[pred_offset] if 0 <= pred_offset < len(logits_list) else math.nan,
                        "candidate_logits_json": json_compact(logits_map),
                        "split": str(split_name),
                        "ce": float(ce_vec[row_idx].detach().cpu().item()),
                        "mean_gate_sum": finite_mean(gate_sum_by_layer.values(), default=0.0)
                        if adapter is not None
                        else "",
                        "mean_matrix_score_sum": finite_mean(matrix_score_sum_by_layer.values(), default=0.0)
                        if adapter is not None
                        else "",
                        "mean_matrix_score_abs_sum": finite_mean(matrix_score_abs_sum_by_layer.values(), default=0.0)
                        if adapter is not None
                        else "",
                        "mean_matrix_score_mean": finite_mean(matrix_score_mean_by_layer.values(), default=0.0)
                        if adapter is not None
                        else "",
                        "mean_matrix_score_abs_mean": finite_mean(matrix_score_abs_mean_by_layer.values(), default=0.0)
                        if adapter is not None
                        else "",
                        "update_norm": float(update_norm) if adapter is not None else "",
                        "message_norm": float(message_norm) if adapter is not None else "",
                        "memory_norm": float(memory_norm) if adapter is not None else "",
                        "gate_values_by_layer_json": json_compact(diag.get("gate_values_by_layer", {}))
                        if adapter is not None
                        else "",
                        "update_norm_by_layer_json": json_compact(update_norm_by_layer) if adapter is not None else "",
                        "message_norm_by_layer_json": json_compact(message_norm_by_layer) if adapter is not None else "",
                        "memory_norm_by_layer_json": json_compact(memory_norm_by_layer) if adapter is not None else "",
                        "gate_sum_by_layer_json": json_compact(gate_sum_by_layer) if adapter is not None else "",
                        "matrix_scores_by_layer_json": json_compact(matrix_scores_by_layer) if adapter is not None else "",
                        "matrix_score_sum_by_layer_json": json_compact(matrix_score_sum_by_layer)
                        if adapter is not None
                        else "",
                        "matrix_score_abs_sum_by_layer_json": json_compact(matrix_score_abs_sum_by_layer)
                        if adapter is not None
                        else "",
                        "matrix_score_mean_by_layer_json": json_compact(matrix_score_mean_by_layer)
                        if adapter is not None
                        else "",
                        "matrix_score_abs_mean_by_layer_json": json_compact(matrix_score_abs_mean_by_layer)
                        if adapter is not None
                        else "",
                        "raw_message_norm_by_layer_json": json_compact(diag.get("raw_message_norm_by_layer", {}))
                        if adapter is not None
                        else "",
                        "message_mode_by_layer_json": json_compact(diag.get("message_mode_by_layer", {}))
                        if adapter is not None
                        else "",
                        "target_positions_json": json_compact(batch.target_positions[row_idx]),
                        "room_char_positions_json": json_compact(batch.target_positions[row_idx])
                        if resolved_token_group == carrier.ROOM_CHAR_TOKEN_GROUP
                        else "",
                        "num_target_positions": len(batch.target_positions[row_idx]),
                        "frame_token_counts_json": json_compact(batch.frame_token_counts[row_idx]),
                        "evidence_frame_mask_json": json_compact(evidence_frame_mask(record)),
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
    checkpoint_path = ckpt_dir / "layer_local_best.pt"
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_acc = -math.inf
    best_val_ce = math.inf
    backward_diag: Dict[str, Any] = {}
    count_min = min(COUNT_VALUES)

    for epoch in range(1, int(args.epochs) + 1):
        adapter.train()
        adapter.enabled = True
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
                    records=batch_records,
                    sample_indices=batch_indices,
                    processor=processor,
                    device=device,
                    token_group=args.token_group,
                )
                if not any(batch.target_positions) or not any(batch.frame_grouping_ok):
                    skipped += 1
                adapter.set_context(
                    target_positions=batch.target_positions,
                    message_target_positions=batch.message_target_positions,
                    query_positions=batch.query_positions,
                    inject_positions=batch.inject_positions,
                    frame_groups=batch.frame_groups,
                )
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
                    print(f"  first backward diagnostics: {json_compact(backward_diag)}")
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
                        f"  layer_local epoch={epoch} step={step}/{len(train_batches)} "
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
            token_group=args.token_group,
        )
        row = {
            "method": LAYER_LOCAL,
            "token_group": str(args.token_group),
            "readout_mode": str(args.readout_mode),
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
            "gamma_json": json_compact([float(x) for x in adapter.gamma.detach().float().cpu().tolist()]),
        }
        history.append(row)
        print(
            f"  layer_local epoch={epoch} train_ce={row['train_ce']:.4f} "
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
                    "token_group": str(args.token_group),
                    "message_mode": str(args.message_mode),
                    "readout_mode": str(args.readout_mode),
                },
                checkpoint_path,
            )
    if best_state is not None:
        adapter.load_state_dict(best_state)
    return adapter, history, backward_diag, checkpoint_path


def prediction_histogram(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    hist = {str(count): 0 for count in COUNT_VALUES}
    for row in rows:
        key = str(int(row["pred_count"]))
        hist[key] = hist.get(key, 0) + 1
    return hist


def infer_readout_mode(rows: Sequence[Dict[str, Any]], default: str = "none") -> str:
    modes = sorted({str(row.get("readout_mode", "")).strip() for row in rows if str(row.get("readout_mode", "")).strip()})
    if not modes:
        return str(default)
    if len(modes) == 1:
        return modes[0]
    return json_compact(modes)


def infer_token_group(rows: Sequence[Dict[str, Any]], default: str = DEFAULT_TOKEN_GROUP) -> str:
    groups = sorted({str(row.get("token_group", "")).strip() for row in rows if str(row.get("token_group", "")).strip()})
    if not groups:
        return str(default)
    if len(groups) == 1:
        return groups[0]
    return json_compact(groups)


def summarize_method(
    rows: Sequence[Dict[str, Any]],
    *,
    method: str,
    train_history: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    correct = [int(row["correct"]) for row in rows]
    train_last = dict(train_history[-1]) if train_history else {}
    readout_mode = infer_readout_mode(rows, default="none")
    token_group = infer_token_group(rows)
    return {
        "method": str(method),
        "token_group": token_group,
        "readout_mode": readout_mode,
        "n": len(rows),
        "accuracy": float(np.mean(correct)) if correct else math.nan,
        "mean_margin": finite_mean(row.get("margin") for row in rows),
        "mean_gold_logit": finite_mean(row.get("gold_logit") for row in rows),
        "mean_pred_count": finite_mean(row.get("pred_count") for row in rows),
        "mean_matrix_score_sum": finite_mean((row.get("mean_matrix_score_sum") for row in rows), default=0.0)
        if method == LAYER_LOCAL
        else 0.0,
        "mean_matrix_score_abs_sum": finite_mean((row.get("mean_matrix_score_abs_sum") for row in rows), default=0.0)
        if method == LAYER_LOCAL
        else 0.0,
        "mean_update_norm": finite_mean((row.get("update_norm") for row in rows), default=0.0)
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
                "token_group": infer_token_group(seq_rows),
                "readout_mode": infer_readout_mode(seq_rows, default="none"),
                "seq_len": int(seq_len),
                "gold_count": int(seq_len),
                "n": len(seq_rows),
                "accuracy": float(np.mean(correct)) if correct else math.nan,
                "mean_margin": finite_mean(row.get("margin") for row in seq_rows),
                "mean_pred_count": finite_mean(row.get("pred_count") for row in seq_rows),
                "prediction_histogram": json_compact(prediction_histogram(seq_rows)),
                "mean_gate_sum": finite_mean((row.get("mean_gate_sum") for row in seq_rows), default=0.0)
                if method == LAYER_LOCAL
                else 0.0,
                "mean_matrix_score_sum": finite_mean(
                    (row.get("mean_matrix_score_sum") for row in seq_rows),
                    default=0.0,
                )
                if method == LAYER_LOCAL
                else 0.0,
                "mean_matrix_score_abs_sum": finite_mean(
                    (row.get("mean_matrix_score_abs_sum") for row in seq_rows),
                    default=0.0,
                )
                if method == LAYER_LOCAL
                else 0.0,
                "mean_update_norm": finite_mean((row.get("update_norm") for row in seq_rows), default=0.0)
                if method == LAYER_LOCAL
                else 0.0,
                "mean_message_norm": finite_mean((row.get("message_norm") for row in seq_rows), default=0.0)
                if method == LAYER_LOCAL
                else 0.0,
                "mean_memory_norm": finite_mean((row.get("memory_norm") for row in seq_rows), default=0.0)
                if method == LAYER_LOCAL
                else 0.0,
            }
        )
    return out


def comparison_by_seq_len(accuracy_rows: Sequence[Dict[str, Any]], seq_lens: Sequence[int]) -> List[Dict[str, Any]]:
    by_key = {(row["method"], int(row["seq_len"])): row for row in accuracy_rows}
    out: List[Dict[str, Any]] = []
    for seq_len in seq_lens:
        base = by_key.get((BASELINE, int(seq_len)), {})
        local = by_key.get((LAYER_LOCAL, int(seq_len)), {})
        base_acc = finite_float(base.get("accuracy"))
        local_acc = finite_float(local.get("accuracy"))
        out.append(
            {
                "seq_len": int(seq_len),
                "gold_count": int(seq_len),
                "baseline_accuracy": "" if base_acc is None else float(base_acc),
                "layer_local_accuracy": "" if local_acc is None else float(local_acc),
                "delta_accuracy": ""
                if base_acc is None or local_acc is None
                else float(local_acc) - float(base_acc),
                "baseline_mean_pred": base.get("mean_pred_count", ""),
                "layer_local_mean_pred": local.get("mean_pred_count", ""),
                "baseline_mean_margin": base.get("mean_margin", ""),
                "layer_local_mean_margin": local.get("mean_margin", ""),
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
    methods = [BASELINE, LAYER_LOCAL]
    plt.figure(figsize=(7.2, 4.5))
    for method in methods:
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
    payload = parse_json_field(row, "candidate_logits_json", {})
    if isinstance(payload, dict):
        return {str(k): float(v) for k, v in payload.items() if finite_float(v) is not None}
    if isinstance(payload, list):
        return {str(i): float(v) for i, v in enumerate(payload) if finite_float(v) is not None}
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
            means.append(finite_mean(vals))
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
    local_readout_mode = infer_readout_mode(
        [row for row in metrics_rows if row.get("method") == LAYER_LOCAL],
        default=SIGMOID_GATE_READOUT,
    )
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
        "Layer-Local Confusion Matrix",
    )
    save_combined_confusions(plots_dir / "combined_confusion_matrices.png", metrics_rows, seq_lens)

    comp = comparison_by_seq_len(accuracy_rows, seq_lens)
    plt.figure(figsize=(7.2, 4.3))
    xs = [int(row["seq_len"]) for row in comp]
    ys = [float(row["delta_accuracy"]) if finite_float(row.get("delta_accuracy")) is not None else math.nan for row in comp]
    colors = ["#2ca02c" if finite_float(y) is not None and float(y) >= 0 else "#d62728" for y in ys]
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
        ("mean_update_norm", "update_norm_vs_seq_len.png", "Mean update norm", "Layer-Local Update Norm vs Seq Len"),
        ("mean_message_norm", "message_norm_vs_seq_len.png", "Mean message norm", "Layer-Local Message Norm vs Seq Len"),
        ("mean_memory_norm", "memory_norm_vs_seq_len.png", "Mean memory norm", "Layer-Local Memory Norm vs Seq Len"),
    ]
    if local_readout_mode == RAW_MATRIX_READOUT:
        diagnostic_specs = [
            (
                "mean_matrix_score_sum",
                "matrix_score_sum_vs_seq_len.png",
                "Mean matrix score sum",
                "Raw Matrix Score Sum vs Seq Len",
            ),
            (
                "mean_matrix_score_abs_sum",
                "matrix_score_abs_sum_vs_seq_len.png",
                "Mean abs matrix score sum",
                "Raw Matrix Abs Score Sum vs Seq Len",
            ),
            *diagnostic_specs,
        ]
    else:
        diagnostic_specs = [
            ("mean_gate_sum", "gate_sum_vs_seq_len.png", "Mean gate sum", "Layer-Local Gate Sum vs Seq Len"),
            *diagnostic_specs,
        ]
    for key, filename, ylabel, title in diagnostic_specs:
        rows = sorted([row for row in accuracy_rows if row["method"] == LAYER_LOCAL], key=lambda row: int(row["seq_len"]))
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
        pred = finite_float(row.get("mean_pred_count"))
        gold = finite_float(row.get("gold_count"))
        if pred is not None and gold is not None:
            vals.append(abs(float(pred) - float(gold)))
    return finite_mean(vals)


def high_count_accuracy(accuracy_rows: Sequence[Dict[str, Any]], method: str, threshold: int = 4) -> float:
    vals = [
        finite_float(row.get("accuracy"))
        for row in accuracy_rows
        if row.get("method") == method and int(row.get("seq_len", 0)) >= int(threshold)
    ]
    return finite_mean(v for v in vals if v is not None)


def low_count_accuracy(accuracy_rows: Sequence[Dict[str, Any]], method: str, threshold: int = 3) -> float:
    vals = [
        finite_float(row.get("accuracy"))
        for row in accuracy_rows
        if row.get("method") == method and int(row.get("seq_len", 0)) <= int(threshold)
    ]
    return finite_mean(v for v in vals if v is not None)


def strongest_prediction_attractor(rows: Sequence[Dict[str, Any]], method: str) -> Tuple[Optional[int], float]:
    preds = [int(row["pred_count"]) for row in rows if row.get("method") == method]
    if not preds:
        return None, math.nan
    count, n = Counter(preds).most_common(1)[0]
    return int(count), float(n) / len(preds)


def write_readme(output_dir: Path, summary_rows: Sequence[Dict[str, Any]], accuracy_rows: Sequence[Dict[str, Any]], metrics_rows: Sequence[Dict[str, Any]]) -> None:
    summary = {row["method"]: row for row in summary_rows}
    token_group = infer_token_group(
        [row for row in metrics_rows if row.get("method") == LAYER_LOCAL],
        default=str(summary.get(LAYER_LOCAL, {}).get("token_group", DEFAULT_TOKEN_GROUP)),
    )
    readout_mode = infer_readout_mode(
        [row for row in metrics_rows if row.get("method") == LAYER_LOCAL],
        default=str(summary.get(LAYER_LOCAL, {}).get("readout_mode", "none")),
    )
    base_acc = finite_float(summary.get(BASELINE, {}).get("accuracy"))
    local_acc = finite_float(summary.get(LAYER_LOCAL, {}).get("accuracy"))
    base_low = low_count_accuracy(accuracy_rows, BASELINE)
    base_high = high_count_accuracy(accuracy_rows, BASELINE)
    local_high = high_count_accuracy(accuracy_rows, LAYER_LOCAL)
    base_declines = finite_float(base_low) is not None and finite_float(base_high) is not None and base_high < base_low - 0.05
    high_delta = local_high - base_high if finite_float(local_high) is not None and finite_float(base_high) is not None else math.nan
    base_mae = mean_pred_mae(accuracy_rows, BASELINE)
    local_mae = mean_pred_mae(accuracy_rows, LAYER_LOCAL)
    better_diagonal = finite_float(base_mae) is not None and finite_float(local_mae) is not None and local_mae < base_mae
    attractor, attractor_frac = strongest_prediction_attractor(metrics_rows, BASELINE)
    layer_low_accuracy = finite_float(local_acc) is not None and float(local_acc) < 0.5
    signal_exists = layer_low_accuracy and better_diagonal
    both_fail = (
        finite_float(base_acc) is not None
        and finite_float(local_acc) is not None
        and float(base_acc) < 0.35
        and float(local_acc) < 0.35
        and (finite_float(high_delta) is None or abs(float(high_delta)) < 0.05)
    )
    strong_positive = finite_float(high_delta) is not None and float(high_delta) >= 0.10 and better_diagonal

    lines = [
        "# Evidence-Only Layer-Local seq_len 1..8 7B",
        "",
        "This is a pure aggregation/cardinality experiment: every frame is evidence, with no non-evidence distractors.",
        "",
        f"Token group: `{token_group}`.",
        f"Layer-local readout mode: `{readout_mode}`.",
    ]
    if token_group == carrier.ALL_QUESTION_TOKEN_GROUP:
        lines.extend(
            [
                "",
                "This run uses all-question-token exact attention messages.",
                "",
                "Memory slot f contains the exact attention-value contribution from frame f into all question tokens, averaged over the question tokens.",
                "",
                "m_f^l = (1 / |Q|) sum_{q in Q} W_O [ sum_{j in I_f} A^l_{q,j} V^l_j ]",
                "",
                "This is more task-agnostic than room+char because it does not manually choose semantic entities. It is not fully task-agnostic yet because it still injects into question tokens rather than the last token. A future variant should test all-question messages -> last-token injection.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "This run uses exact attention messages into the target room+character token group.",
            ]
        )
    if readout_mode == RAW_MATRIX_READOUT:
        lines.extend(
            [
                "",
                "This variant uses a raw matrix-memory readout:",
                "",
                "C = sum_f v_f k_f^T",
                "r = C q = sum_f (k_f^T q) v_f",
                "",
                "The resulting delta is injected into the selected target token group. For all-question runs, that means all question tokens.",
                "",
                "It avoids softmax normalization and also avoids sigmoid slot selection. The readout can grow with the number of useful frame messages, but may be less stable than sigmoid-gated readout because scores can be negative or large.",
            ]
        )
    elif readout_mode == SIGMOID_GATE_READOUT:
        lines.extend(
            [
                "",
                "This variant uses the sigmoid-gated slot readout: each frame slot receives an independent sigmoid retrieval gate before the value vectors are summed.",
            ]
        )
    lines.extend(
        [
            "",
            "## Automatic Interpretation",
            "",
            (
                f"- Does baseline accuracy decline as seq_len/gold_count increases? {bool(base_declines)} "
                f"(low 1..3={base_low:.4f}, high 4..8={base_high:.4f})."
            ),
            (
                f"- Does layer-local improve high counts, especially seq_len 4..8? "
                f"{finite_float(high_delta) is not None and high_delta > 0.0} "
                f"(baseline high={base_high:.4f}, layer-local high={local_high:.4f}, delta={high_delta:.4f})."
            ),
            (
                f"- Does layer-local mean predicted count follow y=x better than baseline? {bool(better_diagonal)} "
                f"(baseline mean-pred MAE={base_mae:.4f}, layer-local={local_mae:.4f})."
            ),
            (
                f"- Is baseline stuck around attractor counts like 2 or 4? "
                f"{attractor in {2, 4} and attractor_frac >= 0.30} "
                f"(top predicted count={attractor}, fraction={attractor_frac:.3f})."
            ),
        ]
    )
    if signal_exists:
        lines.append(
            "- Layer-local accuracy is still low but mean prediction is closer to gold; this suggests the count signal exists but answer-logit translation is imperfect."
        )
    if strong_positive:
        lines.append("- Layer-local is much better on high counts, supporting the pure aggregation bottleneck hypothesis.")
    if both_fail:
        lines.append("- Both methods fail similarly, suggesting the memory messages/update are not sufficient for pure count aggregation yet.")
    if not signal_exists and not strong_positive and not both_fail:
        lines.append("- The result is mixed; inspect the high-count deltas, logit curves, and diagnostics plots before drawing a strong conclusion.")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metrics.csv`: per-sample logits, predictions, and layer-local diagnostics.",
            "- `accuracy_by_seq_len.csv`: accuracy and prediction histograms by count.",
            "- `comparison_by_seq_len.csv`: baseline vs layer-local deltas.",
            "- `plots/`: combined accuracy, margins, mean predicted counts, confusion matrices, deltas, and diagnostics.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diagnostics(
    *,
    output_dir: Path,
    model: Any,
    adapter: Optional[carrier.MessageMemoryCarrierAdapter],
    train_history: Sequence[Dict[str, Any]],
    backward_diag: Dict[str, Any],
    metrics_rows: Sequence[Dict[str, Any]],
    token_group: str,
) -> None:
    model_trainable_tensors = sum(int(param.requires_grad) for param in model.parameters())
    adapter_trainable_tensors = 0 if adapter is None else sum(int(param.requires_grad) for param in adapter.parameters())
    adapter_trainable_params = 0 if adapter is None else sum(int(param.numel()) for param in adapter.parameters() if param.requires_grad)
    failed = [
        row
        for row in metrics_rows
        if int(row.get("token_selection_ok", 0)) == 0 or int(row.get("frame_grouping_ok", 0)) == 0
    ]
    nonfinite_fields: List[Dict[str, Any]] = []
    numeric_fields = [
        "margin",
        "gold_logit",
        "pred_logit",
        "ce",
        "mean_gate_sum",
        "mean_matrix_score_sum",
        "mean_matrix_score_abs_sum",
        "mean_matrix_score_mean",
        "mean_matrix_score_abs_mean",
        "update_norm",
        "message_norm",
        "memory_norm",
    ]
    for row in metrics_rows:
        for field in numeric_fields:
            value = row.get(field, "")
            if value == "":
                continue
            if finite_float(value) is None:
                nonfinite_fields.append({"sample_id": row.get("sample_id"), "method": row.get("method"), "field": field, "value": value})
                if len(nonfinite_fields) >= 50:
                    break
        if len(nonfinite_fields) >= 50:
            break
    target_counts = [len(parse_json_field(row, "target_positions_json", [])) for row in metrics_rows]
    matrix_values: List[float] = []
    for row in metrics_rows:
        matrix_values.extend(flatten_numeric(parse_json_field(row, "matrix_scores_by_layer_json", {})))
    update_values = [float(row.get("update_norm", 0.0)) for row in metrics_rows if finite_float(row.get("update_norm")) is not None]
    payload = {
        "token_group": carrier.canonical_token_group(token_group),
        "qwen_frozen": int(model_trainable_tensors == 0),
        "model_trainable_tensors": int(model_trainable_tensors),
        "adapter_trainable_tensors": int(adapter_trainable_tensors),
        "adapter_trainable_params": int(adapter_trainable_params),
        "readout_mode": "none" if adapter is None else str(getattr(adapter, "readout_mode", "unknown")),
        "only_adapter_params_trainable": int(model_trainable_tensors == 0 and adapter_trainable_tensors > 0)
        if adapter is not None
        else "",
        "hooks_fire_counts": {} if adapter is None else {str(k): int(v) for k, v in sorted(adapter.hook_fire_counts.items())},
        "message_mode_counts": {} if adapter is None else dict(adapter.message_mode_counts),
        "exact_failure_counts": {} if adapter is None else dict(adapter.exact_failure_counts),
        "exact_failure_examples": [] if adapter is None else list(adapter.exact_failure_examples),
        "backward_diagnostics": backward_diag,
        "train_history_last": dict(train_history[-1]) if train_history else {},
        "target_positions_found": int(bool(target_counts) and all(count > 0 for count in target_counts)),
        "avg_num_target_positions": finite_mean(target_counts, default=0.0),
        "min_num_target_positions": min(target_counts) if target_counts else 0,
        "max_num_target_positions": max(target_counts) if target_counts else 0,
        "num_failed_localization_samples": len(failed),
        "failed_localization_sample_ids": [row.get("sample_id") for row in failed[:50]],
        "matrix_score_diagnostics_populated": int(
            any(str(row.get("matrix_scores_by_layer_json", "")).strip() not in {"", "{}"} for row in metrics_rows)
        ),
        "nonzero_matrix_scores": int(any(abs(value) > 0.0 for value in matrix_values)),
        "finite_matrix_scores": int(bool(matrix_values) and all(math.isfinite(value) for value in matrix_values)),
        "nonzero_updates": int(any(abs(value) > 0.0 for value in update_values)),
        "num_nonfinite_numeric_metrics": len(nonfinite_fields),
        "nonfinite_numeric_metric_examples": nonfinite_fields,
    }
    write_json(output_dir / "diagnostics.json", payload)


def print_split_counts(records: Sequence[prev.SampleRecord], splits: Dict[str, List[int]], seq_lens: Sequence[int]) -> None:
    for split, indices in splits.items():
        counts = Counter(len(records[int(idx)].frame_paths) for idx in indices)
        print(f"  {split}: " + ", ".join(f"seq{seq}:{counts.get(int(seq), 0)}" for seq in seq_lens))


def main() -> int:
    args = parse_args()
    args.token_group = carrier.canonical_token_group(args.token_group)
    seq_lens = split_int_tokens(args.seq_lens)
    if not seq_lens:
        raise ValueError("--seq-lens cannot be empty")
    if any(seq_len < 1 or seq_len > 8 for seq_len in seq_lens):
        raise ValueError("This experiment expects seq_lens within 1..8")
    if int(args.layer_end) < int(args.layer_start):
        raise ValueError("--layer-end must be >= --layer-start")
    if not (args.generate_dataset or args.run_baseline or args.run_layer_local or args.run_all):
        args.run_all = True

    should_generate = bool(args.generate_dataset or args.run_all)
    should_run_baseline = bool(args.run_baseline or args.run_all)
    should_run_layer_local = bool(args.run_layer_local or args.run_all)

    if should_generate:
        generate_evidence_only_dataset(
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
    log_handle, old_stdout, old_stderr = setup_logging(output_dir)
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
            "token_group": str(args.token_group),
            "message_mode": str(args.message_mode),
            "readout_mode": str(args.readout_mode),
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
        write_json(output_dir / "run_config.json", run_config)
        print(f"Output dir: {output_dir}")
        print(f"Run config: {json_compact(run_config)}")

        ok, manifest = validate_evidence_only_dataset(Path(args.dataset_root), seq_lens, int(args.samples_per_seq_len))
        write_json(output_dir / "dataset_manifest_snapshot.json", manifest)
        if not ok:
            raise RuntimeError(f"Dataset failed validation: {args.dataset_root}")

        records, by_seq = load_all_records(Path(args.dataset_root), seq_lens)
        splits = make_splits(
            records,
            by_seq,
            seed=int(args.seed),
            max_train_per_seq=int(args.max_train_samples_per_seq_len),
            max_eval_per_seq=int(args.max_eval_samples_per_seq_len),
        )
        print_split_counts(records, splits, seq_lens)
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
                token_group=args.token_group,
            )
            metrics_rows.extend(baseline_eval["rows"])

        if should_run_layer_local:
            print("Training one shared layer-local adapter across all requested seq_lens")
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
            write_json(
                output_dir / "checkpoint.json",
                {
                    "layer_local_best_checkpoint": os.fspath(checkpoint_path),
                    "token_group": str(args.token_group),
                    "readout_mode": str(args.readout_mode),
                    "message_mode": str(args.message_mode),
                    "inject_layers": inject_layers,
                    "d_mem": int(args.d_mem),
                },
            )
            print("Evaluating shared layer-local adapter on test split")
            layer_eval = evaluate_model(
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
                token_group=args.token_group,
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

        write_csv_dynamic(
            output_dir / "metrics.csv",
            metrics_rows,
            [
                "method",
                "sample_id",
                "token_group",
                "readout_mode",
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
                "mean_gate_sum",
                "mean_matrix_score_sum",
                "mean_matrix_score_abs_sum",
                "mean_matrix_score_mean",
                "mean_matrix_score_abs_mean",
                "update_norm",
                "message_norm",
                "memory_norm",
                "gate_values_by_layer_json",
                "matrix_scores_by_layer_json",
                "matrix_score_sum_by_layer_json",
                "matrix_score_abs_sum_by_layer_json",
                "matrix_score_mean_by_layer_json",
                "matrix_score_abs_mean_by_layer_json",
                "update_norm_by_layer_json",
                "message_norm_by_layer_json",
                "memory_norm_by_layer_json",
                "raw_message_norm_by_layer_json",
                "target_positions_json",
            ],
        )
        write_csv_dynamic(
            output_dir / "summary.csv",
            summary_rows,
            [
                "method",
                "token_group",
                "readout_mode",
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
        write_csv_dynamic(
            output_dir / "accuracy_by_seq_len.csv",
            accuracy_rows,
            [
                "method",
                "token_group",
                "readout_mode",
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
        write_csv_dynamic(
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
            write_csv_dynamic(
                output_dir / "train_history.csv",
                train_history,
                [
                    "method",
                    "token_group",
                    "readout_mode",
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
        write_diagnostics(
            output_dir=output_dir,
            model=model,
            adapter=adapter,
            train_history=train_history,
            backward_diag=backward_diag,
            metrics_rows=metrics_rows,
            token_group=args.token_group,
        )
        write_readme(output_dir, summary_rows, accuracy_rows, metrics_rows)
        write_json(
            output_dir / "run_done.json",
            {
                "completed": True,
                "elapsed_seconds": time.time() - started,
                "output_dir": os.fspath(output_dir),
                "token_group": str(args.token_group),
                "readout_mode": str(args.readout_mode),
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

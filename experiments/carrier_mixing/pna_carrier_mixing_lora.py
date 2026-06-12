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
from torch import nn

from models.model import get_layers

try:
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
except Exception:  # pragma: no cover - exercised only when transformers is absent in tiny debug mode
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoModelForImageTextToText = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    BitsAndBytesConfig = None  # type: ignore[assignment]

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
except Exception:  # pragma: no cover - optional, version-dependent import
    apply_multimodal_rotary_pos_emb = None  # type: ignore[assignment]


EXPERIMENT_NAME = "pna_carrier_mixing_lora"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME

FROZEN_QWEN_BASELINE = "frozen_qwen_baseline"
LORA_BASELINE = "lora_baseline"
LORA_MAXMIX = "lora_maxmix_carrier_14_17"
LORA_PNAMIX = "lora_pnamix_carrier_14_17"
VARIANTS = (FROZEN_QWEN_BASELINE, LORA_BASELINE, LORA_MAXMIX, LORA_PNAMIX)
VARIANT_ALIASES = {
    "frozen": FROZEN_QWEN_BASELINE,
    "frozen_qwen": FROZEN_QWEN_BASELINE,
    FROZEN_QWEN_BASELINE: FROZEN_QWEN_BASELINE,
    "lora": LORA_BASELINE,
    "baseline_lora": LORA_BASELINE,
    LORA_BASELINE: LORA_BASELINE,
    "maxmix": LORA_MAXMIX,
    "lora_maxmix": LORA_MAXMIX,
    LORA_MAXMIX: LORA_MAXMIX,
    "pna": LORA_PNAMIX,
    "pnamix": LORA_PNAMIX,
    "lora_pnamix": LORA_PNAMIX,
    LORA_PNAMIX: LORA_PNAMIX,
}

TRAIN_TEMPLATES = ("train_query_first", "train_frames_first", "train_audit", "train_table")
HELDOUT_TEMPLATES = ("heldout_logbook", "heldout_reverse")
COUNT_VALUES = tuple(range(9))
EPS = 1e-8

ROOMS = (
    "kitchen",
    "library",
    "garden",
    "studio",
    "garage",
    "office",
    "pantry",
    "lounge",
    "hallway",
)
CHARACTERS = (
    "Alice",
    "Ben",
    "Carla",
    "Diego",
    "Eve",
    "Finn",
    "Grace",
    "Hana",
    "Ivan",
    "Juno",
)


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
class EvidenceExample:
    prompt_id: str
    split: str
    seq_len: int
    gold_count: int
    template_id: str
    target_character: str
    target_room: str
    frame_lines: Tuple[str, ...]
    evidence_frame_indices: Tuple[int, ...]
    distractor_count: int
    prompt: str
    character_span: Tuple[int, int]
    room_span: Tuple[int, int]
    frame_spans: Tuple[Tuple[int, int], ...]


@dataclass
class EvidenceBatch:
    inputs: Dict[str, Any]
    prompt_last_indices: torch.Tensor
    gold_counts: torch.Tensor
    carrier_positions: List[List[int]]
    frame_groups: List[List[List[int]]]
    token_selection_ok: List[bool]
    token_selection_errors: List[str]
    sample_indices: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen Qwen, vanilla LoRA, MaxMix carrier mixing, and PNA-Mix carrier mixing "
            "on deterministic evidence-counting prompts."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-root", "--output_root", dest="output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-prefix", "--run_prefix", dest="run_prefix", default="")
    parser.add_argument("--variants", nargs="+", default=[",".join(VARIANTS)])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-seed", "--dataset_seed", dest="dataset_seed", type=int, default=12345)
    parser.add_argument("--train-examples-per-bin", "--train_examples_per_bin", dest="train_examples_per_bin", type=int, default=16)
    parser.add_argument("--eval-examples-per-bin", "--eval_examples_per_bin", dest="eval_examples_per_bin", type=int, default=6)
    parser.add_argument("--force-regenerate-dataset", "--force_regenerate_dataset", dest="force_regenerate_dataset", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--grad-accum", "--grad_accum", dest="grad_accum", type=int, default=8)
    parser.add_argument("--grad-clip", "--grad_clip", dest="grad_clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    parser.add_argument("--max-train-examples", "--max_train_examples", dest="max_train_examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", "--max_eval_examples", dest="max_eval_examples", type=int, default=0)

    parser.add_argument("--lora-rank", "--lora_rank", dest="lora_rank", type=int, default=8)
    parser.add_argument("--lora-alpha", "--lora_alpha", dest="lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", "--lora_dropout", dest="lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora-layers", "--lora_layers", dest="lora_layers", nargs="+", default=["14,15,16,17,20,21,22,23,24,25,26,27"])
    parser.add_argument("--carrier-layers", "--carrier_layers", dest="carrier_layers", nargs="+", default=["14,15,16,17"])
    parser.add_argument("--lora-targets", "--lora_targets", dest="lora_targets", nargs="+", default=["q_proj,k_proj,v_proj,o_proj"])
    parser.add_argument("--carrier-gate-init", "--carrier_gate_init", dest="carrier_gate_init", type=float, default=-4.0)
    parser.add_argument("--message-mode", "--message_mode", dest="message_mode", choices=["auto", "exact", "approx"], default="auto")

    parser.add_argument("--candidate-min", "--candidate_min", dest="candidate_min", type=int, default=0)
    parser.add_argument("--candidate-max", "--candidate_max", dest="candidate_max", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--attn-implementation", "--attn_implementation", dest="attn_implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", "--load_in_4bit", dest="load_in_4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-qlora-fallback", "--auto_qlora_fallback", dest="auto_qlora_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pixels", "--max_pixels", dest="max_pixels", type=int, default=None)
    parser.add_argument("--min-pixels", "--min_pixels", dest="min_pixels", type=int, default=None)

    parser.add_argument("--no-plots", "--no_plots", dest="no_plots", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False, help="Write run dirs under outputs/pna_carrier_mixing_lora/debug/.")
    parser.add_argument("--tiny-debug-model", "--tiny_debug_model", dest="tiny_debug_model", action="store_true", default=False)
    parser.add_argument("--tiny-num-layers", "--tiny_num_layers", dest="tiny_num_layers", type=int, default=2)
    parser.add_argument("--tiny-hidden-size", "--tiny_hidden_size", dest="tiny_hidden_size", type=int, default=64)
    parser.add_argument("--tiny-num-heads", "--tiny_num_heads", dest="tiny_num_heads", type=int, default=4)
    parser.add_argument("--submit-mode", "--submit_mode", dest="submit_mode", default="local")
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
    safe = str(value)
    for old, new in (("/", "_"), ("+", "_"), (" ", "_"), (":", "_"), (".", "p"), (",", "_")):
        safe = safe.replace(old, new)
    return safe


def dtype_from_arg(raw: str) -> torch.dtype:
    key = str(raw).lower()
    if key in {"auto", "bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float16", "fp16"}:
        return torch.float16
    if key in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {raw!r}")


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_mean(values: Iterable[Any], default: float = math.nan) -> float:
    vals = [float(value) for value in values if finite_float(value) is not None]
    return float(np.mean(vals)) if vals else float(default)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return math.nan
    return float(sum(int(a) == int(b) for a, b in zip(y_true, y_pred)) / len(y_true))


def mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return math.nan
    return float(sum(abs(int(a) - int(b)) for a, b in zip(y_true, y_pred)) / len(y_true))


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
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def setup_logging(run_dir: Path) -> Tuple[Any, Any, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (run_dir / "run.log").open("a", encoding="utf-8", buffering=1)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)
    return log_handle, old_stdout, old_stderr


def restore_logging(log_handle: Any, old_stdout: Any, old_stderr: Any) -> None:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_handle.close()


def example_to_json(example: EvidenceExample) -> Dict[str, Any]:
    return {
        "prompt_id": example.prompt_id,
        "split": example.split,
        "seq_len": int(example.seq_len),
        "gold_count": int(example.gold_count),
        "template_id": example.template_id,
        "target_character": example.target_character,
        "target_room": example.target_room,
        "frame_lines": list(example.frame_lines),
        "evidence_frame_indices": list(example.evidence_frame_indices),
        "distractor_count": int(example.distractor_count),
        "prompt": example.prompt,
        "character_span": list(example.character_span),
        "room_span": list(example.room_span),
        "frame_spans": [list(span) for span in example.frame_spans],
    }


def example_from_json(row: Dict[str, Any]) -> EvidenceExample:
    return EvidenceExample(
        prompt_id=str(row["prompt_id"]),
        split=str(row["split"]),
        seq_len=int(row["seq_len"]),
        gold_count=int(row["gold_count"]),
        template_id=str(row["template_id"]),
        target_character=str(row["target_character"]),
        target_room=str(row["target_room"]),
        frame_lines=tuple(str(x) for x in row["frame_lines"]),
        evidence_frame_indices=tuple(int(x) for x in row["evidence_frame_indices"]),
        distractor_count=int(row["distractor_count"]),
        prompt=str(row["prompt"]),
        character_span=(int(row["character_span"][0]), int(row["character_span"][1])),
        room_span=(int(row["room_span"][0]), int(row["room_span"][1])),
        frame_spans=tuple((int(span[0]), int(span[1])) for span in row["frame_spans"]),
    )


def choose_other(rng: random.Random, values: Sequence[str], blocked: str) -> str:
    candidates = [value for value in values if value != blocked]
    return rng.choice(candidates)


def make_frame_line(
    *,
    frame_idx: int,
    evidence: bool,
    target_character: str,
    target_room: str,
    rng: random.Random,
    distractor_heavy: bool,
) -> Tuple[str, int]:
    facts: List[str] = []
    hard_distractors = 0
    if evidence:
        facts.append(f"{target_character} is in the {target_room}")
    else:
        hard_mode = rng.choice(("target_char_wrong_room", "other_char_target_room", "both"))
        if hard_mode in {"target_char_wrong_room", "both"}:
            facts.append(f"{target_character} is in the {choose_other(rng, ROOMS, target_room)}")
            hard_distractors += 1
        if hard_mode in {"other_char_target_room", "both"}:
            facts.append(f"{choose_other(rng, CHARACTERS, target_character)} is in the {target_room}")
            hard_distractors += 1

    extra_count = rng.randint(3, 5) if distractor_heavy else rng.randint(1, 2)
    used_pairs = {(target_character, target_room)} if evidence else set()
    for _ in range(extra_count):
        for _attempt in range(20):
            char = choose_other(rng, CHARACTERS, target_character) if rng.random() < 0.75 else target_character
            room = rng.choice(ROOMS)
            if char == target_character and room == target_room:
                room = choose_other(rng, ROOMS, target_room)
            pair = (char, room)
            if pair not in used_pairs:
                used_pairs.add(pair)
                facts.append(f"{char} is in the {room}")
                break
    rng.shuffle(facts)
    return f"Frame {frame_idx + 1}: " + ". ".join(facts) + ".", hard_distractors


def build_prompt_from_template(
    *,
    template_id: str,
    target_character: str,
    target_room: str,
    frame_lines: Sequence[str],
) -> Tuple[str, Tuple[int, int], Tuple[int, int], Tuple[Tuple[int, int], ...]]:
    question_by_template = {
        "train_query_first": f"How many frames show {target_character} in the {target_room}?",
        "train_frames_first": f"Count the frames where {target_character} is located in the {target_room}.",
        "train_audit": f"For the audit, report the number of frames with {target_character} in the {target_room}.",
        "train_table": f"Evidence means {target_character} is in the {target_room}. How many evidence frames are there?",
        "heldout_logbook": f"In this logbook, how often is {target_character} inside the {target_room}?",
        "heldout_reverse": f"What is the count of entries placing {target_character} in the {target_room}?",
    }
    question = question_by_template[template_id]
    frames = "\n".join(frame_lines)
    if template_id == "train_query_first":
        prompt = (
            "You are given numbered frames. Count only frames that match the query.\n"
            f"Query: {question}\n"
            f"{frames}\n"
            "Answer with one integer from 0 to 8.\nAnswer:"
        )
    elif template_id == "train_frames_first":
        prompt = (
            "Frames:\n"
            f"{frames}\n"
            f"Question: {question}\n"
            "Return a single integer.\nAnswer:"
        )
    elif template_id == "train_audit":
        prompt = (
            "Review each frame independently. Distractor facts may mention the same room or the same person.\n"
            f"Audit query: {question}\n"
            f"{frames}\n"
            "Final count:"
        )
    elif template_id == "train_table":
        prompt = (
            "Evidence counting task.\n"
            f"Rule: {question}\n"
            f"{frames}\n"
            "Count:"
        )
    elif template_id == "heldout_logbook":
        prompt = (
            "Logbook entries follow. Some entries are distractors.\n"
            f"{frames}\n"
            f"Logbook question: {question}\n"
            "Numeric answer:"
        )
    elif template_id == "heldout_reverse":
        prompt = (
            "Decide the answer from the entries below.\n"
            f"{frames}\n"
            f"Scoring question: {question}\n"
            "Answer as one digit:"
        )
    else:
        raise ValueError(f"Unknown template_id={template_id!r}")

    question_start = prompt.index(question)
    char_start = question_start + question.index(target_character)
    room_start = question_start + question.index(target_room)
    frame_spans: List[Tuple[int, int]] = []
    search_from = 0
    for line in frame_lines:
        start = prompt.index(line, search_from)
        end = start + len(line)
        frame_spans.append((start, end))
        search_from = end
    return (
        prompt,
        (char_start, char_start + len(target_character)),
        (room_start, room_start + len(target_room)),
        tuple(frame_spans),
    )


def generate_examples_for_split(
    *,
    split: str,
    seq_lens: Sequence[int],
    counts: Sequence[int],
    examples_per_bin: int,
    seed: int,
    templates: Sequence[str],
    distractor_heavy: bool = False,
) -> List[EvidenceExample]:
    rng = random.Random(int(seed))
    examples: List[EvidenceExample] = []
    for seq_len in seq_lens:
        for count in counts:
            if int(count) > int(seq_len):
                continue
            for bin_idx in range(int(examples_per_bin)):
                target_character = rng.choice(CHARACTERS)
                target_room = rng.choice(ROOMS)
                evidence_positions = set(rng.sample(range(int(seq_len)), int(count)))
                frame_lines: List[str] = []
                distractor_count = 0
                for frame_idx in range(int(seq_len)):
                    line, hard = make_frame_line(
                        frame_idx=frame_idx,
                        evidence=frame_idx in evidence_positions,
                        target_character=target_character,
                        target_room=target_room,
                        rng=rng,
                        distractor_heavy=bool(distractor_heavy),
                    )
                    frame_lines.append(line)
                    distractor_count += int(hard)
                template_id = rng.choice(tuple(templates))
                prompt, char_span, room_span, frame_spans = build_prompt_from_template(
                    template_id=template_id,
                    target_character=target_character,
                    target_room=target_room,
                    frame_lines=frame_lines,
                )
                prompt_id = f"{split}_len{int(seq_len)}_count{int(count)}_{bin_idx:04d}"
                examples.append(
                    EvidenceExample(
                        prompt_id=prompt_id,
                        split=str(split),
                        seq_len=int(seq_len),
                        gold_count=int(count),
                        template_id=template_id,
                        target_character=target_character,
                        target_room=target_room,
                        frame_lines=tuple(frame_lines),
                        evidence_frame_indices=tuple(sorted(int(x) for x in evidence_positions)),
                        distractor_count=int(distractor_count),
                        prompt=prompt,
                        character_span=char_span,
                        room_span=room_span,
                        frame_spans=frame_spans,
                    )
                )
    rng.shuffle(examples)
    return examples


def dataset_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "dataset_seed": int(args.dataset_seed),
        "train_examples_per_bin": int(args.train_examples_per_bin),
        "eval_examples_per_bin": int(args.eval_examples_per_bin),
        "train_seq_lens": [3, 4, 5, 6],
        "train_counts": [0, 1, 2, 3, 4, 5],
        "iid_seq_lens": [3, 4, 5, 6],
        "iid_counts": [0, 1, 2, 3, 4, 5],
        "ood_long_seq_lens": [7, 8],
        "ood_long_counts": list(range(9)),
        "ood_high_count_seq_lens": [6, 7, 8],
        "ood_high_count_counts": [5, 6, 7, 8],
        "ood_distractor_heavy_seq_lens": [6, 7, 8],
        "ood_distractor_heavy_counts": list(range(9)),
        "heldout_template_seq_lens": [3, 4, 5, 6],
        "heldout_template_counts": [0, 1, 2, 3, 4, 5],
        "train_templates": list(TRAIN_TEMPLATES),
        "heldout_templates": list(HELDOUT_TEMPLATES),
    }


def dataset_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def ensure_dataset(args: argparse.Namespace, dataset_base: Path) -> Tuple[Path, Dict[str, List[EvidenceExample]], Dict[str, Any]]:
    config = dataset_config(args)
    digest = dataset_hash(config)
    dataset_dir = dataset_base / digest
    manifest_path = dataset_dir / "dataset_manifest.json"
    split_paths = {
        "train": dataset_dir / "train.jsonl",
        "val_iid": dataset_dir / "val_iid.jsonl",
        "test_iid": dataset_dir / "test_iid.jsonl",
        "test_ood_long": dataset_dir / "test_ood_long.jsonl",
        "test_ood_high_count": dataset_dir / "test_ood_high_count.jsonl",
        "test_ood_distractor_heavy": dataset_dir / "test_ood_distractor_heavy.jsonl",
        "test_heldout_template": dataset_dir / "test_heldout_template.jsonl",
    }
    if bool(args.force_regenerate_dataset) or not manifest_path.is_file() or not all(path.is_file() for path in split_paths.values()):
        dataset_dir.mkdir(parents=True, exist_ok=True)
        seed = int(args.dataset_seed)
        generated: Dict[str, List[EvidenceExample]] = {
            "train": generate_examples_for_split(
                split="train",
                seq_lens=config["train_seq_lens"],
                counts=config["train_counts"],
                examples_per_bin=int(args.train_examples_per_bin),
                seed=seed + 11,
                templates=TRAIN_TEMPLATES,
            ),
            "val_iid": generate_examples_for_split(
                split="val_iid",
                seq_lens=config["iid_seq_lens"],
                counts=config["iid_counts"],
                examples_per_bin=int(args.eval_examples_per_bin),
                seed=seed + 23,
                templates=TRAIN_TEMPLATES,
            ),
            "test_iid": generate_examples_for_split(
                split="test_iid",
                seq_lens=config["iid_seq_lens"],
                counts=config["iid_counts"],
                examples_per_bin=int(args.eval_examples_per_bin),
                seed=seed + 31,
                templates=TRAIN_TEMPLATES,
            ),
            "test_ood_long": generate_examples_for_split(
                split="test_ood_long",
                seq_lens=config["ood_long_seq_lens"],
                counts=config["ood_long_counts"],
                examples_per_bin=int(args.eval_examples_per_bin),
                seed=seed + 43,
                templates=TRAIN_TEMPLATES,
            ),
            "test_ood_high_count": generate_examples_for_split(
                split="test_ood_high_count",
                seq_lens=config["ood_high_count_seq_lens"],
                counts=config["ood_high_count_counts"],
                examples_per_bin=int(args.eval_examples_per_bin),
                seed=seed + 53,
                templates=TRAIN_TEMPLATES,
            ),
            "test_ood_distractor_heavy": generate_examples_for_split(
                split="test_ood_distractor_heavy",
                seq_lens=config["ood_distractor_heavy_seq_lens"],
                counts=config["ood_distractor_heavy_counts"],
                examples_per_bin=int(args.eval_examples_per_bin),
                seed=seed + 67,
                templates=TRAIN_TEMPLATES,
                distractor_heavy=True,
            ),
            "test_heldout_template": generate_examples_for_split(
                split="test_heldout_template",
                seq_lens=config["heldout_template_seq_lens"],
                counts=config["heldout_template_counts"],
                examples_per_bin=int(args.eval_examples_per_bin),
                seed=seed + 79,
                templates=HELDOUT_TEMPLATES,
            ),
        }
        for split, rows in generated.items():
            write_jsonl(split_paths[split], [example_to_json(example) for example in rows])
        manifest = {
            "dataset_hash": digest,
            "config": config,
            "splits": {split: {"path": os.fspath(path), "n": len(generated[split])} for split, path in split_paths.items()},
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    examples_by_split = {
        split: [example_from_json(row) for row in read_jsonl(path)]
        for split, path in split_paths.items()
    }
    return dataset_dir, examples_by_split, manifest


class TinyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self.vocab_size = 258

    def _encode_one(self, text: str) -> List[int]:
        ids: List[int] = []
        for char in str(text):
            code = ord(char)
            ids.append(code + 2 if code < 256 else 2)
        return ids

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = self._encode_one(text)
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_id: int) -> str:
        value = int(token_id)
        if value == self.pad_token_id:
            return self.pad_token
        if value == self.eos_token_id:
            return self.eos_token
        return chr(max(0, value - 2))

    def apply_chat_template(self, messages: Sequence[Dict[str, Any]], tokenize: bool = False, add_generation_prompt: bool = True, **_: Any) -> Any:
        text = ""
        for message in messages:
            role = str(message.get("role", "user")).title()
            text += f"{role}: {message.get('content', '')}\n"
        if add_generation_prompt:
            text += "Assistant: "
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    def __call__(
        self,
        texts: Any,
        *,
        padding: bool = False,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in text_list]
        max_len = max(len(ids) for ids in encoded) if encoded else 0
        if padding:
            padded: List[List[int]] = []
            masks: List[List[int]] = []
            for ids in encoded:
                pad = max_len - len(ids)
                if self.padding_side == "left":
                    padded.append([self.pad_token_id] * pad + ids)
                    masks.append([0] * pad + [1] * len(ids))
                else:
                    padded.append(ids + [self.pad_token_id] * pad)
                    masks.append([1] * len(ids) + [0] * pad)
        else:
            padded = encoded
            masks = [[1] * len(ids) for ids in encoded]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded[0] if single else padded, "attention_mask": masks[0] if single else masks}


class TinySelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_key_value_heads = int(num_heads)
        self.head_dim = int(hidden_size) // int(num_heads)
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _hidden = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * float(self.scaling)
        causal = torch.tril(torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
        if torch.is_tensor(attention_mask):
            scores = scores.masked_fill(~attention_mask.bool().view(batch, 1, 1, seq_len), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1).to(dtype=hidden_states.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(out)


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = TinySelfAttention(hidden_size, num_heads)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyDecoderLayer(hidden_size, num_heads) for _ in range(num_layers)])


class TinyQwenLikeForCausalLM(nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=int(hidden_size), vocab_size=int(vocab_size), use_cache=False)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = TinyBackbone(hidden_size, num_layers, num_heads)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, use_cache: bool = False, **_: Any) -> Any:
        del use_cache
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


def load_tiny_debug_model(args: argparse.Namespace, device: str) -> Tuple[nn.Module, TinyTokenizer, bool, str]:
    tokenizer = TinyTokenizer()
    model = TinyQwenLikeForCausalLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(args.tiny_hidden_size),
        num_layers=int(args.tiny_num_layers),
        num_heads=int(args.tiny_num_heads),
    ).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer, False, "tiny_debug_model"


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas" in text and "alloc" in text


def load_hf_model_and_tokenizer(args: argparse.Namespace, device: str, dtype: torch.dtype) -> Tuple[Any, Any, bool, str]:
    if AutoTokenizer is None or AutoProcessor is None:
        raise RuntimeError("transformers is not available; use --tiny-debug-model for local code-path checks")

    def load_once(load_in_4bit: bool) -> Tuple[Any, Any]:
        use_vl = "vl" in str(args.model_name).lower()
        processor_or_tokenizer: Any
        if use_vl and AutoProcessor is not None:
            processor_or_tokenizer = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
            if args.max_pixels is not None and hasattr(processor_or_tokenizer, "image_processor"):
                setattr(processor_or_tokenizer.image_processor, "max_pixels", int(args.max_pixels))
            if args.min_pixels is not None and hasattr(processor_or_tokenizer, "image_processor"):
                setattr(processor_or_tokenizer.image_processor, "min_pixels", int(args.min_pixels))
            tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
            processor_or_tokenizer = tokenizer
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "right"

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": str(args.attn_implementation),
        }
        if load_in_4bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError("bitsandbytes config is unavailable but load_in_4bit=True")
            if not str(device).startswith("cuda"):
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

        if use_vl and AutoModelForImageTextToText is not None:
            model = AutoModelForImageTextToText.from_pretrained(args.model_name, **model_kwargs)
        else:
            if AutoModelForCausalLM is None:
                raise RuntimeError("AutoModelForCausalLM unavailable")
            model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        if not load_in_4bit:
            model.to(device)
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = False
        for param in model.parameters():
            param.requires_grad_(False)
        return model, tokenizer

    try:
        model, tokenizer = load_once(bool(args.load_in_4bit))
        return model, tokenizer, bool(args.load_in_4bit), "requested_4bit" if bool(args.load_in_4bit) else "bf16_or_requested_dtype"
    except RuntimeError as exc:
        if bool(args.load_in_4bit) or not bool(args.auto_qlora_fallback) or not is_oom(exc):
            raise
        print(f"BF16/full-precision model load failed with apparent OOM; falling back to 4-bit QLoRA. Error: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tokenizer = load_once(True)
        return model, tokenizer, True, "auto_qlora_fallback_after_bf16_oom"


def load_model_and_tokenizer(args: argparse.Namespace, device: str, dtype: torch.dtype) -> Tuple[Any, Any, bool, str]:
    if bool(args.tiny_debug_model):
        return load_tiny_debug_model(args, device)
    return load_hf_model_and_tokenizer(args, device, dtype)


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
    return int(getattr(embed, "embedding_dim", embed.weight.shape[-1]))


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


def render_model_text(tokenizer: Any, prompt: str) -> Tuple[str, int]:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            start = str(rendered).find(prompt)
            if start >= 0:
                return str(rendered), int(start)
        except Exception:
            pass
    rendered = f"User: {prompt}\nAssistant:"
    return rendered, rendered.index(prompt)


def token_positions_for_span(tokenizer: Any, rendered_text: str, span_start: int, span_end: int) -> List[int]:
    prefix_ids = tokenizer.encode(rendered_text[: int(span_start)], add_special_tokens=False)
    through_ids = tokenizer.encode(rendered_text[: int(span_end)], add_special_tokens=False)
    return list(range(len(prefix_ids), len(through_ids)))


def move_inputs_to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


def prepare_batch(
    *,
    examples: Sequence[EvidenceExample],
    sample_indices: Sequence[int],
    tokenizer: Any,
    device: str,
) -> EvidenceBatch:
    rendered_payloads = [render_model_text(tokenizer, example.prompt) for example in examples]
    rendered_texts = [payload[0] for payload in rendered_payloads]
    prompt_starts = [int(payload[1]) for payload in rendered_payloads]
    raw_inputs = tokenizer(rendered_texts, padding=True, return_tensors="pt", add_special_tokens=False)
    input_ids = raw_inputs["input_ids"]
    attention_mask = raw_inputs.get("attention_mask")
    prompt_last_indices: List[int] = []
    carrier_positions: List[List[int]] = []
    frame_groups: List[List[List[int]]] = []
    token_selection_ok: List[bool] = []
    token_selection_errors: List[str] = []

    for row, example in enumerate(examples):
        if torch.is_tensor(attention_mask):
            active = attention_mask[row].detach().cpu().nonzero(as_tuple=True)[0]
            prompt_last = int(active[-1].item()) if active.numel() else int(input_ids.shape[1] - 1)
        else:
            prompt_last = int(input_ids.shape[1] - 1)
        prompt_last_indices.append(prompt_last)
        start = prompt_starts[row]
        char_positions = token_positions_for_span(
            tokenizer,
            rendered_texts[row],
            start + int(example.character_span[0]),
            start + int(example.character_span[1]),
        )
        room_positions = token_positions_for_span(
            tokenizer,
            rendered_texts[row],
            start + int(example.room_span[0]),
            start + int(example.room_span[1]),
        )
        carriers = sorted({int(pos) for pos in [*char_positions, *room_positions] if 0 <= int(pos) < prompt_last})
        groups: List[List[int]] = []
        for frame_start, frame_end in example.frame_spans:
            positions = token_positions_for_span(
                tokenizer,
                rendered_texts[row],
                start + int(frame_start),
                start + int(frame_end),
            )
            groups.append([int(pos) for pos in positions if 0 <= int(pos) < prompt_last])
        errors: List[str] = []
        if not carriers:
            errors.append("no carrier room/character tokens located")
        if not groups or any(not group for group in groups):
            errors.append("one or more frame token groups are empty")
        carrier_positions.append(carriers)
        frame_groups.append(groups)
        token_selection_ok.append(not errors)
        token_selection_errors.append("; ".join(errors))

    return EvidenceBatch(
        inputs=move_inputs_to_device(dict(raw_inputs), device),
        prompt_last_indices=torch.tensor(prompt_last_indices, device=device, dtype=torch.long),
        gold_counts=torch.tensor([int(example.gold_count) for example in examples], device=device, dtype=torch.long),
        carrier_positions=carrier_positions,
        frame_groups=frame_groups,
        token_selection_ok=token_selection_ok,
        token_selection_errors=token_selection_errors,
        sample_indices=[int(idx) for idx in sample_indices],
    )


class LoRALinearWrapper(nn.Module):
    def __init__(self, base_layer: nn.Module, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        object.__setattr__(self, "base_layer", base_layer)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(1, int(rank))
        self.dropout = float(dropout)
        weight = getattr(base_layer, "weight", None)
        in_features = getattr(base_layer, "in_features", None)
        out_features = getattr(base_layer, "out_features", None)
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
            self.last_delta_norm = [float(v) for v in delta.detach().float().norm(dim=-1).mean(dim=-1).cpu().tolist()]
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

    def attach(self, model: Any) -> None:
        self.detach()
        layers = get_layers(model)
        for layer_idx in self.inject_layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"LoRA layer={layer_idx} outside [0, {len(layers) - 1}]")
            attn = getattr(layers[int(layer_idx)], "self_attn", None)
            if attn is None:
                raise RuntimeError(f"layer={layer_idx} does not expose self_attn")
            for name in self.target_modules:
                base_layer = getattr(attn, name, None)
                if base_layer is None:
                    raise RuntimeError(f"layer={layer_idx}.self_attn does not expose {name}")
                wrapper = LoRALinearWrapper(base_layer, rank=self.rank, alpha=self.alpha, dropout=self.dropout)
                setattr(attn, name, wrapper)
                self.wrappers.append(wrapper)
                self._wrapped.append((attn, name, base_layer, wrapper, int(layer_idx)))

    def detach(self) -> None:
        for parent, name, original, _wrapper, _layer in reversed(self._wrapped):
            setattr(parent, name, original)
        self._wrapped = []
        self.wrappers = nn.ModuleList()

    def set_context(self, _batch: EvidenceBatch) -> None:
        for wrapper in self.wrappers:
            wrapper.last_delta_norm = []

    def clear_context(self) -> None:
        pass

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        layer_values: Dict[str, List[float]] = defaultdict(list)
        for _parent, name, _original, wrapper, layer_idx in self._wrapped:
            del name
            if row < len(wrapper.last_delta_norm):
                layer_values[str(int(layer_idx))].append(float(wrapper.last_delta_norm[row]))
        by_layer = {layer: finite_mean(values, default=0.0) for layer, values in layer_values.items()}
        return {
            "lora_delta_norm_by_layer": by_layer,
            "lora_delta_norm": finite_mean(by_layer.values(), default=0.0),
        }


class CarrierMixingAdapter(nn.Module):
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
            raise ValueError(f"Unknown carrier mixing method={method!r}")
        self.method = str(method)
        self.hidden_size = int(hidden_size)
        self.inject_layers = [int(layer) for layer in inject_layers]
        self.layer_to_pos = {int(layer): pos for pos, layer in enumerate(self.inject_layers)}
        self.message_mode = str(message_mode)
        in_dim = self.hidden_size if self.method == "maxmix" else self.hidden_size * 6
        self.projections = nn.ModuleList([nn.Linear(in_dim, self.hidden_size, bias=False) for _ in self.inject_layers])
        self.feature_norms = nn.ModuleList([nn.LayerNorm(in_dim) for _ in self.inject_layers])
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

    def set_context(self, batch: EvidenceBatch) -> None:
        self._carrier_positions = [[int(pos) for pos in row] for row in batch.carrier_positions]
        self._frame_groups = [[[int(pos) for pos in group] for group in row] for row in batch.frame_groups]
        self._last_stats = {
            "carrier_update_norm_by_layer": {},
            "carrier_gate_by_layer": {},
            "carrier_feature_norm_by_layer": {},
            "carrier_soft_count_by_layer": {},
            "carrier_message_mode_by_layer": {},
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
    def _replace_hidden_in_args(args: Tuple[Any, ...], kwargs: Dict[str, Any], hidden_states: torch.Tensor) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        if args and torch.is_tensor(args[0]):
            return (hidden_states,) + tuple(args[1:]), kwargs
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = hidden_states
        return args, new_kwargs

    @staticmethod
    def _repeat_kv(states: torch.Tensor, num_heads: int) -> torch.Tensor:
        if int(states.shape[1]) == int(num_heads):
            return states
        repeats = int(num_heads) // int(states.shape[1])
        return states.repeat_interleave(repeats, dim=1)

    def _record_exact_failure(self, reason: str) -> None:
        key = str(reason).split(":", 1)[0][:80]
        self.exact_failure_counts[key] += 1
        if len(self.exact_failure_examples) < 20:
            self.exact_failure_examples.append(str(reason)[:500])

    def _qkv_messages(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        kwargs: Dict[str, Any],
        *,
        require_exact: bool,
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], str]:
        if not hasattr(module, "self_attn"):
            raise RuntimeError("decoder layer does not expose self_attn")
        attn = module.self_attn
        hs = module.input_layernorm(hidden_states) if hasattr(module, "input_layernorm") else hidden_states
        batch, seq_len, _hidden = hs.shape
        q = attn.q_proj(hs)
        k = attn.k_proj(hs)
        v = attn.v_proj(hs)
        head_dim = int(getattr(attn, "head_dim", q.shape[-1] // int(getattr(attn, "num_heads", 1))))
        num_heads = int(getattr(attn, "num_heads", q.shape[-1] // head_dim))
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, -1, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, -1, head_dim).transpose(1, 2)

        mode = "approx"
        if require_exact or self.message_mode in {"auto", "exact"}:
            position_embeddings = kwargs.get("position_embeddings")
            if apply_multimodal_rotary_pos_emb is None or position_embeddings is None or not hasattr(attn, "rope_scaling"):
                if require_exact or self.message_mode == "exact":
                    raise RuntimeError("exact qwen alpha*v unavailable: no multimodal RoPE inputs")
            else:
                q, k = apply_multimodal_rotary_pos_emb(q, k, position_embeddings[0], position_embeddings[1], attn.rope_scaling["mrope_section"])
                mode = "exact"
        k = self._repeat_kv(k, num_heads)
        v = self._repeat_kv(v, num_heads)
        scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
        attention_mask = kwargs.get("attention_mask")
        arange = torch.arange(seq_len, device=hidden_states.device)
        assert self._carrier_positions is not None and self._frame_groups is not None
        z_by_batch: List[List[torch.Tensor]] = []
        soft_counts_by_batch: List[List[torch.Tensor]] = []
        for batch_idx in range(batch):
            sample_z: List[torch.Tensor] = []
            sample_counts: List[torch.Tensor] = []
            carrier_positions = [int(pos) for pos in self._carrier_positions[batch_idx] if 0 <= int(pos) < seq_len]
            frame_positions = sorted(
                {
                    int(pos)
                    for group in self._frame_groups[batch_idx]
                    for pos in group
                    if 0 <= int(pos) < seq_len
                }
            )
            if not carrier_positions or not frame_positions:
                z_by_batch.append(sample_z)
                soft_counts_by_batch.append(sample_counts)
                continue
            c_idx = torch.tensor(carrier_positions, device=hidden_states.device, dtype=torch.long)
            f_idx = torch.tensor(frame_positions, device=hidden_states.device, dtype=torch.long)
            scores = torch.einsum("hcd,hsd->hcs", q[batch_idx, :, c_idx, :].float(), k[batch_idx].float()) * scaling
            causal_allowed = arange.unsqueeze(0) <= c_idx.unsqueeze(1)
            sliding_window = getattr(attn, "sliding_window", None)
            if sliding_window is not None:
                causal_allowed &= arange.unsqueeze(0) >= (c_idx.unsqueeze(1) - int(sliding_window))
            scores = scores.masked_fill(~causal_allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
            if torch.is_tensor(attention_mask):
                if attention_mask.dim() == 4:
                    selected_mask = attention_mask[batch_idx : batch_idx + 1, :, c_idx, :].float()
                    scores = scores + selected_mask.squeeze(0)
                elif attention_mask.dim() == 2:
                    valid = attention_mask[batch_idx].bool()
                    scores = scores.masked_fill(~valid.view(1, 1, -1), torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            selected_probs = probs[:, :, f_idx]
            selected_v = v[batch_idx, :, f_idx, :].float()
            contrib = selected_probs.unsqueeze(-1) * selected_v.unsqueeze(1)
            contrib = contrib.permute(1, 2, 0, 3).contiguous().view(len(carrier_positions), len(frame_positions), num_heads * head_dim)
            soft_counts = selected_probs.sum(dim=-1).mean(dim=0)
            for carrier_row in range(len(carrier_positions)):
                sample_z.append(contrib[carrier_row])
                sample_counts.append(soft_counts[carrier_row])
            z_by_batch.append(sample_z)
            soft_counts_by_batch.append(sample_counts)
        return z_by_batch, soft_counts_by_batch, mode

    def _messages_with_fallback(self, module: Any, hidden_states: torch.Tensor, layer_idx: int, kwargs: Dict[str, Any]) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], str]:
        try:
            return self._qkv_messages(module, hidden_states, kwargs, require_exact=self.message_mode == "exact")
        except Exception as exc:
            self._record_exact_failure(f"layer {layer_idx}: {type(exc).__name__}: {exc}")
            if self.message_mode == "exact":
                raise
            return self._qkv_messages(module, hidden_states, kwargs, require_exact=False)

    def inject_before_layer(self, module: Any, hidden_states: torch.Tensor, layer_idx: int, kwargs: Dict[str, Any]) -> torch.Tensor:
        if self._carrier_positions is None or self._frame_groups is None or int(layer_idx) not in self.layer_to_pos:
            return hidden_states
        layer_pos = self.layer_to_pos[int(layer_idx)]
        self.hook_fire_counts[int(layer_idx)] += 1
        z_by_batch, soft_counts_by_batch, mode = self._messages_with_fallback(module, hidden_states, int(layer_idx), kwargs)
        self.message_mode_counts[f"{int(layer_idx)}:{mode}"] += int(hidden_states.shape[0])

        out = hidden_states.clone()
        batch, seq_len, hidden = hidden_states.shape
        gate = torch.sigmoid(self.gate_logits[layer_pos].float())
        per_batch_update_norm: List[float] = []
        per_batch_feature_norm: List[float] = []
        per_batch_soft_count: List[float] = []
        for batch_idx in range(batch):
            carriers = [int(pos) for pos in self._carrier_positions[batch_idx] if 0 <= int(pos) < seq_len]
            if not carriers:
                per_batch_update_norm.append(0.0)
                per_batch_feature_norm.append(0.0)
                per_batch_soft_count.append(0.0)
                continue
            sample_update_norms: List[float] = []
            sample_feature_norms: List[float] = []
            sample_soft_counts: List[float] = []
            for local_idx, carrier_pos in enumerate(carriers):
                if local_idx >= len(z_by_batch[batch_idx]) or z_by_batch[batch_idx][local_idx].numel() == 0:
                    continue
                z = z_by_batch[batch_idx][local_idx].float()
                soft_count = soft_counts_by_batch[batch_idx][local_idx].float()
                if self.method == "maxmix":
                    features = z.max(dim=0).values
                else:
                    std = z.std(dim=0, unbiased=False) if int(z.shape[0]) > 1 else torch.zeros((hidden,), device=z.device, dtype=z.dtype)
                    count_vec = soft_count.expand(hidden)
                    features = torch.cat(
                        [
                            z.sum(dim=0),
                            z.mean(dim=0),
                            z.max(dim=0).values,
                            z.min(dim=0).values,
                            std,
                            count_vec,
                        ],
                        dim=-1,
                    )
                normalized = self.feature_norms[layer_pos](features.to(dtype=torch.float32))
                update = gate * self.projections[layer_pos](normalized).float()
                out[batch_idx, carrier_pos, :] = out[batch_idx, carrier_pos, :] + update.to(dtype=hidden_states.dtype)
                sample_update_norms.append(float(update.detach().float().norm().cpu().item()))
                sample_feature_norms.append(float(features.detach().float().norm().cpu().item()))
                sample_soft_counts.append(float(soft_count.detach().float().cpu().item()))
            per_batch_update_norm.append(finite_mean(sample_update_norms, default=0.0))
            per_batch_feature_norm.append(finite_mean(sample_feature_norms, default=0.0))
            per_batch_soft_count.append(finite_mean(sample_soft_counts, default=0.0))

        layer_key = str(int(layer_idx))
        self._last_stats["carrier_update_norm_by_layer"][layer_key] = per_batch_update_norm or [0.0 for _ in range(batch)]
        self._last_stats["carrier_gate_by_layer"][layer_key] = [float(gate.detach().cpu().item()) for _ in range(batch)]
        self._last_stats["carrier_feature_norm_by_layer"][layer_key] = per_batch_feature_norm or [0.0 for _ in range(batch)]
        self._last_stats["carrier_soft_count_by_layer"][layer_key] = per_batch_soft_count or [0.0 for _ in range(batch)]
        self._last_stats["carrier_message_mode_by_layer"][layer_key] = [mode for _ in range(batch)]
        return out

    def attach(self, model: Any) -> None:
        self.detach()
        layers = get_layers(model)
        for layer_idx in self.inject_layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"carrier layer={layer_idx} outside [0, {len(layers) - 1}]")

            def hook(module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any], *, layer: int = int(layer_idx)) -> Any:
                hidden = self._hidden_from_args(args, kwargs)
                if hidden is None:
                    return args, kwargs
                new_hidden = self.inject_before_layer(module, hidden, layer, kwargs)
                return self._replace_hidden_in_args(args, kwargs, new_hidden)

            self._handles.append(layers[int(layer_idx)].register_forward_pre_hook(hook, with_kwargs=True))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

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
        out["carrier_update_norm"] = finite_mean(out.get("carrier_update_norm_by_layer", {}).values(), default=0.0)
        out["carrier_gate"] = finite_mean(out.get("carrier_gate_by_layer", {}).values(), default=math.nan)
        out["carrier_feature_norm"] = finite_mean(out.get("carrier_feature_norm_by_layer", {}).values(), default=0.0)
        out["carrier_soft_count"] = finite_mean(out.get("carrier_soft_count_by_layer", {}).values(), default=0.0)
        return out


class VariantAdapter(nn.Module):
    def __init__(self, lora: Optional[MinimalAttentionLoRAAdapter], mixer: Optional[CarrierMixingAdapter]) -> None:
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

    def set_context(self, batch: EvidenceBatch) -> None:
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
    lora = MinimalAttentionLoRAAdapter(
        inject_layers=lora_layers,
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        dropout=float(args.lora_dropout),
        target_modules=lora_targets,
    )
    mixer: Optional[CarrierMixingAdapter] = None
    if variant == LORA_MAXMIX:
        mixer = CarrierMixingAdapter(
            method="maxmix",
            hidden_size=int(hidden_size),
            inject_layers=carrier_layers,
            gate_init=float(args.carrier_gate_init),
            message_mode=str(args.message_mode),
        )
    elif variant == LORA_PNAMIX:
        mixer = CarrierMixingAdapter(
            method="pnamix",
            hidden_size=int(hidden_size),
            inject_layers=carrier_layers,
            gate_init=float(args.carrier_gate_init),
            message_mode=str(args.message_mode),
        )
    return VariantAdapter(lora, mixer)


def count_trainable_parameters(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def select_count_logits(logits: torch.Tensor, prompt_last_indices: torch.Tensor, count_token_ids: Dict[int, int]) -> torch.Tensor:
    batch_idx = torch.arange(int(logits.shape[0]), device=logits.device, dtype=torch.long)
    selected = logits[batch_idx, prompt_last_indices, :].float()
    ordered_ids = [int(count_token_ids[count]) for count in sorted(count_token_ids)]
    token_idx = torch.tensor(ordered_ids, device=selected.device, dtype=torch.long)
    return selected.index_select(dim=-1, index=token_idx)


def candidate_logits_payload(count_logits: torch.Tensor, count_values: Sequence[int]) -> Dict[str, float]:
    values = [float(v) for v in count_logits.detach().float().cpu().tolist()]
    return {str(int(count)): values[pos] for pos, count in enumerate(count_values) if pos < len(values)}


def chunked(values: Sequence[int], chunk_size: int) -> Iterable[List[int]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(values), size):
        yield [int(x) for x in values[start : start + size]]


def limited_indices(examples: Sequence[EvidenceExample], limit: int, seed: int) -> List[int]:
    indices = list(range(len(examples)))
    if int(limit) <= 0 or int(limit) >= len(indices):
        return indices
    rng = random.Random(int(seed))
    by_count: Dict[int, List[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        by_count[int(example.gold_count)].append(int(idx))
    selected: List[int] = []
    per_count = max(1, int(math.ceil(int(limit) / max(1, len(by_count)))))
    for count in sorted(by_count):
        values = list(by_count[count])
        rng.shuffle(values)
        selected.extend(values[:per_count])
    rng.shuffle(selected)
    return selected[: int(limit)]


def batch_indices(indices: Sequence[int], batch_size: int, seed: int, shuffle: bool) -> List[List[int]]:
    values = [int(idx) for idx in indices]
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(values)
    return list(chunked(values, int(batch_size)))


@torch.no_grad()
def evaluate_split(
    *,
    variant: str,
    split_name: str,
    model: Any,
    tokenizer: Any,
    adapter: Optional[VariantAdapter],
    examples: Sequence[EvidenceExample],
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
    batches = batch_indices(indices, int(batch_size), seed=int(seed), shuffle=False)
    for batch_num, idxs in enumerate(batches, start=1):
        batch_examples = [examples[int(idx)] for idx in idxs]
        batch = prepare_batch(examples=batch_examples, sample_indices=idxs, tokenizer=tokenizer, device=device)
        if adapter is not None:
            adapter.set_context(batch)
        outputs = model(**batch.inputs, use_cache=False)
        count_logits = select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
        gold_offsets = batch.gold_counts.long() - int(count_min)
        ce_vec = F.cross_entropy(count_logits, gold_offsets, reduction="none")
        ce_total += float(ce_vec.sum().detach().cpu().item())
        n += int(batch.gold_counts.numel())
        pred_offsets = count_logits.argmax(dim=-1)
        for row, idx in enumerate(idxs):
            example = examples[int(idx)]
            pred = int(pred_offsets[row].detach().cpu().item()) + int(count_min)
            gold = int(example.gold_count)
            stats = adapter.stats_for_row(row) if adapter is not None else {}
            row_payload: Dict[str, Any] = {
                "variant": str(variant),
                "split": str(split_name),
                "prompt_id": example.prompt_id,
                "gold_count": int(gold),
                "predicted_count": int(pred),
                "pred_count": int(pred),
                "correct": int(pred == gold),
                "abs_error": abs(int(pred) - int(gold)),
                "ce": float(ce_vec[row].detach().cpu().item()),
                "seq_len": int(example.seq_len),
                "num_distractors": int(example.distractor_count),
                "template_id": example.template_id,
                "target_character": example.target_character,
                "target_room": example.target_room,
                "evidence_frame_indices": list(example.evidence_frame_indices),
                "raw_answer": str(pred),
                "candidate_logits": candidate_logits_payload(count_logits[row], count_values),
                "token_selection_ok": int(bool(batch.token_selection_ok[row])),
                "token_selection_error": str(batch.token_selection_errors[row]),
                "carrier_positions": list(batch.carrier_positions[row]),
                "frame_token_counts": [len(group) for group in batch.frame_groups[row]],
            }
            for key, value in stats.items():
                if isinstance(value, dict):
                    row_payload[f"{key}_json"] = json_compact(value)
                else:
                    row_payload[key] = value
            rows.append(row_payload)
        if adapter is not None:
            adapter.clear_context()
        if batch_num == 1 or batch_num % 25 == 0:
            print(f"  eval {variant} {split_name}: {min(len(indices), batch_num * int(batch_size))}/{len(indices)}")
    return {
        "rows": rows,
        "ce": ce_total / max(1, n),
        "accuracy": accuracy([int(row["gold_count"]) for row in rows], [int(row["predicted_count"]) for row in rows]),
        "mae": mae([int(row["gold_count"]) for row in rows], [int(row["predicted_count"]) for row in rows]),
    }


def train_adapter(
    *,
    variant: str,
    args: argparse.Namespace,
    run_dir: Path,
    model: Any,
    tokenizer: Any,
    adapter: VariantAdapter,
    train_examples: Sequence[EvidenceExample],
    val_examples: Sequence[EvidenceExample],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    count_token_ids: Dict[int, int],
    count_values: Sequence[int],
    device: str,
) -> Tuple[List[Dict[str, Any]], Path]:
    trainable = [param for param in adapter.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError(f"{variant}: no trainable LoRA/carrier parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "best_adapter.pt"
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_acc = -math.inf
    best_val_ce = math.inf
    history: List[Dict[str, Any]] = []
    count_min = min(int(x) for x in count_values)

    for epoch in range(1, int(args.epochs) + 1):
        model.eval()
        adapter.train()
        batches = batch_indices(train_indices, int(args.batch_size), seed=int(args.seed) + epoch * 1009, shuffle=True)
        optimizer.zero_grad(set_to_none=True)
        train_ce_total = 0.0
        train_loss_total = 0.0
        train_correct = 0
        train_n = 0
        steps = 0
        backward_steps = 0
        for step, idxs in enumerate(batches, start=1):
            batch_examples = [train_examples[int(idx)] for idx in idxs]
            batch = prepare_batch(examples=batch_examples, sample_indices=idxs, tokenizer=tokenizer, device=device)
            adapter.set_context(batch)
            outputs = model(**batch.inputs, use_cache=False)
            count_logits = select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
            gold_offsets = batch.gold_counts.long() - int(count_min)
            ce = F.cross_entropy(count_logits, gold_offsets)
            loss = ce
            torch.autograd.backward(loss / max(1, int(args.grad_accum)))
            preds = count_logits.argmax(dim=-1) + int(count_min)
            train_correct += int((preds == batch.gold_counts.long()).sum().detach().cpu().item())
            train_n += int(batch.gold_counts.numel())
            train_ce_total += float(ce.detach().cpu().item())
            train_loss_total += float(loss.detach().cpu().item())
            steps += 1
            backward_steps += 1
            adapter.clear_context()
            if backward_steps % max(1, int(args.grad_accum)) == 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 25 == 0:
                print(
                    f"  {variant} epoch={epoch} step={step}/{len(batches)} "
                    f"train_ce={train_ce_total / max(1, steps):.4f} "
                    f"train_acc={train_correct / max(1, train_n):.4f}"
                )
        if backward_steps and backward_steps % max(1, int(args.grad_accum)) != 0:
            torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        val_eval = evaluate_split(
            variant=variant,
            split_name="val_iid",
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            examples=val_examples,
            indices=val_indices,
            count_token_ids=count_token_ids,
            count_values=count_values,
            device=device,
            batch_size=int(args.batch_size),
            seed=int(args.seed) + 5000 + epoch,
        )
        row = {
            "variant": str(variant),
            "epoch": int(epoch),
            "train_ce": train_ce_total / max(1, steps),
            "train_loss": train_loss_total / max(1, steps),
            "train_accuracy": train_correct / max(1, train_n),
            "train_steps": int(steps),
            "val_ce": float(val_eval["ce"]),
            "val_accuracy": float(val_eval["accuracy"]),
            "val_mae": float(val_eval["mae"]),
            "trainable_parameters": count_trainable_parameters(adapter),
        }
        history.append(row)
        print(
            f"  {variant} epoch={epoch} train_ce={row['train_ce']:.4f} "
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
                    "variant": str(variant),
                    "note": "Adapter-only checkpoint. Frozen Qwen weights are not stored.",
                },
                checkpoint_path,
            )
    if best_state is not None:
        adapter.load_state_dict(best_state)
    return history, checkpoint_path


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gold = [int(row["gold_count"]) for row in rows]
    pred = [int(row["predicted_count"]) for row in rows]
    return {
        "n": int(len(rows)),
        "accuracy": accuracy(gold, pred),
        "mae": mae(gold, pred),
        "mean_predicted_count": finite_mean(pred, default=math.nan),
        "mean_gold_count": finite_mean(gold, default=math.nan),
    }


def build_metrics(rows: Sequence[Dict[str, Any]], variant: str, splits: Sequence[str], count_values: Sequence[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    split_metrics: List[Dict[str, Any]] = []
    count_metrics: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []
    for split in splits:
        split_rows = [row for row in rows if row.get("split") == split]
        split_metrics.append({"variant": variant, "split": split, **summarize_rows(split_rows)})
        for count in count_values:
            count_rows = [row for row in split_rows if int(row.get("gold_count", -999)) == int(count)]
            preds = [int(row["predicted_count"]) for row in count_rows]
            count_metrics.append(
                {
                    "variant": variant,
                    "split": split,
                    "gold_count": int(count),
                    **summarize_rows(count_rows),
                    "mean_predicted_count": finite_mean(preds, default=math.nan),
                }
            )
        for gold in count_values:
            for pred in count_values:
                n = sum(
                    1
                    for row in split_rows
                    if int(row.get("gold_count", -999)) == int(gold)
                    and int(row.get("predicted_count", -999)) == int(pred)
                )
                confusion_rows.append({"variant": variant, "split": split, "gold_count": int(gold), "predicted_count": int(pred), "n": int(n)})
    return split_metrics, count_metrics, confusion_rows


def plot_accuracy_by_count(path: Path, count_rows: Sequence[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for split in sorted({str(row["split"]) for row in count_rows}):
        rows = sorted([row for row in count_rows if row["split"] == split], key=lambda row: int(row["gold_count"]))
        ax.plot([int(row["gold_count"]) for row in rows], [float(row["accuracy"]) for row in rows], marker="o", label=split)
    ax.set_xlabel("Gold evidence count")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mae_by_count(path: Path, count_rows: Sequence[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for split in sorted({str(row["split"]) for row in count_rows}):
        rows = sorted([row for row in count_rows if row["split"] == split], key=lambda row: int(row["gold_count"]))
        ax.plot([int(row["gold_count"]) for row in rows], [float(row["mae"]) for row in rows], marker="o", label=split)
    ax.set_xlabel("Gold evidence count")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mean_pred_by_gold(path: Path, count_rows: Sequence[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for split in sorted({str(row["split"]) for row in count_rows}):
        rows = sorted([row for row in count_rows if row["split"] == split], key=lambda row: int(row["gold_count"]))
        ax.plot([int(row["gold_count"]) for row in rows], [float(row["mean_predicted_count"]) for row in rows], marker="o", label=split)
    counts = sorted({int(row["gold_count"]) for row in count_rows})
    ax.plot(counts, counts, color="black", linewidth=1.0, linestyle="--", label="ideal")
    ax.set_xlabel("Gold evidence count")
    ax.set_ylabel("Mean predicted count")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(path: Path, rows: Sequence[Dict[str, Any]], count_values: Sequence[int]) -> None:
    matrix = np.zeros((len(count_values), len(count_values)), dtype=float)
    idx = {int(count): pos for pos, count in enumerate(count_values)}
    for row in rows:
        matrix[idx[int(row["gold_count"])], idx[int(row["predicted_count"])]] += float(row["n"])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(count_values)))
    ax.set_xticklabels([str(x) for x in count_values])
    ax.set_yticks(range(len(count_values)))
    ax.set_yticklabels([str(x) for x in count_values])
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("Gold count")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_iid_vs_ood_bar(path: Path, split_metrics: Sequence[Dict[str, Any]]) -> None:
    rows = [row for row in split_metrics if row["split"] != "train"]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [str(row["split"]) for row in rows]
    ax.bar(range(len(rows)), [float(row["accuracy"]) for row in rows], color="#4c78a8")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_train_val_curves(path: Path, history: Sequence[Dict[str, Any]]) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    if history:
        epochs = [int(row["epoch"]) for row in history]
        ax1.plot(epochs, [float(row["train_loss"]) for row in history], marker="o", label="train loss", color="#4c78a8")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train loss")
        ax2 = ax1.twinx()
        ax2.plot(epochs, [float(row["val_accuracy"]) for row in history], marker="s", label="val accuracy", color="#f58518")
        ax2.set_ylabel("Validation accuracy")
        ax2.set_ylim(0, 1)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax1.text(0.5, 0.5, "No training for frozen baseline", ha="center", va="center")
        ax1.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_run_artifacts(
    *,
    run_dir: Path,
    variant: str,
    rows: Sequence[Dict[str, Any]],
    history: Sequence[Dict[str, Any]],
    count_values: Sequence[int],
    no_plots: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    splits = ["train", "val_iid", "test_iid", "test_ood_long", "test_ood_high_count", "test_ood_distractor_heavy", "test_heldout_template"]
    split_metrics, count_metrics, confusion_rows = build_metrics(rows, variant, splits, count_values)
    write_jsonl(run_dir / "predictions.jsonl", list(rows))
    for split in splits:
        write_jsonl(run_dir / f"predictions_{split}.jsonl", [row for row in rows if row.get("split") == split])
    write_csv_dynamic(run_dir / "metrics_by_split.csv", split_metrics, ["variant", "split"])
    write_json(run_dir / "metrics_by_split.json", {"rows": split_metrics})
    write_csv_dynamic(run_dir / "accuracy_by_count.csv", count_metrics, ["variant", "split", "gold_count"])
    write_csv_dynamic(run_dir / "confusion_matrix.csv", confusion_rows, ["variant", "split", "gold_count", "predicted_count"])
    write_csv_dynamic(run_dir / "train_history.csv", list(history), ["variant", "epoch"])
    write_json(run_dir / "train_history.json", {"rows": list(history)})
    if not no_plots:
        plot_accuracy_by_count(run_dir / "accuracy_by_count.png", count_metrics)
        plot_mae_by_count(run_dir / "mae_by_count.png", count_metrics)
        plot_mean_pred_by_gold(run_dir / "mean_predicted_count_by_gold_count.png", count_metrics)
        plot_confusion_matrix(run_dir / "confusion_matrix.png", confusion_rows, count_values)
        plot_iid_vs_ood_bar(run_dir / "iid_vs_ood_accuracy_bar.png", split_metrics)
        plot_train_val_curves(run_dir / "train_val_curves.png", history)
    return split_metrics, count_metrics


def plot_summary_accuracy_by_split(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    splits = [split for split in ["val_iid", "test_iid", "test_ood_long", "test_ood_high_count", "test_ood_distractor_heavy", "test_heldout_template"] if any(row["split"] == split for row in rows)]
    variants = [variant for variant in VARIANTS if any(row["variant"] == variant for row in rows)]
    width = 0.8 / max(1, len(variants))
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(splits))
    for pos, variant in enumerate(variants):
        vals = []
        for split in splits:
            match = next((row for row in rows if row["variant"] == variant and row["split"] == split), None)
            vals.append(float(match["accuracy"]) if match else math.nan)
        ax.bar(x + (pos - (len(variants) - 1) / 2) * width, vals, width=width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_summary_accuracy_by_count(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in [variant for variant in VARIANTS if any(row["variant"] == variant for row in rows)]:
        variant_rows = [
            row
            for row in rows
            if row["variant"] == variant and row["split"] in {"test_ood_high_count", "test_ood_distractor_heavy"}
        ]
        by_count: Dict[int, List[float]] = defaultdict(list)
        for row in variant_rows:
            if finite_float(row.get("accuracy")) is not None:
                by_count[int(row["gold_count"])].append(float(row["accuracy"]))
        xs = sorted(by_count)
        ys = [finite_mean(by_count[count], default=math.nan) for count in xs]
        ax.plot(xs, ys, marker="o", label=variant)
    ax.set_xlabel("Gold evidence count")
    ax.set_ylabel("Accuracy on high-count/distractor OOD")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_summary_ood_gap(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    variants = [variant for variant in VARIANTS if any(row["variant"] == variant for row in rows)]
    gaps: List[float] = []
    for variant in variants:
        iid = next((row for row in rows if row["variant"] == variant and row["split"] == "test_iid"), None)
        ood_vals = [
            float(row["accuracy"])
            for row in rows
            if row["variant"] == variant and str(row["split"]).startswith("test_ood") and finite_float(row.get("accuracy")) is not None
        ]
        gaps.append(float(iid["accuracy"]) - finite_mean(ood_vals, default=math.nan) if iid and ood_vals else math.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(variants)), gaps, color="#e45756")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.set_ylabel("IID accuracy minus mean OOD accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_summary_mae_by_split(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    splits = [split for split in ["test_iid", "test_ood_long", "test_ood_high_count", "test_ood_distractor_heavy", "test_heldout_template"] if any(row["split"] == split for row in rows)]
    variants = [variant for variant in VARIANTS if any(row["variant"] == variant for row in rows)]
    width = 0.8 / max(1, len(variants))
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(splits))
    for pos, variant in enumerate(variants):
        vals = []
        for split in splits:
            match = next((row for row in rows if row["variant"] == variant and row["split"] == split), None)
            vals.append(float(match["mae"]) if match else math.nan)
        ax.bar(x + (pos - (len(variants) - 1) / 2) * width, vals, width=width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=25, ha="right")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(
    *,
    summary_dir: Path,
    split_rows: Sequence[Dict[str, Any]],
    count_rows: Sequence[Dict[str, Any]],
    no_plots: bool,
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv_dynamic(summary_dir / "summary_metrics.csv", split_rows, ["variant", "split"])
    write_json(summary_dir / "summary_metrics.json", {"rows": list(split_rows)})
    write_csv_dynamic(summary_dir / "summary_accuracy_by_count.csv", count_rows, ["variant", "split", "gold_count"])
    if not no_plots:
        plot_summary_accuracy_by_split(summary_dir / "accuracy_by_split.png", split_rows)
        plot_summary_accuracy_by_count(summary_dir / "accuracy_by_count.png", count_rows)
        plot_summary_ood_gap(summary_dir / "ood_gap.png", split_rows)
        plot_summary_mae_by_split(summary_dir / "mae_by_split.png", split_rows)


def resolve_run_base(args: argparse.Namespace) -> Path:
    root = Path(args.output_root).resolve()
    return root / "debug" if bool(args.debug) else root


def run_dir_for_variant(run_base: Path, variant: str, prefix: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix_part = f"{safe_name(prefix)}_" if str(prefix).strip() else ""
    return run_base / f"{stamp}_{prefix_part}{safe_name(variant)}"


def layer_summary(model: Any) -> Dict[str, Any]:
    layers = get_layers(model)
    return {"num_layers": len(layers), "valid_layer_range": [0, len(layers) - 1]}


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    variants = parse_variants(args.variants)
    lora_layers = parse_int_tokens(args.lora_layers)
    carrier_layers = parse_int_tokens(args.carrier_layers)
    lora_targets = split_tokens(args.lora_targets)
    run_base = resolve_run_base(args)
    dataset_base = run_base / "datasets" if bool(args.debug) else Path(args.output_root).resolve() / "datasets"
    run_base.mkdir(parents=True, exist_ok=True)
    dataset_dir, examples_by_split, dataset_manifest = ensure_dataset(args, dataset_base)
    print(f"Dataset: {dataset_dir}")
    for split, rows in examples_by_split.items():
        print(f"  {split}: {len(rows)} examples")

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false. Use --device cpu for local debug.")
    dtype = dtype_from_arg(str(args.dtype))
    model, tokenizer, qlora_used, load_note = load_model_and_tokenizer(args, device=device, dtype=dtype)
    count_style, count_token_ids = candidate_token_ids(tokenizer, int(args.candidate_min), int(args.candidate_max))
    count_values = sorted(count_token_ids)
    hidden_size = hidden_size_from_model(model)
    layers_info = layer_summary(model)
    print(f"Loaded model={args.model_name} hidden_size={hidden_size} {layers_info} qlora_used={qlora_used} load_note={load_note}")
    print(f"Count token style: {count_style} ids={count_token_ids}")

    all_summary_split_rows: List[Dict[str, Any]] = []
    all_summary_count_rows: List[Dict[str, Any]] = []
    split_order = ["train", "val_iid", "test_iid", "test_ood_long", "test_ood_high_count", "test_ood_distractor_heavy", "test_heldout_template"]
    train_indices = limited_indices(examples_by_split["train"], int(args.max_train_examples), int(args.seed) + 100)
    eval_indices_by_split = {
        split: limited_indices(examples_by_split[split], int(args.max_eval_examples), int(args.seed) + 200 + pos)
        for pos, split in enumerate(split_order)
        if split != "train"
    }
    train_eval_indices = train_indices if int(args.max_eval_examples) <= 0 else train_indices[: int(args.max_eval_examples)]

    for variant in variants:
        run_dir = run_dir_for_variant(run_base, variant, str(args.run_prefix))
        log_handle, old_stdout, old_stderr = setup_logging(run_dir)
        adapter: Optional[VariantAdapter] = None
        try:
            print(f"Starting variant={variant} run_dir={run_dir}")
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
                "dtype": str(args.dtype),
                "device": str(args.device),
                "load_in_4bit_requested": bool(args.load_in_4bit),
                "qlora_used": bool(qlora_used),
                "load_note": str(load_note),
                "bf16_preferred": not bool(args.load_in_4bit),
                "lora": {
                    "enabled": variant != FROZEN_QWEN_BASELINE,
                    "layers": list(lora_layers),
                    "target_modules": list(lora_targets),
                    "rank": int(args.lora_rank),
                    "alpha": float(args.lora_alpha),
                    "dropout": float(args.lora_dropout),
                    "layer_filtering": "manual_wrapper",
                },
                "carrier_mixing": {
                    "enabled": variant in {LORA_MAXMIX, LORA_PNAMIX},
                    "layers": list(carrier_layers),
                    "method": "maxmix" if variant == LORA_MAXMIX else "pnamix" if variant == LORA_PNAMIX else "none",
                    "target_tokens": "query room and character tokens",
                    "inject_last_token": False,
                    "message_mode_requested": str(args.message_mode),
                    "implementation_note": (
                        "Uses exact Qwen2.5-VL alpha*v with multimodal RoPE when layer kwargs expose position_embeddings; "
                        "otherwise logs fallback and uses q/k/v attention without RoPE as the closest robust hook."
                    ),
                },
                "training": {
                    "epochs": int(args.epochs) if variant != FROZEN_QWEN_BASELINE else 0,
                    "lr": float(args.lr),
                    "batch_size": int(args.batch_size),
                    "grad_accum": int(args.grad_accum),
                    "grad_clip": float(args.grad_clip),
                    "weight_decay": float(args.weight_decay),
                    "loss": "cross_entropy_over_candidate_count_answer_tokens",
                    "trainable_parameters": int(trainable_parameters),
                },
                "candidate_answer_tokens": {"style": count_style, "ids": count_token_ids},
                "model_layers": layers_info,
                "submit_mode": str(args.submit_mode),
                "debug": bool(args.debug),
            }
            write_json(run_dir / "config.json", config)
            print(f"Trainable parameters: {trainable_parameters}")
            history: List[Dict[str, Any]] = []
            if adapter is not None:
                history, checkpoint_path = train_adapter(
                    variant=variant,
                    args=args,
                    run_dir=run_dir,
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    train_examples=examples_by_split["train"],
                    val_examples=examples_by_split["val_iid"],
                    train_indices=train_indices,
                    val_indices=eval_indices_by_split["val_iid"],
                    count_token_ids=count_token_ids,
                    count_values=count_values,
                    device=device,
                )
                print(f"Best checkpoint: {checkpoint_path}")
            else:
                print("Frozen baseline: no training.")
            all_rows: List[Dict[str, Any]] = []
            eval_plan = [("train", examples_by_split["train"], train_eval_indices)]
            eval_plan.extend(
                (split, examples_by_split[split], eval_indices_by_split[split])
                for split in split_order
                if split != "train"
            )
            for split, examples, indices in eval_plan:
                result = evaluate_split(
                    variant=variant,
                    split_name=split,
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    examples=examples,
                    indices=indices,
                    count_token_ids=count_token_ids,
                    count_values=count_values,
                    device=device,
                    batch_size=int(args.batch_size),
                    seed=int(args.seed) + 7000,
                )
                print(f"  {split}: accuracy={result['accuracy']:.4f} mae={result['mae']:.4f} ce={result['ce']:.4f}")
                all_rows.extend(result["rows"])
            split_rows, count_rows = write_run_artifacts(
                run_dir=run_dir,
                variant=variant,
                rows=all_rows,
                history=history,
                count_values=count_values,
                no_plots=bool(args.no_plots),
            )
            all_summary_split_rows.extend(split_rows)
            all_summary_count_rows.extend(count_rows)
            if adapter is not None and adapter.mixer is not None:
                write_json(
                    run_dir / "carrier_message_diagnostics.json",
                    {
                        "hook_fire_counts": {str(k): int(v) for k, v in sorted(adapter.mixer.hook_fire_counts.items())},
                        "message_mode_counts": {str(k): int(v) for k, v in sorted(adapter.mixer.message_mode_counts.items())},
                        "exact_failure_counts": {str(k): int(v) for k, v in sorted(adapter.mixer.exact_failure_counts.items())},
                        "exact_failure_examples": list(adapter.mixer.exact_failure_examples),
                    },
                )
            print(f"Finished variant={variant}")
        finally:
            if adapter is not None:
                adapter.detach()
            restore_logging(log_handle, old_stdout, old_stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_dir = (Path(args.output_root).resolve() / "summary") if not bool(args.debug) else (run_base / "summary")
    write_summary(
        summary_dir=summary_dir,
        split_rows=all_summary_split_rows,
        count_rows=all_summary_count_rows,
        no_plots=bool(args.no_plots),
    )
    print(f"Summary written to {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

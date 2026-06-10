#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluations.helpers import utils as eval_utils
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base_eval

SUPPORTED_INPUT_FORMATS = ("separate_frames", "mosaic")
SUPPORTED_MOSAIC_SEQ_LENS = tuple(range(2, 10))
ALL_EXACT_SPLITS = {"by_evidence_count", "all_exact", "exact"}

PREDICTION_FIELDS = [
    "input_format",
    "seq_len",
    "evidence_count",
    "sample_id",
    "sample_dir",
    "gold_text",
    "gold_int",
    "predicted_int",
    "correct",
    "raw_prediction",
    "error",
]
METRIC_FIELDS = [
    "input_format",
    "seq_len",
    "evidence_count",
    "n_samples",
    "n_correct",
    "accuracy",
    "n_valid_numeric_predictions",
    "mae",
    "bias",
    "n_valid_predictions_for_avg_prediction",
    "avg_prediction",
]
SUMMARY_FIELDS = ["input_format", "seq_len", "n_samples", "n_correct", "accuracy"]


class EvidenceErrorStats(TypedDict):
    sum_abs_error: float
    sum_signed_error: float
    n_valid: int


class EvidencePredictionStats(TypedDict):
    sum_prediction: float
    n_valid: int


@dataclass
class EvalSample:
    sample_id: str
    sample_dir: Path
    frames: List[Image.Image]
    question: str
    gold_text: str
    gold_int: Optional[int]
    evidence_count: int
    seq_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen 2.5 VL basic MMReD accuracy on either old separate-frame data "
            "or the new one-image mosaic render format."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Rendered MMReD root. Accepts a base root with seq_len_* dirs, a seq_len_* dir, "
            "a by_evidence_count parent, or one exact_* sample bucket."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mmred_basic_acc_mosaic"),
        help="Directory for run logs/CSV outputs.",
    )
    parser.add_argument(
        "--input-format",
        choices=SUPPORTED_INPUT_FORMATS,
        default="mosaic",
        help="mosaic loads only sample/000.png and uses mosaic-layout prompts; separate_frames uses the old behavior.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help=(
            "Optional HF DatasetDict root used to resolve train/val/test membership for old all/ image roots. "
            "Usually unnecessary for mosaic exact_* directories."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all_uniform",
        help=(
            "Subset under each seq_len dir when --data-root is a base root. Use exact_0/exact_1/etc., "
            "or by_evidence_count/all_exact to evaluate every exact_* bucket."
        ),
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=None,
        help="Optional comma-separated seq_len list. Defaults to discovery when --data-root is a base root.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the model with bitsandbytes 4-bit quantization.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--limit",
        "--limit-per-seq",
        dest="limit_per_seq",
        type=int,
        default=None,
        help="Optional cap on evaluated samples per seq_len after sample-dir resolution.",
    )
    parser.add_argument("--model", "--model-id", dest="model_id", type=str, default=base_eval.MODEL_ID)
    return parser.parse_args()


def default_data_root(input_format: str) -> Path:
    if input_format == "mosaic":
        for candidate in (
            Path("data/mmred_mosaic_images"),
            Path("data/mmred_mosaic_images_generated"),
        ):
            if candidate.exists():
                return candidate
        return Path("data/mmred_mosaic_images")
    return base_eval.default_data_root()


def display_split_name(data_root: Path, split: str) -> str:
    if data_root.name.startswith("exact_"):
        return data_root.name
    if data_root.name == "by_evidence_count":
        return "by_evidence_count"
    return split


def infer_seq_len_from_path(path: Path) -> Optional[int]:
    for part in reversed(path.parts):
        match = base_eval.SEQ_LEN_DIR_RE.fullmatch(part)
        if match is not None:
            return int(match.group(1))
    return None


def seq_len_dir_from_path(path: Path, seq_len: int) -> Optional[Path]:
    target_name = f"seq_len_{seq_len}"
    parts = path.parts
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == target_name:
            return Path(*parts[: idx + 1])
    return None


def parse_qa_file(sample_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str, str]:
    qa_path = sample_dir / "qa.txt"
    if not qa_path.is_file():
        raise FileNotFoundError(f"Missing qa.txt: {qa_path}")

    lines = qa_path.read_text(encoding="utf-8").splitlines()
    question_start = next((idx for idx, line in enumerate(lines) if line.strip() == "question:"), -1)
    answer_start = next((idx for idx, line in enumerate(lines) if line.strip() == "answer:"), -1)
    if question_start < 0 or answer_start <= question_start:
        raise RuntimeError(f"Bad qa.txt format: {qa_path}")

    metadata: Dict[str, Any] = {}
    for line in lines[:question_start]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    states: List[Dict[str, Any]] = []
    question_text: Optional[str] = None
    for line in lines[question_start + 1 : answer_start]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            states.append(ast.literal_eval(stripped))
            continue
        question_text = stripped
        break

    if question_text is None:
        raise RuntimeError(f"Could not locate natural-language question in {qa_path}")

    gold_text = next((line.strip() for line in lines[answer_start + 1 :] if line.strip()), "")
    if not gold_text:
        raise RuntimeError(f"Could not locate answer text in {qa_path}")

    if "seq_len" in metadata:
        try:
            metadata["seq_len"] = int(metadata["seq_len"])
        except ValueError as exc:
            raise RuntimeError(f"Bad seq_len in {qa_path}: {metadata['seq_len']!r}") from exc

    return metadata, states, question_text, gold_text


def infer_seq_len_for_sample_dir(sample_dir: Path) -> int:
    metadata, states, _, _ = parse_qa_file(sample_dir)
    if "seq_len" in metadata:
        return int(metadata["seq_len"])
    path_seq_len = infer_seq_len_from_path(sample_dir)
    if path_seq_len is not None:
        return int(path_seq_len)
    return len(states)


def collect_exact_bucket_sample_dirs(parent_dir: Path) -> List[Path]:
    if not parent_dir.is_dir():
        return []

    sample_dirs: List[Path] = []
    for bucket_dir in sorted(path for path in parent_dir.iterdir() if path.is_dir() and path.name.startswith("exact_")):
        sample_dirs.extend(base_eval.iter_sample_dirs(bucket_dir))
    return sample_dirs


def group_sample_dirs_by_seq_len(
    sample_dirs: Sequence[Path],
    seq_filter: Optional[set[int]],
) -> Dict[int, List[Path]]:
    grouped: Dict[int, List[Path]] = {}
    for sample_dir in sorted(sample_dirs):
        seq_len = infer_seq_len_for_sample_dir(sample_dir)
        if seq_filter is not None and seq_len not in seq_filter:
            continue
        grouped.setdefault(seq_len, []).append(sample_dir)
    return grouped


def resolve_base_root_sample_dirs(
    *,
    data_root: Path,
    seq_lens: Sequence[int],
    split: str,
    split_root: Optional[Path],
) -> Dict[int, List[Path]]:
    grouped: Dict[int, List[Path]] = {}
    for seq_len in seq_lens:
        if split in ALL_EXACT_SPLITS:
            sample_dirs = collect_exact_bucket_sample_dirs(data_root / f"seq_len_{seq_len}" / "by_evidence_count")
        else:
            sample_dirs = base_eval.resolve_sample_dirs(
                data_root=data_root,
                seq_len=seq_len,
                split=split,
                split_root=split_root,
            )
        grouped[int(seq_len)] = list(sample_dirs)
    return grouped


def resolve_sample_dirs_by_seq_len(
    *,
    data_root: Path,
    raw_seq_lens: Optional[str],
    split: str,
    split_root: Optional[Path],
) -> Dict[int, List[Path]]:
    explicit_seq_lens = base_eval.parse_seq_lens_arg(raw_seq_lens)
    seq_filter = set(explicit_seq_lens) if explicit_seq_lens is not None else None

    direct_sample_dirs = base_eval.iter_sample_dirs(data_root) if data_root.is_dir() else []
    if direct_sample_dirs:
        return group_sample_dirs_by_seq_len(direct_sample_dirs, seq_filter)

    exact_parent_sample_dirs = collect_exact_bucket_sample_dirs(data_root)
    if exact_parent_sample_dirs:
        return group_sample_dirs_by_seq_len(exact_parent_sample_dirs, seq_filter)

    path_seq_len = infer_seq_len_from_path(data_root)
    if path_seq_len is not None:
        if seq_filter is not None and path_seq_len not in seq_filter:
            return {}

        # If the user accidentally appends /seq_len_X after a concrete sample
        # folder parent such as .../seq_len_2/all_uniform/seq_len_2, recover by
        # evaluating the parent split folder rather than appending seq_len_X again.
        if not data_root.is_dir() and data_root.name == f"seq_len_{path_seq_len}" and data_root.parent.is_dir():
            parent_sample_dirs = base_eval.iter_sample_dirs(data_root.parent)
            if parent_sample_dirs:
                return group_sample_dirs_by_seq_len(parent_sample_dirs, {path_seq_len})

        if split in ALL_EXACT_SPLITS and (data_root / "by_evidence_count").is_dir():
            sample_dirs = collect_exact_bucket_sample_dirs(data_root / "by_evidence_count")
            return group_sample_dirs_by_seq_len(sample_dirs, {path_seq_len})

        seq_len_dir = seq_len_dir_from_path(data_root, path_seq_len)
        if seq_len_dir is not None and data_root != seq_len_dir:
            raise FileNotFoundError(
                f"No sample folders with qa.txt found directly under {data_root}. "
                f"For this mosaic split, use the split directory itself, e.g. {data_root.parent} "
                f"if that contains <sample_id>/qa.txt folders."
            )

        seq_base_root = data_root.parent
        sample_dirs = base_eval.resolve_sample_dirs(
            data_root=seq_base_root,
            seq_len=path_seq_len,
            split=split,
            split_root=split_root,
        )
        return group_sample_dirs_by_seq_len(sample_dirs, {path_seq_len})

    seq_lens = explicit_seq_lens if explicit_seq_lens is not None else base_eval.resolve_seq_lens(data_root, None)
    return resolve_base_root_sample_dirs(
        data_root=data_root,
        seq_lens=seq_lens,
        split=split,
        split_root=split_root,
    )


def load_separate_frame_sample(sample_dir: Path) -> EvalSample:
    sample = base_eval.load_mmred_style_sample(sample_dir)
    return EvalSample(
        sample_id=sample.sample_id,
        sample_dir=sample.sample_dir,
        frames=sample.frames,
        question=sample.question,
        gold_text=sample.gold_text,
        gold_int=sample.gold_int,
        evidence_count=sample.evidence_count,
        seq_len=len(sample.frames),
    )


def load_mosaic_sample(sample_dir: Path) -> EvalSample:
    metadata, states, question_text, gold_text = parse_qa_file(sample_dir)
    seq_len = int(metadata["seq_len"]) if "seq_len" in metadata else len(states)
    if len(states) != seq_len:
        raise RuntimeError(f"qa.txt seq_len={seq_len} but found {len(states)} state lines in {sample_dir / 'qa.txt'}")
    if seq_len not in SUPPORTED_MOSAIC_SEQ_LENS:
        raise ValueError(
            f"Mosaic input supports seq_len {supported_mosaic_seq_lens_text()}; "
            f"got seq_len={seq_len} for {sample_dir}"
        )

    mosaic_path = sample_dir / "000.png"
    png_paths = sorted(sample_dir.glob("*.png"))
    if png_paths != [mosaic_path]:
        found = ", ".join(path.name for path in png_paths) or "<none>"
        raise FileNotFoundError(f"Mosaic sample must contain exactly one image 000.png: {sample_dir} (found: {found})")

    with Image.open(mosaic_path) as image:
        frames = [image.convert("RGB")]

    return EvalSample(
        sample_id=sample_dir.name,
        sample_dir=sample_dir,
        frames=frames,
        question=question_text,
        gold_text=gold_text,
        gold_int=base_eval.extract_first_integer(gold_text),
        evidence_count=len(eval_utils.collect_evidence_frame_indices(question_text, states)),
        seq_len=seq_len,
    )


def load_eval_sample(sample_dir: Path, input_format: str) -> EvalSample:
    if input_format == "separate_frames":
        return load_separate_frame_sample(sample_dir)
    if input_format == "mosaic":
        return load_mosaic_sample(sample_dir)
    raise ValueError(f"Unsupported input_format={input_format!r}")


def supported_mosaic_seq_lens_text() -> str:
    return f"{min(SUPPORTED_MOSAIC_SEQ_LENS)} through {max(SUPPORTED_MOSAIC_SEQ_LENS)}"


def mosaic_layout_description(seq_len: int) -> str:
    if seq_len == 2:
        return "one image containing 2 frames arranged from left to right"
    if seq_len in (3, 4):
        return f"one image containing {seq_len} frames arranged in a 2x2 grid in reading order"
    if 5 <= seq_len <= 9:
        return f"one image containing {seq_len} frames arranged in a 3x3 grid in reading order"
    raise ValueError(
        f"Mosaic prompt supports seq_len {supported_mosaic_seq_lens_text()}; got seq_len={seq_len}"
    )


def mosaic_step_position_text(seq_len: int) -> str:
    if seq_len == 2:
        return "The left half is Step 1 and the right half is Step 2."
    if seq_len in (3, 4):
        lines = [
            "Top-left is Step 1, top-right is Step 2, bottom-left is Step 3, bottom-right is Step 4."
        ]
        if seq_len == 3:
            lines.append("The unused cell is blank / unused.")
        return "\n".join(lines)
    if 5 <= seq_len <= 9:
        return (
            "Row 1: top-left is Step 1, top-middle is Step 2, top-right is Step 3.\n"
            "Row 2: middle-left is Step 4, middle is Step 5, middle-right is Step 6.\n"
            "Row 3: bottom-left is Step 7, bottom-middle is Step 8, bottom-right is Step 9.\n"
            f"Only the first {seq_len} positions are used; any remaining cells are blank / unused."
        )
    raise ValueError(
        f"Mosaic prompt supports seq_len {supported_mosaic_seq_lens_text()}; got seq_len={seq_len}"
    )


def build_mosaic_prompt(question: str, seq_len: int) -> str:
    layout_description = mosaic_layout_description(seq_len)
    step_position_text = mosaic_step_position_text(seq_len)
    return (
        f"You will be shown {layout_description}.\n"
        f"{step_position_text}\n"
        f"Respond with a single integer from 0 to {seq_len}.\n"
        "Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )


def build_prompt(question: str, seq_len: int, input_format: str) -> str:
    if input_format == "separate_frames":
        return base_eval.build_prompt(question, num_frames=seq_len)
    if input_format == "mosaic":
        return build_mosaic_prompt(question, seq_len=seq_len)
    raise ValueError(f"Unsupported input_format={input_format!r}")


def run_generation(
    *,
    model: Any,
    processor: Any,
    sample: EvalSample,
    input_format: str,
    max_new_tokens: int,
    device: str,
) -> str:
    prompt = build_prompt(sample.question, seq_len=sample.seq_len, input_format=input_format)
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": image} for image in sample.frames]
                + [{"type": "text", "text": prompt}]
            ),
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = base_eval.move_inputs_to_device(dict(inputs), device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )

    generated_ids = output_ids[:, prompt_len:]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return str(decoded).strip()


def accuracy_value(correct: int, total: int) -> float:
    return 0.0 if total == 0 else float(correct) / float(total)


def empty_error_stats() -> EvidenceErrorStats:
    return {"sum_abs_error": 0.0, "sum_signed_error": 0.0, "n_valid": 0}


def update_evidence_error_stats(
    evidence_error_stats: Dict[int, EvidenceErrorStats],
    evidence_count: int,
    gold_int: Optional[int],
    predicted_int: Optional[int],
) -> None:
    stats = evidence_error_stats.setdefault(int(evidence_count), empty_error_stats())
    if gold_int is None or predicted_int is None:
        return

    signed_error = int(predicted_int) - int(gold_int)
    stats["sum_abs_error"] += float(abs(signed_error))
    stats["sum_signed_error"] += float(signed_error)
    stats["n_valid"] += 1


def mae_bias_from_error_stats(error_stats: EvidenceErrorStats) -> Tuple[float, float, int]:
    n_valid = int(error_stats["n_valid"])
    if n_valid == 0:
        return float("nan"), float("nan"), 0
    mae = float(error_stats["sum_abs_error"]) / float(n_valid)
    bias = float(error_stats["sum_signed_error"]) / float(n_valid)
    return mae, bias, n_valid


def empty_prediction_stats() -> EvidencePredictionStats:
    return {"sum_prediction": 0.0, "n_valid": 0}


def update_evidence_prediction_stats(
    evidence_prediction_stats: Dict[int, EvidencePredictionStats],
    evidence_count: int,
    predicted_int: Optional[int],
) -> None:
    stats = evidence_prediction_stats.setdefault(int(evidence_count), empty_prediction_stats())
    if predicted_int is None:
        return

    stats["sum_prediction"] += float(predicted_int)
    stats["n_valid"] += 1


def avg_prediction_from_stats(prediction_stats: EvidencePredictionStats) -> Tuple[float, int]:
    n_valid = int(prediction_stats["n_valid"])
    if n_valid == 0:
        return float("nan"), 0
    return float(prediction_stats["sum_prediction"]) / float(n_valid), n_valid


def format_metric_value(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_metric_rows(
    per_seq_evidence_totals: Dict[int, Dict[int, Tuple[int, int]]],
    per_seq_evidence_error_sums: Dict[int, Dict[int, EvidenceErrorStats]],
    per_seq_evidence_prediction_sums: Dict[int, Dict[int, EvidencePredictionStats]],
    input_format: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seq_len in sorted(per_seq_evidence_totals):
        for evidence_count in sorted(per_seq_evidence_totals[seq_len]):
            n_correct, n_samples = per_seq_evidence_totals[seq_len][evidence_count]
            error_stats = per_seq_evidence_error_sums.get(seq_len, {}).get(evidence_count, empty_error_stats())
            mae, bias, n_valid = mae_bias_from_error_stats(error_stats)
            prediction_stats = per_seq_evidence_prediction_sums.get(seq_len, {}).get(
                evidence_count,
                empty_prediction_stats(),
            )
            avg_prediction, n_valid_for_avg_prediction = avg_prediction_from_stats(prediction_stats)
            rows.append(
                {
                    "input_format": input_format,
                    "seq_len": int(seq_len),
                    "evidence_count": int(evidence_count),
                    "n_samples": int(n_samples),
                    "n_correct": int(n_correct),
                    "accuracy": accuracy_value(n_correct, n_samples),
                    "n_valid_numeric_predictions": int(n_valid),
                    "mae": mae,
                    "bias": bias,
                    "n_valid_predictions_for_avg_prediction": int(n_valid_for_avg_prediction),
                    "avg_prediction": avg_prediction,
                }
            )
    return rows


def bucket_metric_lines(
    *,
    seq_len: int,
    evidence_stats: Dict[int, Tuple[int, int]],
    evidence_error_stats: Dict[int, EvidenceErrorStats],
    evidence_prediction_stats: Dict[int, EvidencePredictionStats],
) -> List[str]:
    lines: List[str] = []
    for evidence_count in sorted(evidence_stats):
        correct, total = evidence_stats[evidence_count]
        lines.append(
            f"seq_len={seq_len} evidence_count={evidence_count}: accuracy = "
            f"{base_eval.accuracy_string(correct, total)}"
        )
        mae, bias, n_valid = mae_bias_from_error_stats(
            evidence_error_stats.get(evidence_count, empty_error_stats())
        )
        lines.append(
            f"seq_len={seq_len} evidence_count={evidence_count}: "
            f"mae = {format_metric_value(mae)}, "
            f"bias = {format_metric_value(bias)}, "
            f"n_valid_numeric_predictions = {n_valid}"
        )
        avg_prediction, n_valid_for_avg_prediction = avg_prediction_from_stats(
            evidence_prediction_stats.get(evidence_count, empty_prediction_stats())
        )
        lines.append(
            f"seq_len={seq_len} evidence_count={evidence_count}: "
            f"avg_prediction = {format_metric_value(avg_prediction)}, "
            f"n_valid_predictions_for_avg_prediction = {n_valid_for_avg_prediction}"
        )
    return lines


def build_summary_rows(per_seq_totals: Dict[int, Tuple[int, int]], input_format: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seq_len in sorted(per_seq_totals):
        n_correct, n_samples = per_seq_totals[seq_len]
        rows.append(
            {
                "input_format": input_format,
                "seq_len": int(seq_len),
                "n_samples": int(n_samples),
                "n_correct": int(n_correct),
                "accuracy": accuracy_value(n_correct, n_samples),
            }
        )
    return rows


def plot_metric_vs_evidence_count(
    seq_rows: Sequence[Dict[str, Any]],
    *,
    metric_name: str,
    ylabel: str,
    formula: str,
    output_path: Path,
    input_format: str,
    split_label: str,
    draw_zero_line: bool = False,
    draw_diagonal_reference: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_rows = sorted(seq_rows, key=lambda row: int(row["evidence_count"]))
    x_values = [int(row["evidence_count"]) for row in sorted_rows]
    y_values = [float(row[metric_name]) for row in sorted_rows]
    seq_len = int(sorted_rows[0]["seq_len"])

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.plot(x_values, y_values, marker="o")
    if draw_zero_line:
        ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0, alpha=0.75)
    if draw_diagonal_reference and x_values:
        ax.plot(
            [min(x_values), max(x_values)],
            [min(x_values), max(x_values)],
            linestyle="--",
            color="black",
            linewidth=1.0,
            alpha=0.75,
        )
    ax.set_xlabel("evidence_count")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{input_format} split={split_label} seq_len={seq_len}\n{formula}")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    if x_values:
        ax.set_xticks(x_values)
    if ylim is not None:
        ax.set_ylim(*ylim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_metric_plots(
    metric_rows: Sequence[Dict[str, Any]],
    *,
    output_dir: Path,
    input_format: str,
    split_label: str,
) -> List[Path]:
    rows_by_seq_len: Dict[int, List[Dict[str, Any]]] = {}
    for row in metric_rows:
        rows_by_seq_len.setdefault(int(row["seq_len"]), []).append(row)

    plots_dir = output_dir / "plots"
    plot_paths: List[Path] = []
    for seq_len in sorted(rows_by_seq_len):
        seq_rows = rows_by_seq_len[seq_len]
        if not seq_rows:
            continue
        plot_paths.append(
            plot_metric_vs_evidence_count(
                seq_rows,
                metric_name="accuracy",
                ylabel="Accuracy",
                formula="Acc(j) = mean(1[pred = gold] | evidence_count=j)",
                output_path=plots_dir / f"seq_len_{seq_len}_accuracy_vs_evidence_count.png",
                input_format=input_format,
                split_label=split_label,
                ylim=(0.0, 1.0),
            )
        )
        plot_paths.append(
            plot_metric_vs_evidence_count(
                seq_rows,
                metric_name="avg_prediction",
                ylabel="Average prediction",
                formula="AvgPred(j) = mean(pred | evidence_count=j)",
                output_path=plots_dir / f"seq_len_{seq_len}_avg_prediction_vs_evidence_count.png",
                input_format=input_format,
                split_label=split_label,
                draw_diagonal_reference=True,
            )
        )
        plot_paths.append(
            plot_metric_vs_evidence_count(
                seq_rows,
                metric_name="mae",
                ylabel="MAE",
                formula="MAE(j) = mean(|pred - gold| | evidence_count=j)",
                output_path=plots_dir / f"seq_len_{seq_len}_mae_vs_evidence_count.png",
                input_format=input_format,
                split_label=split_label,
            )
        )
        plot_paths.append(
            plot_metric_vs_evidence_count(
                seq_rows,
                metric_name="bias",
                ylabel="Bias",
                formula="Bias(j) = mean(pred - gold | evidence_count=j)",
                output_path=plots_dir / f"seq_len_{seq_len}_bias_vs_evidence_count.png",
                input_format=input_format,
                split_label=split_label,
                draw_zero_line=True,
            )
        )
    return plot_paths


def maybe_limit_sample_dirs(
    grouped: Dict[int, List[Path]],
    limit_per_seq: Optional[int],
) -> Dict[int, List[Path]]:
    if limit_per_seq is None:
        return grouped
    if limit_per_seq < 0:
        raise ValueError("--limit must be >= 0 when provided.")
    return {seq_len: sample_dirs[:limit_per_seq] for seq_len, sample_dirs in grouped.items()}


def non_empty_grouped_sample_dirs(grouped: Dict[int, List[Path]]) -> Dict[int, List[Path]]:
    return {seq_len: sample_dirs for seq_len, sample_dirs in grouped.items() if sample_dirs}


def main() -> int:
    started_at = time.time()
    args = parse_args()
    data_root = (args.data_root if args.data_root is not None else default_data_root(args.input_format)).resolve()
    output_dir = args.output_dir.resolve()
    split_label = display_split_name(data_root, args.split)
    split_root = args.split_root.resolve() if args.split_root is not None else base_eval.infer_split_root(data_root)
    device = base_eval.resolve_device(args.device)
    dtype = base_eval.resolve_dtype(args.dtype, device)

    grouped_sample_dirs = resolve_sample_dirs_by_seq_len(
        data_root=data_root,
        raw_seq_lens=args.seq_lens,
        split=args.split,
        split_root=split_root,
    )
    grouped_sample_dirs = non_empty_grouped_sample_dirs(maybe_limit_sample_dirs(grouped_sample_dirs, args.limit_per_seq))
    if not grouped_sample_dirs:
        raise RuntimeError(f"No samples found under {data_root} for split={args.split!r}.")
    if args.input_format == "mosaic":
        unsupported_seq_lens = sorted(set(grouped_sample_dirs) - set(SUPPORTED_MOSAIC_SEQ_LENS))
        if unsupported_seq_lens:
            raise SystemExit(
                f"Mosaic input supports seq_len {supported_mosaic_seq_lens_text()}; "
                f"found seq_lens={unsupported_seq_lens}."
            )

    print(f"model: {args.model_id}")
    print(f"input_format: {args.input_format}")
    print(f"data_root: {data_root}")
    print(f"split: {split_label}")
    print(f"seq_lens: {list(sorted(grouped_sample_dirs))}")
    print(f"output_dir: {output_dir}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"load_in_4bit: {args.load_in_4bit}")
    if split_root is not None:
        print(f"split_root: {split_root}")
    elif args.split != "all":
        print("split_root: <none>")
    print()

    load_started = time.time()
    model, processor = base_eval.load_model_and_processor(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )
    print(f"model_load_time_sec: {time.time() - load_started:.2f}")
    print()

    total_correct = 0
    total_count = 0
    per_seq_totals: Dict[int, Tuple[int, int]] = {}
    per_seq_evidence_totals: Dict[int, Dict[int, Tuple[int, int]]] = {}
    per_seq_evidence_error_sums: Dict[int, Dict[int, EvidenceErrorStats]] = {}
    per_seq_evidence_prediction_sums: Dict[int, Dict[int, EvidencePredictionStats]] = {}
    prediction_rows: List[Dict[str, Any]] = []

    for seq_len in sorted(grouped_sample_dirs):
        sample_dirs = grouped_sample_dirs[seq_len]
        seq_correct = 0
        seq_total = 0
        evidence_stats: Dict[int, List[int]] = {}
        evidence_error_stats: Dict[int, EvidenceErrorStats] = {}
        evidence_prediction_stats: Dict[int, EvidencePredictionStats] = {}
        print(f"Evaluating seq_len={seq_len} split={split_label} on {len(sample_dirs)} samples")

        for sample_index, sample_dir in enumerate(sample_dirs, start=1):
            sample_id = sample_dir.name
            sample_seq_len = seq_len
            gold_text = ""
            gold_int: Optional[int] = None
            gold_display = "<unavailable>"
            raw_prediction = ""
            predicted_int: Optional[int] = None
            correct = False
            evidence_count: Optional[int] = None
            error = ""

            try:
                sample = load_eval_sample(sample_dir, input_format=args.input_format)
                sample_id = sample.sample_id
                sample_seq_len = sample.seq_len
                gold_text = sample.gold_text
                gold_int = sample.gold_int
                gold_display = gold_int if gold_int is not None else repr(sample.gold_text)
                evidence_count = int(sample.evidence_count)
                raw_prediction = run_generation(
                    model=model,
                    processor=processor,
                    sample=sample,
                    input_format=args.input_format,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                predicted_int = base_eval.extract_first_integer(raw_prediction)
                correct = predicted_int is not None and gold_int is not None and predicted_int == gold_int
            except Exception as exc:
                error = str(exc)
                raw_prediction = f"<sample_error: {exc}>"
                predicted_int = None
                correct = False

            seq_total += 1
            total_count += 1
            if correct:
                seq_correct += 1
                total_correct += 1

            if evidence_count is None:
                evidence_count = -1
            if evidence_count not in evidence_stats:
                evidence_stats[evidence_count] = [0, 0]
            evidence_stats[evidence_count][1] += 1
            if correct:
                evidence_stats[evidence_count][0] += 1
            update_evidence_error_stats(
                evidence_error_stats=evidence_error_stats,
                evidence_count=evidence_count,
                gold_int=gold_int,
                predicted_int=predicted_int,
            )
            update_evidence_prediction_stats(
                evidence_prediction_stats=evidence_prediction_stats,
                evidence_count=evidence_count,
                predicted_int=predicted_int,
            )

            prediction_rows.append(
                {
                    "input_format": args.input_format,
                    "seq_len": int(sample_seq_len),
                    "evidence_count": int(evidence_count),
                    "sample_id": sample_id,
                    "sample_dir": str(sample_dir),
                    "gold_text": gold_text,
                    "gold_int": gold_int if gold_int is not None else "",
                    "predicted_int": predicted_int if predicted_int is not None else "",
                    "correct": int(correct),
                    "raw_prediction": raw_prediction,
                    "error": error,
                }
            )

            print(
                f"[seq_len={seq_len} {sample_index}/{len(sample_dirs)}] "
                f"sample_id={sample_id} "
                f"evidence_count={evidence_count} "
                f"gold={gold_display} "
                f"pred={predicted_int if predicted_int is not None else 'None'} "
                f"correct={int(correct)} "
                f"raw={raw_prediction!r}"
            )

        per_seq_totals[seq_len] = (seq_correct, seq_total)
        per_seq_evidence_totals[seq_len] = {
            evidence_count: (stats[0], stats[1])
            for evidence_count, stats in evidence_stats.items()
        }
        per_seq_evidence_error_sums[seq_len] = {
            evidence_count: {
                "sum_abs_error": float(stats["sum_abs_error"]),
                "sum_signed_error": float(stats["sum_signed_error"]),
                "n_valid": int(stats["n_valid"]),
            }
            for evidence_count, stats in evidence_error_stats.items()
        }
        per_seq_evidence_prediction_sums[seq_len] = {
            evidence_count: {
                "sum_prediction": float(stats["sum_prediction"]),
                "n_valid": int(stats["n_valid"]),
            }
            for evidence_count, stats in evidence_prediction_stats.items()
        }

        print(f"seq_len={seq_len}: accuracy = {base_eval.accuracy_string(seq_correct, seq_total)}")
        for line in bucket_metric_lines(
            seq_len=seq_len,
            evidence_stats=per_seq_evidence_totals[seq_len],
            evidence_error_stats=per_seq_evidence_error_sums[seq_len],
            evidence_prediction_stats=per_seq_evidence_prediction_sums[seq_len],
        ):
            print(line)
        print()

    if split_label == "all_uniform":
        print("All-uniform summed accuracy by seq_len:")
        for seq_len in sorted(per_seq_totals):
            seq_correct, seq_total = per_seq_totals[seq_len]
            print(f"seq_len={seq_len} all_uniform total: accuracy = {base_eval.accuracy_string(seq_correct, seq_total)}")
        print()

    print(f"Accuracy by seq_len and evidence_count for split={split_label}:")
    for seq_len in sorted(per_seq_evidence_totals):
        for line in bucket_metric_lines(
            seq_len=seq_len,
            evidence_stats=per_seq_evidence_totals[seq_len],
            evidence_error_stats=per_seq_evidence_error_sums.get(seq_len, {}),
            evidence_prediction_stats=per_seq_evidence_prediction_sums.get(seq_len, {}),
        ):
            print(line)
    print()

    metric_rows = build_metric_rows(
        per_seq_evidence_totals,
        per_seq_evidence_error_sums,
        per_seq_evidence_prediction_sums,
        input_format=args.input_format,
    )
    summary_rows = build_summary_rows(per_seq_totals, input_format=args.input_format)
    write_csv(output_dir / "predictions.csv", prediction_rows, PREDICTION_FIELDS)
    write_csv(output_dir / "metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    plot_paths = write_metric_plots(
        metric_rows,
        output_dir=output_dir,
        input_format=args.input_format,
        split_label=split_label,
    )

    print(f"overall: accuracy = {base_eval.accuracy_string(total_correct, total_count)}")
    print(f"predictions_csv={output_dir / 'predictions.csv'}")
    print(f"metrics_csv={output_dir / 'metrics.csv'}")
    print(f"summary_csv={output_dir / 'summary.csv'}")
    print("plot_files=" + ", ".join(str(path) for path in plot_paths))
    print(f"total_runtime: {base_eval.format_duration(time.time() - started_at)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

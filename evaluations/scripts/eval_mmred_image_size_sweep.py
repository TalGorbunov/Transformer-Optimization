#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import utils as eval_utils
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as baseline_eval
from models import model as model_utils
from models.model import DEFAULT_MODEL_ID, VALID_ATTENTION_IMPLEMENTATIONS

DEFAULT_IMAGE_SIZES: Tuple[int, ...] = (224, 280, 336, 448, 504)
PREDICTION_FIELDS: Tuple[str, ...] = (
    "seq_len",
    "evidence_count",
    "image_size",
    "split",
    "sample_id",
    "gold_answer",
    "gold_int",
    "raw_prediction",
    "predicted_int",
    "correct",
    "error",
    "num_frames",
    "image_token_count",
    "image_tokens_per_frame",
    "image_grid_thw",
    "grid_cell_count",
    "grid_cells_per_frame",
)
METRIC_FIELDS: Tuple[str, ...] = (
    "seq_len",
    "evidence_count",
    "image_size",
    "n_samples",
    "n_correct",
    "accuracy",
)
SUMMARY_FIELDS: Tuple[str, ...] = (
    "seq_len",
    "image_size",
    "n_samples",
    "n_correct",
    "accuracy",
    "mean_image_tokens_per_sample",
    "mean_image_tokens_per_frame",
    "mean_grid_cells_per_sample",
    "mean_grid_cells_per_frame",
)


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    sample_dir: Path
    frame_paths: Tuple[Path, ...]
    question: str
    gold_text: str
    gold_int: Optional[int]
    evidence_count: int


@dataclass
class BatchState:
    batch_enabled: bool = True
    warned: bool = False


@dataclass(frozen=True)
class Prediction:
    record: SampleRecord
    seq_len: int
    split: str
    image_size: int
    raw_prediction: str
    predicted_int: Optional[int]
    correct: bool
    error: str = ""
    image_token_count: Optional[int] = None
    image_grid_thw: str = ""
    grid_cell_count: Optional[int] = None


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/mmred_image_size_sweep") / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MMReD accuracy while explicitly resizing every frame to requested square image sizes."
        )
    )
    parser.add_argument(
        "--model",
        "--model-name",
        "--model-id",
        dest="model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Model name or path to load.",
    )
    parser.add_argument(
        "--mmred-dir",
        "--data-root",
        "--data_root",
        dest="mmred_dir",
        type=Path,
        default=baseline_eval.default_data_root(),
        help="Base rendered-image MMReD root containing seq_len_* directories.",
    )
    parser.add_argument(
        "--split-root",
        "--split_root",
        dest="split_root",
        type=Path,
        default=None,
        help="Optional HF DatasetDict root for resolving split membership when rendered samples are under all/.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all_uniform",
        help="Subset folder under each seq_len directory. Common values: all_uniform, all, train, val, test.",
    )
    parser.add_argument(
        "--seq-lens",
        "--seq_lens",
        dest="seq_lens",
        nargs="+",
        default=None,
        help="One or more seq_lens. Accepts either spaces or comma-separated values, e.g. --seq-lens 8 10 or --seq-lens 8,10.",
    )
    parser.add_argument(
        "--image-sizes",
        "--image_sizes",
        dest="image_sizes",
        nargs="+",
        default=None,
        help="Square resize targets in pixels, e.g. --image-sizes 224 280 336 448 504.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=default_output_dir(),
        help="Directory for CSVs, plots, config, and run log.",
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument(
        "--max-samples-per-bin",
        "--max_samples_per_bin",
        dest="max_samples_per_bin",
        type=int,
        default=None,
        help="Optional cap per (seq_len, evidence_count) bin before sweeping image sizes.",
    )
    parser.add_argument(
        "--limit-per-seq",
        "--limit_per_seq",
        dest="limit_per_seq",
        type=int,
        default=None,
        help="Optional cap on rendered sample dirs per seq_len before evidence-bin limiting.",
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
        "--load_in_4bit",
        dest="load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bitsandbytes 4-bit loading on CUDA. Use --no-load-in-4bit to disable.",
    )
    parser.add_argument(
        "--attn-implementation",
        "--attn_implementation",
        dest="attn_implementation",
        type=str,
        default="sdpa",
        choices=list(VALID_ATTENTION_IMPLEMENTATIONS),
        help="Attention backend for normal inference.",
    )
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=4)
    return parser.parse_args()


def parse_positive_ints(raw_values: Optional[Sequence[Any]], *, arg_name: str) -> Optional[List[int]]:
    if raw_values is None:
        return None
    values: List[int] = []
    for raw_value in raw_values:
        for part in str(raw_value).replace(",", " ").split():
            try:
                value = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid integer in {arg_name}: {part!r}") from exc
            if value <= 0:
                raise ValueError(f"{arg_name} values must be positive, got {value}")
            values.append(value)
    if not values:
        raise ValueError(f"{arg_name} must not be empty when provided")
    return sorted(dict.fromkeys(values))


def resolve_seq_lens(data_root: Path, raw_values: Optional[Sequence[Any]]) -> List[int]:
    explicit = parse_positive_ints(raw_values, arg_name="--seq-lens")
    if explicit is not None:
        return explicit
    return baseline_eval.resolve_seq_lens(data_root, None)


def resolve_image_sizes(raw_values: Optional[Sequence[Any]]) -> List[int]:
    explicit = parse_positive_ints(raw_values, arg_name="--image-sizes")
    return explicit if explicit is not None else list(DEFAULT_IMAGE_SIZES)


def setup_run_log(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (output_dir / "run.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)
    return log_handle


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_config(path: Path, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_sample_record(sample_dir: Path) -> SampleRecord:
    qa_path = sample_dir / "qa.txt"
    if not qa_path.is_file():
        raise FileNotFoundError(f"Missing qa.txt: {qa_path}")

    lines = qa_path.read_text(encoding="utf-8").splitlines()
    question_start = next((idx for idx, line in enumerate(lines) if line.strip() == "question:"), -1)
    answer_start = next((idx for idx, line in enumerate(lines) if line.strip() == "answer:"), -1)
    if question_start < 0 or answer_start <= question_start:
        raise RuntimeError(f"Bad qa.txt format: {qa_path}")

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

    frame_paths = tuple(sample_dir / f"{frame_idx:03d}.png" for frame_idx in range(len(states)))
    missing = [path for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frame image: {missing[0]}")

    evidence_count = len(eval_utils.collect_evidence_frame_indices(question_text, states))
    return SampleRecord(
        sample_id=sample_dir.name,
        sample_dir=sample_dir,
        frame_paths=frame_paths,
        question=question_text,
        gold_text=gold_text,
        gold_int=baseline_eval.extract_first_integer(gold_text),
        evidence_count=int(evidence_count),
    )


def cap_records_per_evidence_bin(
    records: Sequence[SampleRecord],
    *,
    max_samples_per_bin: Optional[int],
) -> List[SampleRecord]:
    if max_samples_per_bin is None:
        return list(records)
    if int(max_samples_per_bin) <= 0:
        raise ValueError("--max-samples-per-bin must be positive when provided")

    counts: Dict[int, int] = {}
    capped: List[SampleRecord] = []
    for record in records:
        evidence_count = int(record.evidence_count)
        seen = int(counts.get(evidence_count, 0))
        if seen >= int(max_samples_per_bin):
            continue
        capped.append(record)
        counts[evidence_count] = seen + 1
    return capped


def load_records_for_seq(
    *,
    data_root: Path,
    split_root: Optional[Path],
    seq_len: int,
    split: str,
    limit_per_seq: Optional[int],
    max_samples_per_bin: Optional[int],
) -> List[SampleRecord]:
    sample_dirs = baseline_eval.resolve_sample_dirs(
        data_root=data_root,
        seq_len=int(seq_len),
        split=split,
        split_root=split_root,
    )
    if limit_per_seq is not None:
        if int(limit_per_seq) <= 0:
            raise ValueError("--limit-per-seq must be positive when provided")
        sample_dirs = sample_dirs[: int(limit_per_seq)]

    records: List[SampleRecord] = []
    for sample_dir in sample_dirs:
        try:
            records.append(load_sample_record(sample_dir))
        except Exception as exc:
            print(f"[warn] skipped sample metadata load failure: sample_dir={sample_dir} error={exc}")
    records = cap_records_per_evidence_bin(records, max_samples_per_bin=max_samples_per_bin)
    if not records:
        raise RuntimeError(f"No valid samples found for seq_len={int(seq_len)} split={split!r}")
    return records


def resolve_device_and_dtype(requested_device: str, requested_dtype: str) -> Tuple[str, torch.dtype]:
    device = baseline_eval.resolve_device(requested_device)
    dtype = baseline_eval.resolve_dtype(requested_dtype, device)
    return device, dtype


def load_model_and_processor(
    *,
    model_name: str,
    device: str,
    dtype: torch.dtype,
    load_in_4bit: bool,
    attn_implementation: str,
) -> Tuple[Any, Any, bool]:
    use_4bit = bool(load_in_4bit and str(device).startswith("cuda"))
    device_map: Any = {"": device} if use_4bit else None
    runtime = model_utils.load_model_runtime(
        model_name,
        device_map=device_map,
        device=device if device_map is None else None,
        use_4bit=use_4bit,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        use_fast_processor=False,
    )
    if hasattr(runtime.processor, "tokenizer"):
        runtime.processor.tokenizer.padding_side = "left"
    generation_config = getattr(runtime.model, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.temperature = 1.0
    return runtime.model, runtime.processor, use_4bit


def resize_filter() -> int:
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def load_resized_frames(record: SampleRecord, image_size: int) -> List[Image.Image]:
    frames: List[Image.Image] = []
    target_size = (int(image_size), int(image_size))
    for frame_path in record.frame_paths:
        with Image.open(frame_path) as image:
            rgb_image = image.convert("RGB")
        if rgb_image.size != target_size:
            rgb_image = rgb_image.resize(target_size, resample=resize_filter())
        else:
            rgb_image = rgb_image.copy()
        frames.append(rgb_image)
    return frames


def build_messages(record: SampleRecord, frames: Sequence[Image.Image]) -> List[Dict[str, Any]]:
    prompt = baseline_eval.build_prompt(record.question, num_frames=len(frames))
    return [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": image} for image in frames]
                + [{"type": "text", "text": prompt}]
            ),
        }
    ]


def resolve_image_token_id(processor: Any) -> Optional[int]:
    for holder in (processor, getattr(processor, "tokenizer", None)):
        if holder is None:
            continue
        token_id = getattr(holder, "image_token_id", None)
        if token_id is not None:
            return int(token_id)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    except Exception:
        return None
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        return None
    return int(token_id)


def visual_token_counts(inputs: Dict[str, Any], processor: Any, batch_size: int) -> List[Optional[int]]:
    input_ids = inputs.get("input_ids")
    image_token_id = resolve_image_token_id(processor)
    if image_token_id is None or not torch.is_tensor(input_ids):
        return [None for _ in range(int(batch_size))]
    counts = (input_ids.detach().cpu() == int(image_token_id)).sum(dim=1)
    return [int(value.item()) for value in counts]


def visual_grid_summaries(inputs: Dict[str, Any], frame_counts: Sequence[int]) -> List[Tuple[str, Optional[int]]]:
    grid = inputs.get("image_grid_thw")
    if grid is None:
        grid = inputs.get("video_grid_thw")
    if not torch.is_tensor(grid):
        return [("", None) for _ in frame_counts]

    rows = [[int(value) for value in row] for row in grid.detach().cpu().tolist()]
    summaries: List[Tuple[str, Optional[int]]] = []
    offset = 0
    for frame_count in frame_counts:
        current = rows[offset : offset + int(frame_count)]
        offset += int(frame_count)
        if not current:
            summaries.append(("", None))
            continue
        cell_count = int(sum(math.prod(row) for row in current))
        summaries.append((json.dumps(current, separators=(",", ":")), cell_count))
    return summaries


def pad_token_id(processor: Any) -> int:
    token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if token_id is None:
        token_id = getattr(processor.tokenizer, "eos_token_id", None)
    if token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id for generation.")
    return int(token_id)


def _run_generation_batch(
    *,
    model: Any,
    processor: Any,
    records: Sequence[SampleRecord],
    seq_len: int,
    split: str,
    image_size: int,
    max_new_tokens: int,
    device: str,
) -> List[Prediction]:
    frames_by_record = [load_resized_frames(record, image_size) for record in records]
    conversations = [build_messages(record, frames) for record, frames in zip(records, frames_by_record)]

    if len(conversations) == 1:
        inputs = processor.apply_chat_template(
            conversations[0],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

    model_inputs = baseline_eval.move_inputs_to_device(dict(inputs), device)
    prompt_width = int(model_inputs["input_ids"].shape[-1])
    image_token_counts = visual_token_counts(model_inputs, processor, len(records))
    grid_summaries = visual_grid_summaries(
        model_inputs,
        [len(frames) for frames in frames_by_record],
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=pad_token_id(processor),
            use_cache=True,
        )

    generated_ids = output_ids[:, prompt_width:]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    predictions: List[Prediction] = []
    for idx, (record, raw_prediction) in enumerate(zip(records, decoded)):
        raw_prediction = str(raw_prediction).strip()
        predicted_int = baseline_eval.extract_first_integer(raw_prediction)
        correct = predicted_int is not None and record.gold_int is not None and int(predicted_int) == int(record.gold_int)
        grid_text, grid_count = grid_summaries[idx]
        predictions.append(
            Prediction(
                record=record,
                seq_len=int(seq_len),
                split=split,
                image_size=int(image_size),
                raw_prediction=raw_prediction,
                predicted_int=predicted_int,
                correct=bool(correct),
                image_token_count=image_token_counts[idx],
                image_grid_thw=grid_text,
                grid_cell_count=grid_count,
            )
        )
    return predictions


def error_prediction(
    *,
    record: SampleRecord,
    seq_len: int,
    split: str,
    image_size: int,
    exc: Exception,
) -> Prediction:
    return Prediction(
        record=record,
        seq_len=int(seq_len),
        split=split,
        image_size=int(image_size),
        raw_prediction="",
        predicted_int=None,
        correct=False,
        error=str(exc),
    )


def run_generation_batch(
    *,
    model: Any,
    processor: Any,
    records: Sequence[SampleRecord],
    seq_len: int,
    split: str,
    image_size: int,
    max_new_tokens: int,
    device: str,
    batch_state: BatchState,
) -> List[Prediction]:
    if not records:
        return []
    if len(records) > 1 and not batch_state.batch_enabled:
        predictions: List[Prediction] = []
        for record in records:
            predictions.extend(
                run_generation_batch(
                    model=model,
                    processor=processor,
                    records=[record],
                    seq_len=seq_len,
                    split=split,
                    image_size=image_size,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    batch_state=batch_state,
                )
            )
        return predictions

    try:
        return _run_generation_batch(
            model=model,
            processor=processor,
            records=records,
            seq_len=seq_len,
            split=split,
            image_size=image_size,
            max_new_tokens=max_new_tokens,
            device=device,
        )
    except Exception as exc:
        if len(records) == 1:
            return [
                error_prediction(
                    record=records[0],
                    seq_len=seq_len,
                    split=split,
                    image_size=image_size,
                    exc=exc,
                )
            ]

        batch_state.batch_enabled = False
        if not batch_state.warned:
            print(f"[warn] batched generation failed; falling back to batch_size=1 ({exc})")
            batch_state.warned = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        predictions = []
        for record in records:
            predictions.extend(
                run_generation_batch(
                    model=model,
                    processor=processor,
                    records=[record],
                    seq_len=seq_len,
                    split=split,
                    image_size=image_size,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    batch_state=batch_state,
                )
            )
        return predictions


def chunked(values: Sequence[SampleRecord], chunk_size: int) -> Iterable[Sequence[SampleRecord]]:
    for start in range(0, len(values), int(chunk_size)):
        yield values[start : start + int(chunk_size)]


def evaluate_combo(
    *,
    model: Any,
    processor: Any,
    records: Sequence[SampleRecord],
    seq_len: int,
    split: str,
    image_size: int,
    batch_size: int,
    max_new_tokens: int,
    device: str,
    batch_state: BatchState,
) -> List[Prediction]:
    combo_predictions: List[Prediction] = []
    total = len(records)
    for batch_index, record_batch in enumerate(chunked(records, max(1, int(batch_size))), start=1):
        predictions = run_generation_batch(
            model=model,
            processor=processor,
            records=record_batch,
            seq_len=seq_len,
            split=split,
            image_size=image_size,
            max_new_tokens=max_new_tokens,
            device=device,
            batch_state=batch_state,
        )
        combo_predictions.extend(predictions)
        done = min(len(combo_predictions), total)
        if batch_index == 1 or done == total or batch_index % 20 == 0:
            print(f"  progress seq_len={int(seq_len)} size={int(image_size)} {done}/{total}")
    return combo_predictions


def mean_optional(values: Iterable[Optional[float]]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def prediction_to_row(prediction: Prediction) -> Dict[str, Any]:
    num_frames = len(prediction.record.frame_paths)
    image_tokens_per_frame = (
        float(prediction.image_token_count) / float(num_frames)
        if prediction.image_token_count is not None and num_frames > 0
        else ""
    )
    grid_cells_per_frame = (
        float(prediction.grid_cell_count) / float(num_frames)
        if prediction.grid_cell_count is not None and num_frames > 0
        else ""
    )
    return {
        "seq_len": int(prediction.seq_len),
        "evidence_count": int(prediction.record.evidence_count),
        "image_size": int(prediction.image_size),
        "split": prediction.split,
        "sample_id": prediction.record.sample_id,
        "gold_answer": prediction.record.gold_text,
        "gold_int": "" if prediction.record.gold_int is None else int(prediction.record.gold_int),
        "raw_prediction": prediction.raw_prediction,
        "predicted_int": "" if prediction.predicted_int is None else int(prediction.predicted_int),
        "correct": int(bool(prediction.correct)),
        "error": prediction.error,
        "num_frames": int(num_frames),
        "image_token_count": "" if prediction.image_token_count is None else int(prediction.image_token_count),
        "image_tokens_per_frame": image_tokens_per_frame,
        "image_grid_thw": prediction.image_grid_thw,
        "grid_cell_count": "" if prediction.grid_cell_count is None else int(prediction.grid_cell_count),
        "grid_cells_per_frame": grid_cells_per_frame,
    }


def build_metrics(predictions: Sequence[Prediction]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int, int], List[Prediction]] = {}
    for prediction in predictions:
        key = (int(prediction.seq_len), int(prediction.record.evidence_count), int(prediction.image_size))
        grouped.setdefault(key, []).append(prediction)

    rows: List[Dict[str, Any]] = []
    for seq_len, evidence_count, image_size in sorted(grouped):
        items = grouped[(seq_len, evidence_count, image_size)]
        n_samples = len(items)
        n_correct = sum(1 for item in items if item.correct)
        rows.append(
            {
                "seq_len": int(seq_len),
                "evidence_count": int(evidence_count),
                "image_size": int(image_size),
                "n_samples": int(n_samples),
                "n_correct": int(n_correct),
                "accuracy": float(n_correct / n_samples) if n_samples else float("nan"),
            }
        )
    return rows


def build_summary(predictions: Sequence[Prediction]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int], List[Prediction]] = {}
    for prediction in predictions:
        key = (int(prediction.seq_len), int(prediction.image_size))
        grouped.setdefault(key, []).append(prediction)

    rows: List[Dict[str, Any]] = []
    for seq_len, image_size in sorted(grouped):
        items = grouped[(seq_len, image_size)]
        n_samples = len(items)
        n_correct = sum(1 for item in items if item.correct)
        rows.append(
            {
                "seq_len": int(seq_len),
                "image_size": int(image_size),
                "n_samples": int(n_samples),
                "n_correct": int(n_correct),
                "accuracy": float(n_correct / n_samples) if n_samples else float("nan"),
                "mean_image_tokens_per_sample": mean_optional(item.image_token_count for item in items),
                "mean_image_tokens_per_frame": mean_optional(
                    (
                        float(item.image_token_count) / float(len(item.record.frame_paths))
                        if item.image_token_count is not None and len(item.record.frame_paths) > 0
                        else None
                    )
                    for item in items
                ),
                "mean_grid_cells_per_sample": mean_optional(item.grid_cell_count for item in items),
                "mean_grid_cells_per_frame": mean_optional(
                    (
                        float(item.grid_cell_count) / float(len(item.record.frame_paths))
                        if item.grid_cell_count is not None and len(item.record.frame_paths) > 0
                        else None
                    )
                    for item in items
                ),
            }
        )
    return rows


def accuracy_percent(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{100.0 * float(value):.1f}%"


def heatmap_matrix(
    metric_rows: Sequence[Dict[str, Any]],
    *,
    image_size: int,
) -> Tuple[List[int], List[int], List[List[float]]]:
    rows_for_size = [row for row in metric_rows if int(row["image_size"]) == int(image_size)]
    seq_lens = sorted({int(row["seq_len"]) for row in rows_for_size})
    evidence_counts = sorted({int(row["evidence_count"]) for row in rows_for_size})
    lookup = {
        (int(row["seq_len"]), int(row["evidence_count"])): float(row["accuracy"])
        for row in rows_for_size
    }
    matrix = [
        [lookup.get((seq_len, evidence_count), float("nan")) for evidence_count in evidence_counts]
        for seq_len in seq_lens
    ]
    return seq_lens, evidence_counts, matrix


def draw_heatmap_axis(ax: Any, metric_rows: Sequence[Dict[str, Any]], image_size: int) -> Any:
    import numpy as np

    seq_lens, evidence_counts, matrix = heatmap_matrix(metric_rows, image_size=image_size)
    if not seq_lens or not evidence_counts:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"size={int(image_size)}")
        ax.set_axis_off()
        return None

    array = np.array(matrix, dtype=float)
    masked = np.ma.masked_invalid(array)
    image = ax.imshow(masked, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(f"size={int(image_size)}")
    ax.set_xlabel("Evidence count")
    ax.set_ylabel("Seq len")
    ax.set_xticks(range(len(evidence_counts)))
    ax.set_xticklabels([str(value) for value in evidence_counts])
    ax.set_yticks(range(len(seq_lens)))
    ax.set_yticklabels([str(value) for value in seq_lens])
    for row_idx, seq_len in enumerate(seq_lens):
        for col_idx, evidence_count in enumerate(evidence_counts):
            value = array[row_idx, col_idx]
            if math.isfinite(float(value)):
                color = "white" if float(value) < 0.55 else "black"
                ax.text(col_idx, row_idx, accuracy_percent(value), ha="center", va="center", fontsize=8, color=color)
    return image


def plot_heatmap(metric_rows: Sequence[Dict[str, Any]], *, image_size: int, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    image = draw_heatmap_axis(ax, metric_rows, int(image_size))
    if image is not None:
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_combined_heatmaps(
    metric_rows: Sequence[Dict[str, Any]],
    *,
    image_sizes: Sequence[int],
    output_path: Path,
) -> Optional[Path]:
    if not image_sizes:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = min(3, len(image_sizes))
    nrows = int(math.ceil(len(image_sizes) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.4 * ncols, 4.2 * nrows), dpi=150)
    if not isinstance(axes, (list, tuple)):
        try:
            axes_list = list(axes.ravel())
        except AttributeError:
            axes_list = [axes]
    else:
        axes_list = list(axes)

    last_image = None
    for ax, image_size in zip(axes_list, image_sizes):
        image = draw_heatmap_axis(ax, metric_rows, int(image_size))
        if image is not None:
            last_image = image
    for ax in axes_list[len(image_sizes) :]:
        ax.set_axis_off()
    if last_image is not None:
        fig.colorbar(last_image, ax=axes_list, fraction=0.025, pad=0.02, label="Accuracy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_configuration(
    *,
    args: argparse.Namespace,
    data_root: Path,
    split_root: Optional[Path],
    seq_lens: Sequence[int],
    image_sizes: Sequence[int],
    device: str,
    dtype: torch.dtype,
    use_4bit: bool,
    records_by_seq: Dict[int, Sequence[SampleRecord]],
) -> None:
    print(f"model={args.model}")
    print(f"mmred_dir={data_root}")
    print(f"split={args.split}")
    print(f"split_root={split_root if split_root is not None else '<none>'}")
    print(f"output_dir={args.output_dir}")
    print(f"seq_lens={list(seq_lens)}")
    print(f"image_sizes={list(image_sizes)}")
    print(f"batch_size={int(args.batch_size)}")
    print(f"max_samples_per_bin={args.max_samples_per_bin}")
    print(f"limit_per_seq={args.limit_per_seq}")
    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"use_4bit={use_4bit}")
    print(f"attn_implementation={args.attn_implementation}")
    for seq_len in seq_lens:
        evidence_counts: Dict[int, int] = {}
        for record in records_by_seq[int(seq_len)]:
            evidence_counts[int(record.evidence_count)] = evidence_counts.get(int(record.evidence_count), 0) + 1
        print(
            f"queued seq_len={int(seq_len)} samples={len(records_by_seq[int(seq_len)])} "
            f"evidence_bins={dict(sorted(evidence_counts.items()))}"
        )


def main() -> int:
    start_time = time.perf_counter()
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_handle = setup_run_log(output_dir)

    try:
        data_root = Path(args.mmred_dir).resolve()
        split_root = Path(args.split_root).resolve() if args.split_root is not None else baseline_eval.infer_split_root(data_root)
        seq_lens = resolve_seq_lens(data_root, args.seq_lens)
        image_sizes = resolve_image_sizes(args.image_sizes)
        if int(args.batch_size) <= 0:
            raise ValueError("--batch-size must be positive")
        if int(args.max_new_tokens) <= 0:
            raise ValueError("--max-new-tokens must be positive")

        records_by_seq = {
            int(seq_len): load_records_for_seq(
                data_root=data_root,
                split_root=split_root,
                seq_len=int(seq_len),
                split=str(args.split),
                limit_per_seq=args.limit_per_seq,
                max_samples_per_bin=args.max_samples_per_bin,
            )
            for seq_len in seq_lens
        }

        device, dtype = resolve_device_and_dtype(args.device, args.dtype)
        load_started = time.perf_counter()
        model, processor, use_4bit = load_model_and_processor(
            model_name=str(args.model),
            device=device,
            dtype=dtype,
            load_in_4bit=bool(args.load_in_4bit),
            attn_implementation=str(args.attn_implementation),
        )
        print(f"model_load_time_sec={time.perf_counter() - load_started:.2f}")

        write_config(
            output_dir / "config.json",
            {
                "model": str(args.model),
                "mmred_dir": str(data_root),
                "split_root": None if split_root is None else str(split_root),
                "split": str(args.split),
                "seq_lens": [int(value) for value in seq_lens],
                "image_sizes": [int(value) for value in image_sizes],
                "output_dir": str(output_dir),
                "batch_size": int(args.batch_size),
                "max_samples_per_bin": args.max_samples_per_bin,
                "limit_per_seq": args.limit_per_seq,
                "device": str(device),
                "dtype": str(dtype),
                "load_in_4bit": bool(args.load_in_4bit),
                "use_4bit": bool(use_4bit),
                "attn_implementation": str(args.attn_implementation),
                "max_new_tokens": int(args.max_new_tokens),
                "sample_counts_by_seq": {
                    str(seq_len): len(records_by_seq[int(seq_len)]) for seq_len in seq_lens
                },
            },
        )

        print_configuration(
            args=args,
            data_root=data_root,
            split_root=split_root,
            seq_lens=seq_lens,
            image_sizes=image_sizes,
            device=device,
            dtype=dtype,
            use_4bit=use_4bit,
            records_by_seq=records_by_seq,
        )

        predictions: List[Prediction] = []
        batch_state = BatchState()
        for image_size in image_sizes:
            for seq_len in seq_lens:
                records = records_by_seq[int(seq_len)]
                print(
                    f"Evaluating seq_len={int(seq_len)} image_size={int(image_size)} "
                    f"split={args.split} samples={len(records)}"
                )
                combo_predictions = evaluate_combo(
                    model=model,
                    processor=processor,
                    records=records,
                    seq_len=int(seq_len),
                    split=str(args.split),
                    image_size=int(image_size),
                    batch_size=int(args.batch_size),
                    max_new_tokens=int(args.max_new_tokens),
                    device=device,
                    batch_state=batch_state,
                )
                predictions.extend(combo_predictions)
                n_correct = sum(1 for prediction in combo_predictions if prediction.correct)
                n_total = len(combo_predictions)
                print(
                    f"done seq_len={int(seq_len)} image_size={int(image_size)} "
                    f"accuracy={n_correct}/{n_total} ({100.0 * n_correct / max(1, n_total):.2f}%)"
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        prediction_rows = [prediction_to_row(prediction) for prediction in predictions]
        metric_rows = build_metrics(predictions)
        summary_rows = build_summary(predictions)

        write_csv(output_dir / "predictions.csv", prediction_rows, PREDICTION_FIELDS)
        write_csv(output_dir / "metrics.csv", metric_rows, METRIC_FIELDS)
        write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)

        plot_paths: List[Path] = []
        for image_size in image_sizes:
            plot_paths.append(
                plot_heatmap(
                    metric_rows,
                    image_size=int(image_size),
                    output_path=output_dir / f"heatmap_size_{int(image_size)}.png",
                )
            )
        combined_path = plot_combined_heatmaps(
            metric_rows,
            image_sizes=image_sizes,
            output_path=output_dir / "heatmaps_combined.png",
        )
        if combined_path is not None:
            plot_paths.append(combined_path)

        print("summary:")
        for row in summary_rows:
            print(
                f"  seq_len={row['seq_len']} image_size={row['image_size']} "
                f"accuracy={row['n_correct']}/{row['n_samples']} ({100.0 * float(row['accuracy']):.2f}%) "
                f"mean_image_tokens_per_frame={row['mean_image_tokens_per_frame']}"
            )
        print(f"predictions_csv={output_dir / 'predictions.csv'}")
        print(f"metrics_csv={output_dir / 'metrics.csv'}")
        print(f"summary_csv={output_dir / 'summary.csv'}")
        print(f"config_json={output_dir / 'config.json'}")
        print("plot_files=" + ", ".join(str(path) for path in plot_paths))
        print(eval_utils.format_runtime(time.perf_counter() - start_time))
        return 0
    finally:
        try:
            log_handle.flush()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

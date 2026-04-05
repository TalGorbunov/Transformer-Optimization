#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from datasets import load_from_disk
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SEQ_LENS: Sequence[int] = (2, 4)
INTEGER_RE = re.compile(r"[+-]?\d+")


@dataclass
class Sample:
    sample_id: str
    sample_dir: Path
    frames: List[Image.Image]
    question: str
    gold_text: str
    gold_int: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen 2.5 VL 7B Instruct on MMReD-style seq_len=2 and seq_len=4 data."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Base rendered-image root containing seq_len_2/ and seq_len_4/ directories.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help=(
            "Optional HF DatasetDict root used to resolve train/val/test membership when the image root "
            "stores samples under all/. If omitted, the script will try to infer it from --data-root."
        ),
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
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
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--limit-per-seq",
        type=int,
        default=None,
        help="Optional cap on the number of evaluated samples per seq_len.",
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    return parser.parse_args()


def default_data_root() -> Path:
    for candidate in (
        Path("data/mmred_images_generated"),
        Path("data/mmred_images"),
    ):
        if candidate.exists():
            return candidate
    return Path("data/mmred_images_generated")


def resolve_device(requested: str) -> str:
    requested = str(requested).strip()
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, resolved_device: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(requested).strip().lower()
    if key != "auto":
        return mapping[key]

    if resolved_device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if resolved_device == "mps":
        return torch.float16
    return torch.float32


def infer_split_root(data_root: Path) -> Optional[Path]:
    name = data_root.name
    if "_images" in name:
        candidate = data_root.with_name(name.replace("_images", ""))
        if candidate.exists():
            return candidate
    if "images" in name:
        candidate = data_root.with_name(name.replace("images", "").replace("__", "_").rstrip("_"))
        if candidate.exists():
            return candidate
    return None


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def iter_sample_dirs(sample_root: Path) -> List[Path]:
    return sorted(
        path for path in sample_root.iterdir()
        if path.is_dir() and (path / "qa.txt").exists()
    )


def choose_split_ids(dataset_split: Any, available_ids: set[str]) -> List[str]:
    column_names = set(dataset_split.column_names)
    candidates = []
    for column in ("sample_id", "qid"):
        if column not in column_names:
            continue
        ids = unique_preserve_order(str(value) for value in dataset_split[column])
        missing = [sample_id for sample_id in ids if sample_id not in available_ids]
        candidates.append((len(missing), column, ids, missing))

    if not candidates:
        raise KeyError(
            f"Could not resolve sample ids from split columns {sorted(column_names)}; expected one of ['sample_id', 'qid']."
        )

    candidates.sort(key=lambda item: item[0])
    missing_count, column, ids, missing = candidates[0]
    if missing_count:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Split metadata column {column!r} did not match rendered sample dirs. Missing {missing_count} ids; "
            f"first few: {preview}"
        )
    return ids


def resolve_sample_dirs(
    data_root: Path,
    seq_len: int,
    split: str,
    split_root: Optional[Path],
) -> List[Path]:
    seq_root = data_root / f"seq_len_{seq_len}"
    if not seq_root.is_dir():
        raise FileNotFoundError(f"Missing seq_len directory: {seq_root}")

    direct_split_dir = seq_root / split
    if split != "all" and direct_split_dir.is_dir():
        return iter_sample_dirs(direct_split_dir)

    all_dir = seq_root / "all"
    if split == "all":
        if all_dir.is_dir():
            return iter_sample_dirs(all_dir)
        combined: List[Path] = []
        for split_name in ("train", "val", "test"):
            split_dir = seq_root / split_name
            if split_dir.is_dir():
                combined.extend(iter_sample_dirs(split_dir))
        return combined

    if not all_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find either {direct_split_dir} or {all_dir} for seq_len={seq_len} split={split}."
        )
    if split_root is None:
        raise FileNotFoundError(
            f"Need split metadata to resolve split={split!r} under {all_dir}. "
            "Pass --split-root or use an image root with explicit split folders."
        )

    split_dataset_root = split_root / f"seq_len_{seq_len}"
    if not split_dataset_root.is_dir():
        raise FileNotFoundError(f"Missing split metadata directory: {split_dataset_root}")
    dataset_dict = load_from_disk(str(split_dataset_root))
    if split not in dataset_dict:
        raise KeyError(f"Split {split!r} not found in {split_dataset_root}")

    available_ids = {path.name for path in iter_sample_dirs(all_dir)}
    ordered_ids = choose_split_ids(dataset_dict[split], available_ids)
    sample_dirs = [all_dir / sample_id for sample_id in ordered_ids]
    return sample_dirs


def extract_first_integer(text: Any) -> Optional[int]:
    text = str(text).strip()
    if not text:
        return None
    match = INTEGER_RE.search(text)
    if match is None:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def load_mmred_style_sample(sample_dir: Path) -> Sample:
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

    frames: List[Image.Image] = []
    for frame_idx in range(len(states)):
        frame_path = sample_dir / f"{frame_idx:03d}.png"
        if not frame_path.is_file():
            raise FileNotFoundError(f"Missing frame image: {frame_path}")
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))

    return Sample(
        sample_id=sample_dir.name,
        sample_dir=sample_dir,
        frames=frames,
        question=question_text,
        gold_text=gold_text,
        gold_int=extract_first_integer(gold_text),
    )


def build_prompt(question: str, num_frames: int) -> str:
    return (
        f"You will be shown {num_frames} frames describing steps in a house.\n"
        f"Respond with a single integer from 0 to {num_frames} (0 is allowed). Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )


def move_inputs_to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def load_model_and_processor(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    load_in_4bit: bool,
):
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device.startswith("cuda"):
        model_kwargs["attn_implementation"] = "sdpa"

    if load_in_4bit:
        if not device.startswith("cuda"):
            raise ValueError("--load-in-4bit requires a CUDA device.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map={"": device},
            **model_kwargs,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            **model_kwargs,
        )
        model.to(device)

    model.eval()
    return model, processor


def run_clean_generation(
    model: Any,
    processor: Any,
    sample: Sample,
    max_new_tokens: int,
    device: str,
) -> str:
    prompt = build_prompt(sample.question, num_frames=len(sample.frames))
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
    inputs = move_inputs_to_device(dict(inputs), device)
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


def accuracy_string(correct: int, total: int) -> str:
    percent = 0.0 if total == 0 else (100.0 * float(correct) / float(total))
    return f"{correct} / {total} = {percent:.2f}%"


def format_duration(seconds: float) -> str:
    total_seconds = int(round(float(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> int:
    started_at = time.time()
    args = parse_args()
    data_root = args.data_root.resolve()
    split_root = args.split_root.resolve() if args.split_root is not None else infer_split_root(data_root)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"model: {args.model_id}")
    print(f"data_root: {data_root}")
    print(f"split: {args.split}")
    print(f"seq_lens: {list(SEQ_LENS)}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"load_in_4bit: {args.load_in_4bit}")
    if split_root is not None:
        print(f"split_root: {split_root}")
    elif args.split != "all":
        print("split_root: <none>")
    print()

    load_started = time.time()
    model, processor = load_model_and_processor(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )
    print(f"model_load_time_sec: {time.time() - load_started:.2f}")
    print()

    total_correct = 0
    total_count = 0

    for seq_len in SEQ_LENS:
        sample_dirs = resolve_sample_dirs(
            data_root=data_root,
            seq_len=seq_len,
            split=args.split,
            split_root=split_root,
        )
        if args.limit_per_seq is not None:
            if args.limit_per_seq < 0:
                raise ValueError("--limit-per-seq must be >= 0 when provided.")
            sample_dirs = sample_dirs[: args.limit_per_seq]
        if not sample_dirs:
            raise RuntimeError(f"No samples found for seq_len={seq_len} split={args.split!r}.")

        seq_correct = 0
        seq_total = 0
        print(f"Evaluating seq_len={seq_len} on {len(sample_dirs)} samples")

        for sample_index, sample_dir in enumerate(sample_dirs, start=1):
            sample_id = sample_dir.name
            gold_display = "<unavailable>"
            raw_prediction = ""
            predicted_int: Optional[int] = None
            correct = False

            try:
                sample = load_mmred_style_sample(sample_dir)
                sample_id = sample.sample_id
                gold_display = sample.gold_int if sample.gold_int is not None else repr(sample.gold_text)
                raw_prediction = run_clean_generation(
                    model=model,
                    processor=processor,
                    sample=sample,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                predicted_int = extract_first_integer(raw_prediction)
                correct = predicted_int is not None and sample.gold_int is not None and predicted_int == sample.gold_int
            except Exception as exc:
                raw_prediction = f"<sample_error: {exc}>"
                predicted_int = None
                correct = False

            seq_total += 1
            total_count += 1
            if correct:
                seq_correct += 1
                total_correct += 1

            print(
                f"[seq_len={seq_len} {sample_index}/{len(sample_dirs)}] "
                f"sample_id={sample_id} "
                f"gold={gold_display} "
                f"pred={predicted_int if predicted_int is not None else 'None'} "
                f"correct={int(correct)} "
                f"raw={raw_prediction!r}"
            )

        print(f"seq_len={seq_len}: accuracy = {accuracy_string(seq_correct, seq_total)}")
        print()

    print(f"overall: accuracy = {accuracy_string(total_correct, total_count)}")
    print(f"total_runtime: {format_duration(time.time() - started_at)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

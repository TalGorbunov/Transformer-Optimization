"""
Standalone MMReD evidence-count analysis.

This experiment is not layer-based. It groups results by the exact number of
evidence frames and measures clamped GT-answer drops on clean-top1-correct
samples and absolute GT-answer score differences on all samples.

The implementation intentionally reuses the current MMReD runtime, prompt/input
construction, answer-token extraction, corrupted-sample traversal, and score
computation utilities so the scoring rule remains unchanged.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from models import model as model_utils
from models.model import DEFAULT_MODEL_ID, load_model_runtime

SEQ_LEN_DIR_RE = re.compile(r"^seq_len_(\d+)$")
EXACT_DIR_RE = re.compile(r"^exact_(\d+)$")

PER_FRAME_FIELDS = [
    "seq_len",
    "evidence_count",
    "split",
    "sample_id",
    "frame_idx",
    "is_evidence",
    "gold_answer",
    "clean_top1_answer",
    "clean_top1_is_gold",
    "clean_score_gt",
    "single_corrupt_score_gt",
    "d_i_gt_clamped_influence",
    "d_i_gt_abs_influence",
]

PER_SAMPLE_FIELDS = [
    "seq_len",
    "evidence_count",
    "split",
    "sample_id",
    "gold_answer",
    "clean_top1_answer",
    "clean_top1_is_gold",
    "num_evidence_frames",
    "per_frame_avg_gt_clamped_influence_clean_top1_correct",
    "per_frame_avg_gt_abs_influence_all_samples",
]

AGGREGATE_BY_COUNT_FIELDS = [
    "metric",
    "seq_len",
    "evidence_count",
    "mean",
    "median",
    "std",
    "n_samples",
    "n_frames",
    "mean_ci_low",
    "mean_ci_high",
    "median_ci_low",
    "median_ci_high",
]

AGGREGATE_BY_FRAME_INDEX_FIELDS = [
    "seq_len",
    "frame_idx",
    "mean_d_i_gt_clamped_influence_clean_top1_correct",
    "median_d_i_gt_clamped_influence_clean_top1_correct",
    "std_d_i_gt_clamped_influence_clean_top1_correct",
    "n",
    "mean_ci_low",
    "mean_ci_high",
    "median_ci_low",
    "median_ci_high",
]

METRIC_ORDER = (
    "per_frame_avg_gt_clamped_influence_clean_top1_correct",
    "per_frame_avg_gt_abs_influence_all_samples",
)

MULTIMODAL_BATCH_KEYS = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}


@dataclass(frozen=True)
class ExactBucketSpec:
    seq_len: int
    split: str
    evidence_count: int
    clean_dir: Path
    corrupted_dir: Path
    sample_dirs: Tuple[Path, ...]


def default_data_root() -> Path:
    for candidate in (
        Path("data/mmred_new_images"),
        Path("data/mmred_images"),
        Path("data/mmred_images_generated"),
    ):
        if candidate.exists():
            return candidate
    return Path("data/mmred_images")


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/evidence_count_ablation") / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone MMReD evidence-count clean-ablation analysis across seq_len/*/by_evidence_count/exact_j "
            "directories and write detailed CSVs, aggregate CSVs, and selected plots."
        )
    )
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model_name",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Model name to load.",
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        type=Path,
        default=default_data_root(),
        help="Base image root containing seq_len_* directories.",
    )
    parser.add_argument(
        "--corrupted-root",
        "--corrupted_root",
        dest="corrupted_root",
        type=Path,
        default=None,
        help="Base corrupted root mirroring the clean seq_len/by_evidence_count layout.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=default_output_dir(),
        help="Run output directory.",
    )
    parser.add_argument(
        "--seq-lens",
        "--seq_lens",
        dest="seq_lens",
        type=str,
        default=None,
        help="Optional comma-separated seq_len list. Defaults to all discovered seq_len_* directories.",
    )
    parser.add_argument(
        "--max-samples-per-exact-dir",
        "--max_samples_per_exact_dir",
        dest="max_samples_per_exact_dir",
        type=int,
        default=None,
        help="Optional cap on the number of samples processed from each exact_j directory.",
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
        dest="load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 4-bit loading on CUDA by default.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        "--bootstrap_samples",
        dest="bootstrap_samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--aggregate-every",
        "--aggregate_every",
        dest="aggregate_every",
        type=int,
        default=None,
        help="Optional cadence for rewriting aggregate CSVs every N processed samples. Defaults to end-only.",
    )
    parser.add_argument(
        "--score-batch-size",
        "--score_batch_size",
        dest="score_batch_size",
        type=int,
        default=4,
        help="Micro-batch size for within-sample corrupted GT scoring.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.max_samples_per_exact_dir is not None and int(args.max_samples_per_exact_dir) <= 0:
        raise ValueError("--max-samples-per-exact-dir must be positive when provided")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.aggregate_every is not None and int(args.aggregate_every) <= 0:
        raise ValueError("--aggregate-every must be positive when provided")
    if int(args.score_batch_size) <= 0:
        raise ValueError("--score-batch-size must be positive")
    return args


def parse_seq_lens_arg(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or not str(raw).strip():
        return None
    seq_lens: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seq_lens.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in --seq-lens: {part!r}") from exc
    if not seq_lens:
        raise ValueError("--seq-lens must not be empty when provided")
    return sorted(set(seq_lens))


def discover_seq_lens(data_root: Path) -> List[int]:
    discovered: List[int] = []
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        match = SEQ_LEN_DIR_RE.fullmatch(child.name)
        if match is None:
            continue
        discovered.append(int(match.group(1)))
    if not discovered:
        raise FileNotFoundError(f"No seq_len_* directories found under {data_root}")
    return discovered


def resolve_seq_lens(data_root: Path, raw: Optional[str]) -> List[int]:
    explicit = parse_seq_lens_arg(raw)
    if explicit is not None:
        return explicit
    return discover_seq_lens(data_root)


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


def infer_split_label_from_seq_dir(seq_dir: Path) -> str:
    if (seq_dir / "all_uniform").is_dir():
        return "all_uniform"
    if (seq_dir / "all").is_dir():
        return "all"
    return "by_evidence_count"


def discover_bucket_specs(
    *,
    data_root: Path,
    corrupted_root: Path,
    seq_lens: Sequence[int],
    max_samples_per_exact_dir: Optional[int],
) -> Tuple[List[ExactBucketSpec], Dict[Tuple[int, str], int]]:
    bucket_specs: List[ExactBucketSpec] = []
    exact_zero_counts: Dict[Tuple[int, str], int] = {}

    for seq_len in seq_lens:
        clean_seq_dir = data_root / f"seq_len_{int(seq_len)}"
        corrupted_seq_dir = corrupted_root / f"seq_len_{int(seq_len)}"
        if not clean_seq_dir.is_dir():
            print(f"[warn] missing clean seq_len directory: {clean_seq_dir}")
            continue

        bucket_roots: List[Tuple[str, Path, Path]] = []
        direct_root = clean_seq_dir / "by_evidence_count"
        if direct_root.is_dir():
            bucket_roots.append(
                (
                    infer_split_label_from_seq_dir(clean_seq_dir),
                    direct_root,
                    Path("by_evidence_count"),
                )
            )

        for child in sorted(clean_seq_dir.iterdir()):
            if not child.is_dir() or child.name == "by_evidence_count":
                continue
            nested_root = child / "by_evidence_count"
            if nested_root.is_dir():
                bucket_roots.append(
                    (
                        str(child.name),
                        nested_root,
                        Path(child.name) / "by_evidence_count",
                    )
                )

        if not bucket_roots:
            print(
                f"[warn] seq_len={int(seq_len)} has no by_evidence_count directory under {clean_seq_dir}"
            )
            continue

        for split_label, clean_bucket_root, relative_bucket_root in bucket_roots:
            corrupted_bucket_root = corrupted_seq_dir / relative_bucket_root
            for exact_dir in sorted(clean_bucket_root.iterdir()):
                if not exact_dir.is_dir():
                    continue
                match = EXACT_DIR_RE.fullmatch(exact_dir.name)
                if match is None:
                    continue
                evidence_count = int(match.group(1))
                sample_dirs = tuple(eval_utils.iter_sample_dirs(exact_dir))
                if max_samples_per_exact_dir is not None:
                    sample_dirs = sample_dirs[: int(max_samples_per_exact_dir)]
                if evidence_count == 0:
                    exact_zero_counts[(int(seq_len), str(split_label))] = (
                        int(exact_zero_counts.get((int(seq_len), str(split_label)), 0)) + int(len(sample_dirs))
                    )
                    continue
                bucket_specs.append(
                    ExactBucketSpec(
                        seq_len=int(seq_len),
                        split=str(split_label),
                        evidence_count=int(evidence_count),
                        clean_dir=exact_dir,
                        corrupted_dir=corrupted_bucket_root / exact_dir.name,
                        sample_dirs=sample_dirs,
                    )
                )

    bucket_specs.sort(
        key=lambda spec: (
            int(spec.seq_len),
            int(spec.evidence_count),
            str(spec.split),
            str(spec.clean_dir),
        )
    )
    return bucket_specs, exact_zero_counts


def row_for_fieldnames(row: Dict[str, Any], fieldnames: Sequence[str]) -> Dict[str, Any]:
    return {field: row.get(field) for field in fieldnames}


def initialize_csv(path: Path, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()


def append_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        for row in rows:
            writer.writerow(row_for_fieldnames(row, fieldnames))


def write_csv_atomic(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row_for_fieldnames(row, fieldnames))
    tmp_path.replace(path)


def mean_value(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def median_value(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def bootstrap_stat_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float],
    n_bootstrap: int,
    rng: random.Random,
) -> Tuple[float, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        nan = float("nan")
        return nan, nan
    if len(finite_values) == 1:
        center = float(finite_values[0])
        return center, center

    boot_values: List[float] = []
    sample_count = len(finite_values)
    for _ in range(int(n_bootstrap)):
        sample = [finite_values[rng.randrange(sample_count)] for _ in range(sample_count)]
        boot_values.append(float(statistic(sample)))
    boot_values.sort()
    lo_idx = int(0.025 * (int(n_bootstrap) - 1))
    hi_idx = int(0.975 * (int(n_bootstrap) - 1))
    return float(boot_values[lo_idx]), float(boot_values[hi_idx])


def summarize_values(
    values: Sequence[float],
    *,
    n_bootstrap: int,
    rng: random.Random,
) -> Dict[str, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        nan = float("nan")
        return {
            "mean": nan,
            "median": nan,
            "std": nan,
            "mean_ci_low": nan,
            "mean_ci_high": nan,
            "median_ci_low": nan,
            "median_ci_high": nan,
        }
    mean = mean_value(finite_values)
    median = median_value(finite_values)
    std_value = float(statistics.stdev(finite_values)) if len(finite_values) > 1 else 0.0
    mean_ci_low, mean_ci_high = bootstrap_stat_ci(
        finite_values,
        statistic=mean_value,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
    median_ci_low, median_ci_high = bootstrap_stat_ci(
        finite_values,
        statistic=median_value,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
    return {
        "mean": mean,
        "median": median,
        "std": std_value,
        "mean_ci_low": float(mean_ci_low),
        "mean_ci_high": float(mean_ci_high),
        "median_ci_low": float(median_ci_low),
        "median_ci_high": float(median_ci_high),
    }


def build_runtime(
    *,
    model_name: str,
    requested_device: str,
    requested_dtype: str,
    load_in_4bit: bool,
) -> Tuple[Any, str, torch.dtype, bool]:
    resolved_device = resolve_device(requested_device)
    resolved_dtype = resolve_dtype(requested_dtype, resolved_device)
    use_4bit = bool(load_in_4bit and resolved_device.startswith("cuda"))
    device_map: Any = {"": resolved_device} if use_4bit else None
    runtime = load_model_runtime(
        model_name,
        device_map=device_map,
        device=resolved_device if device_map is None else None,
        use_4bit=use_4bit,
        torch_dtype=resolved_dtype,
    )
    model_utils._DEFAULT_RUNTIME = runtime
    return runtime, resolved_device, resolved_dtype, use_4bit


def build_inputs_for_sample(frames: Sequence[Any], question: str, runtime: Any) -> Tuple[Dict[str, Any], int]:
    inputs = tgi.build_inputs(frames, question, processor=runtime.processor)
    model_inputs = model_utils.move_inputs_to_model_device(inputs, model_obj=runtime.model)
    prompt_len = int(model_inputs["input_ids"].shape[1])
    if prompt_len <= 0:
        raise RuntimeError("Prompt length must be positive.")
    return model_inputs, prompt_len


def answer_token_ids(answer_text: str, runtime: Any) -> List[int]:
    return tgi.token_ids_of_answer(str(answer_text).strip(), processor=runtime.processor)


def score(
    *,
    lm: LanguageModel,
    inputs: Dict[str, Any],
    prompt_len: int,
    answer_token_ids_: Sequence[int],
) -> float:
    scoring_inputs = tgi.append_answer_tokens_for_scoring(
        inputs,
        [int(token_id) for token_id in answer_token_ids_],
    )
    return float(
        tgi.run_clean_sequence_logprob(
            lm=lm,
            scoring_inputs=scoring_inputs,
            prompt_len=int(prompt_len),
            answer_token_ids=[int(token_id) for token_id in answer_token_ids_],
        )
    )


def resolve_pad_token_id(runtime: Any) -> int:
    pad_token_id = getattr(runtime.processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(runtime.processor.tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id for batched scoring.")
    return int(pad_token_id)


def pad_scoring_tensors_for_batch(
    scoring_inputs_list: Sequence[Dict[str, Any]],
    *,
    pad_token_id: int,
) -> Dict[str, Any]:
    if not scoring_inputs_list:
        raise ValueError("scoring_inputs_list must be non-empty")
    if len(scoring_inputs_list) == 1:
        return dict(scoring_inputs_list[0])

    def pad_dim1(value: torch.Tensor, *, pad_value: int) -> torch.Tensor:
        if value.dim() < 2 or int(value.shape[0]) != 1:
            raise ValueError(f"Expected a single-row sequence tensor, got shape={tuple(value.shape)}")
        max_len = max(int(inputs["input_ids"].shape[1]) for inputs in scoring_inputs_list)
        pad_len = int(max_len - int(value.shape[1]))
        if pad_len <= 0:
            return value
        pad_shape = list(value.shape)
        pad_shape[1] = pad_len
        suffix = torch.full(
            tuple(pad_shape),
            int(pad_value),
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat([value, suffix], dim=1)

    out: Dict[str, Any] = {}
    keys = list(scoring_inputs_list[0].keys())
    for key in keys:
        values = [inputs[key] for inputs in scoring_inputs_list]
        first_value = values[0]
        if not torch.is_tensor(first_value):
            out[key] = first_value
            continue
        if key == "input_ids":
            out[key] = torch.cat([pad_dim1(value, pad_value=pad_token_id) for value in values], dim=0)
            continue
        if key == "attention_mask":
            out[key] = torch.cat([pad_dim1(value, pad_value=0) for value in values], dim=0)
            continue
        if key in MULTIMODAL_BATCH_KEYS:
            out[key] = torch.cat(values, dim=0)
            continue
        if first_value.dim() == 0:
            out[key] = torch.stack(values, dim=0)
            continue
        if (
            first_value.dim() >= 2
            and int(first_value.shape[0]) == 1
            and all(int(value.shape[0]) == 1 for value in values)
        ):
            if all(
                int(value.shape[1]) == int(scoring_inputs_list[idx]["input_ids"].shape[1])
                for idx, value in enumerate(values)
            ):
                out[key] = torch.cat([pad_dim1(value, pad_value=0) for value in values], dim=0)
                continue
            if all(tuple(value.shape[1:]) == tuple(first_value.shape[1:]) for value in values):
                out[key] = torch.cat(values, dim=0)
                continue
        raise ValueError(f"Cannot batch input {key!r} with shape={tuple(first_value.shape)}")
    return out


def sequence_logprobs_from_logits_by_prompt_len(
    logits: torch.Tensor,
    *,
    prompt_lens: Sequence[int],
    answer_token_ids_: Sequence[int],
) -> List[float]:
    if logits.dim() != 3:
        raise ValueError(f"Expected logits rank-3 [batch, seq, vocab], got {tuple(logits.shape)}")
    if int(logits.shape[0]) != len(prompt_lens):
        raise ValueError(f"logits batch size {int(logits.shape[0])} does not match prompt_lens={len(prompt_lens)}")
    if not answer_token_ids_:
        raise ValueError("answer_token_ids_ must be non-empty")

    answer_len = len(answer_token_ids_)
    target_token_ids = torch.tensor(answer_token_ids_, device=logits.device, dtype=torch.long)
    scores: List[float] = []
    for batch_idx, prompt_len in enumerate(prompt_lens):
        prompt_len = int(prompt_len)
        if prompt_len <= 0:
            raise ValueError("prompt_len must be >= 1")
        token_positions = torch.arange(answer_len, device=logits.device, dtype=torch.long) + (prompt_len - 1)
        selected_logits = logits[int(batch_idx), token_positions, :]
        log_probs = torch.log_softmax(selected_logits, dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)
        scores.append(float(target_log_probs.sum().item()))
    return scores


def score_batched_same_answer(
    *,
    lm: LanguageModel,
    inputs_and_prompt_lens: Sequence[Tuple[Dict[str, Any], int]],
    answer_token_ids_: Sequence[int],
    score_batch_size: int,
    pad_token_id: int,
) -> List[float]:
    if int(score_batch_size) <= 0:
        raise ValueError("score_batch_size must be positive")
    if not inputs_and_prompt_lens:
        return []

    answer_ids = [int(token_id) for token_id in answer_token_ids_]
    scoring_items = [
        (
            tgi.append_answer_tokens_for_scoring(inputs, answer_ids),
            int(prompt_len),
        )
        for inputs, prompt_len in inputs_and_prompt_lens
    ]

    scores: List[float] = []
    for start_idx in range(0, len(scoring_items), int(score_batch_size)):
        microbatch = scoring_items[start_idx : start_idx + int(score_batch_size)]
        microbatch_inputs = pad_scoring_tensors_for_batch(
            [item[0] for item in microbatch],
            pad_token_id=int(pad_token_id),
        )
        microbatch_prompt_lens = [int(item[1]) for item in microbatch]
        with torch.inference_mode():
            with lm.trace(microbatch_inputs):
                saved_logits = lm.output.logits.save()
        logits = saved_logits.value if hasattr(saved_logits, "value") else saved_logits
        scores.extend(
            sequence_logprobs_from_logits_by_prompt_len(
                logits,
                prompt_lens=microbatch_prompt_lens,
                answer_token_ids_=answer_ids,
            )
        )
    if len(scores) != len(inputs_and_prompt_lens):
        raise RuntimeError(f"Expected {len(inputs_and_prompt_lens)} scores, got {len(scores)}")
    return scores


def normalize_answer_text(answer_text: Any) -> str:
    return str(answer_text).strip()


def process_sample(
    *,
    bucket_spec: ExactBucketSpec,
    sample_dir: Path,
    runtime: Any,
    lm: LanguageModel,
    score_batch_size: int,
    sample_index: int,
    total_samples: int,
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    log_prefix = (
        f"[{sample_index}/{total_samples}] "
        f"seq_len={int(bucket_spec.seq_len)} split={bucket_spec.split} "
        f"evidence_count={int(bucket_spec.evidence_count)} sample_id={sample_dir.name}"
    )
    try:
        sample_id, frames, question, states, answer = eval_utils.load_mmred_sample(sample_dir)
    except Exception as exc:
        print(f"{log_prefix} skipped: load failure ({exc})")
        return None

    evidence_frame_indices = [int(frame_idx) for frame_idx in eval_utils.collect_evidence_frame_indices(question, states)]
    evidence_frame_indices.sort()
    if len(evidence_frame_indices) != int(bucket_spec.evidence_count):
        print(
            f"{log_prefix} skipped: metadata evidence_count={len(evidence_frame_indices)} "
            f"does not match exact bucket {int(bucket_spec.evidence_count)}"
        )
        return None
    if not evidence_frame_indices:
        print(f"{log_prefix} skipped: no evidence frames")
        return None

    try:
        clean_inputs, clean_prompt_len = build_inputs_for_sample(frames, question, runtime)
    except Exception as exc:
        print(f"{log_prefix} skipped: failed to build clean inputs ({exc})")
        return None

    gold_answer = normalize_answer_text(answer)
    try:
        clean_metrics = tgi.score_valid_numeric_answers(
            lm=lm,
            inputs=clean_inputs,
            prompt_len=clean_prompt_len,
            num_frames=len(frames),
        )
    except Exception as exc:
        print(f"{log_prefix} skipped: clean numeric scoring failed ({exc})")
        return None

    clean_top1_answer = normalize_answer_text(clean_metrics["best_answer_text"])
    clean_top1_is_gold = bool(clean_top1_answer == gold_answer)

    try:
        gold_answer_ids = answer_token_ids(gold_answer, runtime)
    except Exception as exc:
        print(f"{log_prefix} skipped: invalid gold answer tokenization ({exc})")
        return None

    clean_score_gt = clean_metrics["scores_by_answer"].get(gold_answer)
    if clean_score_gt is None:
        try:
            clean_score_gt = score(
                lm=lm,
                inputs=clean_inputs,
                prompt_len=clean_prompt_len,
                answer_token_ids_=gold_answer_ids,
            )
        except Exception as exc:
            print(f"{log_prefix} skipped: failed to score clean gold answer ({exc})")
            return None
    clean_score_gt = float(clean_score_gt)

    part1_valid = bool(clean_top1_is_gold and int(bucket_spec.evidence_count) >= 1)
    per_frame_rows: List[Dict[str, Any]] = []
    corrupted_variants: List[Tuple[int, Dict[str, Any], int]] = []

    for frame_idx in evidence_frame_indices:
        corrupted_sample_dir = eval_utils.resolve_corrupted_sample_dir(
            corrupted_data_root=bucket_spec.corrupted_dir,
            sample_id=sample_id,
            frame_idx=int(frame_idx),
        )
        if not corrupted_sample_dir.is_dir():
            print(
                f"{log_prefix} skipped: missing single-corrupt sample for frame_idx={int(frame_idx)} "
                f"at {corrupted_sample_dir}"
            )
            return None

        try:
            _, corrupted_frames, corrupted_question, _, _ = eval_utils.load_mmred_sample(corrupted_sample_dir)
        except Exception as exc:
            print(
                f"{log_prefix} skipped: failed to load single-corrupt sample for frame_idx={int(frame_idx)} ({exc})"
            )
            return None

        if len(corrupted_frames) != len(frames):
            print(
                f"{log_prefix} skipped: frame-count mismatch for frame_idx={int(frame_idx)} "
                f"(clean={len(frames)}, corrupted={len(corrupted_frames)})"
            )
            return None

        try:
            corrupted_inputs, corrupted_prompt_len = build_inputs_for_sample(
                corrupted_frames,
                corrupted_question,
                runtime,
            )
        except Exception as exc:
            print(
                f"{log_prefix} skipped: failed to build single-corrupt inputs for frame_idx={int(frame_idx)} ({exc})"
            )
            return None

        corrupted_variants.append((int(frame_idx), corrupted_inputs, int(corrupted_prompt_len)))

    try:
        single_corrupt_gt_scores = score_batched_same_answer(
            lm=lm,
            inputs_and_prompt_lens=[(inputs, prompt_len) for _, inputs, prompt_len in corrupted_variants],
            answer_token_ids_=gold_answer_ids,
            score_batch_size=int(score_batch_size),
            pad_token_id=resolve_pad_token_id(runtime),
        )
    except Exception as exc:
        print(f"{log_prefix} warning: batched corrupted GT scoring failed ({exc}); falling back to sequential scoring")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        single_corrupt_gt_scores = []
        for frame_idx, corrupted_inputs, corrupted_prompt_len in corrupted_variants:
            try:
                single_corrupt_gt_scores.append(
                    score(
                        lm=lm,
                        inputs=corrupted_inputs,
                        prompt_len=int(corrupted_prompt_len),
                        answer_token_ids_=gold_answer_ids,
                    )
                )
            except Exception as sequential_exc:
                print(
                    f"{log_prefix} skipped: failed scoring single-corrupt frame_idx={int(frame_idx)} "
                    f"after batched fallback ({sequential_exc})"
                )
                return None

    for (frame_idx, _, _), single_corrupt_score_gt in zip(corrupted_variants, single_corrupt_gt_scores):
        single_corrupt_score_gt = float(single_corrupt_score_gt)
        d_i_gt_clamped_influence = float(max(0.0, clean_score_gt - single_corrupt_score_gt))
        d_i_gt_abs_influence = float(abs(clean_score_gt - single_corrupt_score_gt))

        per_frame_rows.append({
            "seq_len": int(bucket_spec.seq_len),
            "evidence_count": int(bucket_spec.evidence_count),
            "split": str(bucket_spec.split),
            "sample_id": str(sample_id),
            "frame_idx": int(frame_idx),
            "is_evidence": True,
            "gold_answer": str(gold_answer),
            "clean_top1_answer": str(clean_top1_answer),
            "clean_top1_is_gold": bool(clean_top1_is_gold),
            "clean_score_gt": float(clean_score_gt),
            "single_corrupt_score_gt": float(single_corrupt_score_gt),
            "d_i_gt_clamped_influence": float(d_i_gt_clamped_influence),
            "d_i_gt_abs_influence": float(d_i_gt_abs_influence),
        })

    if len(per_frame_rows) != len(evidence_frame_indices):
        print(
            f"{log_prefix} skipped: scored corrupted frames mismatch "
            f"expected={len(evidence_frame_indices)} actual={len(per_frame_rows)}"
        )
        return None

    gt_clamped_influences = [
        float(row["d_i_gt_clamped_influence"])
        for row in per_frame_rows
        if math.isfinite(float(row["d_i_gt_clamped_influence"]))
    ]
    per_frame_avg_gt_clamped_influence_clean_top1_correct = (
        float(sum(gt_clamped_influences) / len(gt_clamped_influences))
        if part1_valid and gt_clamped_influences
        else float("nan")
    )
    gt_abs_influences = [
        float(row["d_i_gt_abs_influence"])
        for row in per_frame_rows
        if math.isfinite(float(row["d_i_gt_abs_influence"]))
    ]
    per_frame_avg_gt_abs_influence_all_samples = (
        float(sum(gt_abs_influences) / len(gt_abs_influences))
        if gt_abs_influences
        else float("nan")
    )

    per_sample_row = {
        "seq_len": int(bucket_spec.seq_len),
        "evidence_count": int(bucket_spec.evidence_count),
        "split": str(bucket_spec.split),
        "sample_id": str(sample_id),
        "gold_answer": str(gold_answer),
        "clean_top1_answer": str(clean_top1_answer),
        "clean_top1_is_gold": bool(clean_top1_is_gold),
        "num_evidence_frames": int(len(per_frame_rows)),
        "per_frame_avg_gt_clamped_influence_clean_top1_correct": float(
            per_frame_avg_gt_clamped_influence_clean_top1_correct
        ),
        "per_frame_avg_gt_abs_influence_all_samples": float(per_frame_avg_gt_abs_influence_all_samples),
    }

    print(
        f"{log_prefix} kept: clean_top1={clean_top1_answer!r} clean_top1_is_gold={clean_top1_is_gold} "
        f"num_evidence_frames={len(per_frame_rows)} "
        f"per_frame_avg_gt_clamped_influence_clean_top1_correct="
        f"{per_frame_avg_gt_clamped_influence_clean_top1_correct:.6f} "
        f"per_frame_avg_gt_abs_influence_all_samples={per_frame_avg_gt_abs_influence_all_samples:.6f}"
    )
    return per_sample_row, per_frame_rows


def build_aggregate_by_count_rows(
    *,
    bucket_specs: Sequence[ExactBucketSpec],
    per_sample_rows: Sequence[Dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    bucket_keys = sorted({
        (int(bucket_spec.seq_len), int(bucket_spec.evidence_count))
        for bucket_spec in bucket_specs
        if int(bucket_spec.evidence_count) >= 1
    })

    rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for metric in METRIC_ORDER:
        for seq_len, evidence_count in bucket_keys:
            bucket_sample_rows = [
                row
                for row in per_sample_rows
                if int(row["seq_len"]) == int(seq_len)
                and int(row["evidence_count"]) == int(evidence_count)
            ]
            values = [
                float(row[metric])
                for row in bucket_sample_rows
                if math.isfinite(float(row.get(metric, float("nan"))))
            ]
            summary = summarize_values(values, n_bootstrap=n_bootstrap, rng=rng)
            rows.append({
                "metric": str(metric),
                "seq_len": int(seq_len),
                "evidence_count": int(evidence_count),
                "mean": float(summary["mean"]),
                "median": float(summary["median"]),
                "std": float(summary["std"]),
                "n_samples": int(len(values)),
                "n_frames": int(
                    sum(
                        int(row["num_evidence_frames"])
                        for row in bucket_sample_rows
                        if math.isfinite(float(row.get(metric, float("nan"))))
                    )
                ),
                "mean_ci_low": float(summary["mean_ci_low"]),
                "mean_ci_high": float(summary["mean_ci_high"]),
                "median_ci_low": float(summary["median_ci_low"]),
                "median_ci_high": float(summary["median_ci_high"]),
            })
    return rows


def build_aggregate_by_frame_index_rows(
    *,
    seq_lens: Sequence[int],
    per_frame_rows: Sequence[Dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    for seq_len in sorted(int(value) for value in seq_lens):
        for frame_idx in range(int(seq_len)):
            values = [
                float(row["d_i_gt_clamped_influence"])
                for row in per_frame_rows
                if int(row["seq_len"]) == int(seq_len)
                and int(row["frame_idx"]) == int(frame_idx)
                and bool(row.get("clean_top1_is_gold"))
                and math.isfinite(float(row.get("d_i_gt_clamped_influence", float("nan"))))
            ]
            summary = summarize_values(values, n_bootstrap=n_bootstrap, rng=rng)
            rows.append({
                "seq_len": int(seq_len),
                "frame_idx": int(frame_idx),
                "mean_d_i_gt_clamped_influence_clean_top1_correct": float(summary["mean"]),
                "median_d_i_gt_clamped_influence_clean_top1_correct": float(summary["median"]),
                "std_d_i_gt_clamped_influence_clean_top1_correct": float(summary["std"]),
                "n": int(len(values)),
                "mean_ci_low": float(summary["mean_ci_low"]),
                "mean_ci_high": float(summary["mean_ci_high"]),
                "median_ci_low": float(summary["median_ci_low"]),
                "median_ci_high": float(summary["median_ci_high"]),
            })
    return rows


def plot_metric_vs_evidence_count(
    *,
    aggregate_rows: Sequence[Dict[str, Any]],
    metric: str,
    output_path: Path,
    title: str,
    y_label: str,
    min_evidence_count: int,
    formula_text: Optional[str] = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    seq_lens = sorted({int(row["seq_len"]) for row in aggregate_rows})
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    has_any_line = False
    x_values: set[int] = set()

    for line_index, seq_len in enumerate(seq_lens):
        seq_rows = sorted(
            [
                row
                for row in aggregate_rows
                if str(row["metric"]) == str(metric)
                and int(row["seq_len"]) == int(seq_len)
                and int(row["evidence_count"]) >= int(min_evidence_count)
            ],
            key=lambda row: int(row["evidence_count"]),
        )
        if not seq_rows:
            continue
        x = [int(row["evidence_count"]) for row in seq_rows]
        mean_y = [float(row["mean"]) for row in seq_rows]
        mean_lo = [float(row["mean_ci_low"]) for row in seq_rows]
        mean_hi = [float(row["mean_ci_high"]) for row in seq_rows]
        median_y = [float(row["median"]) for row in seq_rows]
        median_lo = [float(row["median_ci_low"]) for row in seq_rows]
        median_hi = [float(row["median_ci_high"]) for row in seq_rows]
        color = color_cycle[line_index % len(color_cycle)] if color_cycle else None
        line, = ax.plot(
            x,
            mean_y,
            marker="o",
            linewidth=2.2,
            label=f"seq_len={int(seq_len)}",
            color=color,
        )
        ax.fill_between(x, mean_lo, mean_hi, alpha=0.16, color=line.get_color())
        ax.plot(
            x,
            median_y,
            marker="s",
            linewidth=1.8,
            linestyle="--",
            label="_nolegend_",
            color=line.get_color(),
        )
        ax.fill_between(x, median_lo, median_hi, alpha=0.08, color=line.get_color())
        x_values.update(x)
        has_any_line = True

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Evidence count")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if x_values:
        ax.set_xticks(sorted(x_values))
    if has_any_line:
        ax.legend(frameon=True, title="solid=mean, dashed=median")
    else:
        ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")
    if formula_text:
        ax.text(
            0.02,
            0.98,
            formula_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.85",
                "alpha": 0.86,
            },
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_frame_index_metric(
    *,
    aggregate_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    seq_lens = sorted({int(row["seq_len"]) for row in aggregate_rows})
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    has_any_line = False
    x_values: set[int] = set()

    for line_index, seq_len in enumerate(seq_lens):
        seq_rows = sorted(
            [row for row in aggregate_rows if int(row["seq_len"]) == int(seq_len)],
            key=lambda row: int(row["frame_idx"]),
        )
        if not seq_rows:
            continue
        x = [int(row["frame_idx"]) for row in seq_rows]
        mean_y = [float(row["mean_d_i_gt_clamped_influence_clean_top1_correct"]) for row in seq_rows]
        mean_lo = [float(row["mean_ci_low"]) for row in seq_rows]
        mean_hi = [float(row["mean_ci_high"]) for row in seq_rows]
        median_y = [float(row["median_d_i_gt_clamped_influence_clean_top1_correct"]) for row in seq_rows]
        median_lo = [float(row["median_ci_low"]) for row in seq_rows]
        median_hi = [float(row["median_ci_high"]) for row in seq_rows]
        color = color_cycle[line_index % len(color_cycle)] if color_cycle else None
        line, = ax.plot(
            x,
            mean_y,
            marker="o",
            linewidth=2.2,
            label=f"seq_len={int(seq_len)}",
            color=color,
        )
        ax.fill_between(x, mean_lo, mean_hi, alpha=0.16, color=line.get_color())
        ax.plot(
            x,
            median_y,
            marker="s",
            linewidth=1.8,
            linestyle="--",
            label="_nolegend_",
            color=line.get_color(),
        )
        ax.fill_between(x, median_lo, median_hi, alpha=0.08, color=line.get_color())
        x_values.update(x)
        has_any_line = True

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Temporal frame index")
    ax.set_ylabel("Mean GT clamped influence")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if x_values:
        ax.set_xticks(sorted(x_values))
    if has_any_line:
        ax.legend(frameon=True, title="solid=mean, dashed=median")
    else:
        ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_configuration(
    *,
    args: argparse.Namespace,
    corrupted_root: Path,
    resolved_device: str,
    resolved_dtype: torch.dtype,
    use_4bit: bool,
    seq_lens: Sequence[int],
    bucket_specs: Sequence[ExactBucketSpec],
    exact_zero_counts: Dict[Tuple[int, str], int],
) -> None:
    print(f"model={args.model_name}")
    print(f"data_root={args.data_root}")
    print(f"corrupted_root={corrupted_root}")
    print(f"output_dir={args.output_dir}")
    print(f"device={resolved_device}")
    print(f"dtype={resolved_dtype}")
    print(f"use_4bit={use_4bit}")
    print(f"seq_lens={list(seq_lens)}")
    print(f"bootstrap_samples={int(args.bootstrap_samples)} seed={int(args.seed)}")
    print(f"aggregate_every={args.aggregate_every}")
    print(f"score_batch_size={int(args.score_batch_size)}")
    if args.max_samples_per_exact_dir is not None:
        print(f"max_samples_per_exact_dir={int(args.max_samples_per_exact_dir)}")
    for seq_len, split in sorted(exact_zero_counts):
        print(
            f"seq_len={int(seq_len)} split={split} exact_0 skipped_for_metrics sample_count={int(exact_zero_counts[(seq_len, split)])}"
        )
    for bucket_spec in bucket_specs:
        print(
            f"queued seq_len={int(bucket_spec.seq_len)} split={bucket_spec.split} "
            f"evidence_count={int(bucket_spec.evidence_count)} samples={len(bucket_spec.sample_dirs)} "
            f"clean_dir={bucket_spec.clean_dir} corrupted_dir={bucket_spec.corrupted_dir}"
        )


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    data_root = Path(args.data_root)
    corrupted_root = (
        Path(args.corrupted_root)
        if args.corrupted_root is not None
        else eval_utils.infer_corrupted_data_root(data_root)
    )
    seq_lens = resolve_seq_lens(data_root, args.seq_lens)
    bucket_specs, exact_zero_counts = discover_bucket_specs(
        data_root=data_root,
        corrupted_root=corrupted_root,
        seq_lens=seq_lens,
        max_samples_per_exact_dir=args.max_samples_per_exact_dir,
    )
    if not bucket_specs:
        raise RuntimeError("No exact_j buckets with evidence_count >= 1 were found.")

    runtime, resolved_device, resolved_dtype, use_4bit = build_runtime(
        model_name=str(args.model_name),
        requested_device=str(args.device),
        requested_dtype=str(args.dtype),
        load_in_4bit=bool(args.load_in_4bit),
    )
    lm = LanguageModel(runtime.model, tokenizer=runtime.processor.tokenizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_frame_csv_path = output_dir / "per_frame_metrics.csv"
    per_sample_csv_path = output_dir / "per_sample_metrics.csv"
    aggregate_by_count_csv_path = output_dir / "aggregate_by_count.csv"
    aggregate_by_frame_index_csv_path = output_dir / "aggregate_by_frame_index.csv"

    initialize_csv(per_frame_csv_path, PER_FRAME_FIELDS)
    initialize_csv(per_sample_csv_path, PER_SAMPLE_FIELDS)
    write_csv_atomic(aggregate_by_count_csv_path, [], AGGREGATE_BY_COUNT_FIELDS)
    write_csv_atomic(aggregate_by_frame_index_csv_path, [], AGGREGATE_BY_FRAME_INDEX_FIELDS)

    print_configuration(
        args=args,
        corrupted_root=corrupted_root,
        resolved_device=resolved_device,
        resolved_dtype=resolved_dtype,
        use_4bit=use_4bit,
        seq_lens=seq_lens,
        bucket_specs=bucket_specs,
        exact_zero_counts=exact_zero_counts,
    )

    total_samples = sum(len(bucket_spec.sample_dirs) for bucket_spec in bucket_specs)
    if total_samples <= 0:
        raise RuntimeError("No samples remain after applying discovery and per-bucket limits.")
    processed_samples = 0
    skipped_samples = 0
    per_frame_rows: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []

    running_index = 0
    for bucket_spec in bucket_specs:
        for sample_dir in bucket_spec.sample_dirs:
            running_index += 1
            try:
                processed = process_sample(
                    bucket_spec=bucket_spec,
                    sample_dir=sample_dir,
                    runtime=runtime,
                    lm=lm,
                    score_batch_size=int(args.score_batch_size),
                    sample_index=running_index,
                    total_samples=total_samples,
                )
            except Exception as exc:
                skipped_samples += 1
                print(
                    f"[{running_index}/{total_samples}] "
                    f"seq_len={int(bucket_spec.seq_len)} split={bucket_spec.split} "
                    f"evidence_count={int(bucket_spec.evidence_count)} sample_id={sample_dir.name} "
                    f"skipped: unexpected failure ({exc})"
                )
                continue

            if processed is None:
                skipped_samples += 1
                continue

            per_sample_row, sample_frame_rows = processed
            per_sample_rows.append(per_sample_row)
            per_frame_rows.extend(sample_frame_rows)
            append_csv_rows(per_sample_csv_path, [per_sample_row], PER_SAMPLE_FIELDS)
            append_csv_rows(per_frame_csv_path, sample_frame_rows, PER_FRAME_FIELDS)
            processed_samples += 1

            if args.aggregate_every is not None and processed_samples % int(args.aggregate_every) == 0:
                aggregate_by_count_rows = build_aggregate_by_count_rows(
                    bucket_specs=bucket_specs,
                    per_sample_rows=per_sample_rows,
                    n_bootstrap=int(args.bootstrap_samples),
                    seed=int(args.seed),
                )
                aggregate_by_frame_index_rows = build_aggregate_by_frame_index_rows(
                    seq_lens=seq_lens,
                    per_frame_rows=per_frame_rows,
                    n_bootstrap=int(args.bootstrap_samples),
                    seed=int(args.seed),
                )
                write_csv_atomic(
                    aggregate_by_count_csv_path,
                    aggregate_by_count_rows,
                    AGGREGATE_BY_COUNT_FIELDS,
                )
                write_csv_atomic(
                    aggregate_by_frame_index_csv_path,
                    aggregate_by_frame_index_rows,
                    AGGREGATE_BY_FRAME_INDEX_FIELDS,
                )

    final_aggregate_by_count_rows = build_aggregate_by_count_rows(
        bucket_specs=bucket_specs,
        per_sample_rows=per_sample_rows,
        n_bootstrap=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    final_aggregate_by_frame_index_rows = build_aggregate_by_frame_index_rows(
        seq_lens=seq_lens,
        per_frame_rows=per_frame_rows,
        n_bootstrap=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    write_csv_atomic(
        aggregate_by_count_csv_path,
        final_aggregate_by_count_rows,
        AGGREGATE_BY_COUNT_FIELDS,
    )
    write_csv_atomic(
        aggregate_by_frame_index_csv_path,
        final_aggregate_by_frame_index_rows,
        AGGREGATE_BY_FRAME_INDEX_FIELDS,
    )

    plot_paths: List[Path] = []

    plot_paths.append(
        plot_metric_vs_evidence_count(
            aggregate_rows=final_aggregate_by_count_rows,
            metric="per_frame_avg_gt_clamped_influence_clean_top1_correct",
            output_path=output_dir / "mean_per_frame_avg_gt_clamped_influence_vs_evidence_count_clean_top1_correct.png",
            title="Mean Per-Frame Average GT Clamped Influence vs Evidence Count (Clean Top-1 Correct)",
            y_label="Mean per-frame avg GT clamped influence",
            min_evidence_count=1,
            formula_text=(
                "GT clamped per-frame influence:\n"
                "d_i = max(0, s_clean(GT) - s_corrupt_i(GT))\n"
                "avg = (1/k) * sum_i d_i"
            ),
        )
    )
    plot_paths.append(
        plot_metric_vs_evidence_count(
            aggregate_rows=final_aggregate_by_count_rows,
            metric="per_frame_avg_gt_abs_influence_all_samples",
            output_path=output_dir / "mean_per_frame_avg_gt_abs_influence_vs_evidence_count_all_samples.png",
            title="Mean Per-Frame Average GT Absolute Influence vs Evidence Count (All Samples)",
            y_label="Mean per-frame avg GT abs influence",
            min_evidence_count=1,
            formula_text=(
                "GT absolute per-frame influence:\n"
                "d_i = abs(s_clean(GT) - s_corrupt_i(GT))\n"
                "avg = (1/k) * sum_i d_i"
            ),
        )
    )
    plot_paths.append(
        plot_frame_index_metric(
            aggregate_rows=final_aggregate_by_frame_index_rows,
            output_path=output_dir / "mean_gt_clamped_frame_influence_by_frame_index_clean_top1_correct.png",
            title="Mean GT Clamped Frame Influence by Temporal Frame Index (Clean Top-1 Correct)",
        )
    )

    elapsed = time.perf_counter() - start_time
    print(f"processed_samples={processed_samples} skipped_samples={skipped_samples} total_candidates={total_samples}")
    print(f"per_frame_csv={per_frame_csv_path}")
    print(f"per_sample_csv={per_sample_csv_path}")
    print(f"aggregate_by_count_csv={aggregate_by_count_csv_path}")
    print(f"aggregate_by_frame_index_csv={aggregate_by_frame_index_csv_path}")
    print("new_plot_filenames:")
    for path in plot_paths:
        print(f"  {path.name}")
    print("plots=" + ", ".join(str(path) for path in plot_paths))
    print(eval_utils.format_runtime(elapsed))


if __name__ == "__main__":
    main()

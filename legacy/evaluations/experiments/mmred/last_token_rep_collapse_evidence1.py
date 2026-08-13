"""
Standalone MMReD last-token representational collapse analysis.

This experiment compares the final-layer last prompt token hidden state between
clean samples and single-evidence-frame-corrupted variants. It is intentionally
representation-only: no answer scoring or logprob computation is performed.
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
import torch.nn.functional as F
from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from models import model as model_utils
from models.model import DEFAULT_MODEL_ID, get_layers, load_model_runtime

SEQ_LEN_DIR_RE = re.compile(r"^seq_len_(\d+)$")

PER_SAMPLE_FIELDS = [
    "seq_len",
    "evidence_count",
    "split",
    "sample_id",
    "evidence_frame_idx",
    "prompt_len_clean",
    "prompt_len_corrupt",
    "linf",
    "l2",
    "normalized_l2",
    "cosine_distance",
    "score_delta",
]

AGGREGATE_BY_SEQ_LEN_FIELDS = [
    "metric",
    "seq_len",
    "mean",
    "median",
    "std",
    "n_samples",
    "mean_ci_low",
    "mean_ci_high",
    "median_ci_low",
    "median_ci_high",
]

METRIC_ORDER = ("linf", "l2", "normalized_l2", "cosine_distance")
CORRELATION_FIELDS = [
    "metric",
    "n_samples",
    "pearson_r",
    "spearman_rho",
]
SCORE_DELTA_COLUMN_CANDIDATES = (
    "score_delta",
    "score_drop",
    "delta",
    "clean_top1_score_drop",
    "gt_score_drop",
)
SCORE_SEQ_LEN_COLUMN_CANDIDATES = ("seq_len",)
SCORE_FRAME_COLUMN_CANDIDATES = ("evidence_frame_idx", "frame_idx", "corrupt_frame_idx")
SCORE_SAMPLE_ID_COLUMN_CANDIDATES = ("sample_id",)


@dataclass(frozen=True)
class ExactBucketSpec:
    seq_len: int
    split: str
    evidence_count: int
    clean_dir: Path
    corrupted_dir: Path
    sample_dirs: Tuple[Path, ...]


@dataclass(frozen=True)
class ScoreLookup:
    score_csv_path: Path
    score_delta_column: Optional[str]
    usable_rows: int
    by_sample_seq_frame: Dict[Tuple[str, int, int], Optional[float]]
    by_sample_seq: Dict[Tuple[str, int], Optional[float]]
    by_sample_frame: Dict[Tuple[str, int], Optional[float]]
    by_sample: Dict[str, Optional[float]]


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
    return Path("outputs/last_token_rep_collapse_evidence1") / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure MMReD representational collapse by comparing the final-layer "
            "last-token hidden state between clean samples and single-frame-corrupted variants "
            "from by_evidence_count/exact_1 buckets."
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
        default="2,4,8",
        help="Comma-separated seq_len list. Defaults to 2,4,8.",
    )
    parser.add_argument(
        "--max-samples-per-exact-dir",
        "--max_samples_per_exact_dir",
        dest="max_samples_per_exact_dir",
        type=int,
        default=None,
        help="Optional cap on the number of samples processed from each exact_1 directory.",
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
        "--score-csv",
        "--score_csv",
        dest="score_csv",
        type=Path,
        default=None,
        help="Optional CSV with externally-computed score deltas to join by sample_id/seq_len/evidence_frame_idx.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.max_samples_per_exact_dir is not None and int(args.max_samples_per_exact_dir) <= 0:
        raise ValueError("--max-samples-per-exact-dir must be positive when provided")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("--bootstrap-samples must be positive")
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
) -> List[ExactBucketSpec]:
    bucket_specs: List[ExactBucketSpec] = []

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
            exact_dir = clean_bucket_root / "exact_1"
            if not exact_dir.is_dir():
                print(
                    f"[warn] seq_len={int(seq_len)} split={split_label} missing exact_1 directory: {exact_dir}"
                )
                continue
            sample_dirs = tuple(eval_utils.iter_sample_dirs(exact_dir))
            if max_samples_per_exact_dir is not None:
                sample_dirs = sample_dirs[: int(max_samples_per_exact_dir)]
            bucket_specs.append(
                ExactBucketSpec(
                    seq_len=int(seq_len),
                    split=str(split_label),
                    evidence_count=1,
                    clean_dir=exact_dir,
                    corrupted_dir=corrupted_seq_dir / relative_bucket_root / "exact_1",
                    sample_dirs=sample_dirs,
                )
            )

    bucket_specs.sort(
        key=lambda spec: (
            int(spec.seq_len),
            str(spec.split),
            str(spec.clean_dir),
        )
    )
    return bucket_specs


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


def materialize_saved(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


def layer_output_to_hidden_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x:
        return layer_output_to_hidden_tensor(x[0])
    raise TypeError(f"Unsupported layer output type: {type(x)}")


def normalize_column_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def first_present_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized_to_actual = {normalize_column_name(name): str(name) for name in fieldnames}
    for candidate in candidates:
        actual = normalized_to_actual.get(normalize_column_name(candidate))
        if actual is not None:
            return actual
    return None


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            numeric = float(text)
        except ValueError:
            return None
        if not numeric.is_integer():
            return None
        return int(numeric)


def parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def update_unique_float_map(mapping: Dict[Any, Optional[float]], key: Any, value: float) -> None:
    if key is None:
        return
    if key not in mapping:
        mapping[key] = float(value)
        return
    previous = mapping[key]
    if previous is None:
        return
    if float(previous) == float(value):
        return
    mapping[key] = None


def load_score_lookup(score_csv_path: Optional[Path]) -> Optional[ScoreLookup]:
    if score_csv_path is None:
        return None
    score_csv_path = Path(score_csv_path)
    if not score_csv_path.is_file():
        raise FileNotFoundError(f"Score CSV not found: {score_csv_path}")

    with score_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            print(f"[warn] score CSV has no header: {score_csv_path}; score-delta join disabled")
            return ScoreLookup(
                score_csv_path=score_csv_path,
                score_delta_column=None,
                usable_rows=0,
                by_sample_seq_frame={},
                by_sample_seq={},
                by_sample_frame={},
                by_sample={},
            )

        sample_id_column = first_present_column(fieldnames, SCORE_SAMPLE_ID_COLUMN_CANDIDATES)
        seq_len_column = first_present_column(fieldnames, SCORE_SEQ_LEN_COLUMN_CANDIDATES)
        frame_column = first_present_column(fieldnames, SCORE_FRAME_COLUMN_CANDIDATES)
        score_delta_column = first_present_column(fieldnames, SCORE_DELTA_COLUMN_CANDIDATES)

        if sample_id_column is None or score_delta_column is None:
            print(
                f"[warn] score CSV missing required join/value columns "
                f"(sample_id={sample_id_column!r}, score_delta={score_delta_column!r}); "
                "score-delta join disabled"
            )
            return ScoreLookup(
                score_csv_path=score_csv_path,
                score_delta_column=score_delta_column,
                usable_rows=0,
                by_sample_seq_frame={},
                by_sample_seq={},
                by_sample_frame={},
                by_sample={},
            )

        by_sample_seq_frame: Dict[Tuple[str, int, int], Optional[float]] = {}
        by_sample_seq: Dict[Tuple[str, int], Optional[float]] = {}
        by_sample_frame: Dict[Tuple[str, int], Optional[float]] = {}
        by_sample: Dict[str, Optional[float]] = {}
        usable_rows = 0

        for row in reader:
            sample_id = str(row.get(sample_id_column, "")).strip()
            if not sample_id:
                continue
            score_delta = parse_optional_float(row.get(score_delta_column))
            if score_delta is None:
                continue
            seq_len = parse_optional_int(row.get(seq_len_column)) if seq_len_column is not None else None
            frame_idx = parse_optional_int(row.get(frame_column)) if frame_column is not None else None
            usable_rows += 1

            if seq_len is not None and frame_idx is not None:
                update_unique_float_map(by_sample_seq_frame, (sample_id, int(seq_len), int(frame_idx)), score_delta)
            if seq_len is not None:
                update_unique_float_map(by_sample_seq, (sample_id, int(seq_len)), score_delta)
            if frame_idx is not None:
                update_unique_float_map(by_sample_frame, (sample_id, int(frame_idx)), score_delta)
            update_unique_float_map(by_sample, sample_id, score_delta)

    print(
        f"score_csv={score_csv_path} score_delta_column={score_delta_column!r} "
        f"usable_rows={usable_rows}"
    )
    return ScoreLookup(
        score_csv_path=score_csv_path,
        score_delta_column=score_delta_column,
        usable_rows=int(usable_rows),
        by_sample_seq_frame=by_sample_seq_frame,
        by_sample_seq=by_sample_seq,
        by_sample_frame=by_sample_frame,
        by_sample=by_sample,
    )


def maybe_clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_decoder_layers(lm: LanguageModel) -> Any:
    layers = get_layers(lm.model)
    if len(layers) <= 0:
        raise RuntimeError("Model exposes no decoder layers.")
    return layers


def extract_last_token_hidden_state(
    *,
    lm: LanguageModel,
    layer_module: Any,
    inputs: Dict[str, Any],
    prompt_len: int,
) -> torch.Tensor:
    if int(prompt_len) <= 0:
        raise RuntimeError("Prompt length must be positive.")
    last_token_idx = int(prompt_len) - 1
    with torch.inference_mode():
        with lm.trace(inputs):
            saved_hidden = layer_output_to_hidden_tensor(layer_module.output).save()
    hidden = materialize_saved(saved_hidden)
    if not torch.is_tensor(hidden):
        raise TypeError(f"Saved layer output did not materialize to a tensor: {type(hidden)}")
    if hidden.dim() != 3:
        raise RuntimeError(f"Expected rank-3 hidden states [batch, seq, hidden], got {tuple(hidden.shape)}")
    if int(hidden.shape[0]) != 1:
        raise RuntimeError(f"Expected batch size 1, got hidden shape {tuple(hidden.shape)}")
    if last_token_idx < 0 or last_token_idx >= int(hidden.shape[1]):
        raise RuntimeError(
            f"last_token_idx={last_token_idx} is outside hidden shape {tuple(hidden.shape)}"
        )
    return hidden[0, last_token_idx, :].detach().float().cpu()


def compute_rep_metrics(clean_hidden: torch.Tensor, corrupt_hidden: torch.Tensor) -> Dict[str, float]:
    if tuple(clean_hidden.shape) != tuple(corrupt_hidden.shape):
        raise ValueError(
            f"Hidden state shape mismatch: clean={tuple(clean_hidden.shape)} corrupt={tuple(corrupt_hidden.shape)}"
        )
    diff = clean_hidden - corrupt_hidden
    clean_norm_l2 = float(clean_hidden.norm(p=2).item())
    corrupt_norm_l2 = float(corrupt_hidden.norm(p=2).item())
    l2 = float(diff.norm(p=2).item())
    cosine_similarity = F.cosine_similarity(
        clean_hidden.unsqueeze(0),
        corrupt_hidden.unsqueeze(0),
        dim=1,
        eps=1e-12,
    )[0]
    return {
        "linf": float(diff.abs().max().item()),
        "l2": float(l2),
        "normalized_l2": float(l2 / (clean_norm_l2 + 1e-12)),
        "cosine_distance": float((1.0 - cosine_similarity).item()),
        "clean_norm_l2": float(clean_norm_l2),
        "corrupt_norm_l2": float(corrupt_norm_l2),
    }


def join_score_delta(row: Dict[str, Any], score_lookup: Optional[ScoreLookup]) -> Optional[float]:
    if score_lookup is None:
        return None
    sample_id = str(row.get("sample_id", ""))
    seq_len = int(row["seq_len"])
    evidence_frame_idx = int(row["evidence_frame_idx"])

    candidates = (
        score_lookup.by_sample_seq_frame.get((sample_id, seq_len, evidence_frame_idx)),
        score_lookup.by_sample_seq.get((sample_id, seq_len)),
        score_lookup.by_sample_frame.get((sample_id, evidence_frame_idx)),
        score_lookup.by_sample.get(sample_id),
    )
    for candidate in candidates:
        if candidate is not None and math.isfinite(float(candidate)):
            return float(candidate)
    return None


def average_rank(values: Sequence[float]) -> List[float]:
    enumerated = sorted((float(value), idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(enumerated)
    start = 0
    while start < len(enumerated):
        end = start + 1
        while end < len(enumerated) and enumerated[end][0] == enumerated[start][0]:
            end += 1
        avg_rank = (float(start + 1) + float(end)) / 2.0
        for idx in range(start, end):
            ranks[enumerated[idx][1]] = float(avg_rank)
        start = end
    return ranks


def pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if len(x_values) < 2:
        return float("nan")
    x_mean = mean_value(x_values)
    y_mean = mean_value(y_values)
    x_centered = [float(value) - x_mean for value in x_values]
    y_centered = [float(value) - y_mean for value in y_values]
    denom_x = math.sqrt(sum(value * value for value in x_centered))
    denom_y = math.sqrt(sum(value * value for value in y_centered))
    denom = denom_x * denom_y
    if denom <= 0.0:
        return float("nan")
    numerator = sum(x_val * y_val for x_val, y_val in zip(x_centered, y_centered))
    return float(numerator / denom)


def spearman_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if len(x_values) < 2:
        return float("nan")
    return float(pearson_correlation(average_rank(x_values), average_rank(y_values)))


def metric_score_pairs(
    per_sample_rows: Sequence[Dict[str, Any]],
    *,
    metric: str,
) -> Tuple[List[float], List[float]]:
    metric_values: List[float] = []
    score_values: List[float] = []
    for row in per_sample_rows:
        metric_value = parse_optional_float(row.get(metric))
        score_delta = parse_optional_float(row.get("score_delta"))
        if metric_value is None or score_delta is None:
            continue
        metric_values.append(float(metric_value))
        score_values.append(float(score_delta))
    return metric_values, score_values


def build_correlation_summary_rows(
    *,
    per_sample_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric in METRIC_ORDER:
        metric_values, score_values = metric_score_pairs(per_sample_rows, metric=metric)
        rows.append({
            "metric": str(metric),
            "n_samples": int(len(metric_values)),
            "pearson_r": float(pearson_correlation(metric_values, score_values)),
            "spearman_rho": float(spearman_correlation(metric_values, score_values)),
        })
    return rows


def process_sample(
    *,
    bucket_spec: ExactBucketSpec,
    sample_dir: Path,
    runtime: Any,
    lm: LanguageModel,
    final_layer: Any,
    sample_index: int,
    total_samples: int,
) -> Optional[Dict[str, Any]]:
    log_prefix = (
        f"[{sample_index}/{total_samples}] "
        f"seq_len={int(bucket_spec.seq_len)} split={bucket_spec.split} "
        f"evidence_count={int(bucket_spec.evidence_count)} sample_id={sample_dir.name}"
    )
    try:
        sample_id, frames, question, states, _ = eval_utils.load_mmred_sample(sample_dir)
    except Exception as exc:
        print(f"{log_prefix} skipped: load failure ({exc})")
        return None

    evidence_frame_indices = [
        int(frame_idx)
        for frame_idx in eval_utils.collect_evidence_frame_indices(question, states)
    ]
    evidence_frame_indices.sort()
    if len(evidence_frame_indices) != 1:
        print(
            f"{log_prefix} skipped: expected exactly one evidence frame, "
            f"found {len(evidence_frame_indices)}"
        )
        return None
    evidence_frame_idx = int(evidence_frame_indices[0])

    try:
        clean_inputs, clean_prompt_len = build_inputs_for_sample(frames, question, runtime)
    except Exception as exc:
        print(f"{log_prefix} skipped: failed to build clean inputs ({exc})")
        return None

    corrupted_sample_dir = eval_utils.resolve_corrupted_sample_dir(
        corrupted_data_root=bucket_spec.corrupted_dir,
        sample_id=sample_id,
        frame_idx=evidence_frame_idx,
    )
    if not corrupted_sample_dir.is_dir():
        print(
            f"{log_prefix} skipped: missing corrupted sample for evidence_frame_idx={evidence_frame_idx} "
            f"at {corrupted_sample_dir}"
        )
        return None

    try:
        _, corrupted_frames, corrupted_question, _, _ = eval_utils.load_mmred_sample(corrupted_sample_dir)
    except Exception as exc:
        print(
            f"{log_prefix} skipped: failed to load corrupted sample for "
            f"evidence_frame_idx={evidence_frame_idx} ({exc})"
        )
        return None

    if len(corrupted_frames) != len(frames):
        print(
            f"{log_prefix} skipped: frame-count mismatch "
            f"(clean={len(frames)}, corrupted={len(corrupted_frames)})"
        )
        return None

    try:
        corrupt_inputs, corrupt_prompt_len = build_inputs_for_sample(
            corrupted_frames,
            corrupted_question,
            runtime,
        )
    except Exception as exc:
        print(
            f"{log_prefix} skipped: failed to build corrupted inputs for "
            f"evidence_frame_idx={evidence_frame_idx} ({exc})"
        )
        return None

    if int(clean_prompt_len) != int(corrupt_prompt_len):
        print(
            f"{log_prefix} skipped: prompt length mismatch "
            f"(clean={int(clean_prompt_len)}, corrupt={int(corrupt_prompt_len)})"
        )
        return None

    try:
        clean_hidden = extract_last_token_hidden_state(
            lm=lm,
            layer_module=final_layer,
            inputs=clean_inputs,
            prompt_len=int(clean_prompt_len),
        )
        corrupt_hidden = extract_last_token_hidden_state(
            lm=lm,
            layer_module=final_layer,
            inputs=corrupt_inputs,
            prompt_len=int(corrupt_prompt_len),
        )
        metrics = compute_rep_metrics(clean_hidden, corrupt_hidden)
    except Exception as exc:
        maybe_clear_cuda_cache()
        print(
            f"{log_prefix} skipped: failed during hidden-state extraction or metric computation ({exc})"
        )
        return None

    row = {
        "seq_len": int(bucket_spec.seq_len),
        "evidence_count": int(bucket_spec.evidence_count),
        "split": str(bucket_spec.split),
        "sample_id": str(sample_id),
        "evidence_frame_idx": int(evidence_frame_idx),
        "prompt_len_clean": int(clean_prompt_len),
        "prompt_len_corrupt": int(corrupt_prompt_len),
        "linf": float(metrics["linf"]),
        "l2": float(metrics["l2"]),
        "normalized_l2": float(metrics["normalized_l2"]),
        "cosine_distance": float(metrics["cosine_distance"]),
        "score_delta": None,
    }
    print(
        f"{log_prefix} kept: evidence_frame_idx={evidence_frame_idx} "
        f"linf={float(metrics['linf']):.6f} l2={float(metrics['l2']):.6f} "
        f"normalized_l2={float(metrics['normalized_l2']):.6f} "
        f"cosine_distance={float(metrics['cosine_distance']):.6f}"
    )
    return row


def build_aggregate_by_seq_len_rows(
    *,
    seq_lens: Sequence[int],
    per_sample_rows: Sequence[Dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for metric in METRIC_ORDER:
        for seq_len in sorted(int(value) for value in seq_lens):
            values = [
                float(row[metric])
                for row in per_sample_rows
                if int(row["seq_len"]) == int(seq_len)
                and math.isfinite(float(row.get(metric, float("nan"))))
            ]
            summary = summarize_values(values, n_bootstrap=n_bootstrap, rng=rng)
            rows.append({
                "metric": str(metric),
                "seq_len": int(seq_len),
                "mean": float(summary["mean"]),
                "median": float(summary["median"]),
                "std": float(summary["std"]),
                "n_samples": int(len(values)),
                "mean_ci_low": float(summary["mean_ci_low"]),
                "mean_ci_high": float(summary["mean_ci_high"]),
                "median_ci_low": float(summary["median_ci_low"]),
                "median_ci_high": float(summary["median_ci_high"]),
            })
    return rows


def plot_metric_vs_seq_len(
    *,
    aggregate_rows: Sequence[Dict[str, Any]],
    metric: str,
    output_path: Path,
    title: str,
    y_label: str,
    formula_text: Optional[str] = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_rows = sorted(
        [row for row in aggregate_rows if str(row["metric"]) == str(metric)],
        key=lambda row: int(row["seq_len"]),
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    valid_rows = [
        row
        for row in metric_rows
        if int(row.get("n_samples", 0)) > 0 and math.isfinite(float(row.get("mean", float("nan"))))
    ]

    if valid_rows:
        x = [int(row["seq_len"]) for row in valid_rows]
        mean_y = [float(row["mean"]) for row in valid_rows]
        mean_lo = [float(row["mean_ci_low"]) for row in valid_rows]
        mean_hi = [float(row["mean_ci_high"]) for row in valid_rows]
        median_y = [float(row["median"]) for row in valid_rows]

        line, = ax.plot(
            x,
            mean_y,
            marker="o",
            linewidth=2.2,
            label="mean",
        )
        ax.fill_between(x, mean_lo, mean_hi, alpha=0.16, color=line.get_color(), label="95% bootstrap CI")
        ax.plot(
            x,
            median_y,
            marker="s",
            linewidth=1.8,
            linestyle="--",
            color=line.get_color(),
            label="median",
        )
        for row in valid_rows:
            ax.annotate(
                f"n={int(row['n_samples'])}",
                (int(row["seq_len"]), float(row["mean"])),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
    else:
        ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if metric_rows:
        ax.set_xticks([int(row["seq_len"]) for row in metric_rows])
    if valid_rows:
        ax.legend(frameon=True)
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


def plot_metric_vs_score_delta(
    *,
    per_sample_rows: Sequence[Dict[str, Any]],
    metric: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> Optional[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_values, score_values = metric_score_pairs(per_sample_rows, metric=metric)
    if not metric_values:
        return None

    pearson_r = pearson_correlation(metric_values, score_values)
    spearman_rho = spearman_correlation(metric_values, score_values)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    ax.scatter(score_values, metric_values, alpha=0.65, s=28)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("score_delta")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02,
        0.98,
        f"n = {len(metric_values)}\nPearson = {pearson_r:.4f}\nSpearman = {spearman_rho:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
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


def print_configuration(
    *,
    args: argparse.Namespace,
    corrupted_root: Path,
    resolved_device: str,
    resolved_dtype: torch.dtype,
    use_4bit: bool,
    seq_lens: Sequence[int],
    bucket_specs: Sequence[ExactBucketSpec],
    final_layer_idx: int,
    score_lookup: Optional[ScoreLookup],
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
    print(f"final_layer_idx={int(final_layer_idx)}")
    print(f"score_csv={args.score_csv}")
    if score_lookup is not None:
        print(
            f"score_delta_column={score_lookup.score_delta_column!r} "
            f"score_csv_usable_rows={int(score_lookup.usable_rows)}"
        )
    if args.max_samples_per_exact_dir is not None:
        print(f"max_samples_per_exact_dir={int(args.max_samples_per_exact_dir)}")
    for bucket_spec in bucket_specs:
        print(
            f"queued seq_len={int(bucket_spec.seq_len)} split={bucket_spec.split} "
            f"samples={len(bucket_spec.sample_dirs)} clean_dir={bucket_spec.clean_dir} "
            f"corrupted_dir={bucket_spec.corrupted_dir}"
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
    bucket_specs = discover_bucket_specs(
        data_root=data_root,
        corrupted_root=corrupted_root,
        seq_lens=seq_lens,
        max_samples_per_exact_dir=args.max_samples_per_exact_dir,
    )
    if not bucket_specs:
        raise RuntimeError("No exact_1 buckets were found.")

    score_lookup = load_score_lookup(args.score_csv)
    runtime, resolved_device, resolved_dtype, use_4bit = build_runtime(
        model_name=str(args.model_name),
        requested_device=str(args.device),
        requested_dtype=str(args.dtype),
        load_in_4bit=bool(args.load_in_4bit),
    )
    lm = LanguageModel(runtime.model, tokenizer=runtime.processor.tokenizer)
    layers = get_decoder_layers(lm)
    final_layer_idx = int(len(layers) - 1)
    final_layer = layers[final_layer_idx]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_csv_path = output_dir / "per_sample_rep_collapse.csv"
    aggregate_csv_path = output_dir / "aggregate_by_seq_len.csv"
    correlation_csv_path = output_dir / "correlation_summary.csv"

    initialize_csv(per_sample_csv_path, PER_SAMPLE_FIELDS)
    write_csv_atomic(aggregate_csv_path, [], AGGREGATE_BY_SEQ_LEN_FIELDS)

    print_configuration(
        args=args,
        corrupted_root=corrupted_root,
        resolved_device=resolved_device,
        resolved_dtype=resolved_dtype,
        use_4bit=use_4bit,
        seq_lens=seq_lens,
        bucket_specs=bucket_specs,
        final_layer_idx=final_layer_idx,
        score_lookup=score_lookup,
    )

    total_samples = sum(len(bucket_spec.sample_dirs) for bucket_spec in bucket_specs)
    if total_samples <= 0:
        raise RuntimeError("No samples remain after applying discovery and per-bucket limits.")

    processed_samples = 0
    skipped_samples = 0
    per_sample_rows: List[Dict[str, Any]] = []

    running_index = 0
    for bucket_spec in bucket_specs:
        for sample_dir in bucket_spec.sample_dirs:
            running_index += 1
            try:
                per_sample_row = process_sample(
                    bucket_spec=bucket_spec,
                    sample_dir=sample_dir,
                    runtime=runtime,
                    lm=lm,
                    final_layer=final_layer,
                    sample_index=running_index,
                    total_samples=total_samples,
                )
            except Exception as exc:
                skipped_samples += 1
                maybe_clear_cuda_cache()
                print(
                    f"[{running_index}/{total_samples}] "
                    f"seq_len={int(bucket_spec.seq_len)} split={bucket_spec.split} "
                    f"evidence_count={int(bucket_spec.evidence_count)} sample_id={sample_dir.name} "
                    f"skipped: unexpected failure ({exc})"
                )
                continue

            if per_sample_row is None:
                skipped_samples += 1
                continue

            per_sample_row["score_delta"] = join_score_delta(per_sample_row, score_lookup)
            per_sample_rows.append(per_sample_row)
            append_csv_rows(per_sample_csv_path, [per_sample_row], PER_SAMPLE_FIELDS)
            processed_samples += 1

    aggregate_rows = build_aggregate_by_seq_len_rows(
        seq_lens=seq_lens,
        per_sample_rows=per_sample_rows,
        n_bootstrap=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    write_csv_atomic(
        aggregate_csv_path,
        aggregate_rows,
        AGGREGATE_BY_SEQ_LEN_FIELDS,
    )

    produced_plot_paths: List[Path] = []
    for metric, output_name, y_label, formula_text in (
        (
            "linf",
            "last_token_linf_final_layer_vs_seq_len.png",
            "Final-layer last-token clean-vs-corrupt L_inf distance",
            "D_inf = ||h_clean^L - h_corrupt^L||_inf",
        ),
        (
            "l2",
            "last_token_l2_final_layer_vs_seq_len.png",
            "Final-layer last-token clean-vs-corrupt L_2 distance",
            "D_2 = ||h_clean^L - h_corrupt^L||_2",
        ),
        (
            "normalized_l2",
            "last_token_normalized_l2_final_layer_vs_seq_len.png",
            "Final-layer last-token clean-vs-corrupt normalized L_2 distance",
            "D_norm2 = ||h_clean^L - h_corrupt^L||_2 / (||h_clean^L||_2 + eps)",
        ),
        (
            "cosine_distance",
            "last_token_cosine_distance_final_layer_vs_seq_len.png",
            "Final-layer last-token clean-vs-corrupt cosine distance",
            "D_cos = 1 - cos(h_clean^L, h_corrupt^L)",
        ),
    ):
        produced_plot_paths.append(
            plot_metric_vs_seq_len(
                aggregate_rows=aggregate_rows,
                metric=metric,
                output_path=output_dir / output_name,
                title="Last-Token Representational Collapse vs Sequence Length (Evidence Count = 1)",
                y_label=y_label,
                formula_text=formula_text,
            )
        )

    if score_lookup is not None:
        write_csv_atomic(
            correlation_csv_path,
            build_correlation_summary_rows(per_sample_rows=per_sample_rows),
            CORRELATION_FIELDS,
        )
        for metric, output_name, y_label in (
            ("linf", "linf_vs_score_delta.png", "L_inf distance"),
            ("cosine_distance", "cosine_distance_vs_score_delta.png", "Cosine distance"),
            ("normalized_l2", "normalized_l2_vs_score_delta.png", "Normalized L_2 distance"),
        ):
            scatter_path = plot_metric_vs_score_delta(
                per_sample_rows=per_sample_rows,
                metric=metric,
                output_path=output_dir / output_name,
                title=f"Final-layer {metric} vs score_delta",
                y_label=y_label,
            )
            if scatter_path is not None:
                produced_plot_paths.append(scatter_path)

    elapsed = time.perf_counter() - start_time
    print(f"processed_samples={processed_samples} skipped_samples={skipped_samples}")
    print(f"per_sample_csv={per_sample_csv_path}")
    print(f"aggregate_by_seq_len_csv={aggregate_csv_path}")
    if score_lookup is not None:
        print(f"correlation_summary_csv={correlation_csv_path}")
        joined_score_rows = sum(1 for row in per_sample_rows if parse_optional_float(row.get("score_delta")) is not None)
        print(f"joined_score_delta_rows={joined_score_rows}")
    for plot_path in produced_plot_paths:
        print(f"plot={plot_path}")
    print(eval_utils.format_runtime(elapsed))


if __name__ == "__main__":
    main()

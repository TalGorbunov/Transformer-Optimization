import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import plots as plot_utils
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from models.model import get_layers, model as base_model, processor


def format_last_token_score_table(layer_rows: List[tuple[int, float]]) -> str:
    if not layer_rows:
        return "<none>"
    header = "layer".ljust(7) + "patched_score".center(16)
    rows = [header]
    for layer_idx, patched_score in layer_rows:
        rows.append(f"{str(layer_idx).ljust(7)}{f'{patched_score:.4f}'.center(16)}")
    return "\n".join(rows)


def load_sample_components(
    sample_dir: Path,
    sample_index: int,
    total_samples: int,
) -> Optional[tuple[str, List[Any], str, List[Dict[str, Any]], str]]:
    try:
        return load_mmred_sample(sample_dir)
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_dir.name} skipped: load failure ({exc})")
        return None


def build_clean_inputs(
    sample_id: str,
    frames: List[Any],
    question: str,
    sample_index: int,
    total_samples: int,
) -> Optional[tuple[Dict[str, Any], int]]:
    try:
        clean_inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, question))
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to build clean inputs ({exc})")
        return None

    prompt_len = int(clean_inputs["input_ids"].shape[1])
    if prompt_len <= 0:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: prompt has no last token")
        return None
    return clean_inputs, prompt_len


def build_answer_token_ids(
    sample_id: str,
    answer_text: str,
    sample_index: int,
    total_samples: int,
) -> Optional[List[int]]:
    try:
        return tgi.token_ids_of_answer(str(answer_text).strip())
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: invalid answer tokenization ({exc})")
        return None


def score_clean_sample(
    sample_id: str,
    frames: List[Any],
    answer_text: str,
    clean_inputs: Dict[str, Any],
    prompt_len: int,
    min_clean_correct_prob: float,
    lm: LanguageModel,
    sample_index: int,
    total_samples: int,
) -> Optional[Dict[str, Any]]:
    try:
        metrics = tgi.score_valid_numeric_answers(
            lm=lm,
            inputs=clean_inputs,
            prompt_len=prompt_len,
            num_frames=len(frames),
        )
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: clean scoring failed ({exc})")
        return None

    a_star_text = str(answer_text).strip()
    clean_answer_score = float(metrics["scores_by_answer"].get(a_star_text, float("-inf")))
    clean_correct_prob = float(metrics["probs_by_answer"].get(a_star_text, 0.0))
    best_answer_text = str(metrics["best_answer_text"])
    clean_top1_correct = (best_answer_text == a_star_text)

    if not clean_top1_correct:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: clean top-1 is {best_answer_text!r}, "
            f"not correct answer {a_star_text!r}"
        )
        return None
    if clean_correct_prob < min_clean_correct_prob:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"clean_correct_prob={clean_correct_prob:.4f} < threshold={min_clean_correct_prob:.4f}"
        )
        return None

    return {
        "clean_answer_score": clean_answer_score,
        "clean_correct_prob": clean_correct_prob,
        "clean_top1_correct": clean_top1_correct,
        "best_answer_text": best_answer_text,
    }


def build_corrupted_inputs(
    sample_id: str,
    frames: List[Any],
    question: str,
    evidence_frame_indices: List[int],
    corrupted_data_root: Path,
    prompt_len: int,
    sample_index: int,
    total_samples: int,
) -> Optional[Dict[str, Any]]:
    corrupted_frames, corruption_issues = eval_utils.build_composite_corrupted_frames(
        sample_id=sample_id,
        clean_frames=frames,
        evidence_frame_indices=evidence_frame_indices,
        corrupted_data_root=corrupted_data_root,
    )
    if corrupted_frames is None:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to build corrupted frames "
            f"(issues={corruption_issues})"
        )
        return None

    try:
        corrupted_inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(corrupted_frames, question))
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to build corrupted inputs ({exc})")
        return None
    if int(corrupted_inputs["input_ids"].shape[1]) != prompt_len:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: seq_len mismatch "
            f"(clean={prompt_len}, corrupted={int(corrupted_inputs['input_ids'].shape[1])})"
        )
        return None
    return corrupted_inputs


def score_corrupted_sample(
    sample_id: str,
    clean_inputs: Dict[str, Any],
    corrupted_inputs: Dict[str, Any],
    answer_token_ids: List[int],
    prompt_len: int,
    lm: LanguageModel,
    sample_index: int,
    total_samples: int,
) -> Optional[Dict[str, Any]]:
    clean_answer_inputs = tgi.append_answer_tokens_for_scoring(clean_inputs, answer_token_ids)
    corrupted_answer_inputs = tgi.append_answer_tokens_for_scoring(corrupted_inputs, answer_token_ids)
    try:
        corrupted_answer_score = tgi.run_clean_sequence_logprob(
            lm=lm,
            scoring_inputs=corrupted_answer_inputs,
            prompt_len=prompt_len,
            answer_token_ids=answer_token_ids,
        )
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to score corrupted input ({exc})")
        return None

    return {
        "clean_answer_inputs": clean_answer_inputs,
        "corrupted_answer_inputs": corrupted_answer_inputs,
        "corrupted_answer_score": float(corrupted_answer_score),
    }


def compute_last_token_layer_metrics(
    lm: LanguageModel,
    layers: Any,
    selected_layers: List[int],
    corrupted_answer_inputs: Dict[str, Any],
    clean_answer_inputs: Dict[str, Any],
    corrupted_answer_score: float,
    prompt_len: int,
    answer_token_ids: List[int],
) -> tuple[List[Dict[str, Any]], List[tuple[int, float]]]:
    per_layer_metrics: List[Dict[str, Any]] = []
    patched_score_rows: List[tuple[int, float]] = []
    for layer_idx in selected_layers:
        try:
            patched_score = tgi.run_layer_corrupted_sequence_logprob(
                lm=lm,
                layers=layers,
                clean_scoring_inputs=corrupted_answer_inputs,
                control_scoring_inputs=clean_answer_inputs,
                layer_idx=layer_idx,
                clean_token_positions=[-1],
                control_token_positions=[-1],
                prompt_len=prompt_len,
                answer_token_ids=answer_token_ids,
            )
        except Exception as exc:
            print(f"  layer={layer_idx} failed corruption forward ({exc}); using corrupted score")
            patched_score = corrupted_answer_score

        signed_delta = float(patched_score - corrupted_answer_score)
        importance = max(signed_delta, 0.0)
        patched_score_rows.append((layer_idx, patched_score))
        per_layer_metrics.append({
            "layer": int(layer_idx),
            "patched_score": patched_score,
            "corrupted_score": corrupted_answer_score,
            "signed_delta": signed_delta,
            "importance": importance,
        })
    return per_layer_metrics, patched_score_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Measure last-token importance by corrupting all evidence frames in the input, "
            "patching the last prompt token at each layer, and scoring the clean answer."
        )
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--corrupted_root",
        type=str,
        default=None,
        help=(
            "Root directory for corrupted samples. If omitted, inferred from --data_root "
            "(e.g., .../mmred_images/... -> .../mmred_corrupted/...)."
        ),
    )
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--output", type=str, default="outputs")
    ap.add_argument("--batch_size", type=int, default=1, help="Accepted for interface parity; unused here.")
    ap.add_argument(
        "--min_clean_correct_prob",
        type=float,
        default=0.4,
        help="Keep a sample only if the clean correct answer probability among valid numeric answers is at least this value.",
    )
    ap.add_argument("--lambda", dest="lambda_threshold", type=float, default=None)
    ap.add_argument(
        "--min_clean_ld",
        type=float,
        default=None,
        help="Backward-compatible alias for --min_clean_correct_prob.",
    )
    ap.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Optional layer selection. Examples: --layers 32:42, --layers 0:64:2, "
            "--layers 30,32,34,36,38,40"
        ),
    )
    ap.add_argument("--disable_plots", action="store_true")
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer")
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer")
    if args.min_clean_correct_prob < 0.0 or args.min_clean_correct_prob > 1.0:
        raise ValueError("--min_clean_correct_prob must be in [0, 1]")

    min_clean_correct_prob = float(args.min_clean_correct_prob)
    if args.lambda_threshold is not None:
        min_clean_correct_prob = float(args.lambda_threshold)
    elif args.min_clean_ld is not None:
        min_clean_correct_prob = float(args.min_clean_ld)
    args.min_clean_correct_prob = min_clean_correct_prob
    return args


def process_sample(
    sample_dir: Path,
    sample_index: int,
    total_samples: int,
    lm: LanguageModel,
    layers: Any,
    selected_layers: List[int],
    corrupted_data_root: Path,
    min_clean_correct_prob: float,
) -> Optional[Dict[str, Any]]:
    sample_components = load_sample_components(sample_dir, sample_index, total_samples)
    if sample_components is None:
        return None
    sample_id, frames, question, states, answer = sample_components

    evidence_frame_indices = eval_utils.collect_evidence_frame_indices(question, states)
    if len(evidence_frame_indices) < 1:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: no evidence frames")
        return None

    clean_input_result = build_clean_inputs(
        sample_id=sample_id,
        frames=frames,
        question=question,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if clean_input_result is None:
        return None
    clean_inputs, prompt_len = clean_input_result

    a_star_text = str(answer).strip()
    a_star_ids = build_answer_token_ids(sample_id, a_star_text, sample_index, total_samples)
    if a_star_ids is None:
        return None

    clean_answer_metrics = score_clean_sample(
        sample_id=sample_id,
        frames=frames,
        answer_text=a_star_text,
        clean_inputs=clean_inputs,
        prompt_len=prompt_len,
        min_clean_correct_prob=min_clean_correct_prob,
        lm=lm,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if clean_answer_metrics is None:
        return None
    clean_answer_score = float(clean_answer_metrics["clean_answer_score"])
    clean_correct_prob = float(clean_answer_metrics["clean_correct_prob"])
    clean_top1_correct = bool(clean_answer_metrics["clean_top1_correct"])
    best_answer_text = str(clean_answer_metrics["best_answer_text"])

    corrupted_inputs = build_corrupted_inputs(
        sample_id=sample_id,
        frames=frames,
        question=question,
        evidence_frame_indices=evidence_frame_indices,
        corrupted_data_root=corrupted_data_root,
        prompt_len=prompt_len,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if corrupted_inputs is None:
        return None

    corrupted_score_result = score_corrupted_sample(
        sample_id=sample_id,
        clean_inputs=clean_inputs,
        corrupted_inputs=corrupted_inputs,
        answer_token_ids=a_star_ids,
        prompt_len=prompt_len,
        lm=lm,
        sample_index=sample_index,
        total_samples=total_samples,
    )
    if corrupted_score_result is None:
        return None
    clean_answer_inputs = corrupted_score_result["clean_answer_inputs"]
    corrupted_answer_inputs = corrupted_score_result["corrupted_answer_inputs"]
    corrupted_answer_score = float(corrupted_score_result["corrupted_answer_score"])

    print(
        f"[{sample_index}/{total_samples}] sample_id={sample_id} clean_answer_score={clean_answer_score:.4f} "
        f"corrupted_answer_score={corrupted_answer_score:.4f} "
        f"clean_correct_prob={clean_correct_prob:.4f} evidence_frames={evidence_frame_indices} "
        "patched_token_position=-1"
    )

    per_layer_metrics, patched_score_rows = compute_last_token_layer_metrics(
        lm=lm,
        layers=layers,
        selected_layers=selected_layers,
        corrupted_answer_inputs=corrupted_answer_inputs,
        clean_answer_inputs=clean_answer_inputs,
        corrupted_answer_score=corrupted_answer_score,
        prompt_len=prompt_len,
        answer_token_ids=a_star_ids,
    )

    if patched_score_rows:
        print("  Patched score table (rows=layers):")
        print(format_last_token_score_table(patched_score_rows))

    return {
        "sample_id": sample_id,
        "question": question,
        "answer": answer,
        "clean_answer_score": clean_answer_score,
        "corrupted_answer_score": corrupted_answer_score,
        "clean_correct_prob": clean_correct_prob,
        "clean_top1_correct": clean_top1_correct,
        "a_star_text": a_star_text,
        "a_star_ids": a_star_ids,
        "evidence_frames": [int(frame_idx) for frame_idx in evidence_frame_indices],
        "patched_token_position": -1,
        "selected_layers": list(selected_layers),
        "layer_metrics": {"layers": per_layer_metrics},
    }


def finalize_outputs(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    selected_layers: List[int],
    seq_len_label: Optional[str],
    min_clean_correct_prob: float,
    args: argparse.Namespace,
) -> None:
    print(
        f"Processed {len(sample_metrics)} samples "
        f"(target limit={int(args.limit)}, min_clean_correct_prob={min_clean_correct_prob:.4f})."
    )

    if not args.disable_plots:
        lines_path = plot_utils.plot_last_token_importance_lines(
            sample_metrics,
            output_dir,
            selected_layers=selected_layers,
            seq_len_label=seq_len_label,
            title_override="Importance of restoring only the last token on a FULLY corrupted input",
        )
        if lines_path is not None:
            print(f"Wrote last-token importance plot to: {lines_path}")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    min_clean_correct_prob = float(args.min_clean_correct_prob)

    data_root = Path(args.data_root)
    corrupted_data_root = (
        Path(args.corrupted_root) if args.corrupted_root is not None else eval_utils.infer_corrupted_data_root(data_root)
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_len_label = eval_utils.resolve_seq_len_label(data_root)

    lm = LanguageModel(base_model, tokenizer=processor.tokenizer)
    layers = get_layers(lm.model)
    num_layers = len(layers)
    selected_layers = tgi.parse_layer_selection(args.layers, num_layers=num_layers)

    sample_dirs = iter_sample_dirs(data_root)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under: {data_root}")

    processed_samples = 0
    sample_metrics: List[Dict[str, Any]] = []
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= int(args.limit):
            break
        sample_metrics_row = process_sample(
            sample_dir=sample_dir,
            sample_index=idx,
            total_samples=len(sample_dirs),
            lm=lm,
            layers=layers,
            selected_layers=selected_layers,
            corrupted_data_root=corrupted_data_root,
            min_clean_correct_prob=min_clean_correct_prob,
        )
        if sample_metrics_row is None:
            continue
        sample_metrics_row["selected_layers_spec"] = args.layers
        sample_metrics_row["control_debug"] = {
            "control_type": "all_evidence_frames_corrupted",
            "corrupted_root": str(corrupted_data_root),
        }
        sample_metrics.append(sample_metrics_row)
        processed_samples += 1

    finalize_outputs(
        sample_metrics=sample_metrics,
        output_dir=output_dir,
        selected_layers=selected_layers,
        seq_len_label=seq_len_label,
        min_clean_correct_prob=min_clean_correct_prob,
        args=args,
    )

    elapsed = time.perf_counter() - start_time
    print(eval_utils.format_runtime(elapsed))


if __name__ == "__main__":
    main()

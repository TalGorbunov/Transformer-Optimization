import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations import frame_text_group_importance as ftgi
from evaluations import patching_core as tgi
from evaluations import utils as eval_utils
from evaluations.utils import iter_sample_dirs, load_mmred_sample
from models.model import get_layers, model as base_model, processor


def write_sample_report(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    lines: List[str] = []
    for sample in sample_metrics:
        lines.append(f"sample_id={sample['sample_id']}")
        lines.append(f"clean_answer_score={sample['clean_answer_score']:.8f}")
        lines.append(f"corrupted_answer_score={sample['corrupted_answer_score']:.8f}")
        lines.append(f"clean_correct_prob={sample['clean_correct_prob']:.8f}")
        lines.append(f"evidence_frames={sample['evidence_frames']}")
        lines.append(f"last_token_position={sample['last_token_position']}")
        for layer_metrics in sample["layer_metrics"]["layers"]:
            lines.append(
                f"layer={layer_metrics['layer']} "
                f"patched_score={float(layer_metrics['patched_score'][0]):.8f} "
                f"signed_delta={float(layer_metrics['signed_delta'][0]):.8f} "
                f"r={float(layer_metrics['r'][0]):.8f} "
                f"total_importance={float(layer_metrics['total_importance']):.8f}"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_metrics.txt"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


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
    ap.add_argument(
        "--clean_ld_cache_dir",
        type=str,
        default=None,
        help="Directory containing clean_scores.json for loading/updating the clean-answer cache. Defaults to --output.",
    )
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
    clean_score_cache: Dict[str, Dict[str, Any]],
    min_clean_correct_prob: float,
) -> tuple[Optional[Dict[str, Any]], int]:
    # Reuse the same clean-answer gating as the broader patching experiments.
    try:
        sample_id, frames, question, states, answer = load_mmred_sample(sample_dir)
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_dir.name} skipped: load failure ({exc})")
        return None, 0

    evidence_frame_indices = ftgi.collect_evidence_frame_indices(question, states)
    if len(evidence_frame_indices) < 1:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: no evidence frames")
        return None, 0

    try:
        clean_inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, question))
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to build clean inputs ({exc})")
        return None, 0

    prompt_len = int(clean_inputs["input_ids"].shape[1])
    last_token_position = prompt_len - 1
    if last_token_position < 0:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: prompt has no last token")
        return None, 0

    a_star_text = str(answer).strip()
    try:
        a_star_ids = tgi.token_ids_of_answer(a_star_text)
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: invalid answer tokenization ({exc})")
        return None, 0

    try:
        clean_answer_metrics, cache_was_updated = eval_utils.get_or_compute_clean_answer_metrics(
            cache=clean_score_cache,
            sample_id=sample_id,
            num_frames=len(frames),
            answer_text=a_star_text,
            score_fn=lambda: tgi.score_valid_numeric_answers(
                lm=lm,
                inputs=clean_inputs,
                prompt_len=prompt_len,
                num_frames=len(frames),
            ),
        )
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: clean scoring failed ({exc})")
        return None, 0

    clean_answer_score = float(clean_answer_metrics["clean_answer_score"])
    clean_correct_prob = float(clean_answer_metrics["clean_correct_prob"])
    clean_top1_correct = bool(clean_answer_metrics["clean_top1_correct"])
    best_answer_text = str(clean_answer_metrics["best_answer_text"])
    if not clean_top1_correct:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: clean top-1 is {best_answer_text!r}, "
            f"not correct answer {a_star_text!r}"
        )
        return None, int(cache_was_updated)
    if clean_correct_prob < min_clean_correct_prob:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"clean_correct_prob={clean_correct_prob:.4f} < threshold={min_clean_correct_prob:.4f}"
        )
        return None, int(cache_was_updated)

    corrupted_frames, corruption_issues = ftgi.build_composite_corrupted_frames(
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
        return None, int(cache_was_updated)

    try:
        corrupted_inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(corrupted_frames, question))
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to build corrupted inputs ({exc})")
        return None, int(cache_was_updated)
    if int(corrupted_inputs["input_ids"].shape[1]) != prompt_len:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: seq_len mismatch "
            f"(clean={prompt_len}, corrupted={int(corrupted_inputs['input_ids'].shape[1])})"
        )
        return None, int(cache_was_updated)

    clean_answer_inputs = tgi.append_answer_tokens_for_scoring(clean_inputs, a_star_ids)
    corrupted_answer_inputs = tgi.append_answer_tokens_for_scoring(corrupted_inputs, a_star_ids)
    try:
        corrupted_answer_score = tgi.run_clean_sequence_logprob(
            lm=lm,
            scoring_inputs=corrupted_answer_inputs,
            prompt_len=prompt_len,
            answer_token_ids=a_star_ids,
        )
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to score corrupted input ({exc})")
        return None, int(cache_was_updated)

    print(
        f"[{sample_index}/{total_samples}] sample_id={sample_id} clean_answer_score={clean_answer_score:.4f} "
        f"corrupted_answer_score={corrupted_answer_score:.4f} "
        f"clean_correct_prob={clean_correct_prob:.4f} evidence_frames={evidence_frame_indices} "
        f"last_token_position={last_token_position}"
    )

    per_layer_metrics: List[Dict[str, Any]] = []
    patched_score_rows: List[tuple[int, List[float]]] = []
    for layer_idx in selected_layers:
        try:
            patched_scores = tgi.run_layer_multi_group_corrupted_sequence_logprob(
                lm=lm,
                layers=layers,
                clean_batched_scoring_inputs=corrupted_answer_inputs,
                control_batched_scoring_inputs=clean_answer_inputs,
                layer_idx=layer_idx,
                clean_token_positions_by_batch=[[int(last_token_position)]],
                control_token_positions_by_batch=[[int(last_token_position)]],
                prompt_len=prompt_len,
                answer_token_ids=a_star_ids,
            )
            patched_score = float(patched_scores[0].item())
        except Exception as exc:
            print(f"  layer={layer_idx} failed corruption forward ({exc}); using corrupted score")
            patched_score = corrupted_answer_score

        signed_delta = float(patched_score - corrupted_answer_score)
        importance = max(signed_delta, 0.0)
        patched_score_rows.append((layer_idx, [patched_score]))
        probs = [1.0] if importance > 0.0 else [0.0]
        entropy_value = (
            tgi.normalize_entropy(tgi.entropy_from_probabilities(probs), num_groups=1)
            if importance > 0.0 else None
        )
        per_layer_metrics.append({
            "layer": int(layer_idx),
            "groups": ["last_token"],
            "patched_score": [patched_score],
            "corrupted_score": [corrupted_answer_score],
            "signed_delta": [signed_delta],
            "r": [importance],
            "r_normalized": [importance],
            "p": probs,
            "entropy": entropy_value,
            "total_importance": importance,
        })

    if patched_score_rows:
        print("  Patched score table (rows=layers, columns=groups):")
        print(tgi.format_corrupted_score_table(["last_token"], patched_score_rows))

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
        "last_token_position": int(last_token_position),
        "active_groups": ["last_token"],
        "group_token_counts": {"last_token": 1},
        "selected_layers": list(selected_layers),
        "layer_metrics": {"layers": per_layer_metrics},
    }, int(cache_was_updated)


def finalize_outputs(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    data_root: Path,
    corrupted_data_root: Path,
    selected_layers: List[int],
    num_layers: int,
    seq_len_label: Optional[str],
    min_clean_correct_prob: float,
    args: argparse.Namespace,
) -> None:
    text_report_path = write_sample_report(sample_metrics, output_dir)
    sample_json_path = tgi.write_metrics_json(sample_metrics, output_dir)
    aggregate_metrics = ftgi.build_aggregate_metrics(
        sample_metrics,
        num_layers=num_layers,
        selected_layers=selected_layers,
        group_order=["last_token"],
    )
    aggregate_payload = {
        "metadata": {
            "model_name": getattr(base_model.config, "_name_or_path", str(type(base_model).__name__)),
            "dataset_path": str(data_root),
            "corrupted_root": str(corrupted_data_root),
            "selected_layers": list(selected_layers),
            "selected_layers_spec": args.layers,
            "total_layer_count": int(num_layers),
            "corruption_method": "all_evidence_frames_corrupted_last_token_patch",
            "group_names": ["last_token"],
        },
        "aggregate": aggregate_metrics,
    }
    aggregate_json_path = ftgi.write_aggregate_metrics_json(aggregate_payload, output_dir)
    print(f"Wrote sample metrics text report to: {text_report_path}")
    print(f"Wrote sample metrics JSON to: {sample_json_path}")
    print(f"Wrote aggregate metrics JSON to: {aggregate_json_path}")
    print(
        f"Processed {len(sample_metrics)} samples "
        f"(target limit={int(args.limit)}, min_clean_correct_prob={min_clean_correct_prob:.4f})."
    )
    tgi.print_group_summary(["last_token"], sample_metrics)

    if not args.disable_plots:
        lines_path = ftgi.plot_group_importance_lines(
            sample_metrics,
            output_dir,
            num_layers=num_layers,
            selected_layers=selected_layers,
            group_order=["last_token"],
            seq_len_label=seq_len_label,
        )
        if lines_path is not None:
            print(f"Wrote group-importance lines plot to: {lines_path}")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    data_root = Path(args.data_root)
    corrupted_data_root = (
        Path(args.corrupted_root) if args.corrupted_root is not None else ftgi.infer_corrupted_data_root(data_root)
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_ld_cache_dir = Path(args.clean_ld_cache_dir) if args.clean_ld_cache_dir is not None else output_dir
    clean_ld_cache_dir.mkdir(parents=True, exist_ok=True)
    clean_ld_cache_path = clean_ld_cache_dir / "clean_scores.json"
    clean_ld_cache = tgi.load_clean_score_cache(clean_ld_cache_path)
    cache_updates = 0

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
        sample_metrics_row, cache_delta = process_sample(
            sample_dir=sample_dir,
            sample_index=idx,
            total_samples=len(sample_dirs),
            lm=lm,
            layers=layers,
            selected_layers=selected_layers,
            corrupted_data_root=corrupted_data_root,
            clean_score_cache=clean_ld_cache,
            min_clean_correct_prob=min_clean_correct_prob,
        )
        cache_updates += cache_delta
        if sample_metrics_row is None:
            continue
        sample_metrics_row["selected_layers_spec"] = args.layers
        sample_metrics_row["control_debug"] = {
            "control_type": "all_evidence_frames_corrupted",
            "corrupted_root": str(corrupted_data_root),
        }
        sample_metrics.append(sample_metrics_row)
        processed_samples += 1

    print(eval_utils.persist_clean_score_cache(clean_ld_cache_path, clean_ld_cache, cache_updates))
    finalize_outputs(
        sample_metrics=sample_metrics,
        output_dir=output_dir,
        data_root=data_root,
        corrupted_data_root=corrupted_data_root,
        selected_layers=selected_layers,
        num_layers=num_layers,
        seq_len_label=seq_len_label,
        min_clean_correct_prob=min_clean_correct_prob,
        args=args,
    )

    elapsed = time.perf_counter() - start_time
    print(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()

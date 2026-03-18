"""
Scan sliding windows over non-image prompt-side tokens to find missed late-layer carrier tokens.

This script reuses the matched-control activation patching setup and full-answer log-prob
score from the text-group importance experiment, but replaces named text groups with
sliding windows over the last K non-image prompt-side tokens, including the assistant
prefix when present.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import text_controls as tc
from evaluations.helpers import utils as eval_utils
from evaluations.scripts import frame_text_group_importance as ftgi


def parse_layer_selection(raw: Optional[str], num_layers: int) -> List[int]:
    return eval_utils.parse_layer_selection(raw, num_layers)


def locate_non_image_prompt_token_positions(
    inputs: Dict[str, Any],
    question: str,
    num_frames: int,
) -> Tuple[List[int], List[str]]:
    prompt = tgi.build_prompt(question, num_frames=num_frames)
    prompt_token_ids, _ = tc.tokenize_with_offsets_if_available(prompt)
    if not prompt_token_ids:
        return [], ["prompt_tokenization_failed"]

    full_input_ids = [int(tok) for tok in inputs["input_ids"][0].tolist()]
    prompt_start_in_full = tc.find_subsequence(full_input_ids, prompt_token_ids)
    if prompt_start_in_full is None:
        return [], ["prompt_subsequence_not_found_in_input_ids"]

    prompt_positions = list(range(prompt_start_in_full, prompt_start_in_full + len(prompt_token_ids)))

    token_positions_full, _, group_warnings = tc.locate_group_token_positions(
        inputs=inputs,
        question=question,
        num_frames=num_frames,
    )
    assistant_prefix_positions = [int(pos) for pos in token_positions_full.get("assistant_prefix", [])]

    ordered_positions: List[int] = []
    seen_positions = set()
    for position in list(prompt_positions) + assistant_prefix_positions:
        if position not in seen_positions:
            seen_positions.add(position)
            ordered_positions.append(int(position))

    warnings = list(group_warnings)
    if not assistant_prefix_positions:
        warnings.append("assistant_prefix:not_found_for_sliding_scan")
    return ordered_positions, warnings


def build_sliding_windows(
    full_input_ids: Sequence[int],
    prompt_positions: Sequence[int],
    tail_non_image_tokens: int,
    window_size: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    prompt_positions_list = [int(pos) for pos in prompt_positions]
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if tail_non_image_tokens <= 0:
        raise ValueError("tail_non_image_tokens must be positive")

    effective_tail = min(int(tail_non_image_tokens), len(prompt_positions_list))
    if effective_tail < len(prompt_positions_list):
        tail_positions = prompt_positions_list[-effective_tail:]
    else:
        tail_positions = prompt_positions_list

    if not tail_positions:
        return [], ["tail_non_image_prompt_tokens:empty"]
    if len(tail_positions) < window_size:
        warnings.append(
            f"tail_non_image_prompt_tokens:too_short_for_window(window_size={window_size},tail={len(tail_positions)})"
        )
        return [], warnings

    windows: List[Dict[str, Any]] = []
    for start_idx in range(len(tail_positions) - window_size + 1):
        positions = tail_positions[start_idx:start_idx + window_size]
        token_ids = [int(full_input_ids[pos]) for pos in positions]
        decoded_tokens = [
            tgi.processor.tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
            for tok_id in token_ids
        ]
        windows.append({
            "name": f"window_{start_idx}",
            "window_index": start_idx,
            "raw_positions": positions,
            "local_tail_positions": list(range(start_idx, start_idx + window_size)),
            "token_ids": token_ids,
            "decoded_tokens": decoded_tokens,
        })

    return windows, warnings


def global_window_names(tail_non_image_tokens: int, window_size: int) -> List[str]:
    if tail_non_image_tokens <= 0 or window_size <= 0:
        return []
    count = max(tail_non_image_tokens - window_size + 1, 0)
    return [f"window_{idx}" for idx in range(count)]


def write_sample_report(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    lines: List[str] = []
    for sample in sample_metrics:
        lines.append(f"sample_id={sample['sample_id']}")
        lines.append(f"question={sample['question']}")
        lines.append(f"answer={sample['answer']}")
        lines.append(
            f"clean_answer_score={sample['clean_answer_score']:.8f} "
            f"clean_correct_prob={sample['clean_correct_prob']:.8f} "
            f"clean_top1_correct={sample['clean_top1_correct']}"
        )
        lines.append(f"active_groups={sample['active_groups']}")
        lines.append(f"group_token_counts={sample['group_token_counts']}")
        if sample["skipped_groups"]:
            lines.append(f"skipped_groups={sample['skipped_groups']}")
        for layer_metrics in sample["layer_metrics"]["layers"]:
            lines.append(
                f"layer={layer_metrics['layer']} "
                f"groups={layer_metrics['groups']} "
                f"r={[round(float(x), 8) for x in layer_metrics['r']]} "
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
            "Measure late-layer answer-relevant information in sliding windows over non-image "
            "prompt tokens using matched-control activation patching and full-answer log-prob scoring."
        )
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--output", type=str, default="outputs")
    ap.add_argument(
        "--clean_score_cache_dir",
        type=str,
        default=None,
        help="Directory containing clean_scores.json for loading/updating clean-answer filter cache.",
    )
    ap.add_argument(
        "--clean_ld_cache_dir",
        type=str,
        default=None,
        help="Backward-compatible alias for --clean_score_cache_dir.",
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--min_clean_correct_prob",
        type=float,
        default=0.4,
        help="Keep a sample only if the clean correct answer probability among valid numeric answers is at least this value.",
    )
    ap.add_argument(
        "--min_clean_correct_logprob",
        type=float,
        default=None,
        help="Optional fallback filter: require clean correct-answer full-sequence log-prob >= this value.",
    )
    ap.add_argument("--disable_plots", action="store_true")
    ap.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Optional layer selection. Examples: --layers 32:42, --layers 0:42:2, "
            "--layers 30,32,34,36,38,40"
        ),
    )
    ap.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="Number of consecutive non-image prompt tokens per sliding window.",
    )
    ap.add_argument(
        "--tail-non-image-tokens",
        type=int,
        default=20,
        help="Scan only over the last K non-image prompt tokens.",
    )
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer")
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer")
    if args.min_clean_correct_prob < 0.0 or args.min_clean_correct_prob > 1.0:
        raise ValueError("--min_clean_correct_prob must be in [0, 1]")
    if args.window_size <= 0:
        raise ValueError("--window-size must be a positive integer")
    if args.tail_non_image_tokens <= 0:
        raise ValueError("--tail-non-image-tokens must be a positive integer")
    return args


def summarize_window_names(plot_window_names: List[str], sample_metrics: List[Dict[str, Any]]) -> List[str]:
    if plot_window_names:
        return plot_window_names
    seen_window_names: List[str] = []
    for sample in sample_metrics:
        for group_name in sample["active_groups"]:
            if group_name not in seen_window_names:
                seen_window_names.append(group_name)
    return seen_window_names


def process_sample(
    sample_dir: Path,
    sample_index: int,
    total_samples: int,
    args: argparse.Namespace,
    lm: LanguageModel,
    layers: Any,
    selected_layers: List[int],
    clean_score_cache: Dict[str, Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], int]:
    # Keep the same clean-answer filtering logic as the text-group experiment.
    try:
        sample_id, frames, question, states, answer = tgi.load_mmred_sample(sample_dir)
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_dir.name} skipped: load failure ({exc})")
        return None, 0

    if tc.parse_target_character_room(question) is None:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: could not parse question")
        return None, 0

    try:
        inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, question))
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to build clean inputs ({exc})")
        return None, 0

    prompt_len = int(inputs["input_ids"].shape[1])
    a_star_text = str(answer).strip()
    try:
        a_star_ids = tgi.token_ids_of_answer(a_star_text)
        clean_answer_metrics, cache_was_updated = eval_utils.get_or_compute_clean_answer_metrics(
            cache=clean_score_cache,
            sample_id=sample_id,
            num_frames=len(frames),
            answer_text=a_star_text,
            score_fn=lambda: tgi.score_valid_numeric_answers(
                lm=lm,
                inputs=inputs,
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
    if clean_correct_prob < args.min_clean_correct_prob:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"clean_correct_prob={clean_correct_prob:.4f} < threshold={args.min_clean_correct_prob:.4f}"
        )
        return None, int(cache_was_updated)
    if args.min_clean_correct_logprob is not None and clean_answer_score < args.min_clean_correct_logprob:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"clean_answer_score={clean_answer_score:.4f} < threshold={args.min_clean_correct_logprob:.4f}"
        )
        return None, int(cache_was_updated)

    clean_group_positions, clean_group_summaries, clean_group_warnings = tc.locate_group_token_positions(
        inputs=inputs,
        question=question,
        num_frames=len(frames),
    )
    clean_prompt_positions, clean_prompt_warnings = locate_non_image_prompt_token_positions(
        inputs=inputs,
        question=question,
        num_frames=len(frames),
    )
    if not clean_prompt_positions:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: no non-image prompt token positions "
            f"(warnings={clean_prompt_warnings})"
        )
        return None, int(cache_was_updated)

    control_inputs, control_question, control_group_positions, control_group_summaries, control_skip_info, control_reason = tc.choose_best_control(
        frames=frames,
        states=states,
        question=question,
        clean_inputs=inputs,
        clean_group_positions=clean_group_positions,
        include_groups=tc.parse_include_groups(",".join(tc.ALL_GROUPS)),
    )
    if control_inputs is None or control_question is None or control_reason is None:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to find aligned control "
            f"(details={control_skip_info})"
        )
        return None, int(cache_was_updated)

    control_prompt_positions, control_prompt_warnings = locate_non_image_prompt_token_positions(
        inputs=control_inputs,
        question=control_question,
        num_frames=len(frames),
    )
    if not control_prompt_positions or len(clean_prompt_positions) != len(control_prompt_positions):
        return None, int(cache_was_updated)

    clean_windows, clean_window_warnings = build_sliding_windows(
        full_input_ids=[int(tok) for tok in inputs["input_ids"][0].tolist()],
        prompt_positions=clean_prompt_positions,
        tail_non_image_tokens=args.tail_non_image_tokens,
        window_size=args.window_size,
    )
    control_windows, control_window_warnings = build_sliding_windows(
        full_input_ids=[int(tok) for tok in control_inputs["input_ids"][0].tolist()],
        prompt_positions=control_prompt_positions,
        tail_non_image_tokens=args.tail_non_image_tokens,
        window_size=args.window_size,
    )
    if not clean_windows or len(clean_windows) != len(control_windows):
        return None, int(cache_was_updated)

    groups_payload: List[Dict[str, Any]] = []
    skipped_groups: Dict[str, str] = {}
    group_summaries: Dict[str, Dict[str, str]] = {}
    group_token_counts: Dict[str, int] = {}
    window_metadata: List[Dict[str, Any]] = []
    control_windows_by_name = {window["name"]: window for window in control_windows}
    for clean_window in clean_windows:
        group_name = str(clean_window["name"])
        control_window = control_windows_by_name.get(group_name)
        if control_window is None:
            skipped_groups[group_name] = "missing_control_window"
            continue
        clean_positions = [int(pos) for pos in clean_window["raw_positions"]]
        control_positions = [int(pos) for pos in control_window["raw_positions"]]
        if len(clean_positions) != len(control_positions):
            skipped_groups[group_name] = f"token_count_mismatch(clean={len(clean_positions)},control={len(control_positions)})"
            continue
        group_token_counts[group_name] = len(clean_positions)
        group_summaries[group_name] = {
            "clean": tc._normalize_summary_text("".join(clean_window["decoded_tokens"])),
            "control": tc._normalize_summary_text("".join(control_window["decoded_tokens"])),
            "token_count": str(len(clean_positions)),
        }
        groups_payload.append({
            "name": group_name,
            "clean_positions": clean_positions,
            "control_positions": control_positions,
            "control_inputs": control_inputs,
        })
        window_metadata.append({
            "name": group_name,
            "window_index": int(clean_window["window_index"]),
            "raw_positions": clean_positions,
            "local_tail_positions": [int(pos) for pos in clean_window["local_tail_positions"]],
            "clean_token_ids": [int(tok) for tok in clean_window["token_ids"]],
            "clean_decoded_tokens": list(clean_window["decoded_tokens"]),
            "control_token_ids": [int(tok) for tok in control_window["token_ids"]],
            "control_decoded_tokens": list(control_window["decoded_tokens"]),
        })
    if not groups_payload:
        return None, int(cache_was_updated)

    try:
        chunk_data = eval_utils.build_group_patch_batches(
            groups_payload=groups_payload,
            batch_size=args.batch_size,
            clean_inputs=inputs,
            answer_token_ids=a_star_ids,
            repeat_inputs_for_batch=tgi.repeat_inputs_for_batch,
            concatenate_inputs_for_batch=tgi.concatenate_inputs_for_batch,
            append_answer_tokens_for_scoring=tgi.append_answer_tokens_for_scoring,
        )
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: failed to build batched inputs ({exc})")
        return None, int(cache_was_updated)

    per_layer_metrics, all_layer_corrupted_rows = eval_utils.run_group_patch_layer_sweep(
        selected_layers=selected_layers,
        groups_payload=groups_payload,
        chunk_data=chunk_data,
        clean_answer_score=clean_answer_score,
        lm=lm,
        layers=layers,
        prompt_len=prompt_len,
        answer_token_ids=a_star_ids,
        run_layer_patch=tgi.run_layer_multi_group_corrupted_sequence_logprob,
        normalize_to_probabilities=tgi.normalize_to_probabilities,
        entropy_from_probabilities=tgi.entropy_from_probabilities,
        normalize_entropy=tgi.normalize_entropy,
        logger=print,
        include_signed_delta=True,
        normalize_by_token_count=False,
    )
    if all_layer_corrupted_rows:
        print("  Corrupted score table (rows=layers, columns=sliding windows):")
        print(tgi.format_corrupted_score_table([group["name"] for group in groups_payload], all_layer_corrupted_rows))

    return {
        "sample_id": sample_id,
        "answer": answer,
        "question": question,
        "control_question": control_question,
        "control_character": control_reason["control_character"],
        "control_room": control_reason["control_room"],
        "control_answer": control_reason["control_answer"],
        "clean_answer_score": clean_answer_score,
        "clean_correct_prob": clean_correct_prob,
        "clean_top1_correct": clean_top1_correct,
        "best_answer_text": best_answer_text,
        "a_star_text": a_star_text,
        "a_star_ids": a_star_ids,
        "active_groups": [group["name"] for group in groups_payload],
        "group_token_counts": group_token_counts,
        "skipped_groups": skipped_groups,
        "group_summaries": group_summaries,
        "clean_group_warnings": clean_group_warnings,
        "control_group_info": control_skip_info,
        "control_reason": control_reason,
        "window_config": {
            "window_size": int(args.window_size),
            "tail_non_image_tokens": int(args.tail_non_image_tokens),
            "available_non_image_prompt_tokens": len(clean_prompt_positions),
            "scanned_tail_token_count": min(int(args.tail_non_image_tokens), len(clean_prompt_positions)),
        },
        "window_metadata": window_metadata,
        "clean_prompt_warnings": clean_prompt_warnings,
        "control_prompt_warnings": control_prompt_warnings,
        "clean_window_warnings": clean_window_warnings,
        "control_window_warnings": control_window_warnings,
        "layer_metrics": {"layers": per_layer_metrics},
    }, int(cache_was_updated)


def finalize_outputs(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    num_layers: int,
    selected_layers: List[int],
    seq_len_label: Optional[str],
    plot_window_names: List[str],
    args: argparse.Namespace,
) -> None:
    text_report_path = write_sample_report(sample_metrics, output_dir)
    json_path = tgi.write_metrics_json(sample_metrics, output_dir)
    print(f"Wrote sample metrics text report to: {text_report_path}")
    print(f"Wrote sample metrics JSON to: {json_path}")
    print(f"Processed {len(sample_metrics)} samples (target limit={int(args.limit)}).")
    summary_window_names = summarize_window_names(plot_window_names, sample_metrics)
    tgi.print_group_summary(summary_window_names, sample_metrics)
    if not args.disable_plots:
        lines_path = ftgi.plot_group_importance_lines(
            sample_metrics,
            output_dir,
            num_layers=num_layers,
            group_order=summary_window_names,
            selected_layers=selected_layers,
            seq_len_label=seq_len_label,
        )
        if lines_path is not None:
            print(f"Wrote group-importance lines plot to: {lines_path}")
        else:
            print("Skipped group-importance lines plot: insufficient data.")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_score_cache_dir_raw = args.clean_score_cache_dir or args.clean_ld_cache_dir
    clean_score_cache_dir = Path(clean_score_cache_dir_raw) if clean_score_cache_dir_raw is not None else output_dir
    clean_score_cache_dir.mkdir(parents=True, exist_ok=True)
    clean_score_cache_path = clean_score_cache_dir / "clean_scores.json"
    clean_score_cache = tgi.load_clean_score_cache(clean_score_cache_path)
    cache_updates = 0
    if clean_score_cache:
        print(f"Loaded {len(clean_score_cache)} cached clean-score entries from: {clean_score_cache_path}")

    seq_len_label = eval_utils.resolve_seq_len_label(data_root)

    lm = LanguageModel(tgi.base_model, tokenizer=tgi.processor.tokenizer)
    layers = tgi.get_layers(lm.model)
    num_layers = len(layers)
    selected_layers = parse_layer_selection(args.layers, num_layers=num_layers)

    sample_dirs = tgi.iter_sample_dirs(data_root)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under: {data_root}")

    target_processed_samples = int(args.limit)
    processed_samples = 0
    sample_metrics: List[Dict[str, Any]] = []
    plot_window_names = global_window_names(args.tail_non_image_tokens, args.window_size)

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= target_processed_samples:
            break
        sample_metrics_row, cache_delta = process_sample(
            sample_dir=sample_dir,
            sample_index=idx,
            total_samples=len(sample_dirs),
            args=args,
            lm=lm,
            layers=layers,
            selected_layers=selected_layers,
            clean_score_cache=clean_score_cache,
        )
        cache_updates += cache_delta
        if sample_metrics_row is None:
            continue
        sample_metrics_row["selected_layers"] = list(selected_layers)
        sample_metrics_row["selected_layers_spec"] = args.layers
        sample_metrics.append(sample_metrics_row)
        processed_samples += 1

    print(eval_utils.persist_clean_score_cache(clean_score_cache_path, clean_score_cache, cache_updates))
    finalize_outputs(
        sample_metrics=sample_metrics,
        output_dir=output_dir,
        num_layers=num_layers,
        selected_layers=selected_layers,
        seq_len_label=seq_len_label,
        plot_window_names=plot_window_names,
        args=args,
    )

    elapsed = time.perf_counter() - start_time
    print(eval_utils.format_runtime(elapsed))


if __name__ == "__main__":
    main()

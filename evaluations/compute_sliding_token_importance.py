"""
Scan sliding windows over non-image prompt-side tokens to find missed late-layer carrier tokens.

This script reuses the matched-control activation patching setup and full-answer log-prob
score from the text-group importance experiment, but replaces named text groups with
sliding windows over the last K non-image prompt-side tokens, including the assistant
prefix when present.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nnsight import LanguageModel

from evaluations import compute_text_group_importance as tgi


def parse_layer_selection(raw: Optional[str], num_layers: int) -> List[int]:
    if raw is None:
        return list(range(num_layers))

    selected: set[int] = set()
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("--layers must not be empty when provided")

    for part in parts:
        if ":" not in part:
            try:
                selected.add(int(part))
            except ValueError as exc:
                raise ValueError(f"Invalid layer index in --layers: {part!r}") from exc
            continue

        fields = part.split(":")
        if len(fields) not in {2, 3}:
            raise ValueError(
                f"Invalid range in --layers: {part!r}. Expected start:end or start:end:step."
            )
        try:
            start = int(fields[0])
            end = int(fields[1])
            step = int(fields[2]) if len(fields) == 3 else 1
        except ValueError as exc:
            raise ValueError(f"Invalid integer in --layers: {part!r}") from exc
        if step <= 0:
            raise ValueError(f"--layers step must be positive: {part!r}")
        if end <= start:
            raise ValueError(f"--layers range end must be greater than start: {part!r}")
        for layer_idx in range(start, end, step):
            selected.add(int(layer_idx))

    selected_layers = sorted(selected)
    invalid = [layer_idx for layer_idx in selected_layers if layer_idx < 0 or layer_idx >= num_layers]
    if invalid:
        raise ValueError(
            f"--layers contains out-of-bounds layers: {invalid}. Valid range is [0, {num_layers - 1}]."
        )
    return selected_layers


def locate_non_image_prompt_token_positions(
    inputs: Dict[str, Any],
    question: str,
    num_frames: int,
) -> Tuple[List[int], List[str]]:
    prompt = tgi.build_prompt(question, num_frames=num_frames)
    prompt_token_ids, _ = tgi.tokenize_with_offsets_if_available(prompt)
    if not prompt_token_ids:
        return [], ["prompt_tokenization_failed"]

    full_input_ids = [int(tok) for tok in inputs["input_ids"][0].tolist()]
    prompt_start_in_full = tgi.find_subsequence(full_input_ids, prompt_token_ids)
    if prompt_start_in_full is None:
        return [], ["prompt_subsequence_not_found_in_input_ids"]

    prompt_positions = list(range(prompt_start_in_full, prompt_start_in_full + len(prompt_token_ids)))

    token_positions_full, _, group_warnings = tgi.locate_group_token_positions(
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


def main() -> None:
    start_time = time.perf_counter()

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

    seq_len_label = None
    seq_len_match = tgi.re.search(r"(seq_len_\d+)", str(data_root))
    if seq_len_match:
        seq_len_label = seq_len_match.group(1)

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
    layer_sampled_counts = [0 for _ in range(num_layers)]
    layer_invalid_counts = [0 for _ in range(num_layers)]
    plot_window_names = global_window_names(args.tail_non_image_tokens, args.window_size)

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= target_processed_samples:
            break

        try:
            sample_id, frames, question, states, answer = tgi.load_mmred_sample(sample_dir)
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_dir.name} skipped: load failure ({exc})")
            continue

        parsed = tgi.parse_target_character_room(question)
        if parsed is None:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: could not parse question")
            continue

        try:
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, question))
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: failed to build clean inputs ({exc})")
            continue

        prompt_len = int(inputs["input_ids"].shape[1])
        a_star_text = str(answer).strip()
        try:
            a_star_ids = tgi.token_ids_of_answer(a_star_text)
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: invalid answer tokenization ({exc})")
            continue

        cache_entry = clean_score_cache.get(sample_id)
        if cache_entry is not None:
            cached_num_frames = int(cache_entry.get("num_frames", -1))
            cached_answer = str(cache_entry.get("answer_text", ""))
            if cached_num_frames == len(frames) and cached_answer == a_star_text:
                clean_answer_score = float(cache_entry.get("clean_answer_score", float("-inf")))
                clean_correct_prob = float(cache_entry.get("clean_correct_prob", 0.0))
                clean_top1_correct = bool(cache_entry.get("clean_top1_correct", False))
                best_answer_text = str(cache_entry.get("best_answer_text", ""))
            else:
                cache_entry = None

        if cache_entry is None:
            try:
                candidate_scores = tgi.score_valid_numeric_answers(
                    lm=lm,
                    inputs=inputs,
                    prompt_len=prompt_len,
                    num_frames=len(frames),
                )
            except Exception as exc:
                print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: clean scoring failed ({exc})")
                continue

            clean_answer_score = float(candidate_scores["scores_by_answer"].get(a_star_text, float("-inf")))
            clean_correct_prob = float(candidate_scores["probs_by_answer"].get(a_star_text, 0.0))
            best_answer_text = str(candidate_scores["best_answer_text"])
            clean_top1_correct = (best_answer_text == a_star_text)
            clean_score_cache[sample_id] = {
                "num_frames": len(frames),
                "answer_text": a_star_text,
                "clean_answer_score": clean_answer_score,
                "clean_correct_prob": clean_correct_prob,
                "clean_top1_correct": clean_top1_correct,
                "best_answer_text": best_answer_text,
            }
            cache_updates += 1

        if not clean_top1_correct:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: clean top-1 is {best_answer_text!r}, "
                f"not correct answer {a_star_text!r}"
            )
            continue
        if clean_correct_prob < args.min_clean_correct_prob:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: "
                f"clean_correct_prob={clean_correct_prob:.4f} < threshold={args.min_clean_correct_prob:.4f}"
            )
            continue
        if args.min_clean_correct_logprob is not None and clean_answer_score < args.min_clean_correct_logprob:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: "
                f"clean_answer_score={clean_answer_score:.4f} < threshold={args.min_clean_correct_logprob:.4f}"
            )
            continue

        clean_group_positions, clean_group_summaries, clean_group_warnings = tgi.locate_group_token_positions(
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
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: no non-image prompt token positions "
                f"(warnings={clean_prompt_warnings})"
            )
            continue

        control_inputs, control_question, control_group_positions, control_group_summaries, control_skip_info, control_reason = tgi.choose_best_control(
            frames=frames,
            states=states,
            question=question,
            clean_inputs=inputs,
            clean_group_positions=clean_group_positions,
            include_groups=tgi.parse_include_groups(",".join(tgi._ALL_GROUPS)),
        )
        if control_inputs is None or control_question is None or control_reason is None:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: failed to find aligned control "
                f"(details={control_skip_info})"
            )
            continue

        control_prompt_positions, control_prompt_warnings = locate_non_image_prompt_token_positions(
            inputs=control_inputs,
            question=control_question,
            num_frames=len(frames),
        )
        if not control_prompt_positions:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: control has no non-image prompt token positions "
                f"(warnings={control_prompt_warnings})"
            )
            continue
        if len(clean_prompt_positions) != len(control_prompt_positions):
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: non-image prompt token count mismatch "
                f"(clean={len(clean_prompt_positions)}, control={len(control_prompt_positions)})"
            )
            continue

        clean_full_input_ids = [int(tok) for tok in inputs["input_ids"][0].tolist()]
        control_full_input_ids = [int(tok) for tok in control_inputs["input_ids"][0].tolist()]
        clean_windows, clean_window_warnings = build_sliding_windows(
            full_input_ids=clean_full_input_ids,
            prompt_positions=clean_prompt_positions,
            tail_non_image_tokens=args.tail_non_image_tokens,
            window_size=args.window_size,
        )
        control_windows, control_window_warnings = build_sliding_windows(
            full_input_ids=control_full_input_ids,
            prompt_positions=control_prompt_positions,
            tail_non_image_tokens=args.tail_non_image_tokens,
            window_size=args.window_size,
        )
        if not clean_windows:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: no valid sliding windows "
                f"(warnings={clean_window_warnings})"
            )
            continue
        if len(clean_windows) != len(control_windows):
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: sliding window count mismatch "
                f"(clean={len(clean_windows)}, control={len(control_windows)})"
            )
            continue

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
                skipped_groups[group_name] = (
                    f"token_count_mismatch(clean={len(clean_positions)},control={len(control_positions)})"
                )
                continue

            group_token_counts[group_name] = len(clean_positions)
            group_summaries[group_name] = {
                "clean": tgi._normalize_summary_text("".join(clean_window["decoded_tokens"])),
                "control": tgi._normalize_summary_text("".join(control_window["decoded_tokens"])),
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
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: no valid active windows "
                f"(skipped={skipped_groups}, clean_prompt_warnings={clean_prompt_warnings}, "
                f"control_prompt_warnings={control_prompt_warnings})"
            )
            continue

        active_group_names = [group["name"] for group in groups_payload]
        print(
            f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} clean_answer_score={clean_answer_score:.4f} "
            f"clean_correct_prob={clean_correct_prob:.4f} control_question={control_question!r} "
            f"control_character={control_reason['control_character']!r} "
            f"control_room={control_reason['control_room']!r} "
            f"control_answer={control_reason['control_answer']!r} "
            f"active_windows={active_group_names} batch_size={args.batch_size}"
        )
        print(f"  active window token counts: {group_token_counts}")
        if skipped_groups:
            print(f"  skipped windows: {skipped_groups}")

        chunk_size = min(args.batch_size, len(groups_payload))
        group_chunks = [
            groups_payload[start:start + chunk_size]
            for start in range(0, len(groups_payload), chunk_size)
        ]

        chunk_data: List[Dict[str, Any]] = []
        try:
            for group_chunk in group_chunks:
                chunk_len = len(group_chunk)
                repeated_clean_inputs = tgi.repeat_inputs_for_batch(inputs, batch_size=chunk_len)
                clean_scoring_inputs = tgi.append_answer_tokens_for_scoring(repeated_clean_inputs, a_star_ids)
                control_inputs_batch = tgi.concatenate_inputs_for_batch(
                    [group_entry["control_inputs"] for group_entry in group_chunk]
                )
                control_scoring_inputs = tgi.append_answer_tokens_for_scoring(control_inputs_batch, a_star_ids)
                chunk_data.append({
                    "groups": group_chunk,
                    "clean_scoring_inputs": clean_scoring_inputs,
                    "control_scoring_inputs": control_scoring_inputs,
                })
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: failed to build batched inputs ({exc})")
            continue

        per_layer_metrics: List[Dict[str, Any]] = []
        all_layer_corrupted_rows: List[Tuple[int, List[float]]] = []
        for layer_idx in selected_layers:
            layer_sampled_counts[layer_idx] += 1
            per_group_corrupted_score: Dict[str, float] = {}
            per_group_signed_delta: Dict[str, float] = {}
            per_group_importance: Dict[str, float] = {}

            for chunk_idx, packed in enumerate(chunk_data, start=1):
                group_chunk = packed["groups"]
                clean_positions_by_batch = [group["clean_positions"] for group in group_chunk]
                control_positions_by_batch = [group["control_positions"] for group in group_chunk]
                try:
                    corrupted_scores = tgi.run_layer_multi_group_corrupted_sequence_logprob(
                        lm=lm,
                        layers=layers,
                        clean_batched_scoring_inputs=packed["clean_scoring_inputs"],
                        control_batched_scoring_inputs=packed["control_scoring_inputs"],
                        layer_idx=layer_idx,
                        clean_token_positions_by_batch=clean_positions_by_batch,
                        control_token_positions_by_batch=control_positions_by_batch,
                        prompt_len=prompt_len,
                        answer_token_ids=a_star_ids,
                    )
                except Exception as exc:
                    print(
                        f"  layer={layer_idx} failed batched corruption forward "
                        f"(chunk {chunk_idx}/{len(chunk_data)}, {exc}); using clean score for this chunk"
                    )
                    for group in group_chunk:
                        group_name = group["name"]
                        per_group_corrupted_score[group_name] = clean_answer_score
                        per_group_signed_delta[group_name] = 0.0
                        per_group_importance[group_name] = 0.0
                    continue

                for batch_idx, group in enumerate(group_chunk):
                    group_name = group["name"]
                    corrupt_score = float(corrupted_scores[batch_idx].item())
                    signed_delta = float(clean_answer_score - corrupt_score)
                    importance = max(signed_delta, 0.0)
                    per_group_corrupted_score[group_name] = corrupt_score
                    per_group_signed_delta[group_name] = signed_delta
                    per_group_importance[group_name] = importance

            layer_group_order = [group["name"] for group in groups_payload]
            corrupted_score_row = [per_group_corrupted_score.get(group_name, clean_answer_score) for group_name in layer_group_order]
            signed_delta_row = [per_group_signed_delta.get(group_name, 0.0) for group_name in layer_group_order]
            importance_row = [per_group_importance.get(group_name, 0.0) for group_name in layer_group_order]
            all_layer_corrupted_rows.append((layer_idx, list(corrupted_score_row)))

            total_importance = float(sum(importance_row))
            if total_importance > 0.0:
                probs = tgi.normalize_to_probabilities(importance_row)
                entropy_value = tgi.normalize_entropy(
                    tgi.entropy_from_probabilities(probs),
                    num_groups=len(layer_group_order),
                )
            else:
                probs = [0.0 for _ in importance_row]
                entropy_value = None
                layer_invalid_counts[layer_idx] += 1

            per_layer_metrics.append({
                "layer": layer_idx,
                "groups": list(layer_group_order),
                "corrupted_score": corrupted_score_row,
                "signed_delta": signed_delta_row,
                "r": importance_row,
                "p": probs,
                "entropy": entropy_value,
                "total_importance": total_importance,
            })

        if all_layer_corrupted_rows:
            print("  Corrupted score table (rows=layers, columns=sliding windows):")
            print(tgi.format_corrupted_score_table(active_group_names, all_layer_corrupted_rows))

        sample_metrics.append({
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
            "active_groups": active_group_names,
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
            "selected_layers": list(selected_layers),
            "selected_layers_spec": args.layers,
            "clean_prompt_warnings": clean_prompt_warnings,
            "control_prompt_warnings": control_prompt_warnings,
            "clean_window_warnings": clean_window_warnings,
            "control_window_warnings": control_window_warnings,
            "layer_metrics": {"layers": per_layer_metrics},
        })
        processed_samples += 1

    if cache_updates > 0:
        tgi.save_clean_score_cache(clean_score_cache_path, clean_score_cache)
        print(f"Updated clean-score cache at {clean_score_cache_path} ({cache_updates} new/changed entries).")
    elif not clean_score_cache_path.exists():
        tgi.save_clean_score_cache(clean_score_cache_path, clean_score_cache)
        print(f"Wrote empty clean-score cache to: {clean_score_cache_path}")
    else:
        print(f"No clean-score cache updates. Reused existing cache at: {clean_score_cache_path}")

    text_report_path = tgi.write_text_group_report(sample_metrics, output_dir)
    json_path = tgi.write_metrics_json(sample_metrics, output_dir)
    print(f"Wrote sample metrics text report to: {text_report_path}")
    print(f"Wrote sample metrics JSON to: {json_path}")
    print(
        f"Processed {processed_samples} samples "
        f"(target limit={target_processed_samples}, min_clean_correct_prob={args.min_clean_correct_prob:.4f})."
    )
    summary_window_names = plot_window_names
    if not summary_window_names:
        seen_window_names: List[str] = []
        for sample in sample_metrics:
            for group_name in sample["active_groups"]:
                if group_name not in seen_window_names:
                    seen_window_names.append(group_name)
        summary_window_names = seen_window_names
    tgi.print_group_summary(summary_window_names, sample_metrics)

    if not args.disable_plots:
        total_importance_plot_path = tgi.plot_total_importance_mean(
            sample_metrics,
            output_dir,
            num_layers=num_layers,
            seq_len_label=seq_len_label,
        )
        if total_importance_plot_path is not None:
            print(f"Wrote total-importance plot to: {total_importance_plot_path}")
        else:
            print("Skipped total-importance plot: no layer metrics available.")

        invalidity_plot_path = tgi.plot_layer_invalidity_rates(
            layer_sampled_counts,
            layer_invalid_counts,
            output_dir,
            seq_len_label=seq_len_label,
        )
        if invalidity_plot_path is not None:
            print(f"Wrote layer invalidity plot to: {invalidity_plot_path}")
        else:
            print("Skipped layer invalidity plot: no matplotlib available.")

        heatmap_group_order = summary_window_names
        heatmap_path = tgi.plot_group_importance_heatmap(
            sample_metrics,
            output_dir,
            num_layers=num_layers,
            include_groups=heatmap_group_order,
            seq_len_label=seq_len_label,
        )
        if heatmap_path is not None:
            print(f"Wrote group-importance heatmap to: {heatmap_path}")
        else:
            print("Skipped group-importance heatmap: insufficient data.")

        lines_path = tgi.plot_group_importance_lines(
            sample_metrics,
            output_dir,
            num_layers=num_layers,
            include_groups=heatmap_group_order,
            seq_len_label=seq_len_label,
        )
        if lines_path is not None:
            print(f"Wrote group-importance lines plot to: {lines_path}")
        else:
            print("Skipped group-importance lines plot: insufficient data.")

    elapsed = time.perf_counter() - start_time
    print(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()

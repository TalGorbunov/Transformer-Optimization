"""
AF1 frame-group conditional-mean experiment for MMRed with Qwen-VL.

This script is a multimodal adaptation of the AF1 method from:
"All for One: LLMs Solve Mental Math at the Last Token With Information
Transferred From Other Tokens".

What this script does:
1. Runs the clean/base model exactly as the current MMRed evaluation does.
2. Supports three experiment modes:
   - `full_af1`: wait-boundary patching plus ABP masking
   - `wait_only`: wait-boundary patching only
   - `mask_only`: ABP masking only
3. Uses frame-group conditional means plus one all-non-frame prompt
   conditional mean when the selected mode includes wait-boundary patching.
4. Keeps the current target prompt/question fixed while donor hybrids change
   only frame inputs.
5. Applies an ABP-style attention policy after the wait boundary when the
   selected mode includes masking:
   - transfer stage: only the prompt carrier token may attend to earlier
     prompt tokens; frame tokens stay dense within their own frame block;
     instruction tokens use the selected `instruction_mask_mode`
   - post-transfer stage: the carrier token and all instruction tokens also
     become self-only

Important method notes:
- This is not literal token-level CAMA from the paper. It is a multimodal
  adaptation that applies conditional-mean replacement to frame token groups
  plus one all-non-frame prompt token set.
- A single donor is not treated as a conditional mean. We require at least
  two compatible donors and average over up to `k_donors` hybrid contexts.
- `wait_layer` follows the paper's `L_wait` semantics: it is the number of
  waiting layers. Therefore:
  - if `wait_layer > 0`, we replace `x^(L_wait)` by patching the output of
    layer index `wait_layer - 1`
  - if `wait_layer == 0`, we replace `x^(0)` by patching the hidden states
    before layer 0 runs
- ABP starts only after that wait-boundary replacement, matching the paper's
  "replace x^(L_wait), then restrict attention afterward" semantics.

Example:
python evaluations/scripts/af1/full_pipeline.py \
  --split train \
  --seq_len 8 \
  --max_samples 8 \
  --wait_layers 40 \
  --transfer_layers_grid 2 \
  --instruction_mask_mode vision_end_only \
  --k_donors 4 \
  --output_dir outputs/af1_qwen_vl_frame_cama
"""

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.scripts.af1.common import (
    DONOR_POLICY,
    PER_SAMPLE_FIELDS,
    SUMMARY_FIELDS,
    VALID_INSTRUCTION_MASK_MODES,
    VALID_MODES,
    PreparedSample,
    SampleLayout,
)
from evaluations.scripts.af1.kernel import (
    validate_attention_policy,
    build_abp_attention_policy,
    compute_all_frame_group_means_for_sample,
    compute_non_frame_conditional_mean,
    instruction_mask_mode_summary,
    intervention_mode_flags,
    intervention_mode_summary,
)
from evaluations.scripts.af1.layout import (
    choose_reference_layout,
    format_token_debug_rows,
    format_transition_frame_debug,
    inspect_and_validate_layout,
    layout_hash,
    load_and_filter_sample_dirs,
    prepare_sample,
    select_donor_pool,
)
from evaluations.scripts.af1.metrics import (
    evaluated_row,
    materialize_skipped_row,
    run_clean_sample,
    run_intervention_sample,
    skipped_row,
    summarize_grid_point_results,
)
from evaluations.scripts.af1.reporting import (
    csv_header_line,
    format_summary_table,
    plot_metric_heatmap,
    row_for_fieldnames,
    write_csv,
    write_markdown_summary,
)
from models.model import MODEL_ID, get_layers, model as base_model


@dataclass
class GridPointEvaluation:
    sample_rows: List[Dict[str, Any]]
    summary_row: Dict[str, Any]
    cache_note: str
    validation_notes_for_grid_point: List[str]
    per_sample_rows_emitted: int
    expected_per_sample_rows: int


@dataclass
class PreparedEvaluationInputs:
    grid_items: List[Any]
    reference_layout: Optional[SampleLayout]
    compatible_samples: List[PreparedSample]
    compatible_layout_hash: str
    validation_notes: List[str]
    donor_notes: List[str]


@dataclass
class GridRunOutputs:
    all_sample_rows: List[Dict[str, Any]]
    summary_rows: List[Dict[str, Any]]
    validation_notes: List[str]
    cache_notes: List[str]
    per_sample_rows_emitted: int
    expected_per_sample_rows: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run a multimodal AF1 experiment on MMRed/Qwen-VL using frame-group "
            "conditional means at the wait boundary and ABP-style last-token transfer."
        )
    )
    ap.add_argument("--model_name", type=str, default=MODEL_ID)
    ap.add_argument("--data_root_base", type=str, default="data/mmred_images")
    ap.add_argument("--split", type=str, default="all")
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument(
        "--max_samples",
        type=int,
        default=8,
        help=(
            "Maximum number of eligible evaluation targets. Skipped samples do "
            "not count toward this cap; additional shuffled samples are scanned "
            "until the cap is reached or the dataset is exhausted."
        ),
    )
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument(
        "--wait_layers",
        type=str,
        nargs="+",
        required=True,
        help=(
            "AF1 wait boundaries L_wait measured in number of layers. "
            "Accepts integers and inclusive start:end:step ranges like 20:40:2. "
            "If wait_layer > 0, x^(L_wait) is patched at the output of layer wait_layer - 1."
        ),
    )
    ap.add_argument(
        "--transfer_layers_grid",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Transfer-layer counts to sweep after the wait boundary. "
            "Accepts integers and inclusive start:end:step ranges like 1:8:1."
        ),
    )
    ap.add_argument(
        "--k_donors",
        type=int,
        default=4,
        help=(
            "Maximum number of compatible donors to average for each frame-group "
            "or non-frame conditional mean."
        ),
    )
    ap.add_argument("--cache_dir", type=str, default="outputs/af1_frame_cama_cache")
    ap.add_argument("--recompute_cache", action="store_true")
    ap.add_argument("--output_dir", type=str, default="outputs/af1_qwen_vl_frame_cama")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mode",
        type=str,
        default="full_af1",
        choices=list(VALID_MODES),
        help=(
            "Intervention mode: full_af1 = patch + mask, wait_only = patch only, "
            "mask_only = mask only."
        ),
    )
    ap.add_argument("--skip_hallway", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--clean_top1_must_match_gold",
        action="store_true",
        help=(
            "Filter evaluation to samples where the clean model's top-1 answer "
            "matches the gold answer before running intervention evaluation."
        ),
    )
    ap.add_argument(
        "--instruction_mask_mode",
        type=str,
        default="vision_end_only",
        choices=list(VALID_INSTRUCTION_MASK_MODES),
        help=(
            "Instruction-token masking during transfer layers for the prompt "
            "span 'Respond with a single integer from 0 to {num_frames} "
            "(0 is allowed). Output only the integer.\\n': `base` keeps the "
            "base causal/padding mask, `vision_end_only` keeps self plus "
            "earlier `<|vision_end|>` keys, `vision_boundary_only` keeps self "
            "plus earlier `<|vision_start|>`/`<|vision_end|>` keys, "
            "`prompt_only` keeps self plus earlier non-frame non-boundary "
            "prompt keys, and `image_pad_only` keeps self plus earlier "
            "`<|image_pad|>` keys. After transfer, instruction tokens become "
            "self-only in all modes."
        ),
    )
    ap.add_argument("--debug_tokenization", action="store_true")
    return ap.parse_args()


def parse_layer_grid(raw_values: Sequence[str], arg_name: str) -> Tuple[List[int], int]:
    parsed_layers: List[int] = []
    tick_steps: List[int] = []
    for raw_value in raw_values:
        token = str(raw_value).strip()
        if not token:
            continue
        if ":" not in token:
            parsed_layers.append(int(token))
            continue

        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid layer range {token!r} for {arg_name}; expected start:end:step."
            )
        start, end, step = (int(part) for part in parts)
        if step == 0:
            raise ValueError(f"Invalid layer range {token!r} for {arg_name}; step must be non-zero.")
        if step > 0 and start > end:
            raise ValueError(
                f"Invalid layer range {token!r} for {arg_name}; positive step requires start <= end."
            )
        if step < 0 and start < end:
            raise ValueError(
                f"Invalid layer range {token!r} for {arg_name}; negative step requires start >= end."
            )

        tick_steps.append(abs(int(step)))
        stop = end + (1 if step > 0 else -1)
        parsed_layers.extend(list(range(start, stop, step)))

    if not parsed_layers:
        raise ValueError(f"{arg_name} must resolve to at least one layer value.")
    deduped_sorted_layers = sorted(set(int(layer) for layer in parsed_layers))
    tick_step = min(tick_steps) if tick_steps else 1
    return deduped_sorted_layers, int(tick_step)


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(dtype_name).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype={dtype_name!r}")
    return mapping[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seq_len_data_root(data_root_base: Path, seq_len: int, split: str) -> Path:
    return data_root_base / f"seq_len_{seq_len}" / split


def model_runtime_info(requested_device: str, requested_dtype: str) -> Dict[str, str]:
    first_param = next(base_model.parameters())
    actual_device = str(first_param.device)
    actual_dtype = str(first_param.dtype)
    requested_device = str(requested_device).strip()
    if requested_device and requested_device != "auto":
        if requested_device.split(":")[0] != actual_device.split(":")[0]:
            raise RuntimeError(
                f"Requested --device={requested_device!r}, but the current model is loaded on {actual_device!r}."
            )
    return {
        "model_name": MODEL_ID,
        "requested_device": requested_device,
        "actual_model_device": actual_device,
        "requested_dtype": str(requested_dtype),
        "actual_model_dtype": actual_dtype,
    }


def _emit_per_sample_row(
    row: Dict[str, Any],
    per_sample_writer: csv.DictWriter,
    stdout_per_sample_writer: csv.DictWriter,
    per_sample_handle: Any,
) -> None:
    per_sample_writer.writerow(row_for_fieldnames(row, PER_SAMPLE_FIELDS))
    per_sample_handle.flush()
    stdout_per_sample_writer.writerow(row_for_fieldnames(row, PER_SAMPLE_FIELDS))
    sys.stdout.flush()


def sample_passes_clean_top1_filter(
    *,
    args: argparse.Namespace,
    sample: PreparedSample,
    clean_metrics_cache: Dict[Tuple[int, str], Dict[str, Any]],
) -> bool:
    if not bool(args.clean_top1_must_match_gold):
        return True
    clean_cache_key = (args.seq_len, sample.sample_id)
    clean_metrics = clean_metrics_cache.get(clean_cache_key)
    if clean_metrics is None:
        clean_metrics = run_clean_sample(sample)
        clean_metrics_cache[clean_cache_key] = clean_metrics
    clean_pred = str(clean_metrics["best_answer_text"]).strip()
    return clean_pred == sample.gold_answer


def eligible_target_sample_ids(
    *,
    args: argparse.Namespace,
    grid_items: Sequence[Any],
    compatible_samples: Sequence[PreparedSample],
    enable_wait_patch: bool,
    clean_metrics_cache: Dict[Tuple[int, str], Dict[str, Any]],
) -> List[str]:
    eligible_ids: List[str] = []
    for item in grid_items:
        if isinstance(item, dict):
            continue
        if not sample_passes_clean_top1_filter(
            args=args,
            sample=item,
            clean_metrics_cache=clean_metrics_cache,
        ):
            continue
        if enable_wait_patch and len(
            select_donor_pool(
                target_sample=item,
                compatible_samples=compatible_samples,
                k_donors=args.k_donors,
                seed=args.seed,
            )
        ) < 2:
            continue
        eligible_ids.append(item.sample_id)
    return eligible_ids


def trim_grid_items_to_max_used(
    *,
    args: argparse.Namespace,
    prepared_inputs: PreparedEvaluationInputs,
    enable_wait_patch: bool,
    clean_metrics_cache: Dict[Tuple[int, str], Dict[str, Any]],
) -> PreparedEvaluationInputs:
    if int(args.max_samples) <= 0:
        return prepared_inputs

    eligible_ids = eligible_target_sample_ids(
        args=args,
        grid_items=prepared_inputs.grid_items,
        compatible_samples=prepared_inputs.compatible_samples,
        enable_wait_patch=enable_wait_patch,
        clean_metrics_cache=clean_metrics_cache,
    )
    if len(eligible_ids) <= int(args.max_samples):
        return prepared_inputs

    eligible_id_set = set(eligible_ids)
    kept_used_ids = set(eligible_ids[: int(args.max_samples)])
    trimmed_grid_items: List[Any] = []
    for item in prepared_inputs.grid_items:
        if isinstance(item, dict):
            trimmed_grid_items.append(item)
            continue
        if item.sample_id in kept_used_ids:
            trimmed_grid_items.append(item)
            continue
        if item.sample_id not in eligible_id_set:
            trimmed_grid_items.append(item)
            continue

    return PreparedEvaluationInputs(
        grid_items=trimmed_grid_items,
        reference_layout=prepared_inputs.reference_layout,
        compatible_samples=list(prepared_inputs.compatible_samples),
        compatible_layout_hash=prepared_inputs.compatible_layout_hash,
        validation_notes=list(prepared_inputs.validation_notes),
        donor_notes=list(prepared_inputs.donor_notes),
    )


def count_eligible_targets_for_selected_dirs(
    *,
    args: argparse.Namespace,
    sample_dirs: Sequence[Path],
    enable_wait_patch: bool,
    clean_metrics_cache: Dict[Tuple[int, str], Dict[str, Any]],
) -> int:
    prepared_samples: List[PreparedSample] = []
    for sample_dir in sample_dirs:
        prepared_sample, _ = prepare_sample(
            sample_dir,
            skip_hallway=bool(args.skip_hallway),
        )
        if prepared_sample is not None:
            prepared_samples.append(prepared_sample)

    reference_layout = choose_reference_layout(prepared_samples)
    if reference_layout is None:
        return 0

    compatible_samples: List[PreparedSample] = []
    for sample in prepared_samples:
        report = inspect_and_validate_layout(
            reference_layout=reference_layout,
            candidate_layout=sample.layout,
            skip_hallway=bool(args.skip_hallway),
        )
        if report["status"] == "exact_match":
            compatible_samples.append(sample)

    return len(
        eligible_target_sample_ids(
            args=args,
            grid_items=compatible_samples,
            compatible_samples=compatible_samples,
            enable_wait_patch=enable_wait_patch,
            clean_metrics_cache=clean_metrics_cache,
        )
    )


def select_sample_dirs_for_evaluation(
    *,
    args: argparse.Namespace,
    all_sample_dirs: Sequence[Path],
    enable_wait_patch: bool,
    clean_metrics_cache: Dict[Tuple[int, str], Dict[str, Any]],
) -> Tuple[List[Path], List[str]]:
    if int(args.max_samples) <= 0:
        return list(all_sample_dirs), []

    selected_sample_dirs: List[Path] = []
    eligible_target_count = 0
    for sample_dir in all_sample_dirs:
        selected_sample_dirs.append(sample_dir)
        eligible_target_count = count_eligible_targets_for_selected_dirs(
            args=args,
            sample_dirs=selected_sample_dirs,
            enable_wait_patch=enable_wait_patch,
            clean_metrics_cache=clean_metrics_cache,
        )
        if eligible_target_count >= int(args.max_samples):
            return (
                selected_sample_dirs,
                [
                    f"seq_len={args.seq_len}: scanned {len(selected_sample_dirs)} shuffled sample dirs to reach "
                    f"max_samples={int(args.max_samples)} eligible targets after skip filters "
                    f"(eligible_targets={eligible_target_count})"
                ],
            )

    return (
        selected_sample_dirs,
        [
            f"seq_len={args.seq_len}: dataset exhausted after scanning {len(selected_sample_dirs)} shuffled sample dirs; "
            f"eligible_targets={eligible_target_count} < max_samples={int(args.max_samples)}"
        ],
    )


def initial_validation_notes(args: argparse.Namespace) -> List[str]:
    validation_notes = [
        f"mode={args.mode}: {intervention_mode_summary(args.mode)}",
        (
            f"instruction_mask_mode={args.instruction_mask_mode}: "
            f"{instruction_mask_mode_summary(args.instruction_mask_mode)}"
        ),
        (
            f"grid seq_len={args.seq_len}: wait_layers={list(args.wait_layers)} "
            f"transfer_layers_grid={list(args.transfer_layers_grid)}"
        ),
        (
            f"max_samples={int(args.max_samples)} caps eligible evaluation targets; skipped samples "
            "do not count toward this cap."
        ),
        f"clean_top1_must_match_gold={bool(args.clean_top1_must_match_gold)}",
        (
            "wait_layer semantics: if wait_layer > 0, patch x^(L_wait) at layer output wait_layer - 1; "
            "if wait_layer == 0, patch x^(0) before layer 0."
        ),
    ]
    if args.mode == "full_af1":
        validation_notes.append("mode=full_af1: wait-boundary patching and ABP masking are both enabled.")
    elif args.mode == "wait_only":
        validation_notes.append(
            "mode=wait_only: ABP masking is disabled entirely; this isolates when the patched wait-boundary token sets are still needed."
        )
    else:
        validation_notes.append(
            "mode=mask_only: wait-boundary patching is disabled entirely; this isolates the transfer/self-only attention bottleneck."
        )
    return validation_notes


def resolve_grid_combinations(
    args: argparse.Namespace,
    num_model_layers: int,
) -> Tuple[List[Tuple[int, int]], List[int], List[str]]:
    validation_notes: List[str] = []
    valid_grid_combinations: List[Tuple[int, int]] = []
    invalid_combo_notes: List[str] = []
    transfer_layers_candidates = list(args.transfer_layers_grid)
    if args.mode == "wait_only" and transfer_layers_candidates:
        effective_transfer_layers = int(transfer_layers_candidates[0])
        validation_notes.append(
            f"mode=wait_only: collapsing transfer_layers_grid={list(args.transfer_layers_grid)} "
            f"to effective transfer_layers={effective_transfer_layers} because ABP masking is disabled."
        )
        print(
            f"[config] mode=wait_only collapsing transfer_layers_grid={list(args.transfer_layers_grid)} "
            f"to effective transfer_layers={effective_transfer_layers}"
        )
        transfer_layers_candidates = [effective_transfer_layers]
    effective_transfer_layers_grid = [int(value) for value in transfer_layers_candidates]

    for wait_layer in args.wait_layers:
        for transfer_layers in effective_transfer_layers_grid:
            reason: Optional[str] = None
            if int(wait_layer) < 0:
                reason = f"wait_layer={wait_layer} must be >= 0"
            elif int(transfer_layers) < 0:
                reason = f"transfer_layers={transfer_layers} must be >= 0"
            elif int(wait_layer) + int(transfer_layers) > int(num_model_layers):
                reason = (
                    f"wait_layer + transfer_layers must be <= {num_model_layers}; "
                    f"received {wait_layer} + {transfer_layers}"
                )

            if reason is not None:
                note = (
                    f"skipping invalid combo wait_layer={int(wait_layer)} "
                    f"transfer_layers={int(transfer_layers)}: {reason}"
                )
                invalid_combo_notes.append(note)
                print(f"[grid] {note}")
                continue
            valid_grid_combinations.append((int(wait_layer), int(transfer_layers)))

    validation_notes.extend(invalid_combo_notes)
    validation_notes.append(
        f"num_model_layers={num_model_layers}: valid_grid_combinations={len(valid_grid_combinations)}"
    )
    return valid_grid_combinations, effective_transfer_layers_grid, validation_notes


def prepare_evaluation_inputs(
    *,
    args: argparse.Namespace,
    sample_dirs: Sequence[Path],
    enable_wait_patch: bool,
) -> PreparedEvaluationInputs:
    validation_notes: List[str] = []
    donor_notes: List[str] = []
    loaded_items: List[Any] = []
    prepared_samples: List[PreparedSample] = []
    for sample_dir in sample_dirs:
        prepared_sample, skipped_sample_row = prepare_sample(
            sample_dir,
            skip_hallway=bool(args.skip_hallway),
        )
        if skipped_sample_row is not None:
            loaded_items.append(skipped_sample_row)
            validation_notes.append(
                f"seq_len={args.seq_len} sample_id={skipped_sample_row['sample_id']} skipped: {skipped_sample_row['skipped_reason']}"
            )
            continue
        loaded_items.append(prepared_sample)
        prepared_samples.append(prepared_sample)

    reference_layout = choose_reference_layout(prepared_samples)
    compatible_samples: List[PreparedSample] = []
    ordered_items: List[Any] = []
    compatible_layout_hash = ""
    if reference_layout is None:
        validation_notes.append(
            f"seq_len={args.seq_len}: no compatible non-hallway samples remained after filtering"
        )
        donor_notes.append(
            f"seq_len={args.seq_len}: no donor selection was used because no compatible reference layout remained."
        )
        print(
            f"[validation][mode={args.mode}] seq_len={args.seq_len} no compatible reference layout remained"
        )
        grid_items = loaded_items
    else:
        compatible_layout_hash = layout_hash(reference_layout)
        exact_match_count = 0
        for item in loaded_items:
            if isinstance(item, dict):
                ordered_items.append(item)
                continue
            report = inspect_and_validate_layout(
                reference_layout=reference_layout,
                candidate_layout=item.layout,
                skip_hallway=bool(args.skip_hallway),
            )
            if report["status"] != "exact_match":
                incompatible_row = skipped_row(
                    mode=args.mode,
                    sample_id=item.sample_id,
                    seq_len=item.layout.seq_len,
                    gold_answer=item.gold_answer,
                    skipped_reason="layout_incompatible",
                    room_text=item.layout.room_text,
                    layout=item.layout,
                    wait_layer=None,
                    transfer_layers=None,
                    k_donors=None,
                    layout_status=report["status"],
                    layout_details=report["details"],
                )
                ordered_items.append(incompatible_row)
                validation_notes.append(
                    f"seq_len={args.seq_len} sample_id={item.sample_id} incompatible: {report['details']}"
                )
                continue
            ordered_items.append(item)
            compatible_samples.append(item)
            exact_match_count += 1

        validation_notes.append(
            f"seq_len={args.seq_len}: reference_layout sample_id={reference_layout.sample_id} "
            f"prompt_len={reference_layout.prompt_len} carrier_index={reference_layout.carrier_index} "
            f"carrier_token={reference_layout.carrier_token_text!r} "
            f"image_tokens_per_frame={list(reference_layout.image_tokens_per_frame)} "
            f"layout_hash={compatible_layout_hash}"
        )
        validation_notes.append(
            f"seq_len={args.seq_len}: exact_match_samples={exact_match_count} total_selected={len(sample_dirs)}"
        )
        print(
            f"[validation][mode={args.mode}] seq_len={args.seq_len} reference_sample={reference_layout.sample_id} "
            f"prompt_len={reference_layout.prompt_len} carrier_index={reference_layout.carrier_index} "
            f"layout_hash={compatible_layout_hash}"
        )
        print(
            f"[validation][mode={args.mode}] seq_len={args.seq_len} exact_match_samples={exact_match_count} "
            f"total_selected={len(sample_dirs)}"
        )

        if args.debug_tokenization and compatible_samples:
            debug_layout = compatible_samples[0].layout
            print(f"[debug][mode={args.mode}] seq_len={args.seq_len} sample_id={compatible_samples[0].sample_id}")
            print(format_token_debug_rows(debug_layout))

        print(
            f"[validation][mode={args.mode}] seq_len={args.seq_len} compatible_samples={len(compatible_samples)} "
            f"reference_sample={reference_layout.sample_id}"
        )

        if enable_wait_patch:
            donor_notes.append(
                f"seq_len={args.seq_len}: donors come from the same seq_len pool, must pass exact layout validation, "
                f"must not equal the target sample, and are chosen with deterministic seeded shuffle "
                f"under policy={DONOR_POLICY}"
            )
        else:
            donor_notes.append(
                f"seq_len={args.seq_len}: donor selection is not used in mode={args.mode} because no wait-boundary frame patch is applied."
            )
        grid_items = ordered_items

    return PreparedEvaluationInputs(
        grid_items=grid_items,
        reference_layout=reference_layout,
        compatible_samples=compatible_samples,
        compatible_layout_hash=compatible_layout_hash,
        validation_notes=validation_notes,
        donor_notes=donor_notes,
    )


def evaluate_grid_point(
    *,
    args: argparse.Namespace,
    wait_layer: int,
    transfer_layers: int,
    grid_items: Sequence[Any],
    reference_layout: Optional[SampleLayout],
    compatible_samples: Sequence[PreparedSample],
    cache_dir: Path,
    clean_metrics_cache: Dict[Tuple[int, str], Dict[str, Any]],
    compatible_layout_hash: str,
    per_sample_writer: csv.DictWriter,
    stdout_per_sample_writer: csv.DictWriter,
    per_sample_handle: Any,
) -> GridPointEvaluation:
    mode_flags = intervention_mode_flags(args.mode)
    enable_wait_patch = mode_flags["enable_wait_patch"]
    enable_abp_mask = mode_flags["enable_abp_mask"]
    validation_notes_for_grid_point: List[str] = []
    sample_rows: List[Dict[str, Any]] = []
    per_sample_rows_emitted = 0

    if reference_layout is None:
        for item in grid_items:
            if not isinstance(item, dict):
                continue
            row = materialize_skipped_row(
                item,
                mode=args.mode,
                wait_layer=wait_layer,
                transfer_layers=transfer_layers,
                k_donors=args.k_donors,
            )
            sample_rows.append(row)
            _emit_per_sample_row(row, per_sample_writer, stdout_per_sample_writer, per_sample_handle)
            per_sample_rows_emitted += 1

        return GridPointEvaluation(
            sample_rows=sample_rows,
            summary_row=summarize_grid_point_results(
                MODEL_ID,
                mode=args.mode,
                seq_len=args.seq_len,
                wait_layer=wait_layer,
                transfer_layers=transfer_layers,
                sample_rows=sample_rows,
            ),
            cache_note=(
                f"seq_len={args.seq_len} wait_layer={wait_layer} transfer_layers={transfer_layers}: "
                "no conditional-mean cache activity because no compatible reference layout remained."
            ),
            validation_notes_for_grid_point=validation_notes_for_grid_point,
            per_sample_rows_emitted=per_sample_rows_emitted,
            expected_per_sample_rows=len(sample_rows),
        )

    policy = build_abp_attention_policy(
        layout=reference_layout,
        wait_layer=wait_layer,
        transfer_layers=transfer_layers,
        instruction_mask_mode=str(args.instruction_mask_mode),
    )
    if enable_abp_mask:
        note = (
            f"seq_len={args.seq_len} wait_layer={wait_layer} transfer_layers={transfer_layers}: "
            f"instruction_mask_mode={args.instruction_mask_mode} "
            f"instruction_positions={list(reference_layout.instruction_positions)}"
        )
        validation_notes_for_grid_point.append(note)
        print(
            f"[validation][mode={args.mode}] seq_len={args.seq_len} wait_layer={wait_layer} "
            f"transfer_layers={transfer_layers} "
            f"instruction_mask_mode={args.instruction_mask_mode} "
            f"instruction_positions={list(reference_layout.instruction_positions)}"
        )
        if args.debug_tokenization:
            print(
                f"[debug][mode={args.mode}] seq_len={args.seq_len} wait_layer={wait_layer} "
                f"transfer_layers={transfer_layers} {format_transition_frame_debug(reference_layout, policy)}"
            )
        for note in validate_attention_policy(reference_layout, policy=policy):
            validation_notes_for_grid_point.append(
                f"seq_len={args.seq_len} wait_layer={wait_layer} transfer_layers={transfer_layers}: {note}"
            )
            print(
                f"[validation][mode={args.mode}] seq_len={args.seq_len} wait_layer={wait_layer} "
                f"transfer_layers={transfer_layers} {note}"
            )
    else:
        note = (
            f"seq_len={args.seq_len} wait_layer={wait_layer} transfer_layers={transfer_layers}: "
            f"ABP masking disabled for mode={args.mode}, so later attention remains clean."
        )
        validation_notes_for_grid_point.append(note)
        print(
            f"[validation][mode={args.mode}] seq_len={args.seq_len} wait_layer={wait_layer} "
            f"transfer_layers={transfer_layers} ABP masking disabled; later attention remains clean."
        )

    print(
        f"[validation][mode={args.mode}] seq_len={args.seq_len} wait_layer={wait_layer} "
        f"transfer_layers={transfer_layers} compatible_samples={len(compatible_samples)} "
        f"reference_sample={reference_layout.sample_id}"
    )

    seq_cache_hits = 0
    seq_cache_misses = 0
    for item in grid_items:
        if isinstance(item, dict):
            row = materialize_skipped_row(
                item,
                mode=args.mode,
                wait_layer=wait_layer,
                transfer_layers=transfer_layers,
                k_donors=args.k_donors,
            )
            sample_rows.append(row)
            _emit_per_sample_row(row, per_sample_writer, stdout_per_sample_writer, per_sample_handle)
            per_sample_rows_emitted += 1
            print(
                f"[seq_len={args.seq_len}][wait_layer={wait_layer}][transfer_layers={transfer_layers}]"
                f"[mode={args.mode}] sample_id={row['sample_id']} skipped={row['skipped_reason']}"
            )
            continue

        donor_ids: List[str] = []
        frame_group_means = None
        non_frame_prompt_mean = None
        clean_cache_key = (args.seq_len, item.sample_id)
        clean_metrics: Optional[Dict[str, Any]] = None
        if args.clean_top1_must_match_gold:
            clean_metrics = clean_metrics_cache.get(clean_cache_key)
            if clean_metrics is None:
                clean_metrics = run_clean_sample(item)
                clean_metrics_cache[clean_cache_key] = clean_metrics
            clean_pred = str(clean_metrics["best_answer_text"]).strip()
            if clean_pred != item.gold_answer:
                skipped_sample_row = skipped_row(
                    mode=args.mode,
                    sample_id=item.sample_id,
                    seq_len=item.layout.seq_len,
                    gold_answer=item.gold_answer,
                    skipped_reason="clean_top1_not_gold",
                    room_text=item.layout.room_text,
                    layout=item.layout,
                    wait_layer=wait_layer,
                    transfer_layers=transfer_layers,
                    k_donors=args.k_donors,
                    layout_status="exact_match",
                    layout_details="exact_match",
                )
                sample_rows.append(skipped_sample_row)
                _emit_per_sample_row(
                    skipped_sample_row,
                    per_sample_writer,
                    stdout_per_sample_writer,
                    per_sample_handle,
                )
                per_sample_rows_emitted += 1
                validation_notes_for_grid_point.append(
                    f"seq_len={args.seq_len} wait_layer={wait_layer} transfer_layers={transfer_layers} "
                    f"sample_id={item.sample_id} skipped: clean_top1_not_gold "
                    f"(clean_pred={clean_pred!r}, gold_answer={item.gold_answer!r})"
                )
                print(
                    f"[seq_len={args.seq_len}][wait_layer={wait_layer}][transfer_layers={transfer_layers}]"
                    f"[mode={args.mode}] sample_id={item.sample_id} "
                    f"skipped=clean_top1_not_gold clean_pred={clean_pred!r} gold={item.gold_answer!r}"
                )
                continue

        if enable_wait_patch:
            donor_pool = select_donor_pool(
                target_sample=item,
                compatible_samples=compatible_samples,
                k_donors=args.k_donors,
                seed=args.seed,
            )
            donor_ids = [sample.sample_id for sample in donor_pool]
            if len(donor_pool) < 2:
                skipped_sample_row = skipped_row(
                    mode=args.mode,
                    sample_id=item.sample_id,
                    seq_len=item.layout.seq_len,
                    gold_answer=item.gold_answer,
                    skipped_reason="insufficient_compatible_donors",
                    room_text=item.layout.room_text,
                    layout=item.layout,
                    donor_ids=donor_ids,
                    wait_layer=wait_layer,
                    transfer_layers=transfer_layers,
                    k_donors=args.k_donors,
                    layout_status="exact_match",
                    layout_details="exact_match",
                )
                sample_rows.append(skipped_sample_row)
                _emit_per_sample_row(
                    skipped_sample_row,
                    per_sample_writer,
                    stdout_per_sample_writer,
                    per_sample_handle,
                )
                per_sample_rows_emitted += 1
                validation_notes_for_grid_point.append(
                    f"seq_len={args.seq_len} wait_layer={wait_layer} transfer_layers={transfer_layers} "
                    f"sample_id={item.sample_id} skipped: insufficient compatible donors "
                    f"(found={len(donor_pool)}, need>=2)"
                )
                print(
                    f"[seq_len={args.seq_len}][wait_layer={wait_layer}][transfer_layers={transfer_layers}]"
                    f"[mode={args.mode}] sample_id={item.sample_id} "
                    f"skipped=insufficient_compatible_donors donors={donor_ids}"
                )
                continue

            frame_group_means, cache_stats = compute_all_frame_group_means_for_sample(
                target_sample=item,
                donor_samples=donor_pool,
                wait_layer=wait_layer,
                batch_size=args.batch_size,
                cache_dir=cache_dir,
                recompute_cache=bool(args.recompute_cache),
                donor_policy=DONOR_POLICY,
            )
            seq_cache_hits += int(cache_stats["cache_hits"])
            seq_cache_misses += int(cache_stats["cache_misses"])
            non_frame_prompt_mean, non_frame_cache_hit = compute_non_frame_conditional_mean(
                target_sample=item,
                donor_samples=donor_pool,
                wait_layer=wait_layer,
                batch_size=args.batch_size,
                cache_dir=cache_dir,
                recompute_cache=bool(args.recompute_cache),
                donor_policy=DONOR_POLICY,
            )
            seq_cache_hits += int(non_frame_cache_hit)
            seq_cache_misses += int(not non_frame_cache_hit)

        if clean_metrics is None:
            clean_metrics = clean_metrics_cache.get(clean_cache_key)
        if clean_metrics is None:
            clean_metrics = run_clean_sample(item)
            clean_metrics_cache[clean_cache_key] = clean_metrics
        af1_metrics = run_intervention_sample(
            item,
            frame_group_means=frame_group_means,
            non_frame_prompt_mean=non_frame_prompt_mean,
            policy=policy,
            mode=args.mode,
        )
        row = evaluated_row(
            sample=item,
            clean_metrics=clean_metrics,
            af1_metrics=af1_metrics,
            donor_ids=donor_ids,
            policy=policy,
            k_donors_requested=args.k_donors,
            mode=args.mode,
        )
        sample_rows.append(row)
        _emit_per_sample_row(row, per_sample_writer, stdout_per_sample_writer, per_sample_handle)
        per_sample_rows_emitted += 1
        print(
            f"[seq_len={args.seq_len}][wait_layer={wait_layer}][transfer_layers={transfer_layers}]"
            f"[mode={args.mode}] sample_id={item.sample_id} gold={item.gold_answer} "
            f"clean={row['clean_pred']} af1={row['af1_pred']} "
            f"clean_top1_score_drop={float(row['clean_top1_score_drop']):.4f} "
            f"gold_answer_score_drop={float(row['gold_answer_score_drop']):.4f} "
            f"donors={json.dumps(donor_ids)}"
        )

    cache_note = (
        f"seq_len={args.seq_len} wait_layer={wait_layer} transfer_layers={transfer_layers}: "
        f"frame-group+non-frame conditional-mean cache_hits={seq_cache_hits} "
        f"cache_misses={seq_cache_misses} layout_hash={compatible_layout_hash}"
    )
    if not enable_wait_patch:
        cache_note = (
            f"seq_len={args.seq_len} wait_layer={wait_layer} transfer_layers={transfer_layers}: "
            f"no conditional-mean cache activity in mode={args.mode} because wait-boundary patching is disabled."
        )

    return GridPointEvaluation(
        sample_rows=sample_rows,
        summary_row=summarize_grid_point_results(
            MODEL_ID,
            mode=args.mode,
            seq_len=args.seq_len,
            wait_layer=wait_layer,
            transfer_layers=transfer_layers,
            sample_rows=sample_rows,
        ),
        cache_note=cache_note,
        validation_notes_for_grid_point=validation_notes_for_grid_point,
        per_sample_rows_emitted=per_sample_rows_emitted,
        expected_per_sample_rows=len(sample_rows),
    )


def run_grid_evaluation(
    *,
    args: argparse.Namespace,
    per_sample_csv_path: Path,
    valid_grid_combinations: Sequence[Tuple[int, int]],
    prepared_inputs: PreparedEvaluationInputs,
    cache_dir: Path,
    clean_metrics_cache: Dict[Tuple[int, str], Dict[str, Any]],
) -> GridRunOutputs:
    all_sample_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    validation_notes: List[str] = []
    cache_notes: List[str] = []
    per_sample_rows_emitted = 0
    expected_per_sample_rows = 0

    with per_sample_csv_path.open("w", encoding="utf-8", newline="") as per_sample_handle:
        per_sample_writer = csv.DictWriter(
            per_sample_handle,
            fieldnames=list(PER_SAMPLE_FIELDS),
            lineterminator="\n",
        )
        stdout_per_sample_writer = csv.DictWriter(
            sys.stdout,
            fieldnames=list(PER_SAMPLE_FIELDS),
            lineterminator="\n",
        )
        per_sample_writer.writeheader()
        per_sample_handle.flush()
        print(csv_header_line(PER_SAMPLE_FIELDS))
        sys.stdout.flush()

        for wait_layer, transfer_layers in valid_grid_combinations:
            result = evaluate_grid_point(
                args=args,
                wait_layer=wait_layer,
                transfer_layers=transfer_layers,
                grid_items=prepared_inputs.grid_items,
                reference_layout=prepared_inputs.reference_layout,
                compatible_samples=prepared_inputs.compatible_samples,
                cache_dir=cache_dir,
                clean_metrics_cache=clean_metrics_cache,
                compatible_layout_hash=prepared_inputs.compatible_layout_hash,
                per_sample_writer=per_sample_writer,
                stdout_per_sample_writer=stdout_per_sample_writer,
                per_sample_handle=per_sample_handle,
            )
            all_sample_rows.extend(result.sample_rows)
            summary_rows.append(result.summary_row)
            validation_notes.extend(result.validation_notes_for_grid_point)
            cache_notes.append(result.cache_note)
            per_sample_rows_emitted += result.per_sample_rows_emitted
            expected_per_sample_rows += result.expected_per_sample_rows

    return GridRunOutputs(
        all_sample_rows=all_sample_rows,
        summary_rows=summary_rows,
        validation_notes=validation_notes,
        cache_notes=cache_notes,
        per_sample_rows_emitted=per_sample_rows_emitted,
        expected_per_sample_rows=expected_per_sample_rows,
    )


def write_final_reports(
    *,
    args: argparse.Namespace,
    summary_rows: Sequence[Dict[str, Any]],
    summary_csv_path: Path,
    per_sample_csv_path: Path,
    markdown_summary_path: Path,
    output_dir: Path,
    cache_dir: Path,
    effective_transfer_layers_grid: Sequence[int],
    validation_notes: Sequence[str],
    donor_notes: Sequence[str],
    cache_notes: Sequence[str],
    start_time: float,
) -> Tuple[Dict[str, str], List[Tuple[str, str, str]]]:
    summary_table = format_summary_table(summary_rows)
    print(f"\nFinal AF1 Frame-CAMA Table (mode={args.mode}, seq_len={args.seq_len})")
    print(summary_table)
    write_csv(summary_csv_path, summary_rows, fieldnames=SUMMARY_FIELDS)

    heatmap_specs = [
        ("clean_acc", "heatmap_clean_acc.png", "clean_acc"),
        ("af1_acc", "heatmap_af1_acc.png", "af1_acc"),
        ("af1_faith", "heatmap_af1_faith.png", "af1_faith"),
        (
            "mean_clean_top1_score_drop",
            "heatmap_mean_clean_top1_score_drop.png",
            "mean_clean_top1_score_drop",
        ),
        (
            "mean_gold_answer_score_drop",
            "heatmap_mean_gold_answer_score_drop.png",
            "mean_gold_answer_score_drop",
        ),
    ]
    output_notes: List[str] = [
        f"summary_grid.csv: {summary_csv_path}",
        f"per_sample_grid.csv: {per_sample_csv_path}",
    ]
    heatmap_paths: Dict[str, str] = {}
    for value_key, filename, title in heatmap_specs:
        heatmap_path = output_dir / filename
        written_path = plot_metric_heatmap(
            summary_rows=summary_rows,
            wait_layers=args.wait_layers,
            transfer_layers_grid=effective_transfer_layers_grid,
            value_key=value_key,
            output_path=heatmap_path,
            title=title,
            seq_len=args.seq_len,
        )
        heatmap_paths[value_key] = str(written_path or heatmap_path)
        output_notes.append(f"{filename}: {written_path if written_path is not None else 'not_written'}")

    write_markdown_summary(
        markdown_summary_path,
        config={
            "model_name": args.model_name,
            "mode": args.mode,
            "data_root_base": args.data_root_base,
            "split": args.split,
            "seq_len": args.seq_len,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "wait_layers": list(args.wait_layers),
            "wait_layer_tick_step": int(args.wait_layer_tick_step),
            "transfer_layers_grid": list(effective_transfer_layers_grid),
            "transfer_layers_tick_step": int(args.transfer_layers_tick_step),
            "requested_transfer_layers_grid": list(args.transfer_layers_grid),
            "k_donors": args.k_donors,
            "cache_dir": str(cache_dir),
            "recompute_cache": bool(args.recompute_cache),
            "output_dir": str(output_dir),
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
            "skip_hallway": bool(args.skip_hallway),
            "clean_top1_must_match_gold": bool(args.clean_top1_must_match_gold),
            "instruction_mask_mode": str(args.instruction_mask_mode),
            "donor_policy": DONOR_POLICY,
            "wait_layer_semantics": (
                "wait_layer is AF1 L_wait measured in number of waiting layers; "
                "if wait_layer > 0 then x^(L_wait) is patched at layer output wait_layer - 1"
            ),
            "clean_top1_score_drop_semantics": (
                "clean_top1_score_drop = clean_best_score - intervention_score(clean top-1 answer)"
            ),
            "gold_answer_score_drop_semantics": (
                "gold_answer_score_drop = clean_score(gold answer) - intervention_score(gold answer)"
            ),
            "mode_semantics": {
                "full_af1": intervention_mode_summary("full_af1"),
                "wait_only": intervention_mode_summary("wait_only"),
                "mask_only": intervention_mode_summary("mask_only"),
            },
        },
        summary_rows=summary_rows,
        validation_notes=validation_notes,
        donor_notes=donor_notes,
        cache_notes=cache_notes,
        output_notes=output_notes,
        elapsed_seconds=time.time() - start_time,
    )
    return heatmap_paths, heatmap_specs


def main() -> None:
    args = parse_args()
    args.wait_layers, args.wait_layer_tick_step = parse_layer_grid(args.wait_layers, arg_name="--wait_layers")
    args.transfer_layers_grid, args.transfer_layers_tick_step = parse_layer_grid(
        args.transfer_layers_grid,
        arg_name="--transfer_layers_grid",
    )
    mode_flags = intervention_mode_flags(args.mode)
    enable_wait_patch = mode_flags["enable_wait_patch"]
    if args.model_name != MODEL_ID:
        raise ValueError(
            f"This script is pinned to {MODEL_ID!r}; received --model_name={args.model_name!r}"
        )
    if enable_wait_patch and args.k_donors < 2:
        raise ValueError(
            "--k_donors must be at least 2 because one donor is not a conditional mean."
        )
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    parse_dtype(args.dtype)
    set_seed(args.seed)
    start_time = time.time()
    runtime_info = model_runtime_info(requested_device=args.device, requested_dtype=args.dtype)
    num_model_layers = len(get_layers(base_model))
    print(json.dumps(runtime_info, indent=2, sort_keys=True))
    print(
        f"[config] mode={args.mode} seq_len={args.seq_len} skip_hallway={bool(args.skip_hallway)} "
        f"clean_top1_must_match_gold={bool(args.clean_top1_must_match_gold)} "
        f"wait_layers={list(args.wait_layers)} transfer_layers_grid={list(args.transfer_layers_grid)} "
        f"instruction_mask_mode={args.instruction_mask_mode}"
    )

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = output_dir / "summary_grid.csv"
    per_sample_csv_path = output_dir / "per_sample_grid.csv"
    markdown_summary_path = output_dir / "summary.md"

    validation_notes = initial_validation_notes(args)
    donor_notes: List[str] = []
    clean_metrics_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}

    valid_grid_combinations, effective_transfer_layers_grid, grid_validation_notes = resolve_grid_combinations(
        args,
        num_model_layers=num_model_layers,
    )
    validation_notes.extend(grid_validation_notes)

    data_root = seq_len_data_root(Path(args.data_root_base), seq_len=args.seq_len, split=args.split)
    if not data_root.is_dir():
        raise FileNotFoundError(f"seq_len={args.seq_len}: data root not found: {data_root}")

    all_sample_dirs = load_and_filter_sample_dirs(
        data_root=data_root,
        max_samples=0,
        seed=args.seed + args.seq_len,
    )
    if not all_sample_dirs:
        raise RuntimeError(f"seq_len={args.seq_len}: no samples found under {data_root}")

    sample_dirs, sample_selection_notes = select_sample_dirs_for_evaluation(
        args=args,
        all_sample_dirs=all_sample_dirs,
        enable_wait_patch=enable_wait_patch,
        clean_metrics_cache=clean_metrics_cache,
    )
    validation_notes.extend(sample_selection_notes)
    if not sample_dirs:
        raise RuntimeError(f"seq_len={args.seq_len}: no samples selected under {data_root}")

    print(
        f"[seq_len={args.seq_len}][mode={args.mode}] sample_dirs_scanned={len(sample_dirs)} data_root={data_root} "
        f"wait_layers={list(args.wait_layers)} transfer_layers_grid={list(args.transfer_layers_grid)} "
        f"k_donors={args.k_donors} instruction_mask_mode={args.instruction_mask_mode} "
        f"clean_top1_must_match_gold={bool(args.clean_top1_must_match_gold)}"
    )

    prepared_inputs = prepare_evaluation_inputs(
        args=args,
        sample_dirs=sample_dirs,
        enable_wait_patch=enable_wait_patch,
    )
    prepared_inputs = trim_grid_items_to_max_used(
        args=args,
        prepared_inputs=prepared_inputs,
        enable_wait_patch=enable_wait_patch,
        clean_metrics_cache=clean_metrics_cache,
    )
    validation_notes.extend(prepared_inputs.validation_notes)
    donor_notes.extend(prepared_inputs.donor_notes)

    grid_outputs = run_grid_evaluation(
        args=args,
        per_sample_csv_path=per_sample_csv_path,
        valid_grid_combinations=valid_grid_combinations,
        prepared_inputs=prepared_inputs,
        cache_dir=cache_dir,
        clean_metrics_cache=clean_metrics_cache,
    )
    validation_notes.extend(grid_outputs.validation_notes)
    summary_rows = sorted(
        grid_outputs.summary_rows,
        key=lambda row: (int(row["wait_layer"]), int(row["transfer_layers"])),
    )
    heatmap_paths, heatmap_specs = write_final_reports(
        args=args,
        summary_rows=summary_rows,
        summary_csv_path=summary_csv_path,
        per_sample_csv_path=per_sample_csv_path,
        markdown_summary_path=markdown_summary_path,
        output_dir=output_dir,
        cache_dir=cache_dir,
        effective_transfer_layers_grid=effective_transfer_layers_grid,
        validation_notes=validation_notes,
        donor_notes=donor_notes,
        cache_notes=grid_outputs.cache_notes,
        start_time=start_time,
    )

    if len(summary_rows) != len(valid_grid_combinations):
        raise RuntimeError(
            f"Summary row count mismatch: expected {len(valid_grid_combinations)} valid combos, "
            f"found {len(summary_rows)} summary rows"
        )
    if grid_outputs.per_sample_rows_emitted != grid_outputs.expected_per_sample_rows:
        raise RuntimeError(
            f"Per-sample row count mismatch: emitted {grid_outputs.per_sample_rows_emitted}, "
            f"expected {grid_outputs.expected_per_sample_rows}"
        )
    if len(grid_outputs.all_sample_rows) != grid_outputs.expected_per_sample_rows:
        raise RuntimeError(
            f"Accumulated per-sample row count mismatch: stored {len(grid_outputs.all_sample_rows)}, "
            f"expected {grid_outputs.expected_per_sample_rows}"
        )
    missing_heatmaps = [
        str(output_dir / filename)
        for _, filename, _ in heatmap_specs
        if not (output_dir / filename).exists()
    ]
    if missing_heatmaps:
        raise RuntimeError(f"Missing expected heatmaps: {missing_heatmaps}")

    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv_path),
                "per_sample_csv": str(per_sample_csv_path),
                "markdown_summary": str(markdown_summary_path),
                "heatmaps": heatmap_paths,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

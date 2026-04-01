"""
Find full-AF1 masking score-drop curves as a function of transfer length for one start layer.

For each retained MMRed sample:
- define the carrier as the final prompt token before answer generation
- score the clean gold answer
- for one tested start layer l and every tested transfer length t:
  - at layers l .. l + t - 1, apply the frame-aware transition mask
  - at layers >= l + t, apply the post-transfer AF1 mask
- record score_drop = clean_answer_score - ablated_answer_score
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import af1_utils
from evaluations.helpers import patching_core as core
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from models.model import force_eager_attention_backend, get_layers, model as base_model

_RESULTS_KEY = "full_af1_results"
_PLOT_COLOR = "#ff7f0e"
_PER_SAMPLE_FIELDS = [
    "sample_id",
    "seq_len",
    "question",
    "gold_answer",
    "clean_pred",
    "clean_correct",
    "clean_correct_prob",
    "clean_answer_score",
    "transfer_layers",
    "non_frame_prompt_self_only",
    "carrier_index",
    "carrier_token",
    "blocked_token_count",
    "starting_layer",
    "peak_layer_by_transfer_layers",
    "peak_score_drop_by_transfer_layers",
    "ablated_answer_scores_by_transfer_layers",
    "score_drops_by_transfer_layers",
    "missing_layers_by_transfer_layers",
]
_AGGREGATE_FIELDS = [
    "starting_layer",
    "transfer_layers",
    "n_samples",
    "mean_clean_answer_score",
    "mean_ablated_answer_score",
    "mean_score_drop",
    "median_score_drop",
    "max_score_drop",
]


def _parse_bool(raw: str) -> bool:
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {raw!r}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Find full-AF1 score-drop curves by sweeping transfer lengths for one start layer."
        )
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--output", type=str, default="outputs/find_full_af1_mask_transition")
    ap.add_argument("--starting-layer", type=int, required=True, dest="starting_layer")
    ap.add_argument(
        "--non_frame_prompt_self_only",
        type=_parse_bool,
        default=False,
        help=(
            "Whether non-frame prompt tokens should be restricted to self-plus-optional-spared-token attention "
            "during post-transfer masked layers only."
        ),
    )
    ap.add_argument(
        "--transfer_layers",
        type=str,
        required=True,
        help=(
            "Tested transfer lengths. Examples: --transfer_layers 2, --transfer_layers 2,4,6, "
            "--transfer_layers 2:8:2"
        ),
    )
    ap.add_argument(
        "--debug_masks",
        action="store_true",
        help=(
            "Print lightweight frame-mask sanity checks, including transfer-stage frame locality and "
            "post-transfer carrier restrictions."
        ),
    )
    ap.add_argument(
        "--min_clean_correct_prob",
        type=float,
        default=None,
        help="Optional clean-probability filter. Skip samples whose clean gold-answer probability falls below this threshold.",
    )
    ap.add_argument("--disable_plots", action="store_true")
    args = ap.parse_args()

    if int(args.limit) <= 0:
        raise ValueError("--limit must be a positive integer")
    if args.min_clean_correct_prob is not None:
        min_prob = float(args.min_clean_correct_prob)
        if min_prob < 0.0 or min_prob > 1.0:
            raise ValueError("--min_clean_correct_prob must be within [0, 1] when provided")
    return args


def parse_transfer_layer_selection(raw: str) -> List[int]:
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("--transfer_layers must not be empty")

    selected: set[int] = set()
    for part in parts:
        if ":" not in part:
            try:
                value = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid transfer length in --transfer_layers: {part!r}") from exc
            if value <= 0:
                raise ValueError("--transfer_layers values must be positive integers")
            selected.add(int(value))
            continue

        fields = part.split(":")
        if len(fields) not in {2, 3}:
            raise ValueError(
                f"Invalid range in --transfer_layers: {part!r}. Expected start:end or start:end:step."
            )
        try:
            start = int(fields[0])
            end = int(fields[1])
            step = int(fields[2]) if len(fields) == 3 else 1
        except ValueError as exc:
            raise ValueError(f"Invalid integer in --transfer_layers: {part!r}") from exc
        if start <= 0 or end <= 0 or step <= 0:
            raise ValueError("--transfer_layers range values must be positive")
        if end < start:
            raise ValueError(f"--transfer_layers range end must be >= start: {part!r}")
        for value in range(start, end + 1, step):
            selected.add(int(value))

    return sorted(selected)

def resolve_blocked_key_positions(
    carrier_index: int,
) -> List[int]:
    blocked_positions: List[int] = []
    for position in range(int(carrier_index)):
        blocked_positions.append(int(position))
    return blocked_positions


def build_frame_group_by_token(layout: af1_utils.TokenLayout) -> Dict[int, Tuple[int, ...]]:
    frame_group_by_token: Dict[int, Tuple[int, ...]] = {}
    for frame_idx, group in enumerate(layout.frame_groups):
        normalized_group = tuple(int(position) for position in group)
        if not normalized_group:
            continue
        if int(layout.carrier_index) in normalized_group:
            raise RuntimeError(
                f"sample_id={layout.sample_id}: carrier_index={layout.carrier_index} unexpectedly appears "
                f"inside frame group {frame_idx}"
            )
        for position in normalized_group:
            if position < 0 or position >= int(layout.prompt_len):
                raise RuntimeError(
                    f"sample_id={layout.sample_id}: frame group {frame_idx} position={position} is outside "
                    f"prompt_len={layout.prompt_len}"
                )
            if position in frame_group_by_token:
                raise RuntimeError(
                    f"sample_id={layout.sample_id}: overlapping frame groups at token position={position}"
                )
            frame_group_by_token[int(position)] = normalized_group
    return frame_group_by_token

def allowed_transition_prompt_keys(
    query_idx: int,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    non_frame_prompt_self_only: bool = False,
) -> Optional[List[int]]:
    if int(query_idx) == int(layout.carrier_index):
        return list(range(0, int(layout.carrier_index) + 1))

    same_frame_group = frame_group_by_token.get(int(query_idx))
    if same_frame_group is None:
        return None

    return sorted(int(position) for position in same_frame_group)


def allowed_post_transfer_prompt_keys(
    query_idx: int,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    non_frame_prompt_self_only: bool = False,
) -> Optional[List[int]]:
    if int(query_idx) == int(layout.carrier_index):
        return [int(layout.carrier_index)]

    same_frame_group = frame_group_by_token.get(int(query_idx))
    if same_frame_group is None:
        if non_frame_prompt_self_only:
            return [int(query_idx)]
        return None

    return sorted(int(position) for position in same_frame_group)


def build_frame_aware_prompt_attention_mask(
    base_mask: torch.Tensor,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    non_frame_prompt_self_only: bool,
    allowed_prompt_keys_fn: Callable[
        [int, af1_utils.TokenLayout, Dict[int, Tuple[int, ...]], bool],
        Optional[List[int]],
    ],
) -> torch.Tensor:
    if base_mask.dim() != 4:
        raise ValueError(f"Expected rank-4 attention mask, got shape={tuple(base_mask.shape)}")

    batch_size, _, query_len, key_len = base_mask.shape
    if int(layout.carrier_index) < 0 or int(layout.carrier_index) >= int(query_len):
        raise ValueError(
            f"carrier_index={layout.carrier_index} is outside query_len={query_len} for transition masking"
        )

    template = base_mask[0, 0]
    base_allowed = template == 0
    custom_allowed = base_allowed.clone()

    prompt_rows = min(int(layout.prompt_len), int(query_len))
    for query_idx in range(prompt_rows):
        allowed_keys = allowed_prompt_keys_fn(
            query_idx=query_idx,
            layout=layout,
            frame_group_by_token=frame_group_by_token,
            non_frame_prompt_self_only=non_frame_prompt_self_only,
        )
        if allowed_keys is None:
            continue
        allowed_row = torch.zeros(key_len, dtype=torch.bool, device=base_mask.device)
        for key_idx in allowed_keys:
            if 0 <= int(key_idx) < int(key_len):
                allowed_row[int(key_idx)] = True
        custom_allowed[int(query_idx), :] = allowed_row

    final_allowed = base_allowed & custom_allowed
    fill_value = torch.finfo(template.dtype).min if torch.is_floating_point(template) else -1.0e9
    mask_2d = torch.full_like(template, fill_value=fill_value)
    mask_2d[final_allowed] = 0
    return mask_2d.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_len, key_len)


def build_transition_attention_mask(
    base_mask: torch.Tensor,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    non_frame_prompt_self_only: bool,
) -> torch.Tensor:
    return build_frame_aware_prompt_attention_mask(
        base_mask=base_mask,
        layout=layout,
        frame_group_by_token=frame_group_by_token,
        non_frame_prompt_self_only=non_frame_prompt_self_only,
        allowed_prompt_keys_fn=allowed_transition_prompt_keys,
    )


def build_post_transfer_attention_mask(
    base_mask: torch.Tensor,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    non_frame_prompt_self_only: bool,
) -> torch.Tensor:
    return build_frame_aware_prompt_attention_mask(
        base_mask=base_mask,
        layout=layout,
        frame_group_by_token=frame_group_by_token,
        non_frame_prompt_self_only=non_frame_prompt_self_only,
        allowed_prompt_keys_fn=allowed_post_transfer_prompt_keys,
    )


def stage_for_layer(
    layer_idx: int,
    start_layer: int,
    transfer_layers: int,
) -> Optional[str]:
    if int(layer_idx) < int(start_layer):
        return None
    if int(layer_idx) < int(start_layer) + int(transfer_layers):
        return "transition"
    return "post_transfer"


def run_model_with_last_token_mask(
    inputs: Dict[str, torch.Tensor],
    layers: Any,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    non_frame_prompt_self_only: bool,
    start_layer: int,
    transfer_layers: int,
) -> Any:
    def wrapper_factory(layer_idx: int, original_forward: Any) -> Any:
        layer_stage = stage_for_layer(
            layer_idx=int(layer_idx),
            start_layer=int(start_layer),
            transfer_layers=transfer_layers,
        )
        if layer_stage is None:
            return original_forward

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError(f"Layer {layer_idx} forward received no hidden_states")
            base_attention_mask = kwargs.get("attention_mask")
            if base_attention_mask is None:
                raise RuntimeError(
                    f"Layer {layer_idx} did not receive an attention_mask, so transition masking is impossible."
                )
            batch_size = int(hidden_states.shape[0])
            expanded_mask = af1_utils._ensure_mask_tensor(base_attention_mask, batch_size=batch_size)
            if layer_stage == "transition":
                kwargs["attention_mask"] = build_transition_attention_mask(
                    expanded_mask,
                    layout=layout,
                    frame_group_by_token=frame_group_by_token,
                    non_frame_prompt_self_only=non_frame_prompt_self_only,
                )
            elif layer_stage == "post_transfer":
                kwargs["attention_mask"] = build_post_transfer_attention_mask(
                    expanded_mask,
                    layout=layout,
                    frame_group_by_token=frame_group_by_token,
                    non_frame_prompt_self_only=non_frame_prompt_self_only,
                )
            else:
                raise ValueError(f"Unsupported masking stage: {layer_stage!r}")
            return original_forward(*args, **kwargs)

        return wrapped_forward

    force_eager_attention_backend()
    with af1_utils.temporary_layer_wrappers(layers, wrapper_factory):
        with torch.inference_mode():
            return base_model(
                **inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )


def run_window_ablation_logprob(
    scoring_inputs: Dict[str, torch.Tensor],
    layers: Any,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    non_frame_prompt_self_only: bool,
    start_layer: int,
    transfer_layers: int,
    prompt_len: int,
    answer_token_ids: List[int],
) -> float:
    outputs = run_model_with_last_token_mask(
        scoring_inputs,
        layers=layers,
        layout=layout,
        frame_group_by_token=frame_group_by_token,
        non_frame_prompt_self_only=non_frame_prompt_self_only,
        start_layer=start_layer,
        transfer_layers=transfer_layers,
    )
    return af1_utils.sequence_logprob_from_outputs(
        outputs,
        prompt_len=prompt_len,
        answer_token_ids=answer_token_ids,
    )


def format_transition_mask_debug(
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    non_frame_prompt_self_only: bool,
) -> str:
    frame_sizes = [len(group) for group in layout.frame_groups]
    empty_frame_groups = sum(1 for group in layout.frame_groups if not group)
    dense_intra_frame_policy = True
    cross_frame_isolation = True
    for group in layout.frame_groups:
        if not group:
            continue
        group_set = {int(position) for position in group}
        for query_idx in group_set:
            allowed_keys = set(
                allowed_transition_prompt_keys(
                    query_idx=int(query_idx),
                    layout=layout,
                    frame_group_by_token=frame_group_by_token,
                    non_frame_prompt_self_only=non_frame_prompt_self_only,
                )
                or []
            )
            if not group_set.issubset(allowed_keys):
                dense_intra_frame_policy = False
            other_frame_positions = set(frame_group_by_token) - group_set
            if allowed_keys & other_frame_positions:
                cross_frame_isolation = False
    return (
        f"transition_frame_blocks frames={len(layout.frame_groups)} "
        f"tokens_per_frame={frame_sizes} "
        f"empty_frame_groups={empty_frame_groups} "
        f"dense_intra_frame_policy={dense_intra_frame_policy} "
        f"cross_frame_isolation={cross_frame_isolation} "
        f"non_frame_prompt_self_only_post_transfer_only={bool(non_frame_prompt_self_only)} "
        "base_causal_preserved=True"
    )


def format_post_transfer_mask_debug(
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    non_frame_prompt_self_only: bool,
) -> str:
    carrier_allowed_keys = allowed_post_transfer_prompt_keys(
        query_idx=int(layout.carrier_index),
        layout=layout,
        frame_group_by_token=frame_group_by_token,
        non_frame_prompt_self_only=non_frame_prompt_self_only,
    ) or []
    return (
        f"post_transfer_carrier_allowed_keys={carrier_allowed_keys} "
        f"carrier_self_only={set(carrier_allowed_keys) == {int(layout.carrier_index)}} "
        f"non_frame_prompt_self_only={bool(non_frame_prompt_self_only)}"
    )


def _is_present_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _format_float_or_na(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(header), *(len(str(row[col_idx])) for row in rows)) if rows else len(header)
        for col_idx, header in enumerate(headers)
    ]
    header_row = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_row = "|-" + "-|-".join("-" * widths[idx] for idx in range(len(headers))) + "-|"
    data_rows = [
        "| " + " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_row, sep_row] + data_rows)


def active_layers_from_layer_onward(boundary_layer: int, num_layers: int) -> List[int]:
    return list(range(int(boundary_layer), int(num_layers)))


def compute_metrics_for_transfer_length(
    starting_layer: int,
    num_layers: int,
    clean_answer_score: float,
    gold_scoring_inputs: Dict[str, torch.Tensor],
    layers: Any,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    non_frame_prompt_self_only: bool,
    prompt_len: int,
    gold_answer_ids: List[int],
    transfer_layers: int,
    debug_masks: bool,
) -> Dict[str, Any]:
    layer_metrics: List[Dict[str, Any]] = []
    boundary_layer = int(starting_layer)
    if debug_masks:
        debug_prefix = (
            f"  [debug] start_layer={int(boundary_layer)} transfer_layers={int(transfer_layers)}"
        )
        transfer_layer = next(
            (
                layer_idx
                for layer_idx in range(int(boundary_layer), int(num_layers))
                if stage_for_layer(
                    layer_idx=layer_idx,
                    start_layer=int(boundary_layer),
                    transfer_layers=transfer_layers,
                )
                == "transition"
            ),
            None,
        )
        if transfer_layer is not None:
            print(
                f"{debug_prefix} representative_transfer_layer={int(transfer_layer)} "
                f"{format_transition_mask_debug(layout, frame_group_by_token, non_frame_prompt_self_only)}"
            )
        post_transfer_layer = next(
            (
                layer_idx
                for layer_idx in range(int(boundary_layer), int(num_layers))
                if stage_for_layer(
                    layer_idx=layer_idx,
                    start_layer=int(boundary_layer),
                    transfer_layers=transfer_layers,
                )
                == "post_transfer"
            ),
            None,
        )
        if post_transfer_layer is None:
            print(
                f"{debug_prefix} representative_post_transfer_layer=none "
                "post_transfer_not_reached=True"
            )
        else:
            print(
                f"{debug_prefix} representative_post_transfer_layer={int(post_transfer_layer)} "
                f"{format_post_transfer_mask_debug(layout, frame_group_by_token, non_frame_prompt_self_only)}"
            )
    try:
        ablated_answer_score = run_window_ablation_logprob(
            gold_scoring_inputs,
            layers=layers,
            layout=layout,
            frame_group_by_token=frame_group_by_token,
            non_frame_prompt_self_only=non_frame_prompt_self_only,
            start_layer=int(boundary_layer),
            transfer_layers=transfer_layers,
            prompt_len=prompt_len,
            answer_token_ids=gold_answer_ids,
        )
    except Exception as exc:
        print(
            f"  transfer_layers={int(transfer_layers)} start_layer={boundary_layer} failed full-AF1 masking ({exc}); "
            "recording missing values for this layer"
        )
        ablated_answer_score = None

    score_drop = (
        None
        if ablated_answer_score is None
        else float(clean_answer_score - ablated_answer_score)
    )
    layer_metrics.append(
        {
            "layer": int(boundary_layer),
            "clean_answer_score": float(clean_answer_score),
            "ablated_answer_score": (
                None if ablated_answer_score is None else float(ablated_answer_score)
            ),
            "score_drop": None if score_drop is None else float(score_drop),
        }
    )

    valid_rows = [row for row in layer_metrics if row.get("score_drop") is not None]
    peak_row = (
        max(valid_rows, key=lambda row: (float(row["score_drop"]), -int(row["layer"])))
        if valid_rows
        else None
    )
    return {
        "layers": layer_metrics,
        "peak_layer": None if peak_row is None else int(peak_row["layer"]),
        "peak_score_drop": None if peak_row is None else float(peak_row["score_drop"]),
    }


def compute_sample_payload(
    sample_dir: Path,
    sample_index: int,
    total_samples: int,
    layers: Any,
    starting_layer: int,
    non_frame_prompt_self_only: bool,
    min_clean_correct_prob: Optional[float],
    debug_masks: bool,
    transfer_lengths: Sequence[int],
) -> Optional[Dict[str, Any]]:
    try:
        sample_id, frames, question, _, answer_text = load_mmred_sample(sample_dir)
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_dir.name} skipped: load failure ({exc})")
        return None

    try:
        clean_inputs = af1_utils.move_inputs_to_model_device(core.build_inputs(frames, question))
        layout = af1_utils.build_token_layout(
            sample_id=sample_id,
            frames=frames,
            question=question,
            inputs=clean_inputs,
        )
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: input/layout build failed ({exc})")
        return None

    blocked_key_positions = resolve_blocked_key_positions(
        carrier_index=int(layout.carrier_index),
    )
    frame_group_by_token = build_frame_group_by_token(layout)
    if debug_masks:
        print(
            f"[debug][{sample_index}/{total_samples}] sample_id={sample_id} "
            f"{format_transition_mask_debug(layout, frame_group_by_token, non_frame_prompt_self_only)}"
        )

    prompt_len = int(layout.prompt_len)
    gold_answer_text = str(answer_text).strip()
    gold_answer_ids = core.token_ids_of_answer(gold_answer_text)

    clean_metrics = af1_utils.score_valid_numeric_answers_with_runner(
        clean_inputs,
        prompt_len=prompt_len,
        num_frames=len(frames),
        runner=lambda scoring_inputs, answer_ids: af1_utils.run_clean_model(scoring_inputs, output_attentions=False),
    )
    clean_pred = str(clean_metrics["best_answer_text"]).strip()
    clean_correct = int(clean_pred == gold_answer_text)
    if not clean_correct:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"clean top-1 is {clean_pred!r}, not gold answer {gold_answer_text!r}"
        )
        return None

    clean_answer_score = float(clean_metrics["scores_by_answer"].get(gold_answer_text, float("-inf")))
    clean_correct_prob = float(clean_metrics["probs_by_answer"].get(gold_answer_text, 0.0))
    if min_clean_correct_prob is not None and clean_correct_prob < float(min_clean_correct_prob):
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"clean_correct_prob={clean_correct_prob:.4f} < "
            f"min_clean_correct_prob={float(min_clean_correct_prob):.4f}"
        )
        return None

    gold_scoring_inputs = core.append_answer_tokens_for_scoring(clean_inputs, gold_answer_ids)
    num_layers = int(len(layers))

    print(
        f"[{sample_index}/{total_samples}] sample_id={sample_id} "
        f"clean_answer_score={clean_answer_score:.4f} "
        f"clean_correct_prob={clean_correct_prob:.4f} "
        f"starting_layer={int(starting_layer)} "
        f"carrier_index={layout.carrier_index} "
        f"legacy_blocked_tokens={len(blocked_key_positions)} "
        f"transfer_layers={json.dumps([int(value) for value in transfer_lengths])} "
        f"non_frame_prompt_self_only={bool(non_frame_prompt_self_only)} "
    )

    results_by_transfer_length: Dict[str, Dict[str, Any]] = {}
    for transfer_length in transfer_lengths:
        metrics = compute_metrics_for_transfer_length(
            starting_layer=int(starting_layer),
            num_layers=num_layers,
            clean_answer_score=clean_answer_score,
            gold_scoring_inputs=gold_scoring_inputs,
            layers=layers,
            layout=layout,
            frame_group_by_token=frame_group_by_token,
            non_frame_prompt_self_only=non_frame_prompt_self_only,
            prompt_len=prompt_len,
            gold_answer_ids=gold_answer_ids,
            transfer_layers=int(transfer_length),
            debug_masks=debug_masks,
        )
        results_by_transfer_length[str(int(transfer_length))] = metrics
        print(
            "  "
            f"transfer_layers={int(transfer_length)} "
            f"peak_layer={metrics['peak_layer']} "
            f"peak_score_drop={_format_float_or_na(metrics['peak_score_drop'])}"
        )

    print(format_per_sample_score_drop_table({
        "sample_id": sample_id,
        "clean_answer_score": clean_answer_score,
        "clean_correct_prob": clean_correct_prob,
        "starting_layer": int(starting_layer),
        "transfer_lengths": [int(value) for value in transfer_lengths],
        _RESULTS_KEY: results_by_transfer_length,
    }))

    return {
        "sample_id": sample_id,
        "seq_len": int(len(frames)),
        "question": question,
        "gold_answer": gold_answer_text,
        "clean_pred": clean_pred,
        "clean_correct": int(clean_correct),
        "clean_correct_prob": float(clean_correct_prob),
        "clean_answer_score": float(clean_answer_score),
        "non_frame_prompt_self_only": bool(non_frame_prompt_self_only),
        "carrier_index": int(layout.carrier_index),
        "carrier_token": str(layout.carrier_token_text),
        "blocked_token_count": int(len(blocked_key_positions)),
        "starting_layer": int(starting_layer),
        "transfer_lengths": [int(value) for value in transfer_lengths],
        _RESULTS_KEY: results_by_transfer_length,
    }


def build_aggregate_layer_rows(
    sample_payloads: Sequence[Dict[str, Any]],
    starting_layer: int,
    transfer_lengths: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for transfer_length in transfer_lengths:
        transfer_key = str(int(transfer_length))
        clean_values: List[float] = []
        ablated_values: List[float] = []
        score_drop_values: List[float] = []

        for sample in sample_payloads:
            transfer_payload = sample.get(_RESULTS_KEY, {}).get(transfer_key, {})
            layer_row = next(
                (
                    row
                    for row in transfer_payload.get("layers", [])
                    if int(row["layer"]) == int(starting_layer)
                ),
                None,
            )
            if layer_row is None or not _is_present_number(layer_row.get("score_drop")):
                continue
            clean_values.append(float(layer_row["clean_answer_score"]))
            if _is_present_number(layer_row.get("ablated_answer_score")):
                ablated_values.append(float(layer_row["ablated_answer_score"]))
            score_drop_values.append(float(layer_row["score_drop"]))

        if not score_drop_values:
            continue
        rows.append(
            {
                "starting_layer": int(starting_layer),
                "transfer_layers": int(transfer_length),
                "n_samples": int(len(score_drop_values)),
                "mean_clean_answer_score": _mean_or_none(clean_values),
                "mean_ablated_answer_score": _mean_or_none(ablated_values),
                "mean_score_drop": _mean_or_none(score_drop_values),
                "median_score_drop": _median_or_none(score_drop_values),
                "max_score_drop": float(max(score_drop_values)),
            }
        )
    return rows


def build_plot_summary(
    sample_payloads: Sequence[Dict[str, Any]],
    selected_layers: Sequence[int],
    transfer_length: int,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Dict[str, List[float]]]:
    per_layer_values: Dict[int, List[float]] = {int(layer_idx): [] for layer_idx in selected_layers}
    transfer_key = str(int(transfer_length))
    for sample in sample_payloads:
        for layer_row in sample.get(_RESULTS_KEY, {}).get(transfer_key, {}).get("layers", []):
            layer_idx = int(layer_row["layer"])
            if layer_idx not in per_layer_values:
                continue
            value = layer_row.get("score_drop")
            if value is None:
                continue
            per_layer_values[layer_idx].append(float(value))

    if not any(per_layer_values[layer_idx] for layer_idx in per_layer_values):
        return None

    rng = random.Random(seed)
    summary = {
        "layers": [],
        "mean": [],
        "lo": [],
        "hi": [],
        "n_samples": [],
    }
    for layer_idx in selected_layers:
        values = per_layer_values[int(layer_idx)]
        if not values:
            continue
        mean_value = float(sum(values) / len(values))
        if len(values) <= 1:
            lo_value = hi_value = mean_value
        else:
            boot_means: List[float] = []
            for _ in range(int(n_bootstrap)):
                sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
                boot_means.append(float(sum(sample) / len(sample)))
            boot_means.sort()
            lo_idx = int(0.025 * (len(boot_means) - 1))
            hi_idx = int(0.975 * (len(boot_means) - 1))
            lo_value = float(boot_means[lo_idx])
            hi_value = float(boot_means[hi_idx])

        summary["layers"].append(int(layer_idx))
        summary["mean"].append(float(mean_value))
        summary["lo"].append(float(lo_value))
        summary["hi"].append(float(hi_value))
        summary["n_samples"].append(int(len(values)))

    if not summary["layers"]:
        return None
    return summary


def format_per_sample_score_drop_table(sample_payload: Dict[str, Any]) -> str:
    starting_layer = int(sample_payload["starting_layer"])
    headers = ["transfer_layers", f"start_{starting_layer}", "peak_layer", "peak_score_drop"]
    rows: List[List[str]] = []
    for transfer_length in sample_payload.get("transfer_lengths", []):
        transfer_key = str(int(transfer_length))
        transfer_payload = sample_payload.get(_RESULTS_KEY, {}).get(transfer_key, {})
        score_drop_by_layer = {
            int(layer_row["layer"]): layer_row.get("score_drop")
            for layer_row in transfer_payload.get("layers", [])
        }
        rows.append(
            [
                str(int(transfer_length)),
                _format_float_or_na(score_drop_by_layer.get(int(starting_layer))),
                "n/a" if transfer_payload.get("peak_layer") is None else str(int(transfer_payload["peak_layer"])),
                _format_float_or_na(transfer_payload.get("peak_score_drop")),
            ]
        )

    title = (
        f"Per-sample score drops: sample_id={sample_payload['sample_id']} "
        f"clean_score={float(sample_payload['clean_answer_score']):.4f} "
        f"clean_prob={float(sample_payload['clean_correct_prob']):.4f}"
    )
    return f"{title}\n{format_markdown_table(headers, rows)}"


def build_transfer_length_summary_rows(
    sample_payloads: Sequence[Dict[str, Any]],
    transfer_lengths: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for transfer_length in transfer_lengths:
        values: List[float] = []
        for sample in sample_payloads:
            for layer_row in sample.get(_RESULTS_KEY, {}).get(str(int(transfer_length)), {}).get("layers", []):
                if layer_row.get("score_drop") is None:
                    continue
                values.append(float(layer_row["score_drop"]))
        rows.append(
            {
                "transfer_layers": int(transfer_length),
                "n_values": int(len(values)),
                "mean_score_drop": None if not values else float(sum(values) / len(values)),
                "median_score_drop": None if not values else float(statistics.median(values)),
            }
        )
    return rows


def format_transfer_length_summary_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = ["transfer_layers", "n_values", "mean_score_drop", "median_score_drop"]
    values = [
        [
            str(int(row["transfer_layers"])),
            str(int(row["n_values"])),
            _format_float_or_na(row.get("mean_score_drop")),
            _format_float_or_na(row.get("median_score_drop")),
        ]
        for row in rows
    ]
    return format_markdown_table(headers, values)


def plot_transfer_length_summary(
    summary_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    title: str,
    seq_len_label: Optional[str],
) -> Optional[Path]:
    plotted_rows = [row for row in summary_rows if _is_present_number(row.get("mean_score_drop"))]
    if not plotted_rows:
        return None

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    x_values = [int(row["transfer_layers"]) for row in plotted_rows]
    y_values = [float(row["mean_score_drop"]) for row in plotted_rows]
    ax.plot(x_values, y_values, color=_PLOT_COLOR, linewidth=2.0, marker="o", markersize=4.0)

    y_min = min(y_values)
    y_max = max(y_values)
    pad = 0.1 * max(1.0e-6, y_max - y_min) if not math.isclose(y_min, y_max) else 0.1 * max(1.0, abs(y_min))
    ax.set_ylim(float(y_min - pad), float(y_max + pad))

    full_title = title if seq_len_label is None else f"{title} ({seq_len_label})"
    ax.set_title(full_title, fontsize=13, pad=10)
    ax.set_xlabel("Transfer layers")
    ax.set_ylabel("Mean score drop")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_xticks(x_values)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def resolve_shared_y_limits(summaries: Sequence[Optional[Dict[str, List[float]]]]) -> Optional[Tuple[float, float]]:
    values: List[float] = []
    for summary in summaries:
        if summary is None:
            continue
        values.extend(float(value) for value in summary["lo"])
        values.extend(float(value) for value in summary["hi"])
    if not values:
        return None

    y_min = min(values)
    y_max = max(values)
    if math.isclose(y_min, y_max):
        pad = 0.1 * max(1.0, abs(y_min))
    else:
        pad = 0.05 * (y_max - y_min)
    return float(y_min - pad), float(y_max + pad)


def plot_aggregate_summary(
    summary: Dict[str, List[float]],
    output_path: Path,
    title: str,
    y_label: str,
    x_label: str,
    line_label: str,
    color: str,
    seq_len_label: Optional[str],
    y_limits: Optional[Tuple[float, float]],
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    layers = [int(layer_idx) for layer_idx in summary["layers"]]
    mean_values = [float(value) for value in summary["mean"]]
    lo_values = [float(value) for value in summary["lo"]]
    hi_values = [float(value) for value in summary["hi"]]

    ax.plot(layers, mean_values, color=color, linewidth=2.0, label=line_label)
    ax.fill_between(layers, lo_values, hi_values, color=color, alpha=0.16)

    full_title = title if seq_len_label is None else f"{title} ({seq_len_label})"
    ax.set_title(full_title, fontsize=13, pad=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)

    tick_step = max(1, math.ceil(len(layers) / 32))
    xticks = layers[::tick_step]
    if layers[-1] not in xticks:
        xticks.append(layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_aggregate_summaries_by_transfer_length(
    summaries_by_transfer_length: Dict[int, Optional[Dict[str, List[float]]]],
    output_path: Path,
    title: str,
    seq_len_label: Optional[str],
) -> Optional[Path]:
    available_items = [
        (int(transfer_length), summary)
        for transfer_length, summary in summaries_by_transfer_length.items()
        if summary is not None
    ]
    if not available_items:
        return None

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [_PLOT_COLOR])
    y_limits = resolve_shared_y_limits([summary for _, summary in available_items])

    for curve_index, (transfer_length, summary) in enumerate(available_items):
        assert summary is not None
        layers = [int(layer_idx) for layer_idx in summary["layers"]]
        mean_values = [float(value) for value in summary["mean"]]
        lo_values = [float(value) for value in summary["lo"]]
        hi_values = [float(value) for value in summary["hi"]]
        color = color_cycle[curve_index % len(color_cycle)]
        ax.plot(
            layers,
            mean_values,
            color=color,
            linewidth=2.0,
            label=f"transfer_layers={int(transfer_length)}",
        )
        ax.fill_between(layers, lo_values, hi_values, color=color, alpha=0.16)

    full_title = title if seq_len_label is None else f"{title} ({seq_len_label})"
    ax.set_title(full_title, fontsize=13, pad=10)
    ax.set_xlabel("Start layer")
    ax.set_ylabel("Mean score drop")
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)

    all_layers = sorted({int(layer_idx) for _, summary in available_items for layer_idx in summary["layers"]})
    tick_step = max(1, math.ceil(len(all_layers) / 32))
    xticks = all_layers[::tick_step]
    if all_layers and all_layers[-1] not in xticks:
        xticks.append(all_layers[-1])
    if xticks:
        ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_sample_summary_by_transfer_length(
    sample_payload: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    transfer_lengths = [int(value) for value in sample_payload.get("transfer_lengths", [])]
    starting_layer = int(sample_payload["starting_layer"])
    results_by_transfer = sample_payload.get(_RESULTS_KEY, {})
    if not transfer_lengths or not results_by_transfer:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    y_values: List[float] = []
    x_values: List[int] = []
    y_curve: List[float] = []
    for transfer_length in transfer_lengths:
        transfer_payload = results_by_transfer.get(str(int(transfer_length)), {})
        layer_row = next(
            (
                row
                for row in transfer_payload.get("layers", [])
                if int(row["layer"]) == int(starting_layer) and row.get("score_drop") is not None
            ),
            None,
        )
        if layer_row is None:
            continue
        x_values.append(int(transfer_length))
        y_curve.append(float(layer_row["score_drop"]))

    if not x_values:
        plt.close(fig)
        return None

    y_values.extend(y_curve)
    ax.plot(
        x_values,
        y_curve,
        color=_PLOT_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=3.5,
    )

    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        pad = 0.1 * max(1.0e-6, y_max - y_min) if not math.isclose(y_min, y_max) else 0.1 * max(1.0, abs(y_min))
        ax.set_ylim(float(y_min - pad), float(y_max + pad))

    title = (
        f"{sample_payload['sample_id']}\n"
        f"start_layer={int(starting_layer)} | "
        f"clean_score={float(sample_payload['clean_answer_score']):.4f} | "
        f"clean_prob={float(sample_payload['clean_correct_prob']):.4f}"
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Transfer layers")
    ax.set_ylabel("Score drop")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_xticks(sorted(set(int(value) for value in transfer_lengths)))
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_payload['sample_id']}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _serialize_layer_value_map(layer_rows: Sequence[Dict[str, Any]], key: str) -> str:
    payload = {
        str(int(layer_row["layer"])): (
            None if layer_row.get(key) is None else float(layer_row[key])
        )
        for layer_row in layer_rows
    }
    return json.dumps(payload, sort_keys=True)


def _serialize_transfer_to_layer_value_map(sample_payload: Dict[str, Any], key: str) -> str:
    payload = {
        str(int(transfer_length)): json.loads(
            _serialize_layer_value_map(
                sample_payload.get(_RESULTS_KEY, {}).get(str(int(transfer_length)), {}).get("layers", []),
                key,
            )
        )
        for transfer_length in sample_payload.get("transfer_lengths", [])
    }
    return json.dumps(payload, sort_keys=True)


def _serialize_missing_layers_by_transfer(sample_payload: Dict[str, Any]) -> str:
    payload = {
        str(int(transfer_length)): [
            int(layer_row["layer"])
            for layer_row in sample_payload.get(_RESULTS_KEY, {}).get(str(int(transfer_length)), {}).get("layers", [])
            if layer_row.get("score_drop") is None
        ]
        for transfer_length in sample_payload.get("transfer_lengths", [])
    }
    return json.dumps(payload, sort_keys=True)


def build_per_sample_rows(
    sample_payloads: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in sample_payloads:
        peak_layer_by_transfer = {
            str(int(transfer_length)): sample.get(_RESULTS_KEY, {})
            .get(str(int(transfer_length)), {})
            .get("peak_layer")
            for transfer_length in sample.get("transfer_lengths", [])
        }
        peak_score_drop_by_transfer = {
            str(int(transfer_length)): sample.get(_RESULTS_KEY, {})
            .get(str(int(transfer_length)), {})
            .get("peak_score_drop")
            for transfer_length in sample.get("transfer_lengths", [])
        }
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "seq_len": int(sample["seq_len"]),
                "question": sample["question"],
                "gold_answer": sample["gold_answer"],
                "clean_pred": sample["clean_pred"],
                "clean_correct": int(sample["clean_correct"]),
                "clean_correct_prob": float(sample["clean_correct_prob"]),
                "clean_answer_score": float(sample["clean_answer_score"]),
                "transfer_layers": json.dumps(sample.get("transfer_lengths", [])),
                "non_frame_prompt_self_only": int(bool(sample["non_frame_prompt_self_only"])),
                "carrier_index": int(sample["carrier_index"]),
                "carrier_token": sample["carrier_token"],
                "blocked_token_count": int(sample["blocked_token_count"]),
                "starting_layer": int(sample["starting_layer"]),
                "peak_layer_by_transfer_layers": json.dumps(peak_layer_by_transfer, sort_keys=True),
                "peak_score_drop_by_transfer_layers": json.dumps(peak_score_drop_by_transfer, sort_keys=True),
                "ablated_answer_scores_by_transfer_layers": _serialize_transfer_to_layer_value_map(
                    sample,
                    "ablated_answer_score",
                ),
                "score_drops_by_transfer_layers": _serialize_transfer_to_layer_value_map(sample, "score_drop"),
                "missing_layers_by_transfer_layers": _serialize_missing_layers_by_transfer(sample),
            }
        )
    return rows


def _estimate_sustained_crossing(
    layers: Sequence[int],
    values: Sequence[Optional[float]],
    comparator: Any,
    threshold: float,
    window_size: int,
    required_hits: int,
) -> Optional[int]:
    valid = [
        (int(layer_idx), float(value))
        for layer_idx, value in zip(layers, values)
        if value is not None and math.isfinite(float(value))
    ]
    if not valid:
        return None

    for start_idx, (layer_idx, value) in enumerate(valid):
        window = [window_value for _, window_value in valid[start_idx : start_idx + window_size]]
        if not comparator(value, threshold):
            continue
        if sum(1 for window_value in window if comparator(window_value, threshold)) >= min(required_hits, len(window)):
            return int(layer_idx)
    return None


def estimate_transition_window(
    suffix_rows: Sequence[Dict[str, Any]],
    selected_layers: Sequence[int],
) -> Dict[str, Any]:
    suffix_map = {int(row["layer"]): row for row in suffix_rows}
    ordered_layers = [int(layer_idx) for layer_idx in selected_layers]
    suffix_values = [
        None if ordered_layers[idx] not in suffix_map else suffix_map[ordered_layers[idx]].get("mean_score_drop")
        for idx in range(len(ordered_layers))
    ]

    valid_suffix_values = [float(value) for value in suffix_values if _is_present_number(value)]
    prefix_peak = None
    suffix_peak = max(valid_suffix_values) if valid_suffix_values else None

    end_layer_estimate = None
    if suffix_peak is not None and suffix_peak > 0.0:
        end_layer_estimate = _estimate_sustained_crossing(
            ordered_layers,
            suffix_values,
            comparator=lambda value, threshold: value <= threshold,
            threshold=0.1 * float(suffix_peak),
            window_size=4,
            required_hits=3,
        )

    return {
        "start_layer_estimate": None,
        "end_layer_estimate": end_layer_estimate,
        "description": (
            "Suffix-only heuristic on tested layers: end = first layer whose mean score drop falls to at most "
            "10% of the suffix peak and stays that low for at least 3 of the next 4 tested layers."
        ),
        "prefix_peak_mean_score_drop": prefix_peak,
        "suffix_peak_mean_score_drop": suffix_peak,
        "sustain_window_size": 4,
        "sustain_required_hits": 3,
        "prefix_rise_fraction_of_peak": None,
        "suffix_collapse_fraction_of_peak": 0.1,
    }


def write_outputs(
    sample_payloads: List[Dict[str, Any]],
    aggregate_rows: List[Dict[str, Any]],
    output_dir: Path,
    seq_len_label: Optional[str],
    args: argparse.Namespace,
    starting_layer: int,
    transfer_lengths: Sequence[int],
    total_sample_dirs: int,
    scanned_samples: int,
    elapsed_seconds: float,
) -> None:
    per_sample_plot_dir = output_dir / "per_sample_plots_by_transfer_layers"
    per_sample_csv_path = output_dir / "full_af1_per_sample.csv"
    aggregate_csv_path = output_dir / "full_af1_aggregate_by_transfer_layers.csv"
    summary_json_path = output_dir / "summary.json"

    per_sample_rows = build_per_sample_rows(sample_payloads)
    af1_utils.write_csv(per_sample_csv_path, per_sample_rows, fieldnames=_PER_SAMPLE_FIELDS)
    af1_utils.write_csv(aggregate_csv_path, aggregate_rows, fieldnames=_AGGREGATE_FIELDS)

    summary_rows = build_transfer_length_summary_rows(sample_payloads, transfer_lengths)
    aggregate_plot_path: Optional[Path] = None
    if not args.disable_plots:
        aggregate_plot_path = plot_transfer_length_summary(
            summary_rows,
            output_dir / f"full_af1_mean_score_drop_by_transfer_layers{f'_{seq_len_label}' if seq_len_label else ''}.png",
            title=f"Mean full-AF1 score drop by transfer layers (start_layer={int(starting_layer)})",
            seq_len_label=seq_len_label,
        )

    summary_payload = {
        "data_root": args.data_root,
        "output_dir": str(output_dir),
        "limit": int(args.limit),
        "starting_layer": int(starting_layer),
        "transfer_layers": [int(value) for value in transfer_lengths],
        "non_frame_prompt_self_only": bool(args.non_frame_prompt_self_only),
        "min_clean_correct_prob": (
            None if args.min_clean_correct_prob is None else float(args.min_clean_correct_prob)
        ),
        "carrier_definition": "final prompt token before answer generation",
        "masking_definition": (
            "at tested start layer l, apply the frame-aware transition mask at layers "
            "[l, l + transfer_layers), then the post-transfer AF1 mask at layers >= l + transfer_layers"
        ),
        "blocked_keys_definition": (
            "per-sample blocked_token_count is retained as a legacy carrier-only compatibility/debug count of "
            "previous prompt tokens before the carrier; under the active mask, frame-token queries are "
            "restricted to their own frame block, non-frame prompt rows keep the base mask during transfer, "
            "and if non_frame_prompt_self_only is enabled they become self-only only after "
            "transfer; the carrier keeps access to earlier prompt tokens during transfer before becoming "
            "self-only after transfer"
        ),
        "num_total_sample_dirs": int(total_sample_dirs),
        "num_scanned_samples": int(scanned_samples),
        "num_retained_samples": int(len(sample_payloads)),
        "per_sample_plot_dir": None if args.disable_plots else str(per_sample_plot_dir),
        "aggregate_plot": None if aggregate_plot_path is None else str(aggregate_plot_path),
        "per_sample_csv": str(per_sample_csv_path),
        "aggregate_csv": str(aggregate_csv_path),
        "runtime": eval_utils.format_runtime(elapsed_seconds),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if aggregate_plot_path is not None:
        print(f"Wrote aggregate plot to: {aggregate_plot_path}")
    else:
        print("Aggregate plot was not written.")
    print(f"Wrote per-sample CSV to: {per_sample_csv_path}")
    print(f"Wrote aggregate-by-transfer-length CSV to: {aggregate_csv_path}")
    print(f"Wrote summary JSON to: {summary_json_path}")
    print(f"Transfer-length summary across all retained samples at start_layer={int(starting_layer)}:")
    print(format_transfer_length_summary_table(summary_rows))


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_len_label = eval_utils.resolve_seq_len_label(data_root)
    per_sample_plot_dir = output_dir / "per_sample_plots_by_transfer_layers"

    sample_dirs = iter_sample_dirs(data_root)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under: {data_root}")

    layers = get_layers(base_model)
    starting_layer = int(args.starting_layer)
    if starting_layer < 0 or starting_layer >= int(len(layers)):
        raise ValueError(
            f"--starting-layer must be within [0, {int(len(layers)) - 1}], got {starting_layer}"
        )
    transfer_lengths = parse_transfer_layer_selection(args.transfer_layers)

    sample_payloads: List[Dict[str, Any]] = []
    scanned_samples = 0
    for sample_index, sample_dir in enumerate(sample_dirs, start=1):
        if len(sample_payloads) >= int(args.limit):
            break
        scanned_samples = int(sample_index)
        sample_payload = compute_sample_payload(
            sample_dir=sample_dir,
            sample_index=sample_index,
            total_samples=len(sample_dirs),
            layers=layers,
            starting_layer=starting_layer,
            non_frame_prompt_self_only=bool(args.non_frame_prompt_self_only),
            min_clean_correct_prob=args.min_clean_correct_prob,
            debug_masks=bool(args.debug_masks),
            transfer_lengths=transfer_lengths,
        )
        if sample_payload is None:
            continue
        if not args.disable_plots:
            per_sample_plot_path = plot_per_sample_summary_by_transfer_length(
                sample_payload,
                per_sample_plot_dir,
            )
            if per_sample_plot_path is not None:
                print(f"Wrote per-sample transfer-length plot to: {per_sample_plot_path}")
        sample_payloads.append(sample_payload)

    aggregate_rows = build_aggregate_layer_rows(
        sample_payloads,
        starting_layer=starting_layer,
        transfer_lengths=transfer_lengths,
    )
    write_outputs(
        sample_payloads=sample_payloads,
        aggregate_rows=aggregate_rows,
        output_dir=output_dir,
        seq_len_label=seq_len_label,
        args=args,
        starting_layer=starting_layer,
        transfer_lengths=transfer_lengths,
        total_sample_dirs=len(sample_dirs),
        scanned_samples=scanned_samples,
        elapsed_seconds=time.perf_counter() - start_time,
    )

    print(
        f"Retained {len(sample_payloads)} samples "
        f"(target limit={int(args.limit)}, scanned={scanned_samples}/{len(sample_dirs)}, "
        f"starting_layer={int(starting_layer)}, "
        f"transfer_layers={json.dumps([int(value) for value in transfer_lengths])}, "
        f"non_frame_prompt_self_only={bool(args.non_frame_prompt_self_only)}, "
        f"min_clean_correct_prob="
        f"{'none' if args.min_clean_correct_prob is None else f'{float(args.min_clean_correct_prob):.4f}'})."
    )
    print(eval_utils.format_runtime(time.perf_counter() - start_time))


if __name__ == "__main__":
    main()

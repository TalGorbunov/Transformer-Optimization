"""
Find which contiguous groups of non-frame prompt tokens rescue the answer when
reopened during one fixed transition window.

For each retained MMRed sample:
- define the carrier as the final prompt token before answer generation
- score the clean gold answer
- at layers [starting_layer, starting_layer + transition_layers):
  - keep frame-token transition masking identical to the full-AF1 transition script
  - keep carrier transition masking identical to the full-AF1 transition script
  - block non-frame prompt tokens to self-plus-optional-spared-token attention
  - reopen exactly one contiguous non-frame prompt group at a time
- at layers >= starting_layer + transition_layers:
  - reuse the original post-transfer AF1 masking logic unchanged
- record rescue_amount = blocked_baseline_score_drop - candidate_score_drop
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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

_RUNS_KEY = "group_rescue_runs"
_GROUPS_KEY = "non_frame_prompt_groups"
_PLOT_COLOR = "#ff7f0e"
_BASELINE_PLOT_COLOR = "#1f77b4"
_BEST_GROUP_COLOR = "#2ca02c"
_MISSING_BAR_COLOR = "#c7c7c7"
_PER_SAMPLE_FIELDS = [
    "sample_id",
    "seq_len",
    "question",
    "gold_answer",
    "clean_pred",
    "clean_correct",
    "clean_correct_prob",
    "clean_answer_score",
    "starting_layer",
    "transition_layers",
    "carrier_index",
    "carrier_token",
    "num_non_frame_prompt_tokens",
    "num_groups",
    "run_type",
    "group_index",
    "group_start_token_position",
    "group_end_token_position",
    "group_size",
    "group_token_positions_json",
    "group_token_texts_json",
    "ablated_answer_score",
    "score_drop",
    "baseline_score_drop",
    "rescue_amount",
    "bos_handling",
    "spared_position",
]
_AGGREGATE_FIELDS = [
    "group_index",
    "n_samples",
    "mean_group_size",
    "mean_ablated_answer_score",
    "mean_score_drop",
    "median_score_drop",
    "mean_rescue_amount",
    "median_rescue_amount",
    "max_rescue_amount",
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
            "Find which contiguous groups of non-frame prompt tokens rescue the answer "
            "when reopened during one fixed transition window."
        )
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--output", type=str, default="outputs/find_non_frame_prompt_group_rescues")
    ap.add_argument("--starting-layer", type=int, required=True, dest="starting_layer")
    ap.add_argument("--transition-layers", type=int, required=True, dest="transition_layers")
    ap.add_argument("--num-groups", type=int, required=True, dest="num_groups")
    ap.add_argument(
        "--allow_bos_attention",
        type=_parse_bool,
        default=False,
        help="Whether the carrier is allowed to keep attending to BOS while all other previous tokens are masked.",
    )
    ap.add_argument(
        "--allow_first_token_if_no_bos",
        type=_parse_bool,
        default=False,
        help="If BOS attention is allowed but no real BOS token is present, optionally spare token position 0 instead.",
    )
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
        "--debug_masks",
        action="store_true",
        help=(
            "Print lightweight frame-mask sanity checks plus representative non-frame-group "
            "transition rows for one retained sample."
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
    if int(args.transition_layers) <= 0:
        raise ValueError("--transition-layers must be a positive integer")
    if int(args.num_groups) <= 0:
        raise ValueError("--num-groups must be a positive integer")
    if args.min_clean_correct_prob is not None:
        min_prob = float(args.min_clean_correct_prob)
        if min_prob < 0.0 or min_prob > 1.0:
            raise ValueError("--min_clean_correct_prob must be within [0, 1] when provided")
    return args


def resolve_blocked_key_positions(
    carrier_index: int,
    bos_index: Optional[int],
    allow_bos_attention: bool,
    allow_first_token_if_no_bos: bool,
) -> Tuple[List[int], Optional[int], str]:
    spared_position: Optional[int] = None
    spared_token_kind = "neither"
    if allow_bos_attention:
        if bos_index is not None:
            spared_position = int(bos_index)
            spared_token_kind = "real_bos"
        elif allow_first_token_if_no_bos and int(carrier_index) > 0:
            spared_position = 0
            spared_token_kind = "first_token_fallback"

    blocked_positions: List[int] = []
    for position in range(int(carrier_index)):
        if spared_position is not None and int(position) == int(spared_position):
            continue
        blocked_positions.append(int(position))
    return blocked_positions, spared_position, spared_token_kind


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


def collect_non_frame_prompt_positions(
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    spared_position: Optional[int],
) -> List[int]:
    frame_positions = set(int(position) for position in frame_group_by_token.keys())
    positions: List[int] = []
    for position in range(int(layout.prompt_len)):
        if int(position) in frame_positions:
            continue
        if int(position) == int(layout.carrier_index):
            continue
        if spared_position is not None and int(position) == int(spared_position):
            continue
        positions.append(int(position))
    return positions


def split_positions_into_groups(positions: Sequence[int], num_groups: int) -> List[List[int]]:
    if int(num_groups) <= 0:
        raise ValueError("num_groups must be positive")

    normalized_positions = [int(position) for position in positions]
    total_positions = len(normalized_positions)
    base_group_size = total_positions // int(num_groups)
    remainder = total_positions % int(num_groups)

    groups: List[List[int]] = []
    cursor = 0
    for group_index in range(int(num_groups)):
        current_size = int(base_group_size + (1 if group_index < remainder else 0))
        groups.append(normalized_positions[cursor : cursor + current_size])
        cursor += current_size
    return groups


def build_group_records(
    layout: af1_utils.TokenLayout,
    non_frame_prompt_positions: Sequence[int],
    num_groups: int,
) -> List[Dict[str, Any]]:
    groups = split_positions_into_groups(non_frame_prompt_positions, num_groups)
    records: List[Dict[str, Any]] = []
    for group_index, group_positions in enumerate(groups):
        token_texts = [
            af1_utils.sanitize_token_text(str(layout.prompt_decoded_tokens[int(position)]))
            for position in group_positions
        ]
        records.append(
            {
                "group_index": int(group_index),
                "token_positions": [int(position) for position in group_positions],
                "token_texts": token_texts,
                "group_size": int(len(group_positions)),
                "group_start_token_position": (
                    None if not group_positions else int(group_positions[0])
                ),
                "group_end_token_position": (
                    None if not group_positions else int(group_positions[-1])
                ),
            }
        )
    return records


def allowed_transition_prompt_keys(
    query_idx: int,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    spared_position: Optional[int],
    blocked_non_frame_positions: Set[int],
    reopened_non_frame_positions: Set[int],
) -> Optional[List[int]]:
    if int(query_idx) == int(layout.carrier_index):
        allowed = set(range(0, int(layout.carrier_index) + 1))
        if spared_position is not None:
            allowed.add(int(spared_position))
        return sorted(allowed)

    same_frame_group = frame_group_by_token.get(int(query_idx))
    if same_frame_group is not None:
        allowed = {int(position) for position in same_frame_group}
        if spared_position is not None:
            allowed.add(int(spared_position))
        return sorted(allowed)

    if int(query_idx) not in blocked_non_frame_positions:
        return None
    if int(query_idx) in reopened_non_frame_positions:
        return None

    allowed = {int(query_idx)}
    if spared_position is not None:
        allowed.add(int(spared_position))
    return sorted(allowed)


def allowed_post_transfer_prompt_keys(
    query_idx: int,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    spared_position: Optional[int],
    non_frame_prompt_self_only: bool = False,
) -> Optional[List[int]]:
    if int(query_idx) == int(layout.carrier_index):
        allowed = {int(layout.carrier_index)}
        if spared_position is not None:
            allowed.add(int(spared_position))
        return sorted(allowed)

    same_frame_group = frame_group_by_token.get(int(query_idx))
    if same_frame_group is None:
        if non_frame_prompt_self_only:
            allowed = {int(query_idx)}
            if spared_position is not None:
                allowed.add(int(spared_position))
            return sorted(allowed)
        return None

    allowed = {int(position) for position in same_frame_group}
    if spared_position is not None:
        allowed.add(int(spared_position))
    return sorted(allowed)


def build_frame_aware_prompt_attention_mask(
    base_mask: torch.Tensor,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    spared_position: Optional[int],
    non_frame_prompt_self_only: bool,
    allowed_prompt_keys_fn: Callable[
        [int, af1_utils.TokenLayout, Dict[int, Tuple[int, ...]], Optional[int], bool],
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
            spared_position=spared_position,
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
    spared_position: Optional[int],
    blocked_non_frame_positions: Sequence[int],
    reopened_non_frame_positions: Sequence[int],
) -> torch.Tensor:
    blocked_set = {int(position) for position in blocked_non_frame_positions}
    reopened_set = {int(position) for position in reopened_non_frame_positions}

    def allowed_prompt_keys_fn(
        query_idx: int,
        layout: af1_utils.TokenLayout,
        frame_group_by_token: Dict[int, Tuple[int, ...]],
        spared_position: Optional[int],
        non_frame_prompt_self_only: bool,
    ) -> Optional[List[int]]:
        del non_frame_prompt_self_only
        return allowed_transition_prompt_keys(
            query_idx=query_idx,
            layout=layout,
            frame_group_by_token=frame_group_by_token,
            spared_position=spared_position,
            blocked_non_frame_positions=blocked_set,
            reopened_non_frame_positions=reopened_set,
        )

    return build_frame_aware_prompt_attention_mask(
        base_mask=base_mask,
        layout=layout,
        frame_group_by_token=frame_group_by_token,
        spared_position=spared_position,
        non_frame_prompt_self_only=False,
        allowed_prompt_keys_fn=allowed_prompt_keys_fn,
    )


def build_post_transfer_attention_mask(
    base_mask: torch.Tensor,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    spared_position: Optional[int],
    non_frame_prompt_self_only: bool,
) -> torch.Tensor:
    return build_frame_aware_prompt_attention_mask(
        base_mask=base_mask,
        layout=layout,
        frame_group_by_token=frame_group_by_token,
        spared_position=spared_position,
        non_frame_prompt_self_only=non_frame_prompt_self_only,
        allowed_prompt_keys_fn=allowed_post_transfer_prompt_keys,
    )


def stage_for_layer(
    layer_idx: int,
    start_layer: int,
    transition_layers: int,
) -> Optional[str]:
    if int(layer_idx) < int(start_layer):
        return None
    if int(layer_idx) < int(start_layer) + int(transition_layers):
        return "transition"
    return "post_transfer"


def run_model_with_last_token_mask(
    inputs: Dict[str, torch.Tensor],
    layers: Any,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    spared_position: Optional[int],
    blocked_non_frame_positions: Sequence[int],
    reopened_non_frame_positions: Sequence[int],
    non_frame_prompt_self_only: bool,
    start_layer: int,
    transition_layers: int,
) -> Any:
    def wrapper_factory(layer_idx: int, original_forward: Any) -> Any:
        layer_stage = stage_for_layer(
            layer_idx=int(layer_idx),
            start_layer=int(start_layer),
            transition_layers=transition_layers,
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
                    spared_position=spared_position,
                    blocked_non_frame_positions=blocked_non_frame_positions,
                    reopened_non_frame_positions=reopened_non_frame_positions,
                )
            elif layer_stage == "post_transfer":
                kwargs["attention_mask"] = build_post_transfer_attention_mask(
                    expanded_mask,
                    layout=layout,
                    frame_group_by_token=frame_group_by_token,
                    spared_position=spared_position,
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
    spared_position: Optional[int],
    blocked_non_frame_positions: Sequence[int],
    reopened_non_frame_positions: Sequence[int],
    non_frame_prompt_self_only: bool,
    start_layer: int,
    transition_layers: int,
    prompt_len: int,
    answer_token_ids: List[int],
) -> float:
    outputs = run_model_with_last_token_mask(
        scoring_inputs,
        layers=layers,
        layout=layout,
        frame_group_by_token=frame_group_by_token,
        spared_position=spared_position,
        blocked_non_frame_positions=blocked_non_frame_positions,
        reopened_non_frame_positions=reopened_non_frame_positions,
        non_frame_prompt_self_only=non_frame_prompt_self_only,
        start_layer=start_layer,
        transition_layers=transition_layers,
    )
    return af1_utils.sequence_logprob_from_outputs(
        outputs,
        prompt_len=prompt_len,
        answer_token_ids=answer_token_ids,
    )


def format_post_transfer_mask_debug(
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    spared_position: Optional[int],
    non_frame_prompt_self_only: bool,
) -> str:
    carrier_allowed_keys = allowed_post_transfer_prompt_keys(
        query_idx=int(layout.carrier_index),
        layout=layout,
        frame_group_by_token=frame_group_by_token,
        spared_position=spared_position,
        non_frame_prompt_self_only=non_frame_prompt_self_only,
    ) or []
    expected_keys = {int(layout.carrier_index)}
    if spared_position is not None:
        expected_keys.add(int(spared_position))
    return (
        f"post_transfer_carrier_allowed_keys={carrier_allowed_keys} "
        f"carrier_self_only={set(carrier_allowed_keys) == expected_keys} "
        f"non_frame_prompt_self_only={bool(non_frame_prompt_self_only)}"
    )


def _summarize_allowed_keys(allowed_keys: Optional[List[int]], max_items: int = 18) -> str:
    if allowed_keys is None:
        return "base_mask"
    normalized = [int(value) for value in allowed_keys]
    if len(normalized) <= max_items:
        return json.dumps(normalized)
    head = normalized[: max_items // 2]
    tail = normalized[-max_items // 2 :]
    return json.dumps(head + ["..."] + tail)


def format_transition_run_debug(
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    spared_position: Optional[int],
    non_frame_prompt_positions: Sequence[int],
    group_records: Sequence[Dict[str, Any]],
    reopened_group_index: Optional[int],
    reopened_non_frame_positions: Sequence[int],
) -> str:
    blocked_set = {int(position) for position in non_frame_prompt_positions}
    reopened_set = {int(position) for position in reopened_non_frame_positions}
    grouped_positions = [
        [int(position) for position in group_record["token_positions"]]
        for group_record in group_records
    ]

    representative_queries: List[int] = []
    if layout.frame_groups and layout.frame_groups[0]:
        representative_queries.append(int(layout.frame_groups[0][0]))
    if non_frame_prompt_positions:
        representative_queries.append(int(non_frame_prompt_positions[0]))
    if reopened_non_frame_positions:
        representative_queries.append(int(reopened_non_frame_positions[0]))
    representative_queries.append(int(layout.carrier_index))

    unique_queries: List[int] = []
    for query_idx in representative_queries:
        if int(query_idx) not in unique_queries:
            unique_queries.append(int(query_idx))

    lines = [
        f"transition_non_frame_prompt_tokens={len(non_frame_prompt_positions)}",
        f"transition_grouped_token_indices={json.dumps(grouped_positions)}",
        (
            "reopened_group_index=baseline_blocked"
            if reopened_group_index is None
            else f"reopened_group_index={int(reopened_group_index)}"
        ),
    ]
    for query_idx in unique_queries[:5]:
        allowed_keys = allowed_transition_prompt_keys(
            query_idx=int(query_idx),
            layout=layout,
            frame_group_by_token=frame_group_by_token,
            spared_position=spared_position,
            blocked_non_frame_positions=blocked_set,
            reopened_non_frame_positions=reopened_set,
        )
        token_text = af1_utils.sanitize_token_text(str(layout.prompt_decoded_tokens[int(query_idx)]))
        lines.append(
            f"transition_query_idx={int(query_idx)} token={token_text!r} "
            f"allowed_keys={_summarize_allowed_keys(allowed_keys)}"
        )
    return "\n".join(lines)


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


def compute_run_metrics(
    run_type: str,
    clean_answer_score: float,
    gold_scoring_inputs: Dict[str, torch.Tensor],
    layers: Any,
    layout: af1_utils.TokenLayout,
    frame_group_by_token: Dict[int, Tuple[int, ...]],
    spared_position: Optional[int],
    non_frame_prompt_positions: Sequence[int],
    group_records: Sequence[Dict[str, Any]],
    group_record: Optional[Dict[str, Any]],
    non_frame_prompt_self_only: bool,
    prompt_len: int,
    gold_answer_ids: List[int],
    starting_layer: int,
    transition_layers: int,
    debug_masks: bool,
    emit_detailed_debug: bool,
) -> Dict[str, Any]:
    reopened_group_index = None if group_record is None else int(group_record["group_index"])
    reopened_non_frame_positions = [] if group_record is None else list(group_record["token_positions"])
    if debug_masks and emit_detailed_debug:
        debug_prefix = (
            "  [debug] "
            f"run_type={run_type} "
            f"group_index={'none' if reopened_group_index is None else int(reopened_group_index)}"
        )
        print(
            f"{debug_prefix} "
            f"start_layer={int(starting_layer)} transition_layers={int(transition_layers)}"
        )
        for line in format_transition_run_debug(
            layout=layout,
            frame_group_by_token=frame_group_by_token,
            spared_position=spared_position,
            non_frame_prompt_positions=non_frame_prompt_positions,
            group_records=group_records,
            reopened_group_index=reopened_group_index,
            reopened_non_frame_positions=reopened_non_frame_positions,
        ).splitlines():
            print(f"{debug_prefix} {line}")
        print(
            f"{debug_prefix} "
            f"{format_post_transfer_mask_debug(layout, frame_group_by_token, spared_position, non_frame_prompt_self_only)}"
        )

    try:
        ablated_answer_score = run_window_ablation_logprob(
            gold_scoring_inputs,
            layers=layers,
            layout=layout,
            frame_group_by_token=frame_group_by_token,
            spared_position=spared_position,
            blocked_non_frame_positions=non_frame_prompt_positions,
            reopened_non_frame_positions=reopened_non_frame_positions,
            non_frame_prompt_self_only=non_frame_prompt_self_only,
            start_layer=int(starting_layer),
            transition_layers=int(transition_layers),
            prompt_len=prompt_len,
            answer_token_ids=gold_answer_ids,
        )
    except Exception as exc:
        print(
            "  "
            f"run_type={run_type} "
            f"group_index={'none' if reopened_group_index is None else int(reopened_group_index)} "
            f"start_layer={int(starting_layer)} "
            f"transition_layers={int(transition_layers)} "
            f"failed non-frame prompt group rescue masking ({exc}); "
            "recording missing values for this run"
        )
        ablated_answer_score = None

    score_drop = (
        None
        if ablated_answer_score is None
        else float(clean_answer_score - ablated_answer_score)
    )
    return {
        "run_type": str(run_type),
        "group_index": None if group_record is None else int(group_record["group_index"]),
        "group_start_token_position": (
            None if group_record is None else group_record["group_start_token_position"]
        ),
        "group_end_token_position": (
            None if group_record is None else group_record["group_end_token_position"]
        ),
        "group_size": 0 if group_record is None else int(group_record["group_size"]),
        "group_token_positions": [] if group_record is None else list(group_record["token_positions"]),
        "group_token_texts": [] if group_record is None else list(group_record["token_texts"]),
        "ablated_answer_score": None if ablated_answer_score is None else float(ablated_answer_score),
        "score_drop": None if score_drop is None else float(score_drop),
    }


def format_per_sample_results_table(sample_payload: Dict[str, Any]) -> str:
    headers = [
        "run_type",
        "group_index",
        "group_size",
        "ablated_answer_score",
        "score_drop",
        "rescue_amount",
    ]
    rows: List[List[str]] = []
    for run in sample_payload.get(_RUNS_KEY, []):
        rows.append(
            [
                str(run["run_type"]),
                "baseline" if run.get("group_index") is None else str(int(run["group_index"])),
                str(int(run["group_size"])),
                _format_float_or_na(run.get("ablated_answer_score")),
                _format_float_or_na(run.get("score_drop")),
                _format_float_or_na(run.get("rescue_amount")),
            ]
        )
    title = (
        f"Per-sample group rescues: sample_id={sample_payload['sample_id']} "
        f"clean_score={float(sample_payload['clean_answer_score']):.4f} "
        f"clean_prob={float(sample_payload['clean_correct_prob']):.4f} "
        f"baseline_drop={_format_float_or_na(sample_payload.get('blocked_baseline_score_drop'))}"
    )
    return f"{title}\n{format_markdown_table(headers, rows)}"


def compute_sample_payload(
    sample_dir: Path,
    sample_index: int,
    total_samples: int,
    layers: Any,
    starting_layer: int,
    transition_layers: int,
    num_groups: int,
    allow_bos_attention: bool,
    allow_first_token_if_no_bos: bool,
    non_frame_prompt_self_only: bool,
    min_clean_correct_prob: Optional[float],
    debug_masks: bool,
    emit_detailed_debug: bool,
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

    blocked_key_positions, spared_position, spared_token_kind = resolve_blocked_key_positions(
        carrier_index=int(layout.carrier_index),
        bos_index=layout.bos_index,
        allow_bos_attention=allow_bos_attention,
        allow_first_token_if_no_bos=allow_first_token_if_no_bos,
    )
    frame_group_by_token = build_frame_group_by_token(layout)
    non_frame_prompt_positions = collect_non_frame_prompt_positions(
        layout=layout,
        frame_group_by_token=frame_group_by_token,
        spared_position=spared_position,
    )
    group_records = build_group_records(
        layout=layout,
        non_frame_prompt_positions=non_frame_prompt_positions,
        num_groups=int(num_groups),
    )

    prompt_len = int(layout.prompt_len)
    gold_answer_text = str(answer_text).strip()
    try:
        gold_answer_ids = core.token_ids_of_answer(gold_answer_text)
    except Exception as exc:
        print(
            f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: "
            f"answer tokenization failed ({exc})"
        )
        return None

    try:
        clean_metrics = af1_utils.score_valid_numeric_answers_with_runner(
            clean_inputs,
            prompt_len=prompt_len,
            num_frames=len(frames),
            runner=lambda scoring_inputs, answer_ids: af1_utils.run_clean_model(
                scoring_inputs,
                output_attentions=False,
            ),
        )
    except Exception as exc:
        print(f"[{sample_index}/{total_samples}] sample_id={sample_id} skipped: clean scoring failed ({exc})")
        return None

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

    print(
        f"[{sample_index}/{total_samples}] sample_id={sample_id} "
        f"clean_answer_score={clean_answer_score:.4f} "
        f"clean_correct_prob={clean_correct_prob:.4f} "
        f"starting_layer={int(starting_layer)} "
        f"transition_layers={int(transition_layers)} "
        f"carrier_index={layout.carrier_index} "
        f"legacy_blocked_tokens={len(blocked_key_positions)} "
        f"num_non_frame_prompt_tokens={len(non_frame_prompt_positions)} "
        f"num_groups={int(num_groups)} "
        f"group_sizes={json.dumps([int(group['group_size']) for group in group_records])} "
        f"allow_bos_attention={bool(allow_bos_attention)} "
        f"non_frame_prompt_self_only={bool(non_frame_prompt_self_only)} "
        f"bos_handling={spared_token_kind} "
        f"spared_position={'none' if spared_position is None else int(spared_position)}"
    )

    run_records: List[Dict[str, Any]] = []
    baseline_run = compute_run_metrics(
        run_type="baseline_blocked",
        clean_answer_score=clean_answer_score,
        gold_scoring_inputs=gold_scoring_inputs,
        layers=layers,
        layout=layout,
        frame_group_by_token=frame_group_by_token,
        spared_position=spared_position,
        non_frame_prompt_positions=non_frame_prompt_positions,
        group_records=group_records,
        group_record=None,
        non_frame_prompt_self_only=non_frame_prompt_self_only,
        prompt_len=prompt_len,
        gold_answer_ids=gold_answer_ids,
        starting_layer=int(starting_layer),
        transition_layers=int(transition_layers),
        debug_masks=debug_masks,
        emit_detailed_debug=emit_detailed_debug,
    )
    baseline_run["baseline_score_drop"] = baseline_run.get("score_drop")
    baseline_run["rescue_amount"] = None
    run_records.append(baseline_run)
    print(
        "  "
        f"run_type=baseline_blocked "
        f"score_drop={_format_float_or_na(baseline_run.get('score_drop'))} "
        f"ablated_answer_score={_format_float_or_na(baseline_run.get('ablated_answer_score'))}"
    )

    detailed_candidate_emitted = not emit_detailed_debug
    baseline_score_drop = baseline_run.get("score_drop")
    for group_record in group_records:
        candidate_run = compute_run_metrics(
            run_type="group_reopened",
            clean_answer_score=clean_answer_score,
            gold_scoring_inputs=gold_scoring_inputs,
            layers=layers,
            layout=layout,
            frame_group_by_token=frame_group_by_token,
            spared_position=spared_position,
            non_frame_prompt_positions=non_frame_prompt_positions,
            group_records=group_records,
            group_record=group_record,
            non_frame_prompt_self_only=non_frame_prompt_self_only,
            prompt_len=prompt_len,
            gold_answer_ids=gold_answer_ids,
            starting_layer=int(starting_layer),
            transition_layers=int(transition_layers),
            debug_masks=debug_masks,
            emit_detailed_debug=(emit_detailed_debug and not detailed_candidate_emitted),
        )
        candidate_score_drop = candidate_run.get("score_drop")
        rescue_amount = (
            None
            if not (_is_present_number(baseline_score_drop) and _is_present_number(candidate_score_drop))
            else float(float(baseline_score_drop) - float(candidate_score_drop))
        )
        candidate_run["baseline_score_drop"] = baseline_score_drop
        candidate_run["rescue_amount"] = rescue_amount
        run_records.append(candidate_run)
        detailed_candidate_emitted = True
        print(
            "  "
            f"run_type=group_reopened "
            f"group_index={int(group_record['group_index'])} "
            f"group_size={int(group_record['group_size'])} "
            f"score_drop={_format_float_or_na(candidate_score_drop)} "
            f"rescue_amount={_format_float_or_na(rescue_amount)}"
        )

    best_candidate = max(
        [
            run
            for run in run_records
            if run.get("run_type") == "group_reopened" and _is_present_number(run.get("rescue_amount"))
        ],
        key=lambda run: (float(run["rescue_amount"]), -int(run["group_index"])),
        default=None,
    )
    if best_candidate is None:
        print("  best_group_index=none best_rescue_amount=n/a")
    else:
        print(
            "  "
            f"best_group_index={int(best_candidate['group_index'])} "
            f"best_rescue_amount={float(best_candidate['rescue_amount']):.4f}"
        )

    sample_payload = {
        "sample_id": sample_id,
        "seq_len": int(len(frames)),
        "question": question,
        "gold_answer": gold_answer_text,
        "clean_pred": clean_pred,
        "clean_correct": int(clean_correct),
        "clean_correct_prob": float(clean_correct_prob),
        "clean_answer_score": float(clean_answer_score),
        "allow_bos_attention": bool(allow_bos_attention),
        "allow_first_token_if_no_bos": bool(allow_first_token_if_no_bos),
        "non_frame_prompt_self_only": bool(non_frame_prompt_self_only),
        "spared_position": spared_position,
        "spared_token_kind": spared_token_kind,
        "carrier_index": int(layout.carrier_index),
        "carrier_token": str(layout.carrier_token_text),
        "num_non_frame_prompt_tokens": int(len(non_frame_prompt_positions)),
        "num_groups": int(num_groups),
        "starting_layer": int(starting_layer),
        "transition_layers": int(transition_layers),
        "blocked_baseline_answer_score": baseline_run.get("ablated_answer_score"),
        "blocked_baseline_score_drop": baseline_run.get("score_drop"),
        _GROUPS_KEY: group_records,
        _RUNS_KEY: run_records,
    }
    print(format_per_sample_results_table(sample_payload))
    return sample_payload


def build_per_sample_rows(
    sample_payloads: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in sample_payloads:
        for run in sample.get(_RUNS_KEY, []):
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
                    "starting_layer": int(sample["starting_layer"]),
                    "transition_layers": int(sample["transition_layers"]),
                    "carrier_index": int(sample["carrier_index"]),
                    "carrier_token": sample["carrier_token"],
                    "num_non_frame_prompt_tokens": int(sample["num_non_frame_prompt_tokens"]),
                    "num_groups": int(sample["num_groups"]),
                    "run_type": run["run_type"],
                    "group_index": run.get("group_index"),
                    "group_start_token_position": run.get("group_start_token_position"),
                    "group_end_token_position": run.get("group_end_token_position"),
                    "group_size": int(run["group_size"]),
                    "group_token_positions_json": json.dumps(run.get("group_token_positions", [])),
                    "group_token_texts_json": json.dumps(run.get("group_token_texts", [])),
                    "ablated_answer_score": run.get("ablated_answer_score"),
                    "score_drop": run.get("score_drop"),
                    "baseline_score_drop": run.get("baseline_score_drop"),
                    "rescue_amount": run.get("rescue_amount"),
                    "bos_handling": sample["spared_token_kind"],
                    "spared_position": sample.get("spared_position"),
                }
            )
    return rows


def build_aggregate_rows(
    sample_payloads: Sequence[Dict[str, Any]],
    num_groups: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group_index in range(int(num_groups)):
        candidate_rows = [
            run
            for sample in sample_payloads
            for run in sample.get(_RUNS_KEY, [])
            if run.get("run_type") == "group_reopened" and int(run.get("group_index")) == int(group_index)
        ]
        group_sizes = [float(run["group_size"]) for run in candidate_rows]
        ablated_values = [
            float(run["ablated_answer_score"])
            for run in candidate_rows
            if _is_present_number(run.get("ablated_answer_score"))
        ]
        score_drop_values = [
            float(run["score_drop"])
            for run in candidate_rows
            if _is_present_number(run.get("score_drop"))
        ]
        rescue_values = [
            float(run["rescue_amount"])
            for run in candidate_rows
            if _is_present_number(run.get("rescue_amount"))
        ]
        rows.append(
            {
                "group_index": int(group_index),
                "n_samples": int(len(candidate_rows)),
                "mean_group_size": _mean_or_none(group_sizes),
                "mean_ablated_answer_score": _mean_or_none(ablated_values),
                "mean_score_drop": _mean_or_none(score_drop_values),
                "median_score_drop": _median_or_none(score_drop_values),
                "mean_rescue_amount": _mean_or_none(rescue_values),
                "median_rescue_amount": _median_or_none(rescue_values),
                "max_rescue_amount": None if not rescue_values else float(max(rescue_values)),
            }
        )
    return rows


def format_aggregate_group_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "group_index",
        "n_samples",
        "mean_group_size",
        "mean_score_drop",
        "mean_rescue_amount",
        "max_rescue_amount",
    ]
    values = [
        [
            str(int(row["group_index"])),
            str(int(row["n_samples"])),
            _format_float_or_na(row.get("mean_group_size")),
            _format_float_or_na(row.get("mean_score_drop")),
            _format_float_or_na(row.get("mean_rescue_amount")),
            _format_float_or_na(row.get("max_rescue_amount")),
        ]
        for row in rows
    ]
    return format_markdown_table(headers, values)


def build_group_metric_summary(
    sample_payloads: Sequence[Dict[str, Any]],
    num_groups: int,
    metric_key: str,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Dict[str, List[float]]]:
    per_group_values: Dict[int, List[float]] = {int(group_index): [] for group_index in range(int(num_groups))}
    for sample in sample_payloads:
        for run in sample.get(_RUNS_KEY, []):
            if run.get("run_type") != "group_reopened":
                continue
            group_index = run.get("group_index")
            if group_index is None:
                continue
            value = run.get(metric_key)
            if not _is_present_number(value):
                continue
            per_group_values[int(group_index)].append(float(value))

    if not any(per_group_values[group_index] for group_index in per_group_values):
        return None

    rng = random.Random(seed)
    summary = {
        "group_indices": [],
        "mean": [],
        "lo": [],
        "hi": [],
        "n_samples": [],
    }
    for group_index in range(int(num_groups)):
        values = per_group_values[int(group_index)]
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

        summary["group_indices"].append(int(group_index))
        summary["mean"].append(float(mean_value))
        summary["lo"].append(float(lo_value))
        summary["hi"].append(float(hi_value))
        summary["n_samples"].append(int(len(values)))

    if not summary["group_indices"]:
        return None
    return summary


def mean_blocked_baseline_score_drop(sample_payloads: Sequence[Dict[str, Any]]) -> Optional[float]:
    values = [
        float(sample["blocked_baseline_score_drop"])
        for sample in sample_payloads
        if _is_present_number(sample.get("blocked_baseline_score_drop"))
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def plot_group_bar_summary(
    summary: Dict[str, List[float]],
    output_path: Path,
    title: str,
    y_label: str,
    color: str,
    seq_len_label: Optional[str],
    baseline_value: Optional[float] = None,
    baseline_label: Optional[str] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.75), dpi=140)
    group_indices = [int(value) for value in summary["group_indices"]]
    mean_values = [float(value) for value in summary["mean"]]
    lo_values = [float(value) for value in summary["lo"]]
    hi_values = [float(value) for value in summary["hi"]]
    yerr = [
        [float(mean_values[idx] - lo_values[idx]) for idx in range(len(mean_values))],
        [float(hi_values[idx] - mean_values[idx]) for idx in range(len(mean_values))],
    ]

    ax.bar(group_indices, mean_values, color=color, edgecolor="black", linewidth=0.7, alpha=0.88)
    ax.errorbar(
        group_indices,
        mean_values,
        yerr=yerr,
        fmt="none",
        ecolor=color,
        elinewidth=1.2,
        capsize=4.0,
        alpha=0.95,
    )
    if baseline_value is not None:
        ax.axhline(
            float(baseline_value),
            color=_BASELINE_PLOT_COLOR,
            linewidth=1.6,
            linestyle="--",
            label=baseline_label or "Blocked baseline",
        )

    plotted_values = list(lo_values) + list(hi_values)
    if baseline_value is not None:
        plotted_values.append(float(baseline_value))
    if plotted_values:
        y_min = min(plotted_values)
        y_max = max(plotted_values)
        pad = 0.1 * max(1.0e-6, y_max - y_min) if not math.isclose(y_min, y_max) else 0.1 * max(1.0, abs(y_min))
        ax.set_ylim(float(y_min - pad), float(y_max + pad))

    full_title = title if seq_len_label is None else f"{title} ({seq_len_label})"
    ax.set_title(full_title, fontsize=13, pad=10)
    ax.set_xlabel("Group index")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8, axis="y")
    ax.set_xticks(group_indices)
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    if baseline_value is not None:
        ax.legend(frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_sample_rescue_summary(
    sample_payload: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    candidate_rows = [
        run
        for run in sample_payload.get(_RUNS_KEY, [])
        if run.get("run_type") == "group_reopened"
    ]
    if not candidate_rows:
        return None

    candidate_rows = sorted(candidate_rows, key=lambda run: int(run["group_index"]))
    x_values = [int(run["group_index"]) for run in candidate_rows]
    y_values = [
        None if run.get("rescue_amount") is None else float(run["rescue_amount"])
        for run in candidate_rows
    ]
    best_group_index = None
    valid_best_candidates = [
        run for run in candidate_rows if _is_present_number(run.get("rescue_amount"))
    ]
    if valid_best_candidates:
        best_group_index = int(
            max(
                valid_best_candidates,
                key=lambda run: (float(run["rescue_amount"]), -int(run["group_index"])),
            )["group_index"]
        )

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    plotted_y_values: List[float] = []
    bar_colors: List[str] = []
    heights: List[float] = []
    for group_index, rescue_value in zip(x_values, y_values):
        if rescue_value is None:
            bar_colors.append(_MISSING_BAR_COLOR)
            heights.append(0.0)
            continue
        plotted_y_values.append(float(rescue_value))
        if best_group_index is not None and int(group_index) == int(best_group_index):
            bar_colors.append(_BEST_GROUP_COLOR)
        else:
            bar_colors.append(_PLOT_COLOR)
        heights.append(float(rescue_value))

    ax.bar(x_values, heights, color=bar_colors, edgecolor="black", linewidth=0.7, alpha=0.88)

    if plotted_y_values:
        y_min = min(plotted_y_values + [0.0])
        y_max = max(plotted_y_values + [0.0])
        pad = 0.1 * max(1.0e-6, y_max - y_min) if not math.isclose(y_min, y_max) else 0.1 * max(1.0, abs(y_min))
        ax.set_ylim(float(y_min - pad), float(y_max + pad))

    title = (
        f"{sample_payload['sample_id']}\n"
        f"start_layer={int(sample_payload['starting_layer'])} | "
        f"transition_layers={int(sample_payload['transition_layers'])} | "
        f"baseline_drop={_format_float_or_na(sample_payload.get('blocked_baseline_score_drop'))}"
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Group index")
    ax.set_ylabel("Rescue amount")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8, axis="y")
    ax.set_xticks(x_values)
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_payload['sample_id']}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_outputs(
    sample_payloads: List[Dict[str, Any]],
    aggregate_rows: List[Dict[str, Any]],
    output_dir: Path,
    seq_len_label: Optional[str],
    args: argparse.Namespace,
    total_sample_dirs: int,
    scanned_samples: int,
    elapsed_seconds: float,
) -> None:
    per_sample_plot_dir = output_dir / "per_sample_rescue_plots"
    per_sample_csv_path = output_dir / "non_frame_prompt_group_rescues_per_sample.csv"
    aggregate_csv_path = output_dir / "non_frame_prompt_group_rescues_aggregate.csv"
    summary_json_path = output_dir / "summary.json"

    per_sample_rows = build_per_sample_rows(sample_payloads)
    af1_utils.write_csv(per_sample_csv_path, per_sample_rows, fieldnames=_PER_SAMPLE_FIELDS)
    af1_utils.write_csv(aggregate_csv_path, aggregate_rows, fieldnames=_AGGREGATE_FIELDS)

    aggregate_rescue_summary = build_group_metric_summary(
        sample_payloads,
        num_groups=int(args.num_groups),
        metric_key="rescue_amount",
    )
    aggregate_score_drop_summary = build_group_metric_summary(
        sample_payloads,
        num_groups=int(args.num_groups),
        metric_key="score_drop",
    )
    mean_baseline_drop = mean_blocked_baseline_score_drop(sample_payloads)

    aggregate_rescue_plot_path: Optional[Path] = None
    aggregate_score_drop_plot_path: Optional[Path] = None
    if not args.disable_plots:
        if aggregate_rescue_summary is not None:
            aggregate_rescue_plot_path = plot_group_bar_summary(
                aggregate_rescue_summary,
                output_dir
                / (
                    f"mean_rescue_amount_by_group_index"
                    f"{f'_{seq_len_label}' if seq_len_label else ''}.png"
                ),
                title=(
                    "Mean rescue amount by non-frame prompt group "
                    f"(start_layer={int(args.starting_layer)}, transition_layers={int(args.transition_layers)})"
                ),
                y_label="Mean rescue amount",
                color=_PLOT_COLOR,
                seq_len_label=seq_len_label,
            )
        if aggregate_score_drop_summary is not None:
            aggregate_score_drop_plot_path = plot_group_bar_summary(
                aggregate_score_drop_summary,
                output_dir
                / (
                    f"mean_score_drop_by_group_index"
                    f"{f'_{seq_len_label}' if seq_len_label else ''}.png"
                ),
                title=(
                    "Mean score drop by non-frame prompt group "
                    f"(start_layer={int(args.starting_layer)}, transition_layers={int(args.transition_layers)})"
                ),
                y_label="Mean score drop",
                color=_PLOT_COLOR,
                seq_len_label=seq_len_label,
                baseline_value=mean_baseline_drop,
                baseline_label="Mean blocked baseline score drop",
            )

    summary_payload = {
        "cli_args": {
            key: (value if not isinstance(value, Path) else str(value))
            for key, value in vars(args).items()
        },
        "data_root": args.data_root,
        "output_dir": str(output_dir),
        "num_total_sample_dirs": int(total_sample_dirs),
        "num_scanned_samples": int(scanned_samples),
        "num_retained_samples": int(len(sample_payloads)),
        "starting_layer": int(args.starting_layer),
        "transition_layers": int(args.transition_layers),
        "num_groups": int(args.num_groups),
        "carrier_definition": "final prompt token before answer generation",
        "grouping_definition": (
            "For each sample, collect prompt token positions < prompt_len that are not in any frame group, "
            "are not the carrier token, and are not the spared BOS/first-token fallback position when one exists; "
            "then split those positions into num_groups contiguous groups in token order, allowing empty groups "
            "when num_groups exceeds the number of available non-frame prompt tokens."
        ),
        "masking_definition": (
            "During transition layers [starting_layer, starting_layer + transition_layers), frame-token and "
            "carrier masking are identical to find_full_af1_mask_transition.py. Non-frame prompt rows are "
            "blocked to self-plus-optional-spared-token attention, except that exactly one selected contiguous "
            "non-frame group is reopened to the base causal mask in each candidate run. At layers >= "
            "starting_layer + transition_layers, the original post-transfer mask from "
            "find_full_af1_mask_transition.py is reused unchanged."
        ),
        "post_transfer_non_frame_prompt_self_only": bool(args.non_frame_prompt_self_only),
        "mean_blocked_baseline_score_drop": mean_baseline_drop,
        "output_paths": {
            "per_sample_csv": str(per_sample_csv_path),
            "aggregate_csv": str(aggregate_csv_path),
            "per_sample_plot_dir": None if args.disable_plots else str(per_sample_plot_dir),
            "aggregate_rescue_plot": (
                None if aggregate_rescue_plot_path is None else str(aggregate_rescue_plot_path)
            ),
            "aggregate_score_drop_plot": (
                None if aggregate_score_drop_plot_path is None else str(aggregate_score_drop_plot_path)
            ),
        },
        "runtime": eval_utils.format_runtime(elapsed_seconds),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if aggregate_rescue_plot_path is not None:
        print(f"Wrote aggregate rescue plot to: {aggregate_rescue_plot_path}")
    else:
        print("Aggregate rescue plot was not written.")
    if aggregate_score_drop_plot_path is not None:
        print(f"Wrote aggregate score-drop plot to: {aggregate_score_drop_plot_path}")
    else:
        print("Aggregate score-drop plot was not written.")
    print(f"Wrote per-sample CSV to: {per_sample_csv_path}")
    print(f"Wrote aggregate CSV to: {aggregate_csv_path}")
    print(f"Wrote summary JSON to: {summary_json_path}")
    print("Aggregate group summary across all retained samples:")
    print(format_aggregate_group_table(aggregate_rows))


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_len_label = eval_utils.resolve_seq_len_label(data_root)
    per_sample_plot_dir = output_dir / "per_sample_rescue_plots"

    sample_dirs = iter_sample_dirs(data_root)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under: {data_root}")

    layers = get_layers(base_model)
    starting_layer = int(args.starting_layer)
    if starting_layer < 0 or starting_layer >= int(len(layers)):
        raise ValueError(
            f"--starting-layer must be within [0, {int(len(layers)) - 1}], got {starting_layer}"
        )

    sample_payloads: List[Dict[str, Any]] = []
    scanned_samples = 0
    detailed_debug_emitted = False
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
            transition_layers=int(args.transition_layers),
            num_groups=int(args.num_groups),
            allow_bos_attention=bool(args.allow_bos_attention),
            allow_first_token_if_no_bos=bool(args.allow_first_token_if_no_bos),
            non_frame_prompt_self_only=bool(args.non_frame_prompt_self_only),
            min_clean_correct_prob=args.min_clean_correct_prob,
            debug_masks=bool(args.debug_masks),
            emit_detailed_debug=bool(args.debug_masks and not detailed_debug_emitted),
        )
        if sample_payload is None:
            continue
        if not args.disable_plots:
            per_sample_plot_path = plot_per_sample_rescue_summary(
                sample_payload,
                per_sample_plot_dir,
            )
            if per_sample_plot_path is not None:
                print(f"Wrote per-sample rescue plot to: {per_sample_plot_path}")
        detailed_debug_emitted = detailed_debug_emitted or bool(args.debug_masks)
        sample_payloads.append(sample_payload)

    aggregate_rows = build_aggregate_rows(
        sample_payloads,
        num_groups=int(args.num_groups),
    )
    write_outputs(
        sample_payloads=sample_payloads,
        aggregate_rows=aggregate_rows,
        output_dir=output_dir,
        seq_len_label=seq_len_label,
        args=args,
        total_sample_dirs=len(sample_dirs),
        scanned_samples=scanned_samples,
        elapsed_seconds=time.perf_counter() - start_time,
    )

    print(
        f"Retained {len(sample_payloads)} samples "
        f"(target limit={int(args.limit)}, scanned={scanned_samples}/{len(sample_dirs)}, "
        f"starting_layer={int(args.starting_layer)}, "
        f"transition_layers={int(args.transition_layers)}, "
        f"num_groups={int(args.num_groups)}, "
        f"allow_bos_attention={bool(args.allow_bos_attention)}, "
        f"allow_first_token_if_no_bos={bool(args.allow_first_token_if_no_bos)}, "
        f"non_frame_prompt_self_only={bool(args.non_frame_prompt_self_only)}, "
        f"min_clean_correct_prob="
        f"{'none' if args.min_clean_correct_prob is None else f'{float(args.min_clean_correct_prob):.4f}'})."
    )
    print(eval_utils.format_runtime(time.perf_counter() - start_time))


if __name__ == "__main__":
    main()

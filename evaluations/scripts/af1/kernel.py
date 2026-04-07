"""Core AF1 intervention logic, wait-boundary capture, and cache handling."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from evaluations.helpers import patching_core as core
from evaluations.helpers.sdpa_attention import (
    allowed_key_positions_from_mask,
    build_prompt_allow_matrix,
    build_sdpa_layer_mask,
)
from evaluations.scripts.af1.common import (
    VALID_INSTRUCTION_MASK_MODES,
    VALID_MODES,
    AttentionPolicy,
    PreparedSample,
    SampleLayout,
)
from evaluations.scripts.af1.layout import (
    all_non_frame_prompt_positions,
    build_hybrid_sample,
    build_non_frame_hybrid_sample,
    layout_hash,
    sanitize_token_text,
)
from models.model import (
    get_layers,
    move_inputs_to_model_device as move_inputs_to_explicit_model_device,
    prepare_attention_backend_for_forward,
)


class _WaitBoundaryCaptured(RuntimeError):
    pass


_CLEAN_FORWARD_PATH = "af1_clean_forward"
_WAIT_BOUNDARY_CAPTURE_PATH = "af1_donor_hybrid_wait_boundary_capture"
_MODE_TO_INTERVENTION_BACKEND_PATH = {
    "full_af1": "af1_intervention_full_af1",
    "wait_only": "af1_intervention_wait_only",
    "mask_only": "af1_intervention_mask_only",
}


def canonical_model_slug(model_name: str) -> str:
    return model_name.lower().replace("/", "_").replace("-", "")


def intervention_mode_flags(mode: str) -> Dict[str, bool]:
    """Translate the user-facing mode enum into the two intervention switches."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode={mode!r}. Expected one of {VALID_MODES}.")
    return {
        "enable_wait_patch": mode in {"full_af1", "wait_only"},
        "enable_abp_mask": mode in {"full_af1", "mask_only"},
    }


def intervention_mode_summary(mode: str) -> str:
    summaries = {
        "full_af1": "patch frame groups and all non-frame prompt tokens at x^(L_wait) and apply ABP masking afterward",
        "wait_only": "patch frame groups and all non-frame prompt tokens at x^(L_wait) but keep all later attention clean",
        "mask_only": "skip all wait-boundary patching and apply only the ABP transfer/self-only mask",
    }
    return summaries[mode]


def instruction_mask_mode_summary(mode: str) -> str:
    summaries = {
        "base": "transfer=base_causal_padding_mask post_transfer=self_only",
        "vision_end_only": "transfer=self_plus_earlier_vision_end post_transfer=self_only",
        "vision_boundary_only": "transfer=self_plus_earlier_vision_start_end post_transfer=self_only",
        "prompt_only": "transfer=self_plus_earlier_non_frame_non_boundary_prompt post_transfer=self_only",
        "image_pad_only": "transfer=self_plus_earlier_image_pad post_transfer=self_only",
    }
    if mode not in summaries:
        raise ValueError(
            f"Unsupported instruction_mask_mode={mode!r}. "
            f"Expected one of {VALID_INSTRUCTION_MASK_MODES}."
        )
    return summaries[mode]


def move_inputs_to_model_device(
    inputs: Dict[str, torch.Tensor],
    *,
    model_obj: Any,
) -> Dict[str, torch.Tensor]:
    return move_inputs_to_explicit_model_device(inputs, model_obj=model_obj)


def _prepare_forward_backend(
    *,
    path_name: str,
    requires_abp_mask: bool,
    output_attentions: bool = False,
    model_obj: Any,
) -> str:
    return prepare_attention_backend_for_forward(
        path_name=path_name,
        requires_abp_mask=requires_abp_mask,
        output_attentions=output_attentions,
        allow_sdpa_fallback=not requires_abp_mask,
        model_obj=model_obj,
    )


def _backend_cache_component(attention_backend: str) -> str:
    if attention_backend == "eager":
        raise RuntimeError("Eager attention is forbidden in AF1 cache keys.")
    if attention_backend not in {"sdpa", "flash_attention_2"}:
        raise RuntimeError(f"Unsupported attention backend for AF1 cache keys: {attention_backend!r}")
    return f"backend_{attention_backend}"


def batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@contextmanager
def temporary_layer_wrappers(layers: Sequence[Any], wrapper_factory: Any) -> Iterator[None]:
    original_forwards: Dict[int, Any] = {}
    try:
        for layer_idx, layer in enumerate(layers):
            original_forwards[layer_idx] = layer.forward
            layer.forward = wrapper_factory(layer_idx, layer.forward)
        yield
    finally:
        for layer_idx, layer in enumerate(layers):
            if layer_idx in original_forwards:
                layer.forward = original_forwards[layer_idx]


def _to_hidden_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x:
        return _to_hidden_tensor(x[0])
    raise TypeError(f"Unsupported layer output type: {type(x)}")


def _build_frame_group_by_token(layout: SampleLayout) -> Dict[int, Tuple[int, ...]]:
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


def _allowed_prompt_keys(query_idx: int, policy: AttentionPolicy, stage: str) -> List[int]:
    """Return the ABP key set for one prompt query position.

    Transfer stage:
    - frame tokens may attend densely within their own frame block
    - non-frame, non-carrier prompt tokens are self-only by default
    - the carrier token may attend to all earlier prompt tokens and itself

    Post-transfer stage:
    - everyone, including the carrier token and instruction tokens, is self-only
    """
    allowed: set[int] = {int(query_idx)}
    if stage == "transfer":
        if int(query_idx) == int(policy.carrier_index):
            allowed.update(range(0, int(policy.carrier_index) + 1))
        else:
            same_frame_group = policy.frame_group_by_token.get(int(query_idx))
            if same_frame_group is not None:
                allowed.update(int(position) for position in same_frame_group)
    return sorted(allowed)


def _instruction_allowed_prompt_keys(
    query_idx: int,
    policy: AttentionPolicy,
    stage: str,
) -> Optional[List[int]]:
    if stage != "transfer" or int(query_idx) not in policy.instruction_positions:
        return None

    mode = str(policy.instruction_mask_mode)
    if mode == "base":
        return None

    earlier_image_pad = {
        int(position) for position in policy.image_pad_positions if int(position) < int(query_idx)
    }
    earlier_vision_start = {
        int(position) for position in policy.vision_start_positions if int(position) < int(query_idx)
    }
    earlier_vision_end = {
        int(position) for position in policy.vision_end_positions if int(position) < int(query_idx)
    }

    if mode == "vision_end_only":
        return sorted({int(query_idx)} | earlier_vision_end)
    if mode == "vision_boundary_only":
        return sorted({int(query_idx)} | earlier_vision_start | earlier_vision_end)
    if mode == "prompt_only":
        boundary_positions = set(policy.vision_start_positions) | set(policy.vision_end_positions)
        return [
            int(key_idx)
            for key_idx in range(0, int(query_idx) + 1)
            if int(key_idx) == int(query_idx)
            or (
                int(key_idx) not in earlier_image_pad
                and int(key_idx) not in boundary_positions
            )
        ]
    if mode == "image_pad_only":
        return sorted({int(query_idx)} | earlier_image_pad)
    raise ValueError(
        f"Unsupported instruction_mask_mode={mode!r}. "
        f"Expected one of {VALID_INSTRUCTION_MASK_MODES}."
    )


def _instruction_keeps_base_mask(
    query_idx: int,
    policy: AttentionPolicy,
    stage: str,
) -> bool:
    return (
        stage == "transfer"
        and str(policy.instruction_mask_mode) == "base"
        and int(query_idx) in policy.instruction_positions
    )


def _allowed_keys_for_abp_query(
    query_idx: int,
    policy: AttentionPolicy,
    stage: str,
) -> Optional[List[int]]:
    if _instruction_keeps_base_mask(query_idx=query_idx, policy=policy, stage=stage):
        return None
    instruction_allowed_keys = _instruction_allowed_prompt_keys(
        query_idx=query_idx,
        policy=policy,
        stage=stage,
    )
    if instruction_allowed_keys is not None:
        return instruction_allowed_keys
    return _allowed_prompt_keys(query_idx=query_idx, policy=policy, stage=stage)


def build_abp_attention_policy(
    layout: SampleLayout,
    wait_layer: int,
    transfer_layers: int,
    instruction_mask_mode: str,
    *,
    model_obj: Any,
) -> AttentionPolicy:
    """Create the multimodal AF1 ABP schedule.

    `wait_layer` is interpreted as AF1's `L_wait`, i.e. the number of waiting
    layers before the transfer stage begins. The transfer stage then occupies
    the next `transfer_layers` layers, and all later layers use post-transfer
    self-only attention for the carrier as well.
    """
    num_layers = len(get_layers(model_obj))
    if wait_layer < 0 or wait_layer > num_layers:
        raise ValueError(f"wait_layer={wait_layer} must be in [0, {num_layers}]")
    if transfer_layers < 0:
        raise ValueError("transfer_layers must be non-negative")
    if wait_layer + transfer_layers > num_layers:
        raise ValueError(
            f"wait_layer + transfer_layers must be <= {num_layers}; "
            f"received {wait_layer} + {transfer_layers}"
        )
    if str(instruction_mask_mode) not in VALID_INSTRUCTION_MASK_MODES:
        raise ValueError(
            f"instruction_mask_mode={instruction_mask_mode!r} must be one of "
            f"{VALID_INSTRUCTION_MASK_MODES}"
        )
    return AttentionPolicy(
        prompt_len=int(layout.prompt_len),
        carrier_index=int(layout.carrier_index),
        wait_layer=int(wait_layer),
        transfer_layers=int(transfer_layers),
        num_model_layers=int(num_layers),
        frame_group_by_token=_build_frame_group_by_token(layout),
        instruction_positions=tuple(int(position) for position in layout.instruction_positions),
        image_pad_positions=tuple(int(position) for position in layout.image_pad_positions),
        vision_start_positions=tuple(int(position) for position in layout.vision_start_positions),
        vision_end_positions=tuple(int(position) for position in layout.vision_end_positions),
        instruction_mask_mode=str(instruction_mask_mode),
    )


def build_abp_attention_allow_matrix(
    *,
    query_len: int,
    key_len: int,
    policy: AttentionPolicy,
    stage: str,
    device: torch.device,
) -> torch.Tensor:
    """Build the ABP prompt allow-set to intersect with the model's SDPA mask.

    Prompt rows are modified because AF1 is defined over the original prompt
    tokens. Appended answer tokens used for scoring keep the base causal mask
    so sequence log-probability scoring still works normally.
    """
    return build_prompt_allow_matrix(
        query_len=query_len,
        key_len=key_len,
        prompt_len=policy.prompt_len,
        device=device,
        allowed_keys_by_query_fn=lambda query_idx: _allowed_keys_for_abp_query(
            query_idx=query_idx,
            policy=policy,
            stage=stage,
        ),
    )


def build_abp_sdpa_attention_mask(
    *,
    hidden_states: torch.Tensor,
    raw_attention_mask: Optional[torch.Tensor],
    model_config: Any,
    attention_type: str,
    policy: AttentionPolicy,
    stage: str,
    cache_position: Optional[torch.Tensor] = None,
    past_key_values: Optional[Any] = None,
) -> torch.Tensor:
    key_len = int(hidden_states.shape[1]) if raw_attention_mask is None else int(raw_attention_mask.shape[-1])
    allow_matrix = build_abp_attention_allow_matrix(
        query_len=int(hidden_states.shape[1]),
        key_len=key_len,
        policy=policy,
        stage=stage,
        device=hidden_states.device,
    )
    mask = build_sdpa_layer_mask(
        model_config=model_config,
        attention_type=attention_type,
        hidden_states=hidden_states,
        raw_attention_mask=raw_attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        allow_matrix=allow_matrix,
    )
    if mask is None:
        raise RuntimeError("ABP SDPA masking unexpectedly produced no materialized mask.")
    return mask


def validate_attention_policy(
    layout: SampleLayout,
    policy: AttentionPolicy,
    *,
    model_obj: Any,
) -> List[str]:
    frame_prompt_token = next(
        (
            position
            for group in layout.frame_groups
            for position in group
            if position != layout.carrier_index
        ),
        None,
    )
    non_frame_prompt_token = next(
        (
            position
            for position in range(layout.prompt_len)
            if position not in policy.frame_group_by_token
            and position != policy.carrier_index
            and position not in policy.instruction_positions
        ),
        None,
    )
    transfer_last = _allowed_prompt_keys(layout.carrier_index, policy=policy, stage="transfer")
    post_last = _allowed_prompt_keys(layout.carrier_index, policy=policy, stage="post_transfer")

    frame_transfer_note = "ABP frame token transfer connectivity: no frame tokens present"
    if frame_prompt_token is not None:
        transfer_frame = _allowed_prompt_keys(frame_prompt_token, policy=policy, stage="transfer")
        same_frame_group = sorted(policy.frame_group_by_token[int(frame_prompt_token)])
        expected_frame = list(same_frame_group)
        if transfer_frame != expected_frame:
            raise RuntimeError(
                f"ABP validation failed for frame token {frame_prompt_token}: "
                f"expected {expected_frame}, got {transfer_frame}"
            )
        frame_transfer_note = (
            f"ABP frame token transfer connectivity: query={frame_prompt_token} "
            f"frame_size={len(same_frame_group)} dense_intra_frame=yes"
        )

    non_frame_transfer_note = "ABP non-frame token allowed keys at transfer: none"
    if non_frame_prompt_token is not None:
        transfer_non_frame = _allowed_prompt_keys(non_frame_prompt_token, policy=policy, stage="transfer")
        expected_non_frame = [non_frame_prompt_token]
        if transfer_non_frame != expected_non_frame:
            raise RuntimeError(
                f"ABP validation failed for non-frame token {non_frame_prompt_token}: "
                f"expected {expected_non_frame}, got {transfer_non_frame}"
            )
        non_frame_transfer_note = (
            f"ABP non-frame token allowed keys at transfer: "
            f"query={non_frame_prompt_token} keys={transfer_non_frame}"
        )

    instruction_transfer_note = "ABP instruction-token transfer handling: none"
    layers = get_layers(model_obj)
    if not layers:
        raise RuntimeError("AF1 validation could not find any decoder layers.")
    validation_config = getattr(getattr(layers[0], "self_attn", None), "config", None)
    if validation_config is None:
        raise RuntimeError("AF1 validation could not resolve the decoder attention config.")
    prompt_len = int(layout.prompt_len)
    synthetic_hidden_states = torch.zeros((1, prompt_len, 1), dtype=torch.float32)
    synthetic_attention_mask = torch.ones((1, prompt_len), dtype=torch.bool)
    transfer_sdpa_mask = build_abp_sdpa_attention_mask(
        hidden_states=synthetic_hidden_states,
        raw_attention_mask=synthetic_attention_mask,
        model_config=validation_config,
        attention_type="full_attention",
        policy=policy,
        stage="transfer",
    )
    post_transfer_sdpa_mask = build_abp_sdpa_attention_mask(
        hidden_states=synthetic_hidden_states,
        raw_attention_mask=synthetic_attention_mask,
        model_config=validation_config,
        attention_type="full_attention",
        policy=policy,
        stage="post_transfer",
    )
    sdpa_materialization_note = (
        "ABP SDPA mask materialization: "
        f"attention_type=full_attention transfer_mask_dtype={transfer_sdpa_mask.dtype} "
        f"post_transfer_mask_dtype={post_transfer_sdpa_mask.dtype}"
    )
    if policy.instruction_positions:
        instruction_tokens = {
            int(position): sanitize_token_text(layout.prompt_decoded_tokens[int(position)])
            for position in policy.instruction_positions
        }
        instruction_transfer_keys_by_position: Dict[int, List[int]] = {}
        for instruction_prompt_token in policy.instruction_positions:
            actual_instruction_transfer = allowed_key_positions_from_mask(
                transfer_sdpa_mask,
                query_idx=int(instruction_prompt_token),
                key_len=prompt_len,
            )
            expected_instruction_post = [int(instruction_prompt_token)]
            actual_instruction_post = allowed_key_positions_from_mask(
                post_transfer_sdpa_mask,
                query_idx=int(instruction_prompt_token),
                key_len=prompt_len,
            )
            if actual_instruction_post != expected_instruction_post:
                raise RuntimeError(
                    f"ABP validation failed for instruction token {instruction_prompt_token} after transfer: "
                    f"expected {expected_instruction_post}, got {actual_instruction_post}"
                )
            expected_instruction_transfer = _instruction_allowed_prompt_keys(
                int(instruction_prompt_token),
                policy=policy,
                stage="transfer",
            )
            if expected_instruction_transfer is None:
                expected_instruction_transfer = list(range(0, int(instruction_prompt_token) + 1))
            if actual_instruction_transfer != expected_instruction_transfer:
                raise RuntimeError(
                    f"ABP validation failed for instruction token {instruction_prompt_token} during transfer: "
                    f"expected {expected_instruction_transfer}, got {actual_instruction_transfer}"
                )
            instruction_transfer_keys_by_position[int(instruction_prompt_token)] = actual_instruction_transfer
        instruction_transfer_note = (
            "ABP instruction-token transfer handling: "
            f"positions={list(policy.instruction_positions)} tokens={json.dumps(instruction_tokens, sort_keys=True)} "
            f"{instruction_mask_mode_summary(str(policy.instruction_mask_mode))} "
            f"transfer_keys_by_position={json.dumps(instruction_transfer_keys_by_position, sort_keys=True)}"
        )

    expected_transfer_last = list(range(0, layout.carrier_index + 1))
    actual_transfer_last = allowed_key_positions_from_mask(
        transfer_sdpa_mask,
        query_idx=int(layout.carrier_index),
        key_len=prompt_len,
    )
    if transfer_last != expected_transfer_last or actual_transfer_last != expected_transfer_last:
        raise RuntimeError(
            f"ABP validation failed for carrier transfer stage: "
            f"expected {expected_transfer_last}, theoretical={transfer_last}, sdpa_materialized={actual_transfer_last}"
        )

    expected_post_last = [layout.carrier_index]
    actual_post_last = allowed_key_positions_from_mask(
        post_transfer_sdpa_mask,
        query_idx=int(layout.carrier_index),
        key_len=prompt_len,
    )
    if post_last != expected_post_last or actual_post_last != expected_post_last:
        raise RuntimeError(
            f"ABP validation failed for carrier post-transfer stage: "
            f"expected {expected_post_last}, theoretical={post_last}, sdpa_materialized={actual_post_last}"
        )

    return [
        sdpa_materialization_note,
        frame_transfer_note,
        non_frame_transfer_note,
        instruction_transfer_note,
        f"ABP carrier allowed keys at transfer: query={layout.carrier_index} keys={actual_transfer_last}",
        f"ABP carrier allowed keys after transfer: query={layout.carrier_index} keys={actual_post_last}",
        "ABP scoring suffix rows keep the base causal mask when answer tokens are appended for scoring.",
    ]


def _patch_frame_groups(
    hidden_states: torch.Tensor,
    layout: SampleLayout,
    frame_group_means: Dict[int, torch.Tensor],
) -> torch.Tensor:
    patched = hidden_states.clone()
    for frame_idx, positions in enumerate(layout.frame_groups):
        replacement = frame_group_means[frame_idx].to(device=patched.device, dtype=patched.dtype)
        patched[:, list(positions), :] = replacement.unsqueeze(0)
    return patched


def _patch_non_frame_prompt_tokens(
    patched_hidden_states: torch.Tensor,
    layout: SampleLayout,
    non_frame_mean_block: torch.Tensor,
) -> torch.Tensor:
    non_frame_positions = all_non_frame_prompt_positions(layout)
    if not non_frame_positions:
        return patched_hidden_states
    replacement = non_frame_mean_block.to(
        device=patched_hidden_states.device,
        dtype=patched_hidden_states.dtype,
    )
    patched_hidden_states[:, list(non_frame_positions), :] = replacement.unsqueeze(0)
    return patched_hidden_states


def _run_model_forward(
    inputs: Dict[str, torch.Tensor],
    output_hidden_states: bool = False,
    output_attentions: bool = False,
    *,
    path_name: str,
    requires_abp_mask: bool,
    model_obj: Any,
) -> Any:
    _prepare_forward_backend(
        path_name=path_name,
        requires_abp_mask=requires_abp_mask,
        output_attentions=output_attentions,
        model_obj=model_obj,
    )
    with torch.inference_mode():
        return model_obj(
            **inputs,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


def _capture_wait_boundary_blocks(
    inputs: Dict[str, torch.Tensor],
    wait_layer: int,
    positions: Sequence[int],
    *,
    model_obj: Any,
) -> torch.Tensor:
    """Capture hidden states exactly at the AF1 wait boundary `x^(L_wait)`.

    Layer indexing detail:
    - if `wait_layer == 0`, we need `x^(0)`, so we capture the incoming hidden
      states before layer 0 runs
    - if `wait_layer > 0`, we need the output of layer `wait_layer - 1`

    We stop the forward pass immediately after capture because the conditional
    mean only needs the wait-boundary activations for the selected positions.
    """
    layers = get_layers(model_obj)
    if wait_layer < 0 or wait_layer > len(layers):
        raise ValueError(f"wait_layer={wait_layer} must be in [0, {len(layers)}]")
    _prepare_forward_backend(
        path_name=_WAIT_BOUNDARY_CAPTURE_PATH,
        requires_abp_mask=False,
        output_attentions=False,
        model_obj=model_obj,
    )

    capture_positions = [int(position) for position in positions]
    captured: Dict[str, torch.Tensor] = {}
    boundary_output_layer_idx = wait_layer - 1

    def wrapper_factory(layer_idx: int, original_forward: Any) -> Any:
        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError("Layer forward received no hidden_states")

            if wait_layer == 0 and layer_idx == 0:
                captured["block"] = hidden_states[:, capture_positions, :].detach().to(dtype=torch.float32).cpu()
                raise _WaitBoundaryCaptured

            outputs = original_forward(*args, **kwargs)
            if layer_idx == boundary_output_layer_idx:
                hidden_out = _to_hidden_tensor(outputs)
                captured["block"] = hidden_out[:, capture_positions, :].detach().to(dtype=torch.float32).cpu()
                raise _WaitBoundaryCaptured
            return outputs

        return wrapped_forward

    with temporary_layer_wrappers(layers, wrapper_factory):
        with torch.inference_mode():
            try:
                model_obj(
                    **inputs,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            except _WaitBoundaryCaptured:
                pass

    if "block" not in captured:
        raise RuntimeError(
            f"Failed to capture wait-boundary block for wait_layer={wait_layer} positions={capture_positions}"
        )
    return captured["block"]


def _conditional_mean_cache_path(
    cache_dir: Path,
    model_name: str,
    seq_len: int,
    target_sample_id: str,
    frame_idx: int,
    wait_layer: int,
    k_donors_used: int,
    donor_policy: str,
    donor_ids: Sequence[str],
    layout_hash_value: str,
    attention_backend: str,
) -> Path:
    donor_ids_hash = hashlib.sha1(json.dumps(list(donor_ids), sort_keys=True).encode("utf-8")).hexdigest()[:16]
    donor_policy_hash = hashlib.sha1(str(donor_policy).encode("utf-8")).hexdigest()[:10]
    return (
        cache_dir
        / canonical_model_slug(model_name)
        / _backend_cache_component(attention_backend)
        / f"seq_len_{seq_len}"
        / f"wait_{wait_layer}"
        / f"target_{target_sample_id}"
        / (
            f"frame_{frame_idx}_k_{k_donors_used}_policy_{donor_policy_hash}_"
            f"donors_{donor_ids_hash}_layout_{layout_hash_value}.pt"
        )
    )


def _non_frame_conditional_mean_cache_path(
    cache_dir: Path,
    model_name: str,
    seq_len: int,
    target_sample_id: str,
    wait_layer: int,
    k_donors_used: int,
    donor_policy: str,
    donor_ids: Sequence[str],
    layout_hash_value: str,
    attention_backend: str,
) -> Path:
    donor_ids_hash = hashlib.sha1(json.dumps(list(donor_ids), sort_keys=True).encode("utf-8")).hexdigest()[:16]
    donor_policy_hash = hashlib.sha1(str(donor_policy).encode("utf-8")).hexdigest()[:10]
    return (
        cache_dir
        / canonical_model_slug(model_name)
        / _backend_cache_component(attention_backend)
        / f"seq_len_{seq_len}"
        / f"wait_{wait_layer}"
        / f"target_{target_sample_id}"
        / (
            f"non_frame_k_{k_donors_used}_policy_{donor_policy_hash}_"
            f"donors_{donor_ids_hash}_layout_{layout_hash_value}.pt"
        )
    )


def _load_local_cache_payload(cache_path: Path) -> Optional[Dict[str, Any]]:
    """Load one trusted local AF1 cache payload or fall back to recomputation.

    These cache files are written by this script itself and contain a small
    metadata dict plus tensors, so we opt into `weights_only=False` to stay
    compatible with older PyTorch serialization formats. If a cache file is
    unreadable or malformed, we treat it as a cache miss and overwrite it on
    recomputation instead of crashing the full grid run.
    """
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(
            f"[cache] Failed to load local cache {cache_path}: {exc}. "
            "Recomputing and overwriting this cache entry."
        )
        return None
    if not isinstance(payload, dict) or "mean_block" not in payload:
        print(
            f"[cache] Invalid cache payload at {cache_path}: expected a dict with "
            "'mean_block'. Recomputing and overwriting this cache entry."
        )
        return None
    return payload


def compute_frame_group_conditional_mean(
    target_sample: PreparedSample,
    frame_idx: int,
    donor_samples: Sequence[PreparedSample],
    wait_layer: int,
    batch_size: int,
    cache_dir: Path,
    recompute_cache: bool,
    donor_policy: str,
    *,
    runtime: Any,
) -> Tuple[torch.Tensor, bool]:
    """Estimate one frame-group conditional mean for one target sample.

    This is the core CAMA-style adaptation in this script. For frame `j`:
    1. build one hybrid per donor where target frame `j` stays fixed
    2. replace all other frames with donor frames
    3. keep the target text prompt fixed
    4. run each hybrid to the wait boundary `x^(L_wait)`
    5. extract the full activation block for frame group `j`
    6. average those blocks across donors

    A single donor is not treated as a conditional mean, so we require at
    least two donors.
    """
    if len(donor_samples) < 2:
        raise ValueError(
            f"Need at least two donors to estimate a conditional mean, got {len(donor_samples)} "
            f"for target={target_sample.sample_id} frame={frame_idx}"
        )

    donor_ids = [sample.sample_id for sample in donor_samples]
    attention_backend = _prepare_forward_backend(
        path_name=_WAIT_BOUNDARY_CAPTURE_PATH,
        requires_abp_mask=False,
        output_attentions=False,
        model_obj=runtime.model,
    )
    cache_path = _conditional_mean_cache_path(
        cache_dir=cache_dir,
        model_name=runtime.model_name,
        seq_len=target_sample.layout.seq_len,
        target_sample_id=target_sample.sample_id,
        frame_idx=frame_idx,
        wait_layer=wait_layer,
        k_donors_used=len(donor_samples),
        donor_policy=donor_policy,
        donor_ids=donor_ids,
        layout_hash_value=layout_hash(target_sample.layout),
        attention_backend=attention_backend,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not recompute_cache:
        payload = _load_local_cache_payload(cache_path)
        if payload is not None:
            return payload["mean_block"].to(dtype=torch.float32), True

    frame_positions = target_sample.layout.frame_groups[frame_idx]
    blocks: List[torch.Tensor] = []
    for donor_batch in batched(list(donor_samples), batch_size=batch_size):
        hybrid_inputs_list: List[Dict[str, torch.Tensor]] = []
        for donor_sample in donor_batch:
            hybrid = build_hybrid_sample(target_sample=target_sample, donor_sample=donor_sample, frame_idx=frame_idx)
            hybrid_inputs_cpu = core.build_inputs(
                hybrid["frames"],
                hybrid["question"],
                processor=runtime.processor,
            )
            hybrid_inputs_list.append(
                move_inputs_to_model_device(hybrid_inputs_cpu, model_obj=runtime.model)
            )

        # Batched hybrid evaluation is safe only because layout validation
        # guarantees exact prompt/token alignment across compatible donors.
        batched_inputs = core.concatenate_inputs_for_batch(hybrid_inputs_list)
        blocks.append(
            _capture_wait_boundary_blocks(
                batched_inputs,
                wait_layer=wait_layer,
                positions=frame_positions,
                model_obj=runtime.model,
            )
        )

    mean_block = torch.cat(blocks, dim=0).mean(dim=0).to(dtype=torch.float32).cpu()
    torch.save(
        {
            "mean_block": mean_block,
            "metadata": {
                "model_name": runtime.model_name,
                "seq_len": int(target_sample.layout.seq_len),
                "target_sample_id": target_sample.sample_id,
                "frame_idx": int(frame_idx),
                "wait_layer": int(wait_layer),
                "k_donors_requested": int(len(donor_samples)),
                "k_donors_used": int(len(donor_samples)),
                "donor_policy": donor_policy,
                "donor_ids": list(donor_ids),
                "layout_hash": layout_hash(target_sample.layout),
                "attention_backend": attention_backend,
                "cache_semantics": (
                    "frame-group conditional mean at x^(L_wait) estimated from hybrid contexts "
                    "that keep the target frame fixed, replace the other frames with donor frames, "
                    "and keep the target text prompt fixed; backend-specific because "
                    "the clean donor-hybrid capture path may run on flash_attention_2 or sdpa"
                ),
            },
        },
        cache_path,
    )
    return mean_block, False


def compute_non_frame_conditional_mean(
    target_sample: PreparedSample,
    donor_samples: Sequence[PreparedSample],
    wait_layer: int,
    batch_size: int,
    cache_dir: Path,
    recompute_cache: bool,
    donor_policy: str,
    *,
    runtime: Any,
) -> Tuple[torch.Tensor, bool]:
    """Estimate one conditional mean for all non-frame prompt tokens."""
    if len(donor_samples) < 2:
        raise ValueError(
            f"Need at least two donors to estimate a non-frame conditional mean, got {len(donor_samples)} "
            f"for target={target_sample.sample_id}"
        )

    non_frame_positions = all_non_frame_prompt_positions(target_sample.layout)
    if not non_frame_positions:
        raise RuntimeError(f"target={target_sample.sample_id}: no non-frame prompt positions found")

    donor_ids = [sample.sample_id for sample in donor_samples]
    attention_backend = _prepare_forward_backend(
        path_name=_WAIT_BOUNDARY_CAPTURE_PATH,
        requires_abp_mask=False,
        output_attentions=False,
        model_obj=runtime.model,
    )
    cache_path = _non_frame_conditional_mean_cache_path(
        cache_dir=cache_dir,
        model_name=runtime.model_name,
        seq_len=target_sample.layout.seq_len,
        target_sample_id=target_sample.sample_id,
        wait_layer=wait_layer,
        k_donors_used=len(donor_samples),
        donor_policy=donor_policy,
        donor_ids=donor_ids,
        layout_hash_value=layout_hash(target_sample.layout),
        attention_backend=attention_backend,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not recompute_cache:
        payload = _load_local_cache_payload(cache_path)
        if payload is not None:
            return payload["mean_block"].to(dtype=torch.float32), True

    blocks: List[torch.Tensor] = []
    for donor_batch in batched(list(donor_samples), batch_size=batch_size):
        hybrid_inputs_list: List[Dict[str, torch.Tensor]] = []
        for donor_sample in donor_batch:
            hybrid = build_non_frame_hybrid_sample(target_sample=target_sample, donor_sample=donor_sample)
            hybrid_inputs_cpu = core.build_inputs(
                hybrid["frames"],
                hybrid["question"],
                processor=runtime.processor,
            )
            hybrid_inputs_list.append(
                move_inputs_to_model_device(hybrid_inputs_cpu, model_obj=runtime.model)
            )

        # Batched hybrid evaluation is safe only because layout validation
        # guarantees exact prompt/token alignment across compatible donors.
        batched_inputs = core.concatenate_inputs_for_batch(hybrid_inputs_list)
        blocks.append(
            _capture_wait_boundary_blocks(
                batched_inputs,
                wait_layer=wait_layer,
                positions=non_frame_positions,
                model_obj=runtime.model,
            )
        )

    mean_block = torch.cat(blocks, dim=0).mean(dim=0).to(dtype=torch.float32).cpu()
    torch.save(
        {
            "mean_block": mean_block,
            "metadata": {
                "model_name": runtime.model_name,
                "seq_len": int(target_sample.layout.seq_len),
                "target_sample_id": target_sample.sample_id,
                "wait_layer": int(wait_layer),
                "k_donors_requested": int(len(donor_samples)),
                "k_donors_used": int(len(donor_samples)),
                "donor_policy": donor_policy,
                "donor_ids": list(donor_ids),
                "layout_hash": layout_hash(target_sample.layout),
                "num_positions": int(len(non_frame_positions)),
                "attention_backend": attention_backend,
                "cache_semantics": (
                    "all-non-frame prompt conditional mean at x^(L_wait) estimated from hybrid contexts "
                    "that keep the target text prompt fixed and replace the entire frame set with donor frames; "
                    "backend-specific because the clean donor-hybrid capture path may run on "
                    "flash_attention_2 or sdpa"
                ),
            },
        },
        cache_path,
    )
    return mean_block, False


def compute_all_frame_group_means_for_sample(
    target_sample: PreparedSample,
    donor_samples: Sequence[PreparedSample],
    wait_layer: int,
    batch_size: int,
    cache_dir: Path,
    recompute_cache: bool,
    donor_policy: str,
    *,
    runtime: Any,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, int]]:
    """Compute conditional-mean replacements for every frame group in a sample."""
    if len(donor_samples) < 2:
        raise ValueError(
            f"Need at least two donors to estimate conditional means for target={target_sample.sample_id}"
        )

    frame_means: Dict[int, torch.Tensor] = {}
    cache_hits = 0
    cache_misses = 0
    for frame_idx in range(target_sample.layout.seq_len):
        mean_block, cache_hit = compute_frame_group_conditional_mean(
            target_sample=target_sample,
            frame_idx=frame_idx,
            donor_samples=donor_samples,
            wait_layer=wait_layer,
            batch_size=batch_size,
            cache_dir=cache_dir,
            recompute_cache=recompute_cache,
            donor_policy=donor_policy,
            runtime=runtime,
        )
        frame_means[frame_idx] = mean_block
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
    return frame_means, {"cache_hits": cache_hits, "cache_misses": cache_misses}


def run_clean_model(
    inputs: Dict[str, torch.Tensor],
    output_attentions: bool = False,
    *,
    model_obj: Any,
) -> Any:
    return _run_model_forward(
        inputs,
        output_hidden_states=False,
        output_attentions=output_attentions,
        path_name=_CLEAN_FORWARD_PATH,
        requires_abp_mask=False,
        model_obj=model_obj,
    )


def run_model_with_intervention(
    inputs: Dict[str, torch.Tensor],
    layout: SampleLayout,
    frame_group_means: Optional[Dict[int, torch.Tensor]],
    non_frame_prompt_mean: Optional[torch.Tensor],
    policy: AttentionPolicy,
    mode: str,
    output_attentions: bool = False,
    *,
    model_obj: Any,
) -> Any:
    """Run the model with the selected intervention mode.

    Modes:
    - `full_af1`: wait-stage patch + ABP mask
    - `wait_only`: wait-stage patch only
    - `mask_only`: ABP mask only

    Why these modes matter:
    - `wait_only` helps estimate when information in the patched wait-boundary
      token sets has already been transferred away: if replacing those token
      sets at a layer no longer hurts, the transfer may already be complete.
    - `mask_only` isolates the effect of the ABP transfer/self-only bottleneck
      without also removing information from the wait-boundary token sets at the
      boundary.
    """
    mode_flags = intervention_mode_flags(mode)
    enable_wait_patch = mode_flags["enable_wait_patch"]
    enable_abp_mask = mode_flags["enable_abp_mask"]
    if enable_wait_patch and (frame_group_means is None or non_frame_prompt_mean is None):
        raise ValueError(
            f"mode={mode!r} requires both frame_group_means and non_frame_prompt_mean, "
            "but one or both were not provided."
        )

    layers = get_layers(model_obj)
    if int(policy.num_model_layers) != len(layers):
        raise RuntimeError(
            f"Attention policy expected {policy.num_model_layers} layers but model exposes {len(layers)} layers"
        )
    _prepare_forward_backend(
        path_name=_MODE_TO_INTERVENTION_BACKEND_PATH[mode],
        requires_abp_mask=enable_abp_mask,
        output_attentions=output_attentions,
        model_obj=model_obj,
    )

    boundary_output_layer_idx = policy.wait_layer - 1
    raw_attention_mask = inputs.get("attention_mask")

    def wrapper_factory(layer_idx: int, original_forward: Any) -> Any:
        layer_module = layers[layer_idx]
        attention_type = str(getattr(layer_module, "attention_type", "full_attention"))
        attention_config = getattr(getattr(layer_module, "self_attn", None), "config", None)
        if attention_config is None:
            raise RuntimeError(f"Layer {layer_idx} does not expose a self-attention config.")

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError("Layer forward received no hidden_states")

            stage = policy.stage_for_layer(layer_idx)
            if enable_abp_mask and stage in {"transfer", "post_transfer"}:
                kwargs["attention_mask"] = build_abp_sdpa_attention_mask(
                    hidden_states=hidden_states,
                    raw_attention_mask=raw_attention_mask,
                    model_config=attention_config,
                    attention_type=attention_type,
                    policy=policy,
                    stage=stage,
                    cache_position=kwargs.get("cache_position"),
                    past_key_values=kwargs.get("past_key_values"),
                )

            if enable_wait_patch and policy.wait_layer == 0 and layer_idx == 0:
                # `wait_layer == 0` means replace x^(0) before any decoder layer.
                patched_hidden_states = _patch_frame_groups(
                    hidden_states,
                    layout=layout,
                    frame_group_means=frame_group_means or {},
                )
                patched_hidden_states = _patch_non_frame_prompt_tokens(
                    patched_hidden_states,
                    layout=layout,
                    non_frame_mean_block=non_frame_prompt_mean,
                )
                if args:
                    args = (patched_hidden_states,) + tuple(args[1:])
                else:
                    kwargs["hidden_states"] = patched_hidden_states

            outputs = original_forward(*args, **kwargs)
            if enable_wait_patch and policy.wait_layer > 0 and layer_idx == boundary_output_layer_idx:
                # For `wait_layer > 0`, replace x^(L_wait) at the output of
                # layer index `wait_layer - 1`, then let subsequent layers run
                # under the selected mode. In `wait_only`, later attention stays
                # completely clean. In `full_af1`, the later layers use ABP.
                hidden_out = _to_hidden_tensor(outputs)
                patched_hidden_out = _patch_frame_groups(
                    hidden_out,
                    layout=layout,
                    frame_group_means=frame_group_means or {},
                )
                patched_hidden_out = _patch_non_frame_prompt_tokens(
                    patched_hidden_out,
                    layout=layout,
                    non_frame_mean_block=non_frame_prompt_mean,
                )
                return (patched_hidden_out,) + tuple(outputs[1:])
            return outputs

        return wrapped_forward

    with temporary_layer_wrappers(layers, wrapper_factory):
        with torch.inference_mode():
            return model_obj(
                **inputs,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=True,
            )

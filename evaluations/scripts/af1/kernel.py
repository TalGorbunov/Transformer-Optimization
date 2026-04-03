"""Core AF1 intervention logic, wait-boundary capture, and cache handling."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from evaluations.helpers import patching_core as core
from evaluations.scripts.af1.common import (
    NEG_INF,
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
    MODEL_ID,
    force_eager_attention_backend,
    get_layers,
    model as base_model,
)


class _WaitBoundaryCaptured(RuntimeError):
    pass


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
        "base": "transfer=base post_transfer=self_only",
        "no_frame_access": "transfer=base_minus_frames post_transfer=self_only",
        "frame_only": "transfer=self_plus_frames post_transfer=self_only",
    }
    if mode not in summaries:
        raise ValueError(
            f"Unsupported instruction_mask_mode={mode!r}. "
            f"Expected one of {VALID_INSTRUCTION_MASK_MODES}."
        )
    return summaries[mode]


def move_inputs_to_model_device(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    device = next(base_model.parameters()).device
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


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


def _ensure_mask_tensor(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if mask.dim() != 4:
        raise ValueError(f"Expected rank-4 attention mask, got shape={tuple(mask.shape)}")
    if int(mask.shape[0]) == batch_size:
        return mask
    if int(mask.shape[0]) == 1:
        return mask.expand(batch_size, -1, -1, -1)
    raise ValueError(f"Cannot expand attention mask {tuple(mask.shape)} to batch_size={batch_size}")


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


def _query_keeps_base_mask(query_idx: int, policy: AttentionPolicy, stage: str) -> bool:
    return (
        stage == "transfer"
        and str(policy.instruction_mask_mode) == "base"
        and int(query_idx) in policy.instruction_positions
    )


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
    if mode == "no_frame_access":
        return [
            int(key_idx)
            for key_idx in range(0, int(query_idx) + 1)
            if int(key_idx) not in policy.frame_group_by_token
        ]
    if mode == "frame_only":
        allowed = {int(query_idx)}
        allowed.update(
            int(key_idx)
            for key_idx in policy.frame_group_by_token
            if int(key_idx) <= int(query_idx)
        )
        return sorted(allowed)
    raise ValueError(
        f"Unsupported instruction_mask_mode={mode!r}. "
        f"Expected one of {VALID_INSTRUCTION_MASK_MODES}."
    )


def build_abp_attention_policy(
    layout: SampleLayout,
    wait_layer: int,
    transfer_layers: int,
    instruction_mask_mode: str,
) -> AttentionPolicy:
    """Create the multimodal AF1 ABP schedule.

    `wait_layer` is interpreted as AF1's `L_wait`, i.e. the number of waiting
    layers before the transfer stage begins. The transfer stage then occupies
    the next `transfer_layers` layers, and all later layers use post-transfer
    self-only attention for the carrier as well.
    """
    num_layers = len(get_layers(base_model))
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
        instruction_mask_mode=str(instruction_mask_mode),
    )


def build_abp_attention_mask(
    base_mask: torch.Tensor,
    policy: AttentionPolicy,
    stage: str,
) -> torch.Tensor:
    """Rewrite the prompt rows of the causal mask to follow AF1 ABP rules.

    We preserve the base causal/padding constraints and only narrow the allowed
    key set. Prompt rows are modified because AF1 is defined over the original
    prompt tokens; appended answer tokens used for scoring keep the base causal
    mask so sequence log-probability scoring still works normally.
    """
    batch_size, _, query_len, key_len = base_mask.shape
    template = base_mask[0, 0]
    base_allowed = template == 0
    custom_allowed = base_allowed.clone()

    # We only alter the original prompt rows. If answer tokens are appended for
    # scoring, their rows keep the base causal mask so sequence log-probability
    # scoring remains well-defined.
    prompt_rows = min(int(policy.prompt_len), int(query_len))
    for query_idx in range(prompt_rows):
        if _query_keeps_base_mask(query_idx=query_idx, policy=policy, stage=stage):
            continue
        instruction_allowed_keys = _instruction_allowed_prompt_keys(
            query_idx=query_idx,
            policy=policy,
            stage=stage,
        )
        allowed_row = torch.zeros(key_len, dtype=torch.bool, device=base_mask.device)
        allowed_keys = (
            instruction_allowed_keys
            if instruction_allowed_keys is not None
            else _allowed_prompt_keys(query_idx=query_idx, policy=policy, stage=stage)
        )
        for key_idx in allowed_keys:
            if 0 <= int(key_idx) < key_len:
                allowed_row[int(key_idx)] = True
        custom_allowed[query_idx, :] = allowed_row

    final_allowed = base_allowed & custom_allowed
    fill_value = torch.finfo(template.dtype).min if torch.is_floating_point(template) else NEG_INF
    mask_2d = torch.full_like(template, fill_value=fill_value)
    mask_2d[final_allowed] = 0
    return mask_2d.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_len, key_len)


def validate_attention_policy(layout: SampleLayout, policy: AttentionPolicy) -> List[str]:
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
    instruction_prompt_token = next((position for position in policy.instruction_positions), None)
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
    if instruction_prompt_token is not None:
        instruction_tokens = [
            sanitize_token_text(layout.prompt_decoded_tokens[int(position)])
            for position in policy.instruction_positions
        ]
        prompt_len = int(layout.prompt_len)
        synthetic_base_mask = torch.full(
            (1, 1, prompt_len, prompt_len),
            fill_value=NEG_INF,
            dtype=torch.float32,
        )
        causal_allowed = torch.tril(torch.ones(prompt_len, prompt_len, dtype=torch.bool))
        synthetic_base_mask[0, 0][causal_allowed] = 0
        transfer_instruction_mask = build_abp_attention_mask(
            synthetic_base_mask,
            policy=policy,
            stage="transfer",
        )
        post_instruction_mask = build_abp_attention_mask(
            synthetic_base_mask,
            policy=policy,
            stage="post_transfer",
        )
        actual_instruction_transfer = [
            int(key_idx)
            for key_idx in torch.nonzero(
                transfer_instruction_mask[0, 0, int(instruction_prompt_token)] == 0,
                as_tuple=False,
            ).flatten()
        ]
        expected_instruction_post = [int(instruction_prompt_token)]
        actual_instruction_post = [
            int(key_idx)
            for key_idx in torch.nonzero(
                post_instruction_mask[0, 0, int(instruction_prompt_token)] == 0,
                as_tuple=False,
            ).flatten()
        ]
        if actual_instruction_post != expected_instruction_post:
            raise RuntimeError(
                f"ABP validation failed for instruction token {instruction_prompt_token} after transfer: "
                f"expected {expected_instruction_post}, got {actual_instruction_post}"
            )
        if str(policy.instruction_mask_mode) == "base":
            expected_instruction_transfer = list(range(0, int(instruction_prompt_token) + 1))
        else:
            expected_instruction_transfer = _instruction_allowed_prompt_keys(
                int(instruction_prompt_token),
                policy=policy,
                stage="transfer",
            )
            if expected_instruction_transfer is None:
                raise RuntimeError(
                    f"ABP validation failed for instruction token {instruction_prompt_token}: "
                    "expected an explicit transfer-stage key set"
                )
        if actual_instruction_transfer != expected_instruction_transfer:
            raise RuntimeError(
                f"ABP validation failed for instruction token {instruction_prompt_token} during transfer: "
                f"expected {expected_instruction_transfer}, got {actual_instruction_transfer}"
            )
        instruction_transfer_note = (
            "ABP instruction-token transfer handling: "
            f"positions={list(policy.instruction_positions)} tokens={instruction_tokens} "
            f"{instruction_mask_mode_summary(str(policy.instruction_mask_mode))} "
            f"transfer_keys={actual_instruction_transfer}"
        )

    expected_transfer_last = list(range(0, layout.carrier_index + 1))
    if transfer_last != expected_transfer_last:
        raise RuntimeError(
            f"ABP validation failed for carrier transfer stage: "
            f"expected {expected_transfer_last}, got {transfer_last}"
        )

    expected_post_last = [layout.carrier_index]
    if post_last != expected_post_last:
        raise RuntimeError(
            f"ABP validation failed for carrier post-transfer stage: "
            f"expected {expected_post_last}, got {post_last}"
        )

    return [
        frame_transfer_note,
        non_frame_transfer_note,
        instruction_transfer_note,
        f"ABP carrier allowed keys at transfer: query={layout.carrier_index} keys={transfer_last}",
        f"ABP carrier allowed keys after transfer: query={layout.carrier_index} keys={post_last}",
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
) -> Any:
    force_eager_attention_backend()
    with torch.inference_mode():
        return base_model(
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
) -> torch.Tensor:
    """Capture hidden states exactly at the AF1 wait boundary `x^(L_wait)`.

    Layer indexing detail:
    - if `wait_layer == 0`, we need `x^(0)`, so we capture the incoming hidden
      states before layer 0 runs
    - if `wait_layer > 0`, we need the output of layer `wait_layer - 1`

    We stop the forward pass immediately after capture because the conditional
    mean only needs the wait-boundary activations for the selected positions.
    """
    layers = get_layers(base_model)
    if wait_layer < 0 or wait_layer > len(layers):
        raise ValueError(f"wait_layer={wait_layer} must be in [0, {len(layers)}]")

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

    force_eager_attention_backend()
    with temporary_layer_wrappers(layers, wrapper_factory):
        with torch.inference_mode():
            try:
                base_model(
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
) -> Path:
    donor_ids_hash = hashlib.sha1(json.dumps(list(donor_ids), sort_keys=True).encode("utf-8")).hexdigest()[:16]
    donor_policy_hash = hashlib.sha1(str(donor_policy).encode("utf-8")).hexdigest()[:10]
    return (
        cache_dir
        / canonical_model_slug(model_name)
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
) -> Path:
    donor_ids_hash = hashlib.sha1(json.dumps(list(donor_ids), sort_keys=True).encode("utf-8")).hexdigest()[:16]
    donor_policy_hash = hashlib.sha1(str(donor_policy).encode("utf-8")).hexdigest()[:10]
    return (
        cache_dir
        / canonical_model_slug(model_name)
        / f"seq_len_{seq_len}"
        / f"wait_{wait_layer}"
        / f"target_{target_sample_id}"
        / (
            f"non_frame_k_{k_donors_used}_policy_{donor_policy_hash}_"
            f"donors_{donor_ids_hash}_layout_{layout_hash_value}.pt"
        )
    )


def compute_frame_group_conditional_mean(
    target_sample: PreparedSample,
    frame_idx: int,
    donor_samples: Sequence[PreparedSample],
    wait_layer: int,
    batch_size: int,
    cache_dir: Path,
    recompute_cache: bool,
    donor_policy: str,
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
    cache_path = _conditional_mean_cache_path(
        cache_dir=cache_dir,
        model_name=MODEL_ID,
        seq_len=target_sample.layout.seq_len,
        target_sample_id=target_sample.sample_id,
        frame_idx=frame_idx,
        wait_layer=wait_layer,
        k_donors_used=len(donor_samples),
        donor_policy=donor_policy,
        donor_ids=donor_ids,
        layout_hash_value=layout_hash(target_sample.layout),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not recompute_cache:
        payload = torch.load(cache_path, map_location="cpu")
        return payload["mean_block"].to(dtype=torch.float32), True

    frame_positions = target_sample.layout.frame_groups[frame_idx]
    blocks: List[torch.Tensor] = []
    for donor_batch in batched(list(donor_samples), batch_size=batch_size):
        hybrid_inputs_list: List[Dict[str, torch.Tensor]] = []
        for donor_sample in donor_batch:
            hybrid = build_hybrid_sample(target_sample=target_sample, donor_sample=donor_sample, frame_idx=frame_idx)
            hybrid_inputs_cpu = core.build_inputs(hybrid["frames"], hybrid["question"])
            hybrid_inputs_list.append(move_inputs_to_model_device(hybrid_inputs_cpu))

        # Batched hybrid evaluation is safe only because layout validation
        # guarantees exact prompt/token alignment across compatible donors.
        batched_inputs = core.concatenate_inputs_for_batch(hybrid_inputs_list)
        blocks.append(_capture_wait_boundary_blocks(batched_inputs, wait_layer=wait_layer, positions=frame_positions))

    mean_block = torch.cat(blocks, dim=0).mean(dim=0).to(dtype=torch.float32).cpu()
    torch.save(
        {
            "mean_block": mean_block,
            "metadata": {
                "model_name": MODEL_ID,
                "seq_len": int(target_sample.layout.seq_len),
                "target_sample_id": target_sample.sample_id,
                "frame_idx": int(frame_idx),
                "wait_layer": int(wait_layer),
                "k_donors_requested": int(len(donor_samples)),
                "k_donors_used": int(len(donor_samples)),
                "donor_policy": donor_policy,
                "donor_ids": list(donor_ids),
                "layout_hash": layout_hash(target_sample.layout),
                "cache_semantics": (
                    "frame-group conditional mean at x^(L_wait) estimated from hybrid contexts "
                    "that keep the target frame fixed, replace the other frames with donor frames, "
                    "and keep the target text prompt fixed"
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
    cache_path = _non_frame_conditional_mean_cache_path(
        cache_dir=cache_dir,
        model_name=MODEL_ID,
        seq_len=target_sample.layout.seq_len,
        target_sample_id=target_sample.sample_id,
        wait_layer=wait_layer,
        k_donors_used=len(donor_samples),
        donor_policy=donor_policy,
        donor_ids=donor_ids,
        layout_hash_value=layout_hash(target_sample.layout),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not recompute_cache:
        payload = torch.load(cache_path, map_location="cpu")
        return payload["mean_block"].to(dtype=torch.float32), True

    blocks: List[torch.Tensor] = []
    for donor_batch in batched(list(donor_samples), batch_size=batch_size):
        hybrid_inputs_list: List[Dict[str, torch.Tensor]] = []
        for donor_sample in donor_batch:
            hybrid = build_non_frame_hybrid_sample(target_sample=target_sample, donor_sample=donor_sample)
            hybrid_inputs_cpu = core.build_inputs(hybrid["frames"], hybrid["question"])
            hybrid_inputs_list.append(move_inputs_to_model_device(hybrid_inputs_cpu))

        # Batched hybrid evaluation is safe only because layout validation
        # guarantees exact prompt/token alignment across compatible donors.
        batched_inputs = core.concatenate_inputs_for_batch(hybrid_inputs_list)
        blocks.append(
            _capture_wait_boundary_blocks(
                batched_inputs,
                wait_layer=wait_layer,
                positions=non_frame_positions,
            )
        )

    mean_block = torch.cat(blocks, dim=0).mean(dim=0).to(dtype=torch.float32).cpu()
    torch.save(
        {
            "mean_block": mean_block,
            "metadata": {
                "model_name": MODEL_ID,
                "seq_len": int(target_sample.layout.seq_len),
                "target_sample_id": target_sample.sample_id,
                "wait_layer": int(wait_layer),
                "k_donors_requested": int(len(donor_samples)),
                "k_donors_used": int(len(donor_samples)),
                "donor_policy": donor_policy,
                "donor_ids": list(donor_ids),
                "layout_hash": layout_hash(target_sample.layout),
                "num_positions": int(len(non_frame_positions)),
                "cache_semantics": (
                    "all-non-frame prompt conditional mean at x^(L_wait) estimated from hybrid contexts "
                    "that keep the target text prompt fixed and replace the entire frame set with donor frames"
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
        )
        frame_means[frame_idx] = mean_block
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
    return frame_means, {"cache_hits": cache_hits, "cache_misses": cache_misses}


def run_clean_model(
    inputs: Dict[str, torch.Tensor],
    output_attentions: bool = False,
) -> Any:
    return _run_model_forward(
        inputs,
        output_hidden_states=False,
        output_attentions=output_attentions,
    )


def run_model_with_intervention(
    inputs: Dict[str, torch.Tensor],
    layout: SampleLayout,
    frame_group_means: Optional[Dict[int, torch.Tensor]],
    non_frame_prompt_mean: Optional[torch.Tensor],
    policy: AttentionPolicy,
    mode: str,
    output_attentions: bool = False,
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

    layers = get_layers(base_model)
    if int(policy.num_model_layers) != len(layers):
        raise RuntimeError(
            f"Attention policy expected {policy.num_model_layers} layers but model exposes {len(layers)} layers"
        )

    boundary_output_layer_idx = policy.wait_layer - 1

    def wrapper_factory(layer_idx: int, original_forward: Any) -> Any:
        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError("Layer forward received no hidden_states")

            batch_size = int(hidden_states.shape[0])
            stage = policy.stage_for_layer(layer_idx)
            base_attention_mask = kwargs.get("attention_mask")
            if enable_abp_mask and stage in {"transfer", "post_transfer"} and base_attention_mask is not None:
                kwargs["attention_mask"] = build_abp_attention_mask(
                    _ensure_mask_tensor(base_attention_mask, batch_size=batch_size),
                    policy=policy,
                    stage=stage,
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

    force_eager_attention_backend()
    with temporary_layer_wrappers(layers, wrapper_factory):
        with torch.inference_mode():
            return base_model(
                **inputs,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=True,
            )

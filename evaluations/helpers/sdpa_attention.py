"""Shared SDPA mask helpers for AF1-style per-layer prompt restrictions."""

from typing import Any, Callable, Dict, List, Optional

import torch
from transformers.masking_utils import (
    create_causal_mask,
    create_chunked_causal_mask,
    create_sliding_window_causal_mask,
)

_MASK_BUILDERS = {
    "full_attention": create_causal_mask,
    "sliding_attention": create_sliding_window_causal_mask,
    "chunked_attention": create_chunked_causal_mask,
}


def _normalize_2d_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError(
            "Expected the raw model attention mask to be rank-2 when rebuilding an SDPA mask, "
            f"got shape={tuple(attention_mask.shape)}"
        )
    if int(attention_mask.shape[0]) == batch_size:
        return attention_mask.to(device)
    if int(attention_mask.shape[0]) == 1:
        return attention_mask.expand(batch_size, -1).to(device)
    raise ValueError(
        f"Cannot expand raw attention_mask shape={tuple(attention_mask.shape)} to batch_size={batch_size}"
    )


def _resolve_cache_position(hidden_states: torch.Tensor, cache_position: Optional[torch.Tensor]) -> torch.Tensor:
    if cache_position is None:
        return torch.arange(hidden_states.shape[1], device=hidden_states.device)
    return cache_position.to(hidden_states.device)


def _expand_4d_mask_batch(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"Expected a rank-4 SDPA mask, got shape={tuple(mask.shape)}")
    if int(mask.shape[0]) == batch_size:
        return mask
    if int(mask.shape[0]) == 1:
        return mask.expand(batch_size, -1, -1, -1)
    raise ValueError(f"Cannot expand SDPA mask shape={tuple(mask.shape)} to batch_size={batch_size}")


def build_prompt_allow_matrix(
    *,
    query_len: int,
    key_len: int,
    prompt_len: int,
    device: torch.device,
    allowed_keys_by_query_fn: Callable[[int], Optional[List[int]]],
) -> torch.Tensor:
    # Rows beyond the original prompt belong to answer tokens appended for
    # scoring. Those rows must keep the base causal mask exactly as-is.
    allow_matrix = torch.ones((query_len, key_len), dtype=torch.bool, device=device)
    prompt_rows = min(int(prompt_len), int(query_len))
    for query_idx in range(prompt_rows):
        allowed_keys = allowed_keys_by_query_fn(int(query_idx))
        if allowed_keys is None:
            continue
        allow_row = torch.zeros(key_len, dtype=torch.bool, device=device)
        for key_idx in allowed_keys:
            if 0 <= int(key_idx) < key_len:
                allow_row[int(key_idx)] = True
        allow_matrix[int(query_idx), :] = allow_row
    return allow_matrix


def mask_function_from_allow_matrix(allow_matrix: torch.Tensor) -> Callable[[Any, Any, Any, Any], torch.Tensor]:
    def inner_mask(batch_idx: Any, head_idx: Any, q_idx: Any, kv_idx: Any) -> torch.Tensor:
        del batch_idx, head_idx
        return allow_matrix[q_idx, kv_idx]

    return inner_mask


def build_sdpa_layer_mask(
    *,
    model_config: Any,
    attention_type: str,
    hidden_states: torch.Tensor,
    raw_attention_mask: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    past_key_values: Optional[Any],
    allow_matrix: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if attention_type not in _MASK_BUILDERS:
        raise ValueError(
            f"Unsupported attention_type={attention_type!r}. "
            f"Expected one of {sorted(_MASK_BUILDERS)}."
        )

    batch_size = int(hidden_states.shape[0])
    mask_builder = _MASK_BUILDERS[attention_type]
    normalized_attention_mask = _normalize_2d_attention_mask(
        attention_mask=raw_attention_mask,
        batch_size=batch_size,
        device=hidden_states.device,
    )
    resolved_cache_position = _resolve_cache_position(hidden_states, cache_position)
    and_mask_function = None
    if allow_matrix is not None:
        and_mask_function = mask_function_from_allow_matrix(allow_matrix.to(hidden_states.device))

    # We rebuild the layer mask through the Transformers SDPA mask factory so
    # the result preserves the model's own causal/sliding/padding semantics and
    # only intersects them with the AF1/ABP allow-set.
    mask = mask_builder(
        config=model_config,
        input_embeds=hidden_states,
        attention_mask=normalized_attention_mask,
        cache_position=resolved_cache_position,
        past_key_values=past_key_values,
        and_mask_function=and_mask_function,
    )
    if mask is None:
        if allow_matrix is not None:
            raise RuntimeError(
                "Expected an explicit SDPA mask when applying a custom prompt restriction, "
                "but the mask builder returned None."
            )
        return None
    if isinstance(mask, torch.Tensor):
        return _expand_4d_mask_batch(mask, batch_size=batch_size)
    return mask


def allowed_key_positions_from_mask(mask: torch.Tensor, query_idx: int, key_len: Optional[int] = None) -> List[int]:
    if mask.ndim != 4:
        raise ValueError(f"Expected rank-4 mask when inspecting allowed keys, got shape={tuple(mask.shape)}")
    limit = int(mask.shape[-1]) if key_len is None else min(int(key_len), int(mask.shape[-1]))
    row = mask[0, 0, int(query_idx), :limit]
    if row.dtype == torch.bool:
        return [int(idx) for idx in torch.nonzero(row, as_tuple=False).flatten().tolist()]
    return [int(idx) for idx in torch.nonzero(row == 0, as_tuple=False).flatten().tolist()]

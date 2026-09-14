"""Ordinary adaptation controls for the native aggregation attribution study.

The hidden-only adapter receives the normalized pre-attention hidden state and
adds its result after the attention output projection, before the residual sum.
It delegates attention and KV-cache updates to the original model forward.
The placement control combines existing LoRA hooks at one middle layer and the
upper layers. Neither control changes the vocabulary head or V1 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import types
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .carriers import Lora, attach_lora


class HiddenOnlyAdapter(nn.Module):
    """Delta(h) = up(SiLU(down(h))); zero output at initialization."""

    def __init__(self, hidden_size: int, *, rank: int = 96):
        super().__init__()
        if min(hidden_size, rank) < 1:
            raise ValueError("Hidden size and rank must be positive")
        self.rank = rank
        self.mode = "all"
        self.down = nn.Linear(hidden_size, rank, bias=True)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)
        self._remove_callback = None

    def forward(self, hidden):
        return self.up(F.silu(self.down(hidden.to(self.up.weight.dtype)))).to(hidden.dtype)

    def remove(self):
        if self._remove_callback is not None:
            self._remove_callback()
            self._remove_callback = None


def _forward_with_hidden_adapter(attn, hidden_states, position_embeddings=None,
                                 attention_mask=None, past_key_values=None,
                                 cache_position=None, **kwargs):
    branch = attn.hidden_only_adapter
    if past_key_values is None:
        past_key_values = kwargs.pop("past_key_value", None)
    if branch.mode not in ("all", "off", "prefill", "decode"):
        raise ValueError("Unknown hidden-adapter mode")
    # Inspect history before the original forward performs its cache update.
    is_decode = past_key_values is not None and past_key_values.get_seq_length(attn.layer_idx) > 0
    enabled = (branch.mode == "all"
               or (branch.mode == "decode" and is_decode)
               or (branch.mode == "prefill" and not is_decode))
    result = attn._hidden_only_original_forward(
        hidden_states=hidden_states, position_embeddings=position_embeddings,
        attention_mask=attention_mask, past_key_values=past_key_values,
        cache_position=cache_position, **kwargs,
    )
    if enabled:
        return (result[0] + branch(hidden_states), *result[1:])
    return result


def attach_hidden_only_adapter(model, layer_index: int, *, rank: int = 96) -> HiddenOnlyAdapter:
    """Attach at the same residual site as V1; freeze the base model first.

    Registered parameters live at self_attn.hidden_only_adapter. Save its state
    dictionary with the constructor rank and layer index. The normalized input
    is exactly the hidden_states argument to the selected attention module.
    """
    from .runtime import get_layers

    layers = model.layers if hasattr(model, "layers") else get_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError("Invalid adapter layer index")
    attn = layers[layer_index].self_attn
    if type(attn).__name__ not in ("Qwen2Attention", "Qwen3Attention", "Qwen2_5_VLAttention"):
        raise TypeError(f"Unsupported attention class: {type(attn).__name__}")
    if attn.config._attn_implementation != "sdpa" or attn.sliding_window is not None:
        raise ValueError("Initial attribution control requires SDPA and full attention")
    if hasattr(attn, "native_aggregation") or hasattr(attn, "hidden_only_adapter"):
        raise ValueError("An adapter is already attached at this attention layer")
    branch = HiddenOnlyAdapter(attn.q_proj.in_features, rank=rank)
    branch.to(device=attn.q_proj.weight.device, dtype=torch.float32)
    attn.add_module("hidden_only_adapter", branch)
    attn._hidden_only_original_forward = attn.forward
    attn.forward = types.MethodType(_forward_with_hidden_adapter, attn)

    def remove():
        attn.forward = attn._hidden_only_original_forward
        del attn._hidden_only_original_forward
        del attn.hidden_only_adapter

    branch._remove_callback = remove
    return branch


@dataclass
class LoraGroup:
    """A flat checkpoint/optimizer interface for disjoint LoRA placements.

    Scaling remains captured by each original hook; groups can have different
    ranks and alphas. State keys retain the actual decoder layer indices.
    """

    groups: tuple[Lora, ...]
    params: dict[tuple[int, str], tuple[nn.Parameter, nn.Parameter]] = field(init=False)

    def __post_init__(self):
        self.params = {}
        for group in self.groups:
            if self.params.keys() & group.params.keys():
                raise ValueError("LoRA placements must not overlap")
            self.params.update(group.params)

    def parameters(self):
        return [parameter for pair in self.params.values() for parameter in pair]

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def state(self):
        return {f"{index}.{name}": (a.detach().cpu(), b.detach().cpu())
                for (index, name), (a, b) in self.params.items()}

    def remove(self):
        for group in self.groups:
            group.remove()


def attach_middle_and_upper_lora(
    layers: Any,
    *,
    middle_layer_index: int,
    upper_layers: int = 4,
    middle_rank: int = 32,
    middle_alpha: float = 64.0,
    upper_rank: int = 8,
    upper_alpha: float = 16.0,
    device: Any,
) -> LoraGroup:
    """Use the legacy LoRA operation at one middle layer and upper layers.

    Reset the seed before calling, as for attach_lora. Constructing the upper
    group first preserves its initialization relative to other experiment arms.
    For Qwen2.5-VL-7B defaults each group has 720896 trainable parameters.
    """
    if not 1 <= upper_layers <= len(layers):
        raise ValueError("Invalid number of upper LoRA layers")
    upper_start = len(layers) - upper_layers
    if not 0 <= middle_layer_index < upper_start:
        raise ValueError("Middle layer must be valid and below the upper LoRA layers")
    if min(middle_rank, upper_rank) < 1 or min(middle_alpha, upper_alpha) <= 0:
        raise ValueError("LoRA ranks and alphas must be positive")
    upper = attach_lora(layers, upper_start, rank=upper_rank,
                        alpha=upper_alpha, device=device)
    try:
        middle = attach_lora([layers[middle_layer_index]], 0, rank=middle_rank,
                             alpha=middle_alpha, device=device)
    except Exception:
        upper.remove()
        raise
    middle.params = {(middle_layer_index, name): pair
                     for (_, name), pair in middle.params.items()}
    middle.l_open = middle_layer_index
    return LoraGroup((upper, middle))

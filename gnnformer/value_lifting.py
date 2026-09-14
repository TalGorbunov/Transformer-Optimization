"""Native-width value lifting before/after attention: a controlled diagnostic.

The frozen/native attention path, vocabulary head, positions and KV cache remain
unchanged. This branch adds one normalized SDPA read. It recomputes lifted values
from the native cached V on every forward; it has no transformed-value cache.
The nonlinear-placement comparison is inspired by existing feature-map-before-
pooling methods and makes no new attention-family or counting guarantee.
"""
from __future__ import annotations

import types

import torch
from torch import nn
from torch.nn import functional as F

from .native_aggregation import _forward_with_aggregation, _repeat_kv


def _mask_for_sdpa(query, key, attention_mask, query_positions):
    """Preserve absolute cache causality, padding and additive attention biases."""
    batch, _, queries, _ = query.shape
    keys = key.shape[-2]
    expected = torch.arange(keys - queries, keys, device=query.device)
    positions = expected if query_positions is None else query_positions
    if positions.ndim == 1:
        positions = positions[None]
    if positions.ndim != 2 or positions.shape[-1] != queries or positions.shape[0] not in (1, batch):
        raise ValueError("Invalid absolute query positions")
    trailing = torch.equal(positions, expected[None].expand_as(positions))
    if attention_mask is None and trailing and (queries == keys or queries == 1):
        valid = torch.ones(batch, queries, dtype=torch.bool, device=query.device)
        return None, queries == keys and queries > 1, valid
    allowed = torch.arange(keys, device=query.device)[None, None, None] <= positions[:, None, :, None]
    additive = None
    if attention_mask is not None:
        mask = attention_mask[..., :keys]
        if mask.shape[-1] != keys:
            raise ValueError("Attention mask is shorter than the key cache")
        if mask.ndim == 2:
            allowed = allowed & mask[:, None, None, :].bool()
        elif mask.ndim == 4:
            if mask.shape[-2] not in (1, queries):
                raise ValueError("Attention mask has the wrong query length")
            if mask.dtype == torch.bool:
                allowed = allowed & mask
            elif mask.is_floating_point():
                allowed = allowed & torch.isfinite(mask) & (mask > torch.finfo(mask.dtype).min / 2)
                # SDPA accepts float32 masks with low-precision query tensors.
                dtype = mask.dtype if mask.dtype in (torch.float32, query.dtype) else query.dtype
                additive = torch.where(allowed, mask.to(dtype), -torch.inf)
            else:
                raise ValueError("4D masks must be Boolean or floating point")
        else:
            raise ValueError("Attention mask must be 2D or 4D")
    valid = allowed.any(dim=-1).any(dim=1).expand(batch, -1)
    return allowed if additive is None else additive, False, valid


class ValueLifting(nn.Module):
    """Same outer adapter, with SiLU before or after a native-width SDPA read."""

    def __init__(self, hidden_size, read_size, kv_size, head_dim, *,
                 rank=64, lift_rank=8, placement="before", activation="silu"):
        super().__init__()
        if min(hidden_size, read_size, kv_size, head_dim, rank, lift_rank) < 1:
            raise ValueError("Dimensions and ranks must be positive")
        if read_size % head_dim or kv_size % head_dim or read_size % kv_size:
            raise ValueError("Expected complete native query/KV heads and integral GQA ratio")
        if placement not in ("before", "after") or activation not in ("silu", "identity"):
            raise ValueError("Unknown placement or lift activation")
        self.rank, self.lift_rank = rank, lift_rank
        self.head_dim, self.kv_size, self.read_size = head_dim, kv_size, read_size
        self.placement, self.activation = placement, activation
        self.mode = "all"
        # Match the V2 global adapter's common initialization order exactly.
        self.query_down = nn.Linear(hidden_size, rank, bias=False)
        self.read_down = nn.Linear(read_size, rank, bias=True)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)
        # One shared tokenwise map on concatenated native KV heads.
        self.lift_down = nn.Linear(kv_size, lift_rank, bias=False)
        self.lift_up = nn.Linear(lift_rank, kv_size, bias=True)
        nn.init.zeros_(self.lift_up.weight)
        nn.init.zeros_(self.lift_up.bias)
        self._remove_callback = None

    def lift_values(self, value):
        batch, heads, keys, dim = value.shape
        if heads * dim != self.kv_size or dim != self.head_dim:
            raise ValueError("Value tensor does not match the native KV layout")
        packed = value.transpose(1, 2).flatten(-2).to(self.lift_down.weight.dtype)
        lifted = packed + self.lift_up(self.lift_down(packed))
        return lifted.view(batch, keys, heads, dim).transpose(1, 2)

    def attention_read(self, query, key, value, attention_mask=None,
                       query_positions=None, scaling=None):
        if (key.shape != value.shape or query.shape[-1] != self.head_dim
                or query.shape[1] * self.head_dim != self.read_size
                or not 1 <= query.shape[-2] <= key.shape[-2]):
            raise ValueError("Invalid native query/key/value layout")
        mask, causal, valid = _mask_for_sdpa(query, key, attention_mask, query_positions)
        lifted = self.lift_values(value)
        if self.placement == "before" and self.activation == "silu":
            lifted = F.silu(lifted)
        heads = query.shape[1]
        read = F.scaled_dot_product_attention(
            query, _repeat_kv(key, heads), _repeat_kv(lifted.to(query.dtype), heads),
            attn_mask=mask, dropout_p=0.0, is_causal=causal,
            scale=self.head_dim ** -0.5 if scaling is None else scaling,
        ).to(self.up.weight.dtype)
        if self.placement == "after" and self.activation == "silu":
            read = F.silu(read)
        return read, valid

    def forward(self, hidden, query, key, value, attention_mask, query_positions, scaling):
        read, valid = self.attention_read(query, key, value, attention_mask, query_positions, scaling)
        read = read.transpose(1, 2).flatten(-2)
        messages = self.read_down(read) + self.query_down(hidden.to(self.up.weight.dtype))
        merged = F.silu(messages) * valid[..., None]
        return self.up(merged).to(hidden.dtype)

    def remove(self):
        if self._remove_callback is not None:
            self._remove_callback()
            self._remove_callback = None


def attach_value_lifting(model, layer_index, *, rank=64, lift_rank=8,
                         placement="before", activation="silu"):
    """Attach using the existing V1 wrapper, without copying attention code.

    The reused wrapper in gnnformer/native_aggregation.py owns projections, RoPE,
    ordinary SDPA and the single native cache update. Only its registered branch
    is new. Freeze the base before attachment; save branch state and constructor
    parameters separately. No dummy adapter is created or retained.
    """
    from .runtime import get_layers
    layers = model.layers if hasattr(model, "layers") else get_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError("Invalid layer index")
    attn = layers[layer_index].self_attn
    if type(attn).__name__ not in ("Qwen2Attention", "Qwen3Attention", "Qwen2_5_VLAttention"):
        raise TypeError(f"Unsupported attention class: {type(attn).__name__}")
    if attn.config._attn_implementation != "sdpa" or attn.sliding_window is not None:
        raise ValueError("Value-lifting comparison requires SDPA and full attention")
    if hasattr(attn, "native_aggregation") or hasattr(attn, "hidden_only_adapter"):
        raise ValueError("An adapter is already attached")
    branch = ValueLifting(attn.q_proj.in_features, attn.o_proj.in_features,
                          attn.v_proj.out_features, attn.head_dim, rank=rank,
                          lift_rank=lift_rank, placement=placement, activation=activation)
    branch.to(device=attn.q_proj.weight.device, dtype=torch.float32)
    attn.add_module("native_aggregation", branch)
    attn._native_original_forward = attn.forward
    attn.forward = types.MethodType(_forward_with_aggregation, attn)

    def remove():
        attn.forward = attn._native_original_forward
        del attn._native_original_forward
        del attn.native_aggregation

    branch._remove_callback = remove
    return branch

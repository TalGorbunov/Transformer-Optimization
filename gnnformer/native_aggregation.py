"""Causal local attention reads with a trainable message before the merge.

The ordinary SDPA path and native vocabulary head are retained. This research
adapter supports the pinned transformers 4.57 Qwen2, Qwen3 and Qwen2.5-VL text
attention implementations, with full attention and the ordinary HF KV cache.
It does not fence tokens, reset positions, or assign numerical coordinates.
"""
from __future__ import annotations

import types
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _repeat_kv(x: torch.Tensor, heads: int) -> torch.Tensor:
    if heads % x.shape[1]:
        raise ValueError("Query heads must be a multiple of KV heads")
    return x.repeat_interleave(heads // x.shape[1], dim=1)


def blockwise_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    query_positions: Optional[torch.Tensor] = None,
    block_size: int = 64,
    scaling: Optional[float] = None,
):
    """Return reads [B,H,Q,R,D], log normalizers and validity [B,H,Q,R].

    Blocks use absolute cache slots, including padding slots. Four-dimensional
    bool masks follow SDPA (True means allowed); float masks are additive. A 2D
    mask is a key-validity mask. Causality is enforced in addition to the mask.
    With no explicit positions the queries are the last Q entries of the cache.
    Accumulation is fp32 (fp64 retained for numerical tests). Empty reads are zero.
    Call on query tiles rather than materializing all query-region messages.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    b, h, q, d = query.shape
    k = key.shape[-2]
    if not k or value.shape != key.shape or q > k:
        raise ValueError("Expected nonempty matching K/V and Q <= K")
    key, value = _repeat_kv(key, h), _repeat_kv(value, h)
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    logits = torch.matmul(query.to(dtype), key.to(dtype).transpose(-1, -2))
    logits = logits * (d ** -0.5 if scaling is None else scaling)
    if query_positions is None:
        query_positions = torch.arange(k - q, k, device=query.device)
    if query_positions.ndim == 1:
        query_positions = query_positions[None, :]
    if query_positions.shape[-1] != q:
        raise ValueError("query_positions length must equal the query length")
    visible = torch.arange(k, device=query.device)[None, None, None, :] <= query_positions[:, None, :, None]
    if attention_mask is not None:
        mask = attention_mask[..., :k]
        if mask.ndim == 2:
            visible = visible & mask[:, None, None, :].bool()
        elif mask.ndim == 4:
            if mask.dtype == torch.bool:
                visible = visible & mask
            else:
                # HF uses finfo.min for excluded keys, rather than -inf.
                floor = torch.finfo(mask.dtype).min / 2
                visible = visible & (mask > floor) & torch.isfinite(mask)
                logits = logits + mask.to(dtype)
        else:
            raise ValueError("attention_mask must have 2 or 4 dimensions")
    logits = logits.masked_fill(~visible, -torch.inf)
    regions = (k + block_size - 1) // block_size
    pad = regions * block_size - k
    scores = F.pad(logits, (0, pad), value=-torch.inf).view(b, h, q, regions, block_size)
    valid = torch.isfinite(scores).any(dim=-1)
    # Avoid softmax(-inf,...,-inf), including its undefined backward.
    safe_scores = torch.where(valid[..., None], scores, torch.zeros_like(scores))
    log_z = torch.logsumexp(safe_scores, dim=-1).masked_fill(~valid, -torch.inf)
    weights = torch.softmax(safe_scores, dim=-1) * valid[..., None]
    values = F.pad(value.to(dtype), (0, 0, 0, pad)).view(b, h, regions, block_size, d)
    reads = torch.einsum("bhqrw,bhrwd->bhqrd", weights, values)
    return reads.to(query.dtype), log_z, valid


def reconstruct_global_attention(reads: torch.Tensor, log_z: torch.Tensor) -> torch.Tensor:
    """Exact segmented softmax reference, before dropout/output projection."""
    any_valid = torch.isfinite(log_z).any(dim=-1, keepdim=True)
    safe = torch.where(any_valid, log_z, torch.zeros_like(log_z))
    weights = torch.softmax(safe, dim=-1) * any_valid
    return (weights[..., None] * reads.to(weights.dtype)).sum(dim=-2).to(reads.dtype)


class NativeAggregation(nn.Module):
    def __init__(self, hidden_size: int, read_size: int, *, block_size: int = 64,
                 rank: int = 64, nonlinear: bool = True, merge: str = "sum",
                 query_chunk_size: int = 32, center_messages: bool = False,
                 variant: str = "local_pre", routing_rank: int = 8,
                 head_dim: Optional[int] = None):
        super().__init__()
        if merge not in ("sum", "mean"):
            raise ValueError("merge must be sum or mean")
        if variant not in ("local_pre", "local_post", "global", "hierarchical"):
            raise ValueError("Unknown aggregation variant")
        if variant in ("global", "hierarchical") and merge != "mean":
            raise ValueError("Global and hierarchical fusion require merge='mean'")
        if center_messages and variant != "local_pre":
            raise ValueError("Centering is supported only for the legacy local_pre variant")
        if min(block_size, rank, query_chunk_size, routing_rank) < 1:
            raise ValueError("block size, ranks and query chunk size must be positive")
        self.block_size, self.rank = block_size, rank
        self.nonlinear, self.merge = nonlinear, merge
        self.query_chunk_size = query_chunk_size
        self.center_messages = center_messages
        self.variant, self.routing_rank = variant, routing_rank
        self.mode = "all"
        # Keep common parameter construction/initialization identical across arms.
        self.query_down = nn.Linear(hidden_size, rank, bias=False)
        self.read_down = nn.Linear(read_size, rank, bias=True)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)
        if variant == "hierarchical":
            if head_dim is None or head_dim < 1 or read_size % head_dim:
                raise ValueError("hierarchical requires a valid head_dim")
            self.route_query = nn.Linear(head_dim, routing_rank, bias=False)
            self.route_read = nn.Linear(head_dim, routing_rank, bias=True)
            self.route_out = nn.Linear(routing_rank, 1, bias=False)
            nn.init.zeros_(self.route_out.weight)
        self._remove_callback = None

    def hierarchical_weights(self, query, reads, log_z, valid):
        """Per-head normalized block weights; zero score correction recovers global.

        Scorer weights are shared across heads, but queries, reads and softmax
        distributions are separate. This dense fusion control is not HiLS.
        """
        dtype = self.up.weight.dtype
        local = self.route_read(reads.to(dtype))
        current = self.route_query(query.to(dtype)).unsqueeze(-2)
        correction = self.route_out(F.silu(local + current)).squeeze(-1)
        scores = (log_z.to(dtype) + correction).masked_fill(~valid, -torch.inf)
        any_valid = valid.any(dim=-1, keepdim=True)
        safe = torch.where(any_valid, scores, torch.zeros_like(scores))
        return torch.softmax(safe, dim=-1) * any_valid

    def forward(self, hidden, query, key, value, attention_mask, query_positions, scaling):
        pieces = []
        q_total = query.shape[2]
        for start in range(0, q_total, self.query_chunk_size):
            end = min(q_total, start + self.query_chunk_size)
            mask = attention_mask
            if mask is not None and mask.ndim == 4 and mask.shape[-2] > 1:
                mask = mask[..., start:end, :]
            positions = query_positions[..., start:end]
            tile_query = query[:, :, start:end]
            reads, log_z, valid = blockwise_attention(
                tile_query, key, value, mask, positions, self.block_size, scaling)
            dtype = self.up.weight.dtype
            query_term = self.query_down(hidden[:, start:end].to(dtype))
            visible_regions = valid.any(dim=1)
            counts = visible_regions.sum(dim=-1, keepdim=True)
            if self.variant in ("global", "hierarchical"):
                if self.variant == "global":
                    fused = reconstruct_global_attention(reads, log_z)
                else:
                    weights = self.hierarchical_weights(tile_query, reads, log_z, valid)
                    fused = (weights[..., None] * reads.to(dtype)).sum(dim=-2).to(reads.dtype)
                fused = fused.transpose(1, 2).flatten(-2)
                merged = self.read_down(fused.to(dtype)) + query_term
                if self.nonlinear:
                    merged = F.silu(merged)
                merged = merged * (counts > 0)
            else:
                # Concatenate heads within each region before the shared map.
                reads = reads.permute(0, 2, 3, 1, 4).flatten(-2)
                if self.variant == "local_post":
                    # The affine projection commutes with averaging. Execute it
                    # once/query so POST receives its natural compute advantage.
                    averaged = (reads.to(dtype) * visible_regions[..., None]).sum(dim=2)
                    averaged = averaged / counts.clamp_min(1)
                    merged = self.read_down(averaged) + query_term
                    if self.nonlinear:
                        merged = F.silu(merged)
                    merged = merged * (counts if self.merge == "sum" else counts > 0)
                else:
                    messages = self.read_down(reads.to(dtype)) + query_term[:, :, None]
                    if self.nonlinear:
                        messages = F.silu(messages)
                    if self.center_messages:
                        baseline = query_term[:, :, None] + self.read_down.bias
                        if self.nonlinear:
                            baseline = F.silu(baseline.expand_as(messages).contiguous())
                        messages = messages - baseline
                    messages = messages * visible_regions[..., None]
                    merged = messages.sum(dim=2)
                    if self.merge == "mean":
                        merged = merged / counts.clamp_min(1)
            pieces.append(self.up(merged).to(hidden.dtype))
        return torch.cat(pieces, dim=1)

    def remove(self):
        if self._remove_callback is not None:
            self._remove_callback()
            self._remove_callback = None


def _forward_with_aggregation(attn, hidden_states, position_embeddings=None,
                              attention_mask=None, past_key_values=None,
                              cache_position=None, **kwargs):
    branch = attn.native_aggregation
    if past_key_values is None:
        past_key_values = kwargs.pop("past_key_value", None)
    if branch.mode not in ("all", "off", "prefill", "decode"):
        raise ValueError("Unknown aggregation mode")
    # Use cache history, not Q==1, to distinguish a one-token prefill.
    is_decode = past_key_values is not None and past_key_values.get_seq_length(attn.layer_idx) > 0
    enabled = branch.mode == "all" or (branch.mode == "decode" and is_decode) or (branch.mode == "prefill" and not is_decode)
    if not enabled:
        return attn._native_original_forward(
            hidden_states=hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values,
            cache_position=cache_position, **kwargs)
    shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
    query = attn.q_proj(hidden_states).view(shape)
    key = attn.k_proj(hidden_states).view(shape)
    value = attn.v_proj(hidden_states).view(shape).transpose(1, 2)
    if hasattr(attn, "q_norm"):
        query, key = attn.q_norm(query), attn.k_norm(key)
    query, key = query.transpose(1, 2), key.transpose(1, 2)
    cos, sin = position_embeddings
    if type(attn).__name__ == "Qwen2_5_VLAttention":
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
        query, key = apply_multimodal_rotary_pos_emb(query, key, cos, sin, attn.rope_scaling["mrope_section"])
    else:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, attn.layer_idx,
                                           {"sin": sin, "cos": cos, "cache_position": cache_position})
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    output, weights = sdpa_attention_forward(
        attn, query, key, value, attention_mask,
        dropout=attn.attention_dropout if attn.training else 0.0,
        scaling=attn.scaling, **kwargs)
    output = attn.o_proj(output.reshape(*hidden_states.shape[:-1], -1).contiguous())
    if cache_position is None:
        cache_position = torch.arange(key.shape[-2] - query.shape[-2], key.shape[-2], device=query.device)
    return output + branch(hidden_states, query, key, value, attention_mask, cache_position, attn.scaling), weights


def attach_native_aggregation(model, layer_index: int, **kwargs) -> NativeAggregation:
    """Attach once to one decoder layer. Freeze the base model *before* calling.

    Parameters are registered at self_attn.native_aggregation; save its state_dict
    along with constructor kwargs and layer index. No backbone weights are copied.
    Base projections/RoPE and cache update execute once in the selected layer.
    """
    from .runtime import get_layers
    layers = model.layers if hasattr(model, "layers") else get_layers(model)
    attn = layers[layer_index].self_attn
    if type(attn).__name__ not in ("Qwen2Attention", "Qwen3Attention", "Qwen2_5_VLAttention"):
        raise TypeError(f"Unsupported attention class: {type(attn).__name__}")
    if attn.config._attn_implementation != "sdpa" or attn.sliding_window is not None:
        raise ValueError("Initial adapter requires SDPA and full attention")
    if hasattr(attn, "native_aggregation"):
        raise ValueError("An aggregation branch is already attached here")
    branch = NativeAggregation(attn.q_proj.in_features, attn.o_proj.in_features,
                               head_dim=attn.head_dim, **kwargs)
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

"""Software-only differentiable replay of the actual frozen final Qwen block.

The caller supplies frozen hidden states immediately before the final decoder
block, complete native token/mask/position layouts, and live FP32 residuals.
This module does not load a model, construct a dataset, fit parameters, or alter
native modules. It must be called outside any active native fusion controller.

Unlike a final-norm-only write, a pre_last write changes the final block's K/V.
Training replays the complete sequence without a cache, keeping every previous
assistant-query write in the graph. Native cached execution is a separate gate.
"""
from __future__ import annotations

import os

PLACEMENTS = ('pre_last', 'post_last')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def require_slurm():
    need(os.environ.get('SLURM_JOB_ID')
         and os.environ.get('SLURM_JOB_PARTITION') in ('cpu', 'gpu'),
         'Run all tensor/model work through CPU or GPU Slurm')


def _torch():
    require_slurm()
    import torch
    return torch


def _frozen_tensor(torch, value, name, dimensions):
    need(isinstance(value, torch.Tensor) and value.is_floating_point()
         and value.ndim == dimensions and all(x > 0 for x in value.shape)
         and not value.requires_grad and not torch.is_inference(value)
         and bool(torch.isfinite(value).all()),
         name+' must be a finite, frozen, ordinary floating tensor')


def _queries(torch, hidden_states, query_indices, deltas, attention_mask=None):
    batch, length, width = hidden_states.shape
    need(isinstance(query_indices, (list, tuple)) and len(query_indices) == batch,
         'One explicit query-index list per batch row is required')
    rows, positions, offsets = [], [], [0]
    for row, indices in enumerate(query_indices):
        need(isinstance(indices, (list, tuple))
             and all(type(x) is int and 0 <= x < length for x in indices)
             and list(indices) == sorted(set(indices)),
             'Query indices must be distinct, increasing, in-range integers')
        rows.extend([row]*len(indices)); positions.extend(indices); offsets.append(len(rows))
    need(rows, 'At least one query is required; empty local-row lists are allowed')
    need(isinstance(deltas, torch.Tensor) and deltas.dtype == torch.float32
         and deltas.shape == (len(rows), width) and deltas.device == hidden_states.device
         and not torch.is_inference(deltas) and bool(torch.isfinite(deltas).all()),
         'Deltas must be finite ordinary FP32 [sum_queries,H], on the native device')
    r = torch.tensor(rows, dtype=torch.long, device=hidden_states.device)
    p = torch.tensor(positions, dtype=torch.long, device=hidden_states.device)
    if attention_mask is not None:
        need(bool(attention_mask[r, p].bool().all()), 'A query cannot refer to a padding position')
    return r, p, offsets


def _scatter(hidden_states, rows, positions, deltas):
    # Keep the existing deployed cast/add order. Adding in FP32 and then casting
    # is a different numerical operation and is intentionally not used.
    updated = hidden_states.clone()
    updated[rows, positions] = hidden_states[rows, positions] + deltas.to(hidden_states.dtype)
    return updated


def reconstruct_global_sequence(prompt_hidden, prompt_input_ids, strict_prefix_states,
                                target_ids, prefix_ids, *, eos_token_id):
    """Rebuild one unpadded global sequence without future-token leakage.

    prompt_hidden: frozen [P,H] before the final block, for the complete prompt.
    strict_prefix_states: frozen [T,H], whose row t is the last-query state for
      the exact prefix target_ids[:t]. Row 0 must equal prompt_hidden[-1].
    target_ids: complete native target, including exactly one terminal EOS.
    prefix_ids: explicit prefix lists independently supplied by the cache plan.

    The returned sequence is prompt + target[:-1], with query positions P-1+t.
    No EOS target is placed into the teacher-forced input. Position IDs must be
    supplied/verified independently through the actual native position helper.
    """
    torch = _torch()
    _frozen_tensor(torch, prompt_hidden, 'prompt_hidden', 2)
    _frozen_tensor(torch, strict_prefix_states, 'strict_prefix_states', 2)
    need(isinstance(prompt_input_ids, torch.Tensor) and prompt_input_ids.ndim == 1
         and prompt_input_ids.shape[0] == prompt_hidden.shape[0]
         and prompt_input_ids.dtype in (torch.int32, torch.int64)
         and prompt_input_ids.device == prompt_hidden.device and bool((prompt_input_ids >= 0).all()),
         'Complete native prompt IDs differ from prompt states')
    need(type(eos_token_id) is int and eos_token_id >= 0
         and isinstance(target_ids, (list, tuple)) and len(target_ids) > 0
         and all(type(x) is int and x >= 0 for x in target_ids)
         and target_ids[-1] == eos_token_id and eos_token_id not in target_ids[:-1],
         'Target must contain exactly one terminal declared EOS')
    targets = list(target_ids); expected = [targets[:t] for t in range(len(targets))]
    need(isinstance(prefix_ids, (list, tuple)) and list(prefix_ids) == expected,
         'Cache prefixes must be the exact strict prefixes of the complete target')
    need(strict_prefix_states.shape == (len(targets), prompt_hidden.shape[1])
         and strict_prefix_states.device == prompt_hidden.device
         and strict_prefix_states.dtype == prompt_hidden.dtype
         and torch.equal(strict_prefix_states[0], prompt_hidden[-1]),
         'Strict-prefix states differ in layout/dtype or empty-prefix identity')
    hidden = torch.cat((prompt_hidden, strict_prefix_states[1:]), dim=0)
    ids = torch.cat((prompt_input_ids, prompt_input_ids.new_tensor(targets[:-1])))
    start = prompt_hidden.shape[0]-1
    return dict(hidden_states=hidden, input_ids=ids, attention_mask=torch.ones_like(ids),
                query_indices=list(range(start, start+len(targets))), target_ids=targets,
                prefix_ids=expected, prompt_length=prompt_hidden.shape[0])


def replay_last_block(last_block, final_norm, lm_head, rotary_emb, *, hidden_states,
                      input_ids, attention_mask, position_ids, query_indices,
                      deltas, placement):
    """Replay the actual frozen final decoder block with live residual leaves.

    hidden_states [B,L,H] are frozen states before that block; input_ids and
    attention_mask are native [B,L] arrays. position_ids is native [3|4,B,L].
    query_indices is a ragged list per batch row (empty local rows are valid).
    deltas [sum(T),H] is FP32 in batch-row/query order. Nothing is detached.

    Results include live queried logits and full final_norm_input for native
    head-batch-shape comparisons. Queried-only logits do not claim bitwise
    equality to a differently shaped native vocabulary GEMM.
    """
    torch = _torch()
    from transformers.masking_utils import create_causal_mask
    need(placement in PLACEMENTS, 'Unknown write placement')
    _frozen_tensor(torch, hidden_states, 'hidden_states', 3)
    batch, length, width = hidden_states.shape
    need(hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32),
         'Keep the actual native hidden dtype (FP32 is permitted for CPU fixtures)')
    need(isinstance(input_ids, torch.Tensor) and input_ids.shape == (batch, length)
         and input_ids.dtype in (torch.int32, torch.int64)
         and input_ids.device == hidden_states.device and bool((input_ids >= 0).all()),
         'Native input IDs must cover the complete replayed sequence')
    need(isinstance(attention_mask, torch.Tensor) and attention_mask.shape == input_ids.shape
         and attention_mask.device == hidden_states.device
         and bool(((attention_mask == 0) | (attention_mask == 1)).all())
         and bool(attention_mask.bool().any(dim=1).all()),
         'Require a nonempty native 2D binary padding mask for every row')
    need(isinstance(position_ids, torch.Tensor) and position_ids.ndim == 3
         and position_ids.shape[0] in (3, 4) and position_ids.shape[1:] == (batch, length)
         and position_ids.dtype in (torch.int32, torch.int64)
         and position_ids.device == hidden_states.device,
         'Require explicit complete native 3-axis or 4-axis positions')
    text_positions = position_ids[0] if position_ids.shape[0] == 4 else None
    rotary_positions = position_ids[-3:]
    valid = attention_mask.bool()
    need(bool((rotary_positions[:, valid] >= 0).all()), 'Negative valid-token mRoPE positions')
    if text_positions is not None:
        expected_text = attention_mask.long().cumsum(-1)-1
        need(torch.equal(text_positions[valid], expected_text[valid]),
             'Full-sequence native text positions disagree with the padding mask')
    for module in (last_block, final_norm, lm_head, rotary_emb):
        need(isinstance(module, torch.nn.Module) and not module.training
             and not any(p.requires_grad for p in module.parameters()),
             'All native modules must remain frozen/eval; do not disable input autograd')
    need(last_block.hidden_size == width and last_block.attention_type == 'full_attention'
         and last_block.self_attn.config._attn_implementation == 'sdpa',
         'This bounded prototype requires the actual full-attention Qwen SDPA block')
    need(final_norm.weight.dtype == lm_head.weight.dtype == hidden_states.dtype,
         'Preserve the actual native final norm/head dtype')
    rows, positions, offsets = _queries(torch, hidden_states, query_indices, deltas, attention_mask)
    block_input = (_scatter(hidden_states, rows, positions, deltas)
                   if placement == 'pre_last' else hidden_states.clone())
    cache_position = torch.arange(length, device=hidden_states.device, dtype=torch.long)
    mask = create_causal_mask(config=last_block.self_attn.config, input_embeds=block_input,
        attention_mask=attention_mask, cache_position=cache_position,
        past_key_values=None, position_ids=text_positions)
    position_embeddings = rotary_emb(block_input, rotary_positions)
    need(len(position_embeddings) == 2 and all(x.dtype == hidden_states.dtype
         and bool(torch.isfinite(x).all()) for x in position_embeddings), 'Native rotary embeddings differ')
    outputs = last_block(hidden_states=block_input, attention_mask=mask,
        position_ids=text_positions, past_key_values=None, output_attentions=False,
        use_cache=False, cache_position=cache_position, position_embeddings=position_embeddings)
    need(isinstance(outputs, tuple) and len(outputs) == 1 and outputs[0].shape == hidden_states.shape,
         'Unexpected native decoder result or cache output')
    block_output = outputs[0]
    final_norm_input = (_scatter(block_output, rows, positions, deltas)
                        if placement == 'post_last' else block_output)
    native_query_hidden = block_output[rows, positions]
    fused_query_hidden = final_norm_input[rows, positions]
    logits = lm_head(final_norm(fused_query_hidden))
    need(logits.shape[0] == len(rows) and logits.ndim == 2
         and logits.dtype == hidden_states.dtype and bool(torch.isfinite(logits).all()),
         'Nonfinite or incorrectly shaped native vocabulary output')
    return dict(logits=logits, native_query_hidden=native_query_hidden,
                fused_query_hidden=fused_query_hidden, final_norm_input=final_norm_input,
                block_input=block_input, block_output=block_output, deltas=deltas,
                offsets=offsets, flat_batch_indices=rows, flat_query_indices=positions,
                attention_mask=mask, input_attention_mask=attention_mask,
                position_embeddings=position_embeddings, cache_position=cache_position,
                position_ids=position_ids, placement=placement, use_cache=False)

"""Full causal final-block replay with independent write and loss positions.

All historical residual writes remain live. Only loss queries reach the frozen
native norm/head. Call outside active fusion controllers; the caller binds the
common lower states, actual modules and native token/position provenance.
No model, cache, dataset or optimizer is constructed here.
"""
from __future__ import annotations

from scripts.native_vision_v11_last_block import (
    PLACEMENTS, need, _torch, _frozen_tensor, _queries, _scatter,
    reconstruct_global_sequence,
)

OWN=('scripts/native_learned_memory_training.py','scripts/native_vision_v11_last_block.py')


def replay_learned_memory(last_block, final_norm, lm_head, rotary_emb, *,
                          hidden_states, input_ids, attention_mask, position_ids,
                          write_indices, loss_indices, deltas, placement):
    """Replay [B,L,H] common frozen lower states without a detached KV cache.

    ``write_indices`` and ``loss_indices`` are ragged, increasing, unique lists
    of physical sequence indices per batch row. Each loss must also be a write
    in the same row. Empty individual rows are allowed; at least one write and
    one loss are required globally. The caller supplies FP32 ``deltas`` in
    batch-row/write order, including every live historical assistant query.

    Position IDs retain the native [3|4,B,L] layout. Native FP16 residuals are
    cast before addition; other actual matching native dtypes retain the frozen
    V11 helper's contract, including FP32 CPU fixtures. No tensor is detached.
    logits/native_query_hidden/fused_query_hidden and the legacy index aliases
    refer to loss queries. Explicit write fields cover all historical writes.
    Selected-only vocabulary GEMMs do not claim bitwise equality with a
    differently shaped native full-batch head call.
    """
    torch=_torch()
    from transformers.masking_utils import create_causal_mask
    need(placement in PLACEMENTS,'Unknown write placement')
    _frozen_tensor(torch,hidden_states,'hidden_states',3)
    batch,length,width=hidden_states.shape
    need(hidden_states.dtype in (torch.float16,torch.bfloat16,torch.float32),'Unsupported native hidden dtype')
    need(isinstance(input_ids,torch.Tensor) and input_ids.shape==(batch,length)
         and input_ids.dtype in (torch.int32,torch.int64) and input_ids.device==hidden_states.device
         and bool((input_ids>=0).all()),'Complete native input IDs required')
    need(isinstance(attention_mask,torch.Tensor) and attention_mask.shape==input_ids.shape
         and attention_mask.device==hidden_states.device
         and bool(((attention_mask==0)|(attention_mask==1)).all())
         and bool(attention_mask.bool().any(dim=1).all()),'Nonempty native binary padding rows required')
    need(isinstance(position_ids,torch.Tensor) and position_ids.ndim==3
         and position_ids.shape[0] in (3,4) and position_ids.shape[1:]==(batch,length)
         and position_ids.dtype in (torch.int32,torch.int64) and position_ids.device==hidden_states.device,
         'Explicit complete native three/four-axis positions required')
    text_positions=position_ids[0] if position_ids.shape[0]==4 else None
    rotary_positions=position_ids[-3:];valid=attention_mask.bool()
    need(bool((rotary_positions[:,valid]>=0).all()),'Negative valid-token mRoPE position')
    if text_positions is not None:
        expected_text=attention_mask.long().cumsum(-1)-1
        need(torch.equal(text_positions[valid],expected_text[valid]),'Native text positions disagree with padding')
    for module in (last_block,final_norm,lm_head,rotary_emb):
        need(isinstance(module,torch.nn.Module) and not module.training
             and not any(p.requires_grad for p in module.parameters()),'Native modules must remain frozen/eval')
    need(last_block.hidden_size==width and last_block.attention_type=='full_attention'
         and last_block.self_attn.config._attn_implementation=='sdpa','Actual full-attention Qwen SDPA block required')
    need(final_norm.weight.dtype==lm_head.weight.dtype==hidden_states.dtype,'Native norm/head dtype differs')
    wr,wp,write_offsets=_queries(torch,hidden_states,write_indices,deltas,attention_mask)
    need(isinstance(loss_indices,(list,tuple)) and len(loss_indices)==batch,'One loss-index list per row required')
    loss_to_write=[]
    for row,indices in enumerate(loss_indices):
        need(isinstance(indices,(list,tuple)) and all(type(i) is int and 0<=i<length for i in indices)
             and list(indices)==sorted(set(indices)),'Loss indices must be increasing unique in-range integers')
        mapping={position:write_offsets[row]+i for i,position in enumerate(write_indices[row])}
        need(all(position in mapping for position in indices),'Every loss position must be a write in its own row')
        loss_to_write.extend(mapping[position] for position in indices)
    need(loss_to_write,'At least one native loss query is required')
    loss_to_write=torch.tensor(loss_to_write,dtype=torch.long,device=hidden_states.device)
    loss_deltas=deltas[loss_to_write]
    lr,lp,loss_offsets=_queries(torch,hidden_states,loss_indices,loss_deltas,attention_mask)

    # Reuse the immutable native cast-before-add scatter. Every write, including
    # an unsupervised earlier query, participates in the one full causal block.
    block_input=_scatter(hidden_states,wr,wp,deltas) if placement=='pre_last' else hidden_states.clone()
    need(bool(torch.isfinite(block_input).all()),'Native residual cast/add overflowed before final block')
    cache_position=torch.arange(length,device=hidden_states.device,dtype=torch.long)
    mask=create_causal_mask(config=last_block.self_attn.config,input_embeds=block_input,
        attention_mask=attention_mask,cache_position=cache_position,past_key_values=None,position_ids=text_positions)
    position_embeddings=rotary_emb(block_input,rotary_positions)
    need(len(position_embeddings)==2 and all(x.dtype==hidden_states.dtype and bool(torch.isfinite(x).all())
         for x in position_embeddings),'Native rotary embeddings differ')
    outputs=last_block(hidden_states=block_input,attention_mask=mask,position_ids=text_positions,
        past_key_values=None,output_attentions=False,use_cache=False,cache_position=cache_position,
        position_embeddings=position_embeddings)
    need(isinstance(outputs,tuple) and len(outputs)==1 and outputs[0].shape==hidden_states.shape
         and outputs[0].dtype==hidden_states.dtype and bool(torch.isfinite(outputs[0]).all()),
         'Unexpected or nonfinite native final-block result')
    block_output=outputs[0]
    final_norm_input=_scatter(block_output,wr,wp,deltas) if placement=='post_last' else block_output
    need(bool(torch.isfinite(final_norm_input).all()),'Native residual cast/add overflowed before final norm')
    native_query_hidden=block_output[lr,lp];fused_query_hidden=final_norm_input[lr,lp]
    logits=lm_head(final_norm(fused_query_hidden))
    need(logits.ndim==2 and logits.shape[0]==len(lr) and logits.dtype==hidden_states.dtype
         and bool(torch.isfinite(logits).all()),'Native loss-query vocabulary output differs')
    return dict(logits=logits,native_query_hidden=native_query_hidden,fused_query_hidden=fused_query_hidden,
        block_input=block_input,block_output=block_output,final_norm_input=final_norm_input,
        deltas=deltas,loss_deltas=loss_deltas,write_offsets=write_offsets,loss_offsets=loss_offsets,
        flat_write_batch_indices=wr,flat_write_indices=wp,flat_loss_batch_indices=lr,flat_loss_indices=lp,
        loss_to_write_indices=loss_to_write,write_native_query_hidden=block_output[wr,wp],
        write_fused_query_hidden=final_norm_input[wr,wp],offsets=loss_offsets,
        flat_batch_indices=lr,flat_query_indices=lp,
        layout=dict(write_indices=[list(x) for x in write_indices],loss_indices=[list(x) for x in loss_indices],
                    write_offsets=write_offsets,loss_offsets=loss_offsets),
        attention_mask=mask,input_attention_mask=attention_mask,position_embeddings=position_embeddings,
        cache_position=cache_position,position_ids=position_ids,placement=placement,use_cache=False)

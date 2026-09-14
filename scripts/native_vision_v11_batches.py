"""Whole global teacher-forced sequences for V11 live final-block training.

Source preparation only. Frozen penultimate states are supplied by a separately
verified cache. Every assistant prediction position remains in the live replay;
no prefix is decoded in isolation and no target EOS enters a model input.
"""
from __future__ import annotations
from scripts import train_native_vision_v10 as previous
from scripts.native_vision_v11_last_block import reconstruct_global_sequence, require_slurm
need=previous.need


def batch_states(torch, cache, feature_plan, states, index, prompts, sids):
    require_slurm()
    local, global_states, layout=previous.batch_states(torch,cache,states,index,sids)
    rows=[];positions=[]
    for i,sid in enumerate(sids):
        scene=cache['scenes'][sid];gids=scene['global_feature_ids'];prompt=prompts[gids[0]]
        a,b=layout['offsets'][i:i+2]
        item=reconstruct_global_sequence(prompt['hidden_states'],prompt['input_ids'],
            global_states[a:b],scene['target_ids'],scene['target_prefixes'],eos_token_id=151645)
        full=feature_plan['layouts'][feature_plan['features'][gids[-1]]['layout_id']]
        need(item['input_ids'].tolist()==full['input_ids'],
             'Reconstructed global teacher prefix differs from native feature layout')
        position=torch.tensor(full['position_ids'],dtype=torch.long,device=states.device)
        need(position.shape==(3,1,item['input_ids'].numel()),'Require native unpadded 3-axis global positions')
        rows.append(item);positions.append(position[:,0])
    length=max(row['hidden_states'].shape[0] for row in rows);width=global_states.shape[-1]
    hidden=states.new_zeros((len(rows),length,width))
    ids=torch.full((len(rows),length),feature_plan['pad_token_id'],dtype=torch.long,device=states.device)
    mask=torch.zeros_like(ids);position=torch.ones((4,len(rows),length),dtype=torch.long,device=states.device)
    queries=[]
    for i,(row,axes) in enumerate(zip(rows,positions)):
        size=row['input_ids'].numel();pad=length-size
        hidden[i,pad:]=row['hidden_states'];ids[i,pad:]=row['input_ids'];mask[i,pad:]=1
        position[0,i,pad:]=torch.arange(size,device=states.device);position[1:,i,pad:]=axes
        indices=[pad+p for p in row['query_indices']];queries.append(indices)
        a,b=layout['offsets'][i:i+2]
        need(torch.equal(hidden[i,indices],global_states[a:b]),'Packed query states differ from the common frozen reads')
    return local,global_states,layout,dict(hidden_states=hidden,input_ids=ids,attention_mask=mask,
                                          position_ids=position,query_indices=queries)


def forward_objectives(torch,branch,local,g,layout,replay_inputs,last_block,norm,head,rotary,placement,eps=1e-6):
    from scripts.native_vision_v11_last_block import replay_last_block
    from gnnformer.aggregation_path_bound import native_path_aggregate
    from gnnformer.paired_sequence_objectives import sequence_objectives
    delta=branch(local,g,output_dtype=torch.float32)
    z=native_path_aggregate(branch,local,g)
    output=replay_last_block(last_block,norm,head,rotary,**replay_inputs,deltas=delta,placement=placement)
    result=sequence_objectives(output['logits'],delta,z,branch.aggregate_projection.weight,
                               branch.up.weight,g,layout,eps=eps)
    result.update(total=result['ce']+result['residual'],delta=delta,replay=output)
    need(all(bool(torch.isfinite(result[k])) for k in ('total','ce','residual','path')),'Nonfinite V11 loss')
    return result

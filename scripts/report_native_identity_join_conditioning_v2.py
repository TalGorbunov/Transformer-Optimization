"""Independent CPU audit of two fixed local-conditioning, cached-head fits.

This reports the fixed first-query resource screen, never native whole answers.
All functional recomputation is independent of the conditioning hook/core forward.
"""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from collections import Counter,defaultdict
import math
import os
import random
import subprocess
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha

OWN=('scripts/report_native_identity_join_conditioning_v2.py','slurm/native_identity_join_conditioning_v2_report.sbatch')
DIAGNOSTIC_STEPS=(1,2,32,128,300,600)
CORE_ATOL=1e-4
CORE_RTOL=1e-4


def driver():
    from scripts import diagnose_native_identity_join_conditioning_v2 as module
    return module


def bind(file,digest):
    need(sha(Path(file))==digest,'Changed bound file: '+str(file))


def close(a,b,atol=2e-6,rtol=2e-6):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=atol+rtol*max(abs(a),abs(b))


def expected_order(pairs):
    need(len(pairs)==54 and len({p['pair_id'] for p in pairs})==54,'Pair inventory differs')
    rng=random.Random(24);result=[];cycle=0
    while len(result)<4800:
        cycle+=1;slots=list(range(54));rng.shuffle(slots)
        result.extend(dict(cycle=cycle,pair_slot=i,pair_id=pairs[i]['pair_id'],sids=pairs[i]['sids']) for i in slots)
    return result[:4800]


def layout_for(sequences,paired=True):
    sequences=[list(x) for x in sequences];offsets=[0]
    need(sequences and all(x and all(type(t) is int and t>=0 for t in x) for x in sequences),'Invalid target sequences')
    for ids in sequences:offsets.append(offsets[-1]+len(ids))
    result=dict(target_sequences=sequences,targets=[t for ids in sequences for t in ids],offsets=offsets,
        scene_weights=[1/(len(sequences)*len(ids)) for ids in sequences for _ in ids],
        prefixes=[[ids[:j] for j in range(len(ids))] for ids in sequences])
    if paired:
        need(len(sequences)%2==0 and all(sequences[i]==sequences[i+1] for i in range(0,len(sequences),2)),'Paired targets differ')
        result.update(left=[],right=[],pair_ids=[],pair_weights=[])
        for pair,i in enumerate(range(0,len(sequences),2)):
            length=len(sequences[i]);result['left'].extend(range(offsets[i],offsets[i+1]));result['right'].extend(range(offsets[i+1],offsets[i+2]))
            result['pair_ids'].extend([pair]*length);result['pair_weights'].extend([1/(len(sequences)//2*length)]*length)
    return result


def first_token_criterion(rows):
    need(len(rows)==108 and len({r['sid'] for r in rows})==108,'First-token denominator differs')
    families=defaultdict(list)
    for row in rows:
        need(type(row['first_token_correct']) is bool,'Correctness must be Boolean');families[row['contrast_id']].append(row)
    need(len(families)==18 and all(len(v)==6 and {(r['variant'],r['n_frames']) for r in v}==
        {(i,n) for i in range(3) for n in (8,16)} for v in families.values()),'Complete-family coverage differs')
    correct=sum(r['first_token_correct'] for r in rows);complete=sum(all(r['first_token_correct'] for r in v) for v in families.values())
    return dict(passed=correct>=103 and complete>=16,correct=correct,contexts=108,complete_families=complete,families=18,
        thresholds=dict(correct=103,contexts=108,complete_families=16,families=18),first_token_only=True,cached_first_query_only=True)


def fp64_nll(torch,logits,targets):
    need(logits.ndim==2 and len(targets)==len(logits) and bool(torch.isfinite(logits).all()),'Invalid NLL rows')
    z=logits.double();z=z-z.amax(-1,keepdim=True)
    return torch.logsumexp(z,-1)-z[torch.arange(len(z)),torch.as_tensor(targets,dtype=torch.long)]


def rms(torch,x):
    x=x.float();return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)


def independent_statistics(torch,x,indices,count):
    need(x.dtype==torch.float32 and indices.dtype==torch.int64 and indices.shape==(len(x),),'Statistics input types differ')
    need(set(indices.tolist())==set(range(count)) and bool(torch.isfinite(x).all()),'Statistics question coverage differs')
    means=torch.stack([x[indices==i].double().mean(0) for i in range(count)])
    global_mean=x.double().mean(0)
    scale=((x.double()-global_mean).square().mean()+1e-6).sqrt()
    return means,means.float(),global_mean,global_mean.float(),scale,scale.float()


def functional_core(torch,h,g,valid,means,scale,w):
    """No module/hook call; global query uses original RMS, local uses fixed affine transform."""
    f=torch.nn.functional;original=rms(torch,h);conditioned=(original-means.unsqueeze(0))/scale
    q=f.linear(rms(torch,g),w['query.weight'])
    payload=torch.tanh(f.linear(conditioned,w['local.weight'])+q.unsqueeze(0)+w['local_bias'])
    scores=f.linear(payload,w['selection_weight'].unsqueeze(0)).squeeze(-1)+w['selection_bias']
    gates=torch.sigmoid(4*(scores-.5))*valid.float();messages=gates.unsqueeze(-1)*payload
    aggregate=messages.sum(0);preactivation=f.linear(aggregate,w['aggregate_projection.weight'],w['aggregate_projection.bias'])+q
    delta=f.linear(f.silu(preactivation),w['up.weight'])
    return dict(original_local_rms_input=original,conditioned_local_rms_input=conditioned,query=q,payload=payload,
        scores=scores,gates=gates,messages=messages,aggregate=aggregate,preactivation=preactivation,delta=delta)


def tensor_error(torch,a,b,*,atol=CORE_ATOL,rtol=CORE_RTOL):
    need(a.shape==b.shape and a.dtype==b.dtype and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'Capture shape/dtype/finite check failed')
    error=(a.double()-b.double()).abs();bound=atol+rtol*b.double().abs()
    return dict(max_absolute=float(error.max()) if error.numel() else 0.,rms=float(error.square().mean().sqrt()) if error.numel() else 0.,
        passed=bool((error<=bound).all()),atol=atol,rtol=rtol)


def replay_metric(torch,reference,replay):
    need(reference.shape==replay.shape and reference.ndim==2 and reference.dtype==replay.dtype==torch.float16
         and bool(torch.isfinite(reference).all()) and bool(torch.isfinite(replay).all()),'Invalid native replay shape/dtype')
    a=reference.float();b=replay.float();tv=.5*(a.softmax(-1)-b.softmax(-1)).abs().sum(-1);same=a.argmax(-1)==b.argmax(-1)
    return [dict(full_vocabulary_tv=float(t),argmax_exact=bool(m),passed=float(t)<=.02 and bool(m)) for t,m in zip(tv,same)]


def gather(torch,scenes,states,index,sids,first_only=False):
    sequences=[[scenes[sid]['target_ids'][0]] if first_only else scenes[sid]['target_ids'] for sid in sids]
    layout=layout_for(sequences,paired=not first_only);q=layout['offsets'][-1];n=max(scenes[sid]['n_frames'] for sid in sids)
    h=torch.zeros((n,q,states.shape[-1]),dtype=states.dtype);g=torch.empty((q,states.shape[-1]),dtype=states.dtype)
    valid=torch.zeros((n,q),dtype=torch.bool)
    for sid,a,b in zip(sids,layout['offsets'],layout['offsets'][1:]):
        scene=scenes[sid];nlocal=scene['n_frames'];t=b-a
        need(len(scene['local_feature_ids'])==nlocal and all(len(ids)==len(scene['target_ids']) for ids in scene['local_feature_ids']), 'Feature/prefix shape differs')
        ids=[[index[fid] for fid in row[:t]] for row in scene['local_feature_ids']]
        h[:nlocal,a:b]=states[torch.tensor(ids)];g[a:b]=states[torch.tensor([index[fid] for fid in scene['global_feature_ids'][:t]])]
        valid[:nlocal,a:b]=True
    return h,g,valid,layout


def audit_inputs(torch,plan):
    p=driver();rows=read(plan['rows_file']);scenes=read(plan['scenes_file']);pairs=read(plan['pairs_file']);order=read(plan['order_file'])
    parent=read(plan['uniform_plan']['file']);bind(plan['uniform_plan']['file'],plan['uniform_plan']['sha256'])
    originals=read(parent['rows_file'])['train'];questions=sorted({r['sample']['question'] for r in originals});expected=[]
    for item in originals:
        s=item['sample'];need(s['split']=='train' and item['cell']=='train_N'+str(s['n_frames']),'Nontraining scene')
        expected.append({**{k:s[k] for k in ('sid','contrast_id','pair_id','variant','n_frames','question','gold','target_ids')},
            'first_token_id':s['target_ids'][0],'question_index':questions.index(s['question'])})
    need(rows==expected and scenes==read(parent['scenes_file']) and pairs==read(parent['pairs_file'])
         and order==read(parent['order_file'])==expected_order(pairs),'Original inputs/order changed')
    need(len(rows)==len(scenes)==108 and len(questions)==6 and Counter(r['gold'] for r in rows)==Counter({name:12 for name in p.oracle.PEOPLE}), 'Balanced training subset differs')
    for r in rows:
        s=scenes[r['sid']];ids=r['target_ids']
        need(s['target_ids']==ids and ids[-1]==151645 and len(ids)==(3 if r['gold'] in ('Sandra','Noah') else 2)
             and s['target_prefixes']==[ids[:i] for i in range(len(ids))] and s['question']==r['question'],'Full name/EOS or strict prefix differs')
    packet=torch.load(plan['features_file'],map_location='cpu',weights_only=True);states=packet['states'];ids=packet['feature_ids']
    need(states.dtype==torch.float16 and states.shape==(2144,3584) and len(set(ids))==len(ids)==2144
         and p.tensor_info(states)==plan['feature_tensor'] and bool(torch.isfinite(states).all()),'Cached feature packet differs')
    need(plan['features_sha256']==parent['features_sha256'] and plan['initial_sha256']==parent['initial_sha256'], 'Original feature/initial bytes differ')
    index={fid:i for i,fid in enumerate(ids)}
    for pair in pairs:
        a,b=(scenes[sid] for sid in pair['sids'])
        need([a['n_frames'],b['n_frames']]==[8,16] and a['target_ids']==b['target_ids'] and a['question']==b['question'],'Paired scene mismatch')
        need(all(torch.equal(states[index[x]],states[index[y]]) for x,y in zip(a['global_feature_ids'],b['global_feature_ids'])),'Paired global states differ')
    stats=torch.load(plan['stats_file'],map_location='cpu',weights_only=True)
    occurrences=[dict(sid=r['sid'],local_index=i,feature_id=fids[0],question_index=r['question_index']) for r in rows
                 for i,fids in enumerate(scenes[r['sid']]['local_feature_ids'])]
    need(stats['schema_version']==1 and stats['questions']==questions and stats['occurrences']==occurrences
         and len(occurrences)==1296 and Counter(x['question_index'] for x in occurrences)==Counter({i:216 for i in range(6)}), 'Statistic occurrence ownership differs')
    need(all(set(x)=={'sid','local_index','feature_id','question_index'} for x in stats['occurrences']), 'Labels entered statistic packet')
    x=rms(torch,states[torch.tensor([index[o['feature_id']] for o in occurrences])]);qi=torch.tensor([o['question_index'] for o in occurrences])
    need(torch.equal(stats['x'],x) and torch.equal(stats['question_indices'],qi),'Original cached RMS statistic input differs')
    reference=independent_statistics(torch,x,qi,6)
    for key,value in zip(('means_fp64','means','global_mean_fp64','global_mean','scale_fp64','scale'),reference):
        need(torch.equal(stats[key],value),'Frozen statistic arithmetic differs: '+key)
    need({k:p.tensor_info(stats[k]) for k in ('x','question_indices','means_fp64','means','global_mean_fp64','global_mean','scale_fp64','scale')}==plan['stats_tensors'], 'Statistic tensor identities differ')
    need(stats['scale'].ndim==0 and float(stats['scale'])>0,'Invalid fixed pooled scale')
    return rows,scenes,pairs,order,states,index,stats


def audit_training(directory,summary,scenes,order,denominators):
    bind(summary['training_file'],summary['training_sha256']);logs=read(summary['training_file']);need(len(logs)==600,'Update coverage differs')
    total=0;maximum=0;expected_calls=[]
    for step,row in enumerate(logs,1):
        batch=order[(step-1)*8:step*8];sids=[sid for pair in batch for sid in pair['sids']]
        sequences=[scenes[sid]['target_ids'] for sid in sids];layout=layout_for(sequences);lengths=list(map(len,sequences));offsets=layout['offsets']
        rate=.001*step/50 if step<=50 else .00001+(.001-.00001)*(1+math.cos(math.pi*(step-50)/550))/2
        need(row['step']==step and row['sids']==sids and row['pair_ids']==[r['pair_id'] for r in batch]
             and row['cycles']==[r['cycle'] for r in batch] and row['target_ids']==layout['targets']
             and row['prefix_ids']==[prefix for group in layout['prefixes'] for prefix in group]
             and close(row['lr'],rate,atol=1e-12,rtol=1e-12),'Training schedule or target ownership differs')
        need(row['scene_lengths']==lengths and row['scene_offsets']==offsets and row['pair_lengths']==lengths[::2], 'Scene/prefix segmentation differs')
        ce=row['position_losses']['ce'];reg=row['position_losses']['consistency']
        need(len(ce)==len(layout['targets']) and len(reg)==len(layout['left']) and all(math.isfinite(x) and x>=0 for x in ce+reg), 'Position losses invalid')
        means=[sum(ce[a:b])/(b-a) for a,b in zip(offsets,offsets[1:])];regularizers=[];cursor=0
        for length in lengths[::2]:regularizers.append(sum(reg[cursor:cursor+length])/length);cursor+=length
        need(len(row['per_scene_ce'])==16 and len(row['per_pair_consistency'])==8
             and all(close(a,b) for a,b in zip(means,row['per_scene_ce'])) and all(close(a,b) for a,b in zip(regularizers,row['per_pair_consistency']))
             and close(sum(means)/16,row['ce_loss']) and close(sum(regularizers)/8,row['consistency_loss'])
             and row['consistency_coefficient']==0. and row['weighted_consistency_loss']==0. and close(row['loss'],row['ce_loss']), 'Full-name mean-scene CE objective differs')
        den=[denominators[fid] for sid in sids[::2] for fid in scenes[sid]['global_feature_ids']]
        need(len(den)==len(reg)==len(row['pair_position_denominator'])==len(row['residual_difference_norms'])
             and all(close(a,b,atol=1e-4,rtol=1e-6) for a,b in zip(den,row['pair_position_denominator']))
             and all(close(a*a/b,c) for a,b,c in zip(row['residual_difference_norms'],row['pair_position_denominator'],reg)), 'Frozen global denominator/residual diagnostic differs')
        valid=[scenes[sid]['n_frames'] for sid,ids in zip(sids,sequences) for _ in ids]
        need(row['valid_item_count_by_position']==valid and row['valid_gate_count']==sum(valid)
             and row['gate_sum_by_position']==[n*.5 for n in valid] and row['gate_min']==row['gate_mean']==row['gate_max']==.5
             and row['gate_zero_count']==row['gate_one_count']==0 and row['padding_messages_exact_zero'] is True
             and row['closed_message_coordinates_exact_zero'] is True and 0<=row['payload_max_abs']<=1, 'Fixed uniform gate or padding differs')
        need(all(row[k] is True for k in ('native_frozen','statistics_frozen','frozen_selector_exact'))
             and row['conditioned_input_shape']==[16,len(layout['targets']),3584]
             and math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0 and row['clipped']==(row['gradient_norm']>1)
             and math.isfinite(row['seconds']) and row['seconds']>0, 'Gradient/timing metadata invalid')
        q=len(layout['targets']);total+=q;maximum=max(maximum,q)
        for module in ('norm','head'):
            expected_calls.append(dict(module=module,phase='training',step=step,input_shape=[1,q,3584],
                output_shape=[1,q,3584 if module=='norm' else 152064],input_dtype='torch.float16',output_dtype='torch.float16'))
    need(total==21334 and maximum==44 and logs[0]['consistency_loss']==0, 'Full training call-row inventory differs')
    for batch in range(1,8):
        q=12 if batch==7 else 16
        for module in ('norm','head'):
            expected_calls.append(dict(module=module,phase='evaluation',step=batch,input_shape=[1,q,3584],
                output_shape=[1,q,3584 if module=='norm' else 152064],input_dtype='torch.float16',output_dtype='torch.float16'))
    return logs,expected_calls,dict(passed=True,updates=600,pair_presentations=4800,scene_presentations=9600,
        training_target_positions=total,maximum_training_positions_per_call=maximum,final_first_query_positions=108,total_head_rows=21442,
        terminal_ce=logs[-1]['ce_loss'],terminal_residual_diagnostic=logs[-1]['consistency_loss'],consistency_coefficient=0.,
        full_sequence_loss_reductions_verified=True,no_independent_optimizer_trajectory_replay=True)


def check_weights(torch,weights,initial):
    need(set(weights)==set(initial) and all(v.dtype==torch.float32 and v.shape==initial[k].shape and bool(torch.isfinite(v).all()) for k,v in weights.items())
         and sum(v.numel() for v in weights.values())==1041697,'Core parameter table differs')
    need(torch.equal(weights['selection_weight'],initial['selection_weight']) and bool((weights['selection_weight']==0).all())
         and torch.equal(weights['selection_bias'],initial['selection_bias']) and bool((weights['selection_bias']==.5).all()),'Fixed selector changed')


def analyze_first_captures(torch,records):
    """Descriptive saved first-query geometry, with sampling occurrences preserved."""
    groups=defaultdict(list)
    for row in records:groups[row['question']].append(row)
    return dict(scope='fixed final cached training first queries only; no native generation or causal attribution',
        scenes=len(records),questions=len(groups),saturation_derivative_threshold=.01,
        rows=records,by_question=[dict(labels=dict(question=q),metrics=dict(scenes=len(rows),
            local_occurrences=sum(x['local_items'] for x in rows),payload_coordinates=sum(x['payload_coordinates'] for x in rows),
            saturated_coordinates=sum(x['saturated_coordinates'] for x in rows),
            saturated_fraction=sum(x['saturated_coordinates'] for x in rows)/sum(x['payload_coordinates'] for x in rows),
            mean_preactivation_norm=sum(x['preactivation_norm'] for x in rows)/len(rows))) for q,rows in sorted(groups.items())])


def self_test(torch):
    rows=[dict(sid=f'{f}_{v}_{n}',contrast_id=f,variant=v,n_frames=n,first_token_correct=True)
          for f in range(18) for v in range(3) for n in (8,16)]
    need(first_token_criterion(rows)['passed'],'Complete-family positive fixture failed')
    for i in (0,6,12):rows[i]['first_token_correct']=False
    value=first_token_criterion(rows);need(value['correct']==105 and value['complete_families']==15 and not value['passed'],'Complete-family rejection failed')
    pairs=[dict(pair_id=str(i),sids=[f'{i}a',f'{i}b']) for i in range(54)];order=expected_order(pairs)
    need(len(order)==4800 and len({x['pair_slot'] for x in order[:54]})==54 and order==expected_order(pairs),'Persistent order fixture failed')
    layout=layout_for([[2,3],[2,3],[4,5,6],[4,5,6]])
    ce=[2.,4.,2.,4.,3.,6.,9.,3.,6.,9.]
    need(close(sum(a*b for a,b in zip(ce,layout['scene_weights'])),4.5)
         and not close(sum(ce)/len(ce),4.5) and layout['left']==[0,1,4,5,6] and layout['right']==[2,3,7,8,9], 'Unequal-length CE weighting failed')
    x=torch.tensor([[1.,2.],[3.,4.],[5.,4.],[7.,6.]],dtype=torch.float32);qi=torch.tensor([0,0,1,1])
    m,rounded,gm,gmr,s,sf=independent_statistics(torch,x,qi,2)
    need(torch.equal(m,torch.tensor([[2.,3.],[6.,5.]],dtype=torch.float64)) and torch.equal(gm,torch.tensor([4.,4.],dtype=torch.float64)) and abs(float(s)-math.sqrt(3.5+1e-6))<1e-12
         and rounded.dtype==sf.dtype==torch.float32,'Pooled statistic fixture failed')
    h=torch.tensor([[[.2,.7]],[[.4,-.8]]],dtype=torch.float16);g=torch.tensor([[.1,.4]],dtype=torch.float16);valid=torch.tensor([[True],[False]])
    weights={'query.weight':torch.eye(2),'local.weight':torch.eye(2),'local_bias':torch.zeros(2),
        'selection_weight':torch.zeros(2),'selection_bias':torch.tensor(.5),'aggregate_projection.weight':torch.eye(2),
        'aggregate_projection.bias':torch.zeros(2),'up.weight':torch.eye(2)}
    mean=torch.tensor([[.3,-.2]]);scale=torch.tensor(2.);result=functional_core(torch,h,g,valid,mean,scale,weights)
    need(torch.equal(result['query'],rms(torch,g)) and torch.equal(result['conditioned_local_rms_input'],(rms(torch,h)-mean)/scale)
         and torch.equal(result['gates'],torch.tensor([[.5],[0.]])) and bool((result['messages'][1]==0).all())
         and not torch.equal(result['conditioned_local_rms_input'],rms(torch,result['conditioned_local_rms_input'])), 'Local-only placement or padding fixture failed')
    logits=torch.tensor([[2.,1.,0.],[0.,1.,2.]],dtype=torch.float16)
    need(all(x['passed'] for x in replay_metric(torch,logits,logits)) and not all(x['passed'] for x in replay_metric(torch,logits,logits.flip(-1)))
         and torch.allclose(fp64_nll(torch,logits,[0,2]),fp64_nll(torch,logits+16,[0,2]),atol=1e-12,rtol=0), 'Native replay/NLL fixture failed')
    need(tensor_error(torch,result['delta'],result['delta'])['passed']
         and not tensor_error(torch,result['delta']+1,result['delta'])['passed'],'Core tolerance failure fixture failed')
    return dict(passed=True,groups=6,complete_family_threshold=True,persistent_order=True,unequal_target_weighting=True,
        pooled_unrounded_statistics=True,local_only_transform_and_padding=True,replay_and_nll=True)


def chosen_means(torch,stats,rows,sids,layout,arm):
    q=layout['offsets'][-1]
    if arm=='global':return stats['global_mean'].unsqueeze(0).expand(q,-1)
    need(arm=='question','Unknown conditioning arm')
    by_sid={r['sid']:r for r in rows}
    return stats['means'][torch.tensor([by_sid[sid]['question_index'] for sid,ids in zip(sids,layout['target_sequences']) for _ in ids])]


def audit_capture(torch,out,record,arm,plan,rows,scenes,states,index,stats,initial,final,logs,norm,head):
    p=driver();info=p.tensor_info;phase=record['phase'];step=record['step'];training=phase=='training'
    need(phase in ('training','evaluation') and record['arm']==arm,'Unknown capture phase/arm')
    bind(record['file'],record['sha256']);bind(record['weights_file'],record['weights_sha256'])
    cap=torch.load(record['file'],map_location='cpu',weights_only=True)
    weights_packet=torch.load(record['weights_file'],map_location='cpu',weights_only=True);weights=weights_packet['branch']
    check_weights(torch,weights,initial)
    if training:
        need(weights_packet['arm']==arm and step in DIAGNOSTIC_STEPS and record['weights_step']==step-1 and weights_packet['step']==step-1
             and weights_packet['optimizer_step']==step and weights_packet['position']=='before_update'
             and weights_packet['source_sha256']==plan['source_sha256'],'Pre-update capture weights differ')
        sids=logs[step-1]['sids']
        if step==1:need(all(torch.equal(weights[k],initial[k]) for k in initial),'First captured state differs from original initialization')
    else:
        need(1<=step<=7 and record['weights_step']==weights_packet['step']==600
             and all(torch.equal(weights[k],final[k]) for k in final),'Final capture does not use fixed endpoint')
        sids=[r['sid'] for r in rows[(step-1)*16:step*16]]
    need(cap['schema_version']==1 and cap['arm']==arm and cap['phase']==phase and cap['step']==step and cap['sids']==record['sids']==sids,'Capture scene/phase ownership differs')
    h,g,valid,layout=gather(torch,scenes,states,index,sids,first_only=not training)
    if training:need(cap['layout']==layout,'Training capture full-prefix layout differs')
    else:
        full=layout_for([scenes[sid]['target_ids'] for sid in sids]);expected=dict(first_query_only=True,
            target_ids=layout['targets'],full_target_ids=full['target_sequences'],first_indices=full['offsets'][:-1],original_full_layout=full)
        need(cap['layout']==expected,'Final capture original empty-prefix ownership differs')
    for key,value in (('local_states',h),('global_states',g),('valid_mask',valid)):
        need(cap[key].dtype==value.dtype and torch.equal(cap[key],value),'Actual captured frozen state differs: '+key)
    means=chosen_means(torch,stats,rows,sids,layout,arm)
    need(cap['question_means'].dtype==torch.float32 and torch.equal(cap['question_means'],means),'Fixed arm mean was not broadcast at every prefix')
    observed=cap['capture'];expected=functional_core(torch,h,g,valid,means,stats['scale'],weights)
    need(set(observed)==set(expected),'Core capture coverage differs')
    metrics={k:tensor_error(torch,observed[k],expected[k]) for k in expected}
    need(observed['delta'].dtype==torch.float32 and cap['fused_global'].dtype==cap['normalized'].dtype==cap['logits'].dtype==torch.float16
         and cap['fused_global'].shape==cap['normalized'].shape==g.shape and cap['logits'].shape==(len(g),152064)
         and all(bool(torch.isfinite(cap[k]).all()) for k in ('fused_global','normalized','logits')), 'Actual native readout dtype/shape differs')
    need(torch.equal(g+observed['delta'].half(),cap['fused_global']), 'Actual FP16 cast-before-add differs')
    need(torch.equal(observed['gates'],valid.float()*.5) and bool((observed['scores']==.5).all())
         and torch.equal(observed['messages'],observed['payload']*observed['gates'].unsqueeze(-1))
         and bool((observed['messages'][~valid]==0).all()) and bool((observed['payload'].abs()<=1).all()), 'Native fixed gate/message arithmetic differs')
    need({k:info(cap[k]) for k in ('fused_global','normalized','logits')}==record['tensors'],'Captured native tensor identities differ')
    if training and step==1:need(bool((observed['delta']==0).all()) and torch.equal(cap['fused_global'],g),'Zero-U initial identity differs')
    head_metrics=[];normalized_error=None
    if not training:
        normalized=norm(cap['fused_global'].unsqueeze(0))[0];replayed=head(normalized.unsqueeze(0))[0]
        head_metrics=replay_metric(torch,cap['logits'],replayed)
        normalized_error=float((normalized.float()-cap['normalized'].float()).abs().max())
    nll=fp64_nll(torch,cap['logits'],layout['targets'])
    if training:
        logged=logs[step-1]['position_losses']['ce']
        nll_metrics=[dict(position=i,fp64=float(a),gpu_fp32=b,passed=close(float(a),b)) for i,(a,b) in enumerate(zip(nll,logged))]
    else:nll_metrics=[]
    result=dict(arm=arm,phase=phase,step=step,sids=sids,capture_file=record['file'],capture_sha256=record['sha256'],
        weights_file=record['weights_file'],weights_sha256=record['weights_sha256'],functional_core=metrics,
        native_head=head_metrics,normalized_max_absolute=normalized_error,
        nll=nll_metrics,fp16_cast_before_add_exact=True,captured_rows=len(g),head_rows=0 if training else len(g),
        cpu_norm_calls=0 if training else 1,cpu_head_calls=0 if training else 1,
        passed=all(x['passed'] for x in metrics.values()) and all(x['passed'] for x in head_metrics+nll_metrics))
    save(out/f'{arm}_{phase}_{step:04d}_capture_audit.json',result)
    need(result['passed'],'Captured functional core/native head/NLL replay failed; metrics and raw evidence retained')
    geometry=[]
    if not training:
        by_sid={r['sid']:r for r in rows}
        for j,sid in enumerate(sids):
            row=by_sid[sid];payload=observed['payload'][:row['n_frames'],j].double();derivative=1-payload.square()
            geometry.append(dict(sid=sid,question=row['question'],contrast_id=row['contrast_id'],variant=row['variant'],n_frames=row['n_frames'],
                local_items=row['n_frames'],payload_coordinates=payload.numel(),saturated_coordinates=int((derivative<=.01).sum()),
                preactivation_norm=float(observed['preactivation'][j].double().norm()),
                aggregate_norm=float(observed['aggregate'][j].double().norm()),delta_norm=float(observed['delta'][j].double().norm())))
    return result,cap['logits'],geometry


def allocation_rows(raw,job_names):
    records=[]
    for line in raw.splitlines():
        if not line.strip():continue
        f=line.split('|');need(len(f)==9,'Scheduler row schema differs')
        if f[1] not in job_names:continue
        fields=[x.split('=',1) for x in f[6].split(',') if '=' in x];need(len(dict(fields))==len(fields),'Duplicate TRES field')
        values=dict(fields);typed=[int(v) for k,v in values.items() if k.startswith('gres/gpu:')]
        gpus=int(values['gres/gpu']) if 'gres/gpu' in values else sum(typed)
        need(gpus in (0,1) and (not typed or sum(typed)==gpus) and int(f[5])>=0,'Invalid GPU allocation')
        records.append(dict(job_id=f[0],name=f[1],partition=f[2],state=f[3],exit_code=f[4],seconds=int(f[5]),gpus=gpus,gpu_seconds=int(f[5])*gpus,
            start=f[7],end=f[8]))
    need(len({r['job_id'] for r in records})==len(records),'Duplicate scheduler allocation')
    return records


def resources(out,runs):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'all_user_sacct.psv';file.write_text(raw)
    job_arms={name:arm for arm,name in driver().RUN_JOBS.items()};names=set(job_arms);records=allocation_rows(raw,names)
    need(len(records)==2 and {r['name'] for r in records}==names,'Exactly one attempt per conditioning arm required; all failed/zero allocations retained')
    for row in records:
        arm=job_arms[row['name']]
        need(row['job_id']==runs[arm]['slurm_job_id'] and row['partition']=='gpu' and row['state']=='COMPLETED'
             and row['exit_code']=='0:0' and row['gpus']==1 and row['seconds']<=90,'Conditioning allocation identity/cap differs')
    from datetime import datetime
    events=[]
    for row in records:
        a,b=map(datetime.fromisoformat,(row['start'],row['end']));need(b>=a,'Invalid allocation interval')
        if b>a:events.extend(((a,1),(b,-1)))
    current=peak=0
    for _,d in sorted(events):current+=d;need(current>=0,'Invalid allocation overlap');peak=max(peak,current)
    total=sum(r['gpu_seconds'] for r in records);need(current==0 and peak<=2 and total<=180,'Conditioning campaign cap exceeded')
    return dict(passed=True,jobs=records,allocated_gpu_seconds=total,maximum_concurrent_gpus=peak,
        failed_and_zero_allocations_retained=True,file=str(file),sha256=sha(file),command=command)


def audit_run(torch,out,directory,plan_path,plan,frozen,rows,scenes,order,states,index,stats,initial,norm,head):
    p=driver();directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json');arm=config['arm']
    need(arm in ('global','question') and directory.parent==p.OUT and directory.name==config['run_id']==f'run_{arm}_{config["slurm_job_id"]}'
         and config['protocol']==p.PROTOCOL and config['policy']==p.POLICY and config['seed']==24
         and config['source_sha256']==frozen==plan['source_sha256'] and all(summary.get(k)==v for k,v in config.items())
         and Path(config['plan_file']).resolve()==Path(plan_path).resolve() and config['plan_sha256']==sha(plan_path),'Run/config/source/plan join differs')
    for name,h in frozen.items():bind(directory/'source'/name.replace('/','_'),h)
    for flag in ('passed','completed','computational_integrity_passed','checkpoint_roundtrip_passed','native_weights_unchanged',
                 'statistics_frozen','cached_first_query_only','no_pretrained_backbone_loaded','no_dev_or_test','no_native_or_whole_answer_claim'):
        need(summary[flag] is True,'Run gate failed: '+flag)
    need(summary['phase']=='run' and summary['steps']==600 and summary['training_head_rows']==21334
         and summary['final_head_rows']==108 and summary['total_head_rows']==21442,'Run endpoint/call rows differ')
    need(config['hardware']['gpu']=='NVIDIA B200' and config['hardware']['matmul_allow_tf32'] is False
         and config['hardware']['torch_version']==str(torch.__version__),'Native hardware/precision source differs')
    for key in ('native_identity','native_identity_sha256','native_module_identity','native_model_file','native_model_sha256',
                'stats_file','stats_sha256','stats_tensors','oracle_report','uniform_plan'):
        need(config[key]==plan[key],'Fixed plan/native/statistics binding differs: '+key)
    need(config['initialized']==plan['initial_state'] and config['initial_checkpoint_sha256']==plan['initial_sha256'],'Shared original initialization differs')
    bind(config['initial_checkpoint'],config['initial_checkpoint_sha256'])
    actual_initial=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=True)
    need(all(torch.equal(actual_initial['branch'][k],initial[k]) for k in initial),'Run actual initial tensors differ')
    mask=plan['trainability_mask'];trainable=[k for k in initial if k not in ('selection_weight','selection_bias')]
    need(mask==dict(trainable_parameter_names=trainable,frozen_selector_parameter_names=['selection_weight','selection_bias'],
        effective_trainable_parameters=1041600,retained_state_parameters=1041697)
         and all(config[k]==v for k,v in mask.items()),'Optimizer selector exclusion/parameter counts differ')
    for key in ('training','captures','raw','predictions','calls','endpoint','roundtrip'):bind(summary[key+'_file'],summary[key+'_sha256'])
    for key in ('rows','presentations','stats_inventory'):bind(config[key+'_file'],config[key+'_sha256'])
    need(read(config['rows_file'])==rows and read(config['presentations_file'])==order and config['order_sha256']==object_sha(order)
         and read(config['stats_inventory_file'])==read(plan['stats_inventory_file']), 'Copied input/inventory/order changed')
    checkpoint_file=Path(summary['checkpoint']);bind(checkpoint_file,summary['checkpoint_sha256'])
    need(checkpoint_file==p.CKPT/directory.name/'final.pt' and Path(config['checkpoint_directory'])==checkpoint_file.parent
         and Path(config['initial_checkpoint'])==checkpoint_file.parent/'initial.pt' and Path(config['data_directory'])==p.DATA/directory.name, 'Authorized data/checkpoint storage differs')
    packet=torch.load(checkpoint_file,map_location='cpu',weights_only=True)
    need(set(packet)=={'branch','step','config'} and packet['step']==600 and packet['config']==config,'Final-only checkpoint differs')
    final=packet['branch'];check_weights(torch,final,initial);table={k:p.tensor_info(v) for k,v in final.items()}
    endpoint=read(summary['endpoint_file']);roundtrip=read(summary['roundtrip_file'])
    native_table=dict(norm=plan['native_identity']['norm_weight'],head=plan['native_identity']['head_weight'])
    need(endpoint['before_evaluation']==endpoint['after_evaluation']==table and endpoint['parameter_sha256']==summary['selected_parameter_sha256']==object_sha(table)
         and endpoint['native_before']==endpoint['native_after']==config['native_weight_identity']==native_table
         and endpoint['native_weights_unchanged'] is True and endpoint['passed'] is True and endpoint['hooks_removed'] is True,'Final deployed/frozen endpoint differs')
    need(roundtrip['passed'] is True and roundtrip['initial']==plan['initial_state'] and roundtrip['final']==roundtrip['reloaded']==table
         and roundtrip['step']==600 and roundtrip['config_sha256']==sha(directory/'config.json'),'Actual reset/restricted reload differs')
    for value in (endpoint,roundtrip):
        need(value['checkpoint']==str(checkpoint_file) and value['checkpoint_sha256']==summary['checkpoint_sha256'],'Endpoint checkpoint binding differs')
    need(all(endpoint[k]==plan[k] for k in ('stats_file','stats_sha256','stats_tensors'))
         and endpoint['working_statistics_before']==endpoint['working_statistics_after']=={k:plan['stats_tensors'][k] for k in ('means','global_mean','scale')}
         and endpoint['statistics_versions_unchanged'] is True,'Actual deployed fixed statistics changed')
    denominators={fid:float(states[index[fid]].float().square().sum()+1e-6) for scene in scenes.values() for fid in scene['global_feature_ids']}
    logs,expected_calls,training=audit_training(directory,summary,scenes,order,denominators)
    need([len(r['target_ids']) for r in logs]==plan['head_rows_by_update'] and close(summary['training_seconds'],sum(r['seconds'] for r in logs)), 'Training row/timing totals differ')
    counters=dict(core=607,conditioning=607,norm=607,head=607,vlm=0,vision=0);calls=read(summary['calls_file'])
    need(calls==expected_calls and summary['counters']==endpoint['counters']==counters
         and read(directory/'final_counters.json')==dict(counters=counters,recorded_calls=calls,partial_outputs_retained=True), 'Actual work inventory differs')
    records=read(summary['captures_file']);expected_positions=[('training',step) for step in DIAGNOSTIC_STEPS]+[('evaluation',step) for step in range(1,8)]
    need([(r['phase'],r['step']) for r in records]==expected_positions and all(r['arm']==arm for r in records), 'Captured batch coverage/arm differs')
    need(sorted(f.name for f in checkpoint_file.parent.glob('*.pt'))==sorted(['initial.pt','final.pt']+[f'training_core_{step:04d}.pt' for step in DIAGNOSTIC_STEPS]),'Checkpoint inventory differs')
    need(sorted(f.name for f in directory.glob('progress_*.json'))==[f'progress_{step:04d}.json' for step in range(100,601,100)],'Immutable progress coverage differs')
    for step in range(100,601,100):
        need(read(directory/f'progress_{step:04d}.json')==dict(step=step,training=logs[:step],captures=[r for r in records if r['phase']=='training' and r['step']<=step],
            counters=dict(core=step,conditioning=step,norm=step,head=step,vlm=0,vision=0)),'Immutable progress prefix changed')
    raw=torch.load(summary['raw_file'],map_location='cpu',weights_only=True);logits=raw['logits']
    need(raw['schema_version']==1 and raw['sids']==[r['sid'] for r in rows] and logits.dtype==torch.float16 and logits.shape==(108,152064)
         and bool(torch.isfinite(logits).all()) and p.tensor_info(logits)==summary['raw_tensor']
         and raw['batches']==summary['raw_batches'] and len(raw['batches'])==7,'Final full-vocabulary raw output inventory differs')
    need(Path(summary['raw_file'])==Path(config['data_directory'])/'final_logits.pt','Raw storage root differs')
    capture_audits=[];geometry=[];gpu_nll=[];cursor=0
    with torch.no_grad():
        for record in records:
            phase=record['phase'];step=record['step'];need(record['arm']==arm,'Capture arm differs')
            expected_weight=checkpoint_file.parent/f'training_core_{step:04d}.pt' if phase=='training' else checkpoint_file
            need(Path(record['weights_file'])==expected_weight and Path(record['file'])==Path(config['data_directory'])/f'{phase}_capture_{step:04d}.pt','Capture/checkpoint path differs')
            audit,captured_logits,scene_geometry=audit_capture(torch,out,record,arm,plan,rows,scenes,states,index,stats,initial,final,logs,norm,head)
            capture_audits.append(audit);geometry+=scene_geometry
            if phase=='evaluation':
                count=len(record['sids']);batch=raw['batches'][step-1]
                need(batch['capture_file']==record['file'] and batch['capture_sha256']==record['sha256'] and batch['sids']==record['sids']
                     and batch['logits']==record['tensors']['logits'] and len(batch['gpu_fp32_nll'])==count
                     and torch.equal(captured_logits,logits[cursor:cursor+count]),'Final batch raw/capture ownership differs')
                gpu_nll+=batch['gpu_fp32_nll'];cursor+=count
    need(cursor==108 and len(geometry)==108,'Final captured row coverage differs')
    ids=logits.argmax(-1).tolist();nll=fp64_nll(torch,logits,[r['first_token_id'] for r in rows]);saved=read(summary['predictions_file'])
    need(len(saved)==108,'Prediction denominator differs')
    outcomes=[dict(r,argmax_id=ids[i],first_token_correct=ids[i]==r['first_token_id'],nll=float(nll[i]),gpu_fp32_nll=gpu_nll[i]) for i,r in enumerate(rows)]
    precision=[dict(sid=r['sid'],fp64_nll=r['nll'],gpu_fp32_nll=r['gpu_fp32_nll'],absolute_difference=abs(r['nll']-r['gpu_fp32_nll']),
        passed=close(r['nll'],r['gpu_fp32_nll'])) for r in outcomes]
    save(out/f'{arm}_outcomes.json',outcomes);save(out/f'{arm}_nll_precision.json',precision)
    need(all(set(a)==set(b) and all(a[k]==b[k] for k in a if k!='nll') and close(a['nll'],b['nll']) for a,b in zip(outcomes,saved))
         and all(r['passed'] for r in precision),'Independent final argmax/NLL/metadata rescore differs')
    criterion=first_token_criterion(outcomes);need(criterion==summary['first_token_fit'],'Fixed cached first-token screen differs')
    descriptive=analyze_first_captures(torch,geometry);save(out/f'{arm}_first_query_geometry.json',descriptive)
    strata=[]
    for key in ('gold','question','n_frames','contrast_id'):
        groups=defaultdict(list)
        for row in outcomes:groups[row[key]].append(row)
        for label,group in sorted(groups.items(),key=lambda x:str(x[0])):
            strata.append(dict(labels={key:label},metrics=dict(correct=sum(r['first_token_correct'] for r in group),contexts=len(group),
                mean_nll=sum(r['nll'] for r in group)/len(group),all_correct=all(r['first_token_correct'] for r in group))))
    save(out/f'{arm}_strata.json',strata)
    return dict(passed=True,completed=True,arm=arm,slurm_job_id=config['slurm_job_id'],run_directory=str(directory),run_summary_sha256=sha(directory/'summary.json'),
        checkpoint=str(checkpoint_file),checkpoint_sha256=summary['checkpoint_sha256'],selected_parameter_sha256=object_sha(table),
        first_token_fit=criterion,training=training,counters=counters,capture_audits=capture_audits,
        cpu_head_calls=7,cpu_norm_calls=7,cpu_head_rows=sum(r['head_rows'] for r in capture_audits),
        maximum_native_replay_tv=max(m['full_vocabulary_tv'] for r in capture_audits for m in r['native_head']),
        outcomes_file=str(out/f'{arm}_outcomes.json'),outcomes_sha256=sha(out/f'{arm}_outcomes.json'),
        nll_precision_file=str(out/f'{arm}_nll_precision.json'),nll_precision_sha256=sha(out/f'{arm}_nll_precision.json'),
        geometry_file=str(out/f'{arm}_first_query_geometry.json'),geometry_sha256=sha(out/f'{arm}_first_query_geometry.json'),
        strata_file=str(out/f'{arm}_strata.json'),strata_sha256=sha(out/f'{arm}_strata.json'))


def report(args,out,frozen):
    import torch
    torch.set_num_threads(4);p=driver();plan=p.verify_plan(args.plan,ancestors=True)
    _,trigger=p.conditional_gate({});need(trigger==plan['oracle_report'],'Conditioning trigger changed')
    need(frozen==plan['source_sha256']==p.sources() and len(args.runs)==2,'Source or two-arm report inventory differs')
    rows,scenes,pairs,order,states,index,stats=audit_inputs(torch,plan)
    inventory=read(plan['stats_inventory_file'])
    need(inventory==dict(questions=stats['questions'],occurrences=stats['occurrences'],training_sids=[r['sid'] for r in rows],
        original_cached_empty_prefix_only=True,no_fitted_captures=True,no_local_labels=True),'Statistic inventory provenance differs')
    need(p.oracle.native_module_identity(torch,plan['native_identity'])==plan['native_module_identity'],'Actual installed native normalization/head source differs')
    initial_packet=torch.load(plan['initial_file'],map_location='cpu',weights_only=True);initial=initial_packet['branch']
    need(initial_packet['seed']==24 and {k:p.tensor_info(v) for k,v in initial.items()}==plan['initial_state']
         and bool((initial['up.weight']==0).all()),'Original initialization tensor table differs')
    check_weights(torch,initial,initial)
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);unfitted=ParallelLocalLearnedSelection(mode='sigmoid')
    need(p.base.v7.state_info(unfitted)==plan['initial_state'],'Independent seeded initialization differs');del unfitted
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=p.oracle.native_modules(torch,packet,plan['native_identity'],'cpu');del packet
    native_before=dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight))
    runs={}
    for directory in args.runs:
        value=audit_run(torch,out,directory,args.plan,plan,frozen,rows,scenes,order,states,index,stats,initial,norm,head)
        need(value['arm'] not in runs,'Duplicate conditioning arm');runs[value['arm']]=value
    need(set(runs)=={'global','question'} and native_before==dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight))
         and norm.weight.grad is head.weight.grad is None and not norm.weight.requires_grad and not head.weight.requires_grad,'Two-arm/native frozen audit incomplete')
    budget=resources(out,runs);outcomes={arm:read(runs[arm]['outcomes_file']) for arm in runs}
    transitions=Counter((a['first_token_correct'],b['first_token_correct']) for a,b in zip(outcomes['global'],outcomes['question']))
    comparison=dict(scope='paired cached training first queries, descriptive; shared original initialization/order/global pooled scale',
        contexts=108,global_to_question_transitions={f'{int(a)}_to_{int(b)}':transitions[a,b] for a in (False,True) for b in (False,True)},
        question_minus_global_correct=runs['question']['first_token_fit']['correct']-runs['global']['first_token_fit']['correct'],
        prior_uniform_first_token_correct=36,prior_uniform_contexts=108,no_new_baseline_fit=True)
    result=dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),oracle_report=trigger,uniform_plan=plan['uniform_plan'],
        stats_file=plan['stats_file'],stats_sha256=plan['stats_sha256'],stats_tensors=plan['stats_tensors'],runs=runs,comparison=comparison,
        resources=budget,first_token_screen={arm:runs[arm]['first_token_fit'] for arm in runs},
        cpu_head_calls=14,cpu_norm_calls=14,cpu_head_rows=sum(r['cpu_head_rows'] for r in runs.values()),
        cached_first_query_only=True,no_native_or_whole_answer_claim=True,no_generalization_or_reasoning_claim=True,
        no_new_fit_or_model_forward=True,all_outcomes_retained=True)
    save(out/'analysis.json',result)
    lines=['# Fixed local-conditioning diagnostic','',
        'Independent computation and provenance audits passed. These are cached training first tokens, not native whole answers.','',
        '| Mean | Correct /108 | Complete families /18 | Fixed screen |','|---|---:|---:|---|']
    for arm in ('global','question'):
        c=runs[arm]['first_token_fit'];lines.append(f"| {arm} | {c['correct']} | {c['complete_families']} | {'PASS' if c['passed'] else 'FAIL'} |")
    lines+=['','Both arms used the same original initialization,600 full-name-plus-EOS CE updates and global pooled scale. '
        'Each retained607 actual native norm/head batch calls and21442 rows; zero VLM/vision calls. '
        'Only the mean varies between arms. Centering and scale relative to the old uniform model remain a combined intervention.','',
        'A screen pass permits preparation of a separately audited native evaluation of this unchanged endpoint. '
        'It does not establish name completion, EOS behavior, generalization or reasoning composition. '
        'The question arm retains a fixed six-question diagnostic lookup; the global arm does not.','',
        f"Allocated GPU cost: {budget['allocated_gpu_seconds']} seconds. CPU replay:14 final norm/head batches, with fixed TV<=.02 and exact argmax.\n"]
    (out/'REPORT.md').write_text('\n'.join(lines))
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),first_token_screen=result['first_token_screen'],
        resources=budget,cached_first_query_only=True,no_native_or_whole_answer_claim=True)


def main():
    import argparse,time
    p=driver();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs=2,required=True);args=parser.parse_args()
    p.native.require_slurm(gpu=False)
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only independent report')
    out=p.OUT/f'report_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    frozen=p.snapshot(out);save(out/'request.json',dict(plan=str(args.plan),runs=list(map(str,args.runs)),source_sha256=frozen))
    try:
        result=report(args,out,frozen);need(p.sources()==frozen,'Source changed during report')
        result['elapsed_seconds']=time.perf_counter()-started;save(out/'summary.json',result)
        print(__import__('json').dumps(dict(passed=True,completed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

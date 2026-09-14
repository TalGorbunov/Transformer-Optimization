"""Independent fixed-budget orientation-coverage training audit.

Audits the full216-scene bank, alternating original base-pair schedule and
only the fixed6000 endpoint. Native whole-answer inference is a separate gate.
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
from scripts import report_native_identity_join_factor_binding as old
OWN=('scripts/report_native_identity_join_factor_orientation_training.py',)
ARMS=('product','additive')
DIAGNOSTIC_STEPS=(1,2,32,128,300,600,2000,6000)
ENDPOINTS=(6000,)
PROGRESS=(600,2000,4000,6000)
FROZEN_NAMES=('selection_weight','selection_bias')
close=old.close
layout_for=old.layout_for
fp64_nll=old.fp64_nll
tensor_error=old.tensor_error
replay_metric=old.replay_metric
allocation_rows=old.allocation_rows
check_weights=old.check_weights
gather=old.gather
functional_core=old.functional_core
bind=old.bind


def driver():
    from scripts import diagnose_native_identity_join_factor_orientation_training as module
    return module


def expected_order(base_pairs,pairs):
    need(len(base_pairs)==54 and len({x['pair_id'] for x in base_pairs})==54
         and len(pairs)==108 and len({x['pair_id'] for x in pairs})==108,'Base/counterpart pair inventory differs')
    lookup={(r['base_pair_id'],r['orientation_version']):r for r in pairs}
    need(set(lookup)=={(r['pair_id'],o) for r in base_pairs for o in ('original','flipped')}
         and [(r['base_pair_id'],r['orientation_version']) for r in pairs]==[(r['pair_id'],o) for r in base_pairs for o in ('original','flipped')],
         'Each base pair needs two canonical-order orientations')
    rng=random.Random(24);plain=[];cycle=0
    while len(plain)<48000:
        cycle+=1;slots=list(range(54));rng.shuffle(slots)
        plain.extend(dict(cycle=cycle,pair_slot=i,pair_id=base_pairs[i]['pair_id'],sids=base_pairs[i]['sids']) for i in slots)
    seen=Counter();result=[]
    for row in plain[:48000]:
        base=row['pair_id'];orientation='original' if seen[base]%2==0 else 'flipped';seen[base]+=1
        selected=lookup[base,orientation]
        result.append(dict(row,base_pair_id=base,base_sids=row['sids'],orientation_version=orientation,base_pair_visit=seen[base],
                           pair_id=selected['pair_id'],sids=selected['sids']))
    return result


def first_token_criterion(rows):
    need(len(rows)==len({r['sid'] for r in rows})==216,'All216 unique outcomes required')
    orientations=('original','flipped');counts={}
    for o in orientations:
        selected=[r for r in rows if r['orientation_version']==o]
        need(len(selected)==108,'Each orientation requires108 outcomes')
        counts[o]=sum(bool(r['first_token_correct']) for r in selected)
    families=defaultdict(list)
    for r in rows:families[r['base_contrast_id']].append(r)
    expected={(v,n,o) for v in range(3) for n in (8,16) for o in orientations}
    need(len(families)==18 and all(len(group)==12 and {(r['variant'],r['n_frames'],r['orientation_version']) for r in group}==expected
                                 for group in families.values()),'Every family needs its twelve distinct contexts')
    complete=sum(all(r['first_token_correct'] for r in group) for group in families.values());total=sum(counts.values())
    return dict(passed=total>=206 and all(v>=103 for v in counts.values()) and complete>=16,first_correct=total,contexts=216,
        orientation_correct=counts,complete_families=complete,families=18,thresholds=dict(pooled=206,per_orientation=103,complete_families=16),
        cached_first_query_only=True,whole_answer_claim=False)


def learning_rate(step):
    need(type(step) is int and 1<=step<=6000,'Update outside fixed horizon')
    return .001*step/50 if step<=50 else .00001+(.001-.00001)*(1+math.cos(math.pi*(step-50)/5950))/2


def primary_screen(endpoint_rows):
    need(set(endpoint_rows)==set(ENDPOINTS),'Only the fixed6000 endpoint is authorized')
    return first_token_criterion(endpoint_rows[6000])


def audit_inputs(torch,plan):
    p=driver();reference=plan['factor_plan'];bind(reference['file'],reference['sha256'])
    parent=old.driver().verify_plan(reference['file'],ancestors=True)
    original_rows,original_scenes,base_pairs,_,original_states,original_index,stats=old.audit_inputs(torch,parent)
    bank=p.bridge.verify_cache(plan['cache_report']['file']);bind(bank['summary_file'],plan['cache_report']['sha256'])
    stage=p.bridge.training.verify_stage(plan['training_stage']['file'])
    confirmation=p.confirmation.verify_stage(plan['confirmation_report']['file'])
    proof=read(plan['confirmation_report']['file']);bind(plan['confirmation_report']['file'],plan['confirmation_report']['sha256'])
    bind(plan['confirmation_report']['manifest_file'],plan['confirmation_report']['manifest_sha256'])
    need(proof['manifest_file']==plan['confirmation_report']['manifest_file'] and proof['manifest_sha256']==plan['confirmation_report']['manifest_sha256']
         and confirmation['seed']==91726342 and confirmation['manifest_required_before_fitting'] is True
         and confirmation['new_model_trainability_established'] is False
         and confirmation['training_stage']==plan['training_stage']==bank['plan']['training_stage'],
         'Frozen independent confirmation and exact training-stage identities must precede fitting')
    need(stage['plan']==bank['plan']['training_plan'] and stage['summary_file']==plan['training_stage']['file']
         and stage['summary_sha256']==plan['training_stage']['sha256'],'Training/bridge stage ownership differs')
    rows=read(plan['rows_file']);scenes=read(plan['scenes_file']);pairs=read(plan['pairs_file']);order=read(plan['order_file'])
    qindex={r['question']:r['question_index'] for r in original_rows}
    need(rows==[dict(r,first_token_id=r['target_ids'][0],question_index=qindex[r['question']]) for r in bank['rows']] and scenes==bank['scenes']
         and pairs==bank['pairs'] and order==bank['order']==expected_order(base_pairs,pairs), 'Complete stage/bank/order ownership differs')
    need(len(rows)==len(scenes)==216 and len(pairs)==108 and len(order)==48000
         and Counter(r['gold'] for r in rows)==Counter({name:24 for name in p.PEOPLE})
         and {r['question'] for r in rows}==set(stats['questions']),'Balanced original support differs')
    lookup={r['sid']:r for r in rows}
    for r in original_rows:
        row=lookup[r['sid']]
        need(all(row[k]==v for k,v in r.items()) and stats['questions'][r['question_index']]==row['question'] and row['orientation_version']=='original' and row['base_sid']==r['sid']
             and row['base_pair_id']==r['pair_id'] and row['base_contrast_id']==r['contrast_id'], 'Original108 semantic reference changed')
    first_token_criterion([dict(r,first_token_correct=True) for r in rows])
    originals={r['sid']:r for r in original_rows}
    for r in rows:
        reference=originals[r['base_sid']]
        need(all(r[k]==reference[k] for k in ('question','question_index','n_frames','gold','target_ids'))
             and r['base_pair_id']==reference['pair_id'] and r['base_contrast_id']==reference['contrast_id'],
             'The room-label swap changed original targets or support')
        scene=scenes[r['sid']];ids=r['target_ids']
        need(scene['split']=='train' and scene['target_ids']==ids and scene['question']==r['question'] and scene['n_frames']==r['n_frames']
             and ids[-1]==151645 and len(ids)==(3 if r['gold'] in ('Sandra','Noah') else 2)
             and scene['target_prefixes']==[ids[:i] for i in range(len(ids))]
             and len(scene['local_feature_ids'])==r['n_frames'] and all(len(group)==len(ids) for group in scene['local_feature_ids'])
             and len(scene['global_feature_ids'])==len(ids),'Full-name/EOS strict-prefix feature layout differs')
    for pair in pairs:
        a,b=(lookup[sid] for sid in pair['sids'])
        need([a['n_frames'],b['n_frames']]==[8,16] and a['target_ids']==b['target_ids']==pair['target_ids']
             and a['question']==b['question']==pair['question'] and a['pair_id']==b['pair_id']==pair['pair_id']
             and a['orientation_version']==b['orientation_version']==pair['orientation_version']
             and a['base_pair_id']==b['base_pair_id']==pair['base_pair_id'],'Exact paired N8/N16 ownership differs')
    for key in ('stats_sha256','stats_tensors','initial_sha256','initial_state','native_identity','native_identity_sha256','native_module_identity',
                'native_model_sha256','uniform_plan','oracle_report','conditioning_plan','conditioning_report','trainability_mask'):
        need(plan[key]==parent[key],'Original factor/native/statistics input differs: '+key)
    for field,digest_field in (('stats_file','stats_sha256'),('initial_file','initial_sha256'),('native_model_file','native_model_sha256')):
        bind(plan[field],plan[digest_field]);need(sha(plan[field])==sha(parent[field]),'Original consumed bytes differ: '+field)
    bind(plan['stats_inventory_file'],parent['runtime_bindings'][parent['stats_inventory_file']])
    need(read(plan['stats_inventory_file'])==dict(questions=stats['questions'],occurrences=stats['occurrences'],training_sids=[r['sid'] for r in original_rows],
         original_cached_empty_prefix_only=True,no_fitted_captures=True,no_local_labels=True)
         and len(stats['occurrences'])==1296,'Statistics must retain original108-scene occurrence weighting')
    need(bank['native_identity']==plan['native_identity'] and bank['native_identity_sha256']==plan['native_identity_sha256']
         and all(bank[k]==plan[k] for k in ('features_file','features_sha256','feature_tensor')),'Merged native feature identity differs')
    packet=torch.load(plan['features_file'],map_location='cpu',weights_only=True);states=packet['states'];index=packet['index'];ids=packet['feature_ids']
    need(packet['schema_version']==1 and ids==bank['feature_ids'] and index==bank['index']=={fid:i for i,fid in enumerate(ids)}
         and states.dtype==torch.float16 and states.shape==(len(ids),3584) and bool(torch.isfinite(states).all())
         and p.tensor_info(states)==plan['feature_tensor'],'Complete actual native feature bank differs')
    need(set(ids)=={fid for scene in scenes.values() for group in scene['local_feature_ids'] for fid in group}
                  |{fid for scene in scenes.values() for fid in scene['global_feature_ids']},'Feature bank has missing or unrelated rows')
    for fid in ids:
        pointer=bank['cache']['features'][fid]
        need(pointer==dict(file=plan['features_file'],file_sha256=plan['features_sha256'],row=index[fid],state_sha256=p.tensor_info(states[index[fid]])['sha256']),
             'Actual merged row differs from independently audited native ownership')
    for sid,scene in original_scenes.items():
        for key in ('local_feature_ids','global_feature_ids','target_prefixes','target_ids','question','n_frames'):
            need(scenes[sid][key]==scene[key],'Original feature layout changed: '+key)
    for fid in original_index:
        need(fid in index and torch.equal(states[index[fid]],original_states[original_index[fid]]),'Original native feature bytes changed')
    del original_states,packet
    for pair in pairs:
        a,b=(scenes[sid] for sid in pair['sids'])
        need(a['global_feature_ids']==b['global_feature_ids'],'Paired global native history differs')
    packet=torch.load(plan['initial_file'],map_location='cpu',weights_only=True);initial=packet['branch']
    need(set(packet)=={'seed','branch'} and packet['seed']==24 and {k:p.tensor_info(v) for k,v in initial.items()}==plan['initial_state']
         and bool((initial['up.weight']==0).all()),'Original unfitted factor state differs');check_weights(torch,initial,initial)
    counts=[sum(len(scenes[sid]['target_ids']) for pair in order[i:i+8] for sid in pair['sids']) for i in range(0,48000,8)]
    need(counts==plan['head_rows_by_update'] and counts[:600]==parent['head_rows_by_update'] and max(counts)==44
         and sum(counts)==plan['training_head_rows']==213330 and plan['total_head_rows']==213546
         and read(plan['head_rows_file'])==dict(by_update=counts,total=sum(counts),maximum=44,endpoint_rows=216,all_head_rows=213546),
         'Unchanged full-target row inventory differs')
    return rows,scenes,order,states,index,stats,initial


def audit_training(directory,summary,scenes,order,denominators):
    bind(summary['training_file'],summary['training_sha256']);logs=read(summary['training_file']);need(len(logs)==6000,'Update coverage differs')
    total=0;maximum=0;expected_calls=[]
    for step,row in enumerate(logs,1):
        batch=order[(step-1)*8:step*8];sids=[sid for pair in batch for sid in pair['sids']]
        sequences=[scenes[sid]['target_ids'] for sid in sids];layout=layout_for(sequences);lengths=list(map(len,sequences));offsets=layout['offsets']
        rate=learning_rate(step)
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
        q=len(layout['targets']);need(q<=44,'Per-call target row cap exceeded');total+=q;maximum=max(maximum,q)
        for module in ('norm','head'):
            expected_calls.append(dict(module=module,phase='training',step=step,endpoint_step=None,input_shape=[1,q,3584],
                output_shape=[1,q,3584 if module=='norm' else 152064],input_dtype='torch.float16',output_dtype='torch.float16'))
        if step in ENDPOINTS:
            for batch in range(1,15):
                q=8 if batch==14 else 16
                for module in ('norm','head'):
                    expected_calls.append(dict(module=module,phase='evaluation',step=batch,endpoint_step=step,input_shape=[1,q,3584],
                        output_shape=[1,q,3584 if module=='norm' else 152064],input_dtype='torch.float16',output_dtype='torch.float16'))
    independent_total=sum(len(scenes[sid]['target_ids']) for pair in order for sid in pair['sids'])
    need(total==independent_total and maximum==max(len(r['target_ids']) for r in logs) and logs[0]['consistency_loss']==0,
         'Full training call-row inventory differs')
    return logs,expected_calls,dict(passed=True,updates=6000,pair_presentations=48000,scene_presentations=96000,
        training_target_positions=total,maximum_training_positions_per_call=maximum,endpoint_first_query_positions=216,total_head_rows=total+216,
        terminal_ce=logs[-1]['ce_loss'],terminal_residual_diagnostic=logs[-1]['consistency_loss'],consistency_coefficient=0.,
        full_sequence_loss_reductions_verified=True,no_independent_optimizer_trajectory_replay=True)


def audit_capture(torch,out,record,arm,plan,rows,scenes,states,index,stats,initial,endpoints,logs,norm,head):
    p=driver();phase=record['phase'];step=record['step'];training=phase=='training';endpoint_step=record['endpoint_step'];cyclic=False
    need(endpoint_step is None if training else endpoint_step in ENDPOINTS,'Invalid capture endpoint association')
    need(phase in ('training','evaluation') and record['arm']==arm and arm in ARMS, 'Invalid capture phase/arm')
    bind(record['file'],record['sha256']);bind(record['weights_file'],record['weights_sha256'])
    cap=torch.load(record['file'],map_location='cpu',weights_only=True);wp=torch.load(record['weights_file'],map_location='cpu',weights_only=True);w=wp['branch']
    check_weights(torch,w,initial)
    if training:
        need(wp['arm']==arm and step in DIAGNOSTIC_STEPS and wp['step']==record['weights_step']==step-1
             and wp['optimizer_step']==step and wp['position']=='before_update' and wp['source_sha256']==plan['source_sha256'], 'Pre-update capture weight binding differs')
        sids=logs[step-1]['sids']
        if step==1:need(all(torch.equal(w[k],initial[k]) for k in initial),'First captured state differs from fresh initialization')
    else:
        need(1<=step<=14 and wp['step']==record['weights_step']==endpoint_step and all(torch.equal(w[k],endpoints[endpoint_step][k]) for k in initial), 'Final capture does not use fixed endpoint')
        sids=[r['sid'] for r in rows[(step-1)*16:step*16]]
    need(cap['schema_version']==1 and cap['arm']==arm and cap['phase']==phase and cap['step']==step
         and cap['sids']==record['sids']==sids and cap['pairing']==record['pairing']=='paired' and cap['endpoint_step']==endpoint_step, 'Capture scene/phase ownership differs')
    h,g,valid,layout=gather(torch,scenes,states,index,sids,first_only=not training)
    if training:need(cap['layout']==layout,'Training capture full-prefix layout differs')
    else:
        full=layout_for([scenes[sid]['target_ids'] for sid in sids]);expected=dict(first_query_only=True,target_ids=layout['targets'],
            full_target_ids=full['target_sequences'],first_indices=full['offsets'][:-1],original_full_layout=full)
        need(cap['layout']==expected,'Final capture original empty-prefix ownership differs')
    for key,value in (('local_states',h),('global_states',g),('valid_mask',valid)):
        need(cap[key].dtype==value.dtype and torch.equal(cap[key],value),'Actual frozen feature ownership differs: '+key)
    mean=stats['global_mean'].unsqueeze(0).expand(len(g),-1)
    need(cap['question_means'].dtype==torch.float32 and torch.equal(cap['question_means'],mean), 'Global mean not fixed at all prefixes')
    observed=cap['capture'];reference=functional_core(torch,h,g,valid,mean,stats['scale'],w,arm,cyclic)
    need(set(observed)==set(reference),'Factor capture coverage differs')
    need(observed['factor_permutation_indices'].dtype==torch.int64
         and torch.equal(observed['factor_permutation_indices'],reference['factor_permutation_indices']), 'Prespecified factor-pairing indices differ')
    metrics={key:tensor_error(torch,observed[key],reference[key]) for key in reference if key!='factor_permutation_indices'}
    need(torch.equal(observed['factor_b_used'],observed['factor_b'].gather(0,reference['factor_permutation_indices'].unsqueeze(-1).expand_as(observed['factor_b']))), 'Actual factor permutation differs')
    payload=observed['factor_a']*observed['factor_b_used'] if arm=='product' else .5*(observed['factor_a']+observed['factor_b_used'])
    need(torch.equal(observed['payload'],payload) and torch.equal(observed['gates'],valid.float()*.5)
         and bool((observed['scores']==.5).all()) and torch.equal(observed['messages'],payload*observed['gates'].unsqueeze(-1))
         and bool((observed['messages'][~valid]==0).all()) and bool((payload.abs()<=1).all()), 'Captured factor/gate/message arithmetic differs')
    need(observed['delta'].dtype==torch.float32 and cap['fused_global'].dtype==cap['normalized'].dtype==cap['logits'].dtype==torch.float16
         and cap['fused_global'].shape==cap['normalized'].shape==g.shape and cap['logits'].shape==(len(g),152064)
         and all(bool(torch.isfinite(cap[k]).all()) for k in ('fused_global','normalized','logits')), 'Actual native dtype/shape differs')
    need(torch.equal(g+observed['delta'].half(),cap['fused_global'])
         and {k:p.tensor_info(cap[k]) for k in ('fused_global','normalized','logits')}==record['tensors'], 'Native cast-before-add or tensor identities differ')
    if training and step==1:need(bool((observed['delta']==0).all()) and torch.equal(cap['fused_global'],g),'Zero-U identity differs')
    head_metrics=[];normal_error=None
    if not training:
        normalized=norm(cap['fused_global'].unsqueeze(0))[0];replayed=head(normalized.unsqueeze(0))[0]
        head_metrics=replay_metric(torch,cap['logits'],replayed);normal_error=float((normalized.float()-cap['normalized'].float()).abs().max())
    nll=fp64_nll(torch,cap['logits'],layout['targets'])
    nll_metrics=[dict(position=i,fp64=float(a),gpu_fp32=b,passed=close(float(a),b)) for i,(a,b) in
        enumerate(zip(nll,logs[step-1]['position_losses']['ce']))] if training else []
    result=dict(arm=arm,phase=phase,step=step,endpoint_step=endpoint_step,sids=sids,capture_file=record['file'],capture_sha256=record['sha256'],
        weights_file=record['weights_file'],weights_sha256=record['weights_sha256'],functional_core=metrics,
        native_head=head_metrics,normalized_max_absolute=normal_error,nll=nll_metrics,fp16_cast_before_add_exact=True,
        factor_pairing='paired',captured_rows=len(g),
        head_rows=0 if training else len(g),cpu_norm_calls=0 if training else 1,cpu_head_calls=0 if training else 1,
        passed=all(x['passed'] for x in metrics.values()) and all(x['passed'] for x in head_metrics+nll_metrics))
    save(out/(f'{arm}_training_{step:04d}_capture_audit.json' if training else f'{arm}_evaluation_{endpoint_step:04d}_{step:04d}_capture_audit.json'),result)
    need(result['passed'],'Factor/core/native/NLL audit failed; metrics and raw evidence retained')
    return result,cap['logits']



def optimizer_audit(torch,value,step,weights):
    need(set(value)=={'parameter_ids','optimizer_parameter_ids','requires_grad','state_tensor_info','param_group_hyperparameters'},'Optimizer snapshot schema differs')
    names=list(weights);trainable=[k for k in names if k not in FROZEN_NAMES];ids=value['parameter_ids']
    need(len(names)==8 and len(trainable)==6 and set(ids)==set(names) and all(type(v) is int and v>0 for v in ids.values())
         and len(set(ids.values()))==8 and value['optimizer_parameter_ids']==[ids[k] for k in trainable]
         and value['requires_grad']=={k:k in trainable for k in names},'Optimizer must retain six live parameters and two frozen selector objects')
    need(set(value['state_tensor_info'])==set(trainable),'Adam state must exclude the fixed selector')
    expected_step=driver().tensor_info(torch.tensor(float(step),dtype=torch.float32))
    for name,state in value['state_tensor_info'].items():
        need(set(state)=={'step','exp_avg','exp_avg_sq'} and state['step']==expected_step,'Adam update counter changed or restarted')
        for key in ('exp_avg','exp_avg_sq'):
            info=state[key];need(set(info)=={'shape','dtype','sha256'} and info['shape']==list(weights[name].shape) and info['dtype']=='torch.float32'
                and isinstance(info['sha256'],str) and len(info['sha256'])==64,'Adam moment tensor metadata differs')
    groups=value['param_group_hyperparameters'];need(len(groups)==1,'Single matched AdamW parameter group required');group=groups[0]
    need(close(group['lr'],learning_rate(step),atol=1e-12,rtol=1e-12) and group['betas']==[.9,.999]
         and group['eps']==1e-8 and group['weight_decay']==0 and group['amsgrad'] is False and group['maximize'] is False,
         'Actual AdamW hyperparameters differ at endpoint')
    return ids


def checkpoint_path(root,step):return root/('final.pt' if step==6000 else f'step_{step:04d}.pt')


def training_loss_descriptions(logs,scenes):
    roles=defaultdict(list);cycles=defaultdict(lambda:dict(pair_presentations=0,scene_ce=[],roles=defaultdict(list)))
    for row in logs:
        cursor=0
        for i,sid in enumerate(row['sids']):
            count=len(scenes[sid]['target_ids']);values=row['position_losses']['ce'][cursor:cursor+count];cursor+=count
            group=cycles[row['cycles'][i//2]];group['scene_ce'].append(sum(values)/count)
            if i%2==0:group['pair_presentations']+=1
            for name,items in (('first_token',values[:1]),('EOS',values[-1:]),('name_continuation',values[1:-1])):
                roles[name].extend(items);group['roles'][name].extend(items)
    def metrics(groups):
        return [dict(labels=dict(role=k),metrics=dict(positions=len(v),mean_nll=None if not v else sum(v)/len(v))) for k,v in sorted(groups.items())]
    return dict(descriptive_only=True,token_roles=metrics(roles),cycles=[dict(cycle=k,pair_presentations=v['pair_presentations'],
        complete_pair_cycle=v['pair_presentations']==54,mean_scene_ce=sum(v['scene_ce'])/len(v['scene_ce']),token_roles=metrics(v['roles']))
        for k,v in sorted(cycles.items())],no_checkpoint_selection=True,horizon_and_annealing_unchanged=True,
        all_complete_cycles_and_final_partial_retained=True,no_convergence_certificate=True)


def self_test(torch):
    base=[dict(pair_id=f'p{i}',sids=[f'a{i}',f'b{i}']) for i in range(54)]
    pairs=[dict(base_pair_id=r['pair_id'],orientation_version=o,pair_id=r['pair_id']+'/'+o,
                sids=[sid+'/'+o for sid in r['sids']]) for r in base for o in ('original','flipped')]
    order=expected_order(base,pairs);cycles=Counter(x['cycle'] for x in order);seen=Counter()
    for row in order:
        bid=row['base_pair_id'];expected='original' if seen[bid]%2==0 else 'flipped';seen[bid]+=1
        need(row['orientation_version']==expected and row['base_pair_visit']==seen[bid]
             and row['sids']==[sid+'/'+expected for sid in row['base_sids']],'Per-base visit alternation fixture failed')
    projected=[{k:(r['base_pair_id'] if k=='pair_id' else r['base_sids'] if k=='sids' else r[k])
                for k in ('cycle','pair_slot','pair_id','sids')} for r in order]
    need(projected[:4800]==old.expected_order(base) and cycles[889]==48 and all(cycles[i]==54 for i in range(1,889))
         and sum(r['orientation_version']=='original' for r in order)==24024,'Fixed schedule/original target-position fixture failed')
    need(close(learning_rate(1),.00002,atol=1e-15,rtol=1e-15) and learning_rate(50)==.001
         and learning_rate(6000)==.00001 and learning_rate(600)>.0009,'Unchanged warmup/cosine fixture failed')
    rows=[dict(sid=f'{f}/{v}/{n}/{o}',base_contrast_id=f,variant=v,n_frames=n,orientation_version=o,first_token_correct=True)
          for f in range(18) for v in range(3) for n in (8,16) for o in ('original','flipped')]
    def scored(failures):return [dict(r,first_token_correct=i not in failures) for i,r in enumerate(rows)]
    boundary=primary_screen({6000:scored(set(range(6))|set(range(12,16)))})
    need(boundary['passed'] and boundary['first_correct']==206 and boundary['orientation_correct']==dict(original=103,flipped=103)
         and boundary['complete_families']==16,'Exact registered acceptance boundary differs')
    orientation=first_token_criterion(scored(set(range(0,12,2))))
    family=first_token_criterion(scored({0,12,24}))
    need(not orientation['passed'] and orientation['first_correct']==210 and orientation['orientation_correct']['original']==102
         and not family['passed'] and family['first_correct']==213 and family['complete_families']==15,
         'Pooled success must not bypass orientation or whole-family requirements')
    for bad in (rows[:-1],rows[:-1]+[rows[0]]):
        try:first_token_criterion(bad)
        except ValueError:pass
        else:raise AssertionError('Missing/duplicate outcomes must fail')
    try:primary_screen({600:rows,6000:rows})
    except ValueError:pass
    else:raise AssertionError('Extra endpoint cannot select acceptance')
    names=('local_bias','selection_weight','selection_bias','query.weight','local.weight','aggregate_projection.weight','aggregate_projection.bias','up.weight')
    parameters={k:torch.nn.Parameter(torch.ones(2,1),requires_grad=k not in FROZEN_NAMES) for k in names}
    live=[v for v in parameters.values() if v.requires_grad];optimizer=torch.optim.AdamW(live,lr=learning_rate(3),weight_decay=0.)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        for v in live:v.grad=torch.ones_like(v)
        optimizer.step()
    info=driver().tensor_info
    snapshot=dict(parameter_ids={k:id(v) for k,v in parameters.items()},optimizer_parameter_ids=[id(v) for v in live],
        requires_grad={k:v.requires_grad for k,v in parameters.items()},
        state_tensor_info={k:{name:info(value) for name,value in optimizer.state[v].items()} for k,v in parameters.items() if v.requires_grad},
        param_group_hyperparameters=[{k:v for k,v in optimizer.param_groups[0].items() if k!='params'}])
    snapshot=__import__('json').loads(__import__('json').dumps(snapshot));optimizer_audit(torch,snapshot,3,parameters)
    snapshot['state_tensor_info']['local_bias']['step']=info(torch.tensor(2.))
    try:optimizer_audit(torch,snapshot,3,parameters)
    except Exception as exc:need('Adam update counter' in str(exc),'Unexpected optimizer-reset fixture failure')
    else:raise AssertionError('A reset Adam counter must fail')
    snapshot['state_tensor_info']['local_bias']['step']=info(torch.tensor(3.));snapshot['requires_grad']['selection_weight']=True
    try:optimizer_audit(torch,snapshot,3,parameters)
    except Exception as exc:need('six live parameters' in str(exc),'Unexpected selector fixture failure')
    else:raise AssertionError('A trainable selector must fail')
    toy_scenes={'a':{'target_ids':[1,2]},'b':{'target_ids':[3,4,2]}}
    toy=[dict(sids=['a','b'],cycles=[1],position_losses={'ce':[2.,0.,3.,6.,0.]})]
    description=training_loss_descriptions(toy,toy_scenes);need(description['cycles'][0]['mean_scene_ce']==2.,'Mean-scene length weighting fixture failed')
    return dict(passed=True,groups=7,alternating_orientation_original_base_order=True,unchanged_cosine_schedule=True,
        exact_orientation_and_family_boundaries=True,complete_outcome_denominator=True,
        final6000_only_screen=True,actual_Adam_counter_selector_and_reset_rejection=True,ragged_scene_loss_weights=True)


def audit_run(torch,out,directory,plan_path,plan,frozen,rows,scenes,order,states,index,stats,initial,norm,head):
    p=driver();directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json');arm=config['arm']
    need(directory.parent==p.OUT and directory.name==config['run_id']==f'run_{arm}_{config["slurm_job_id"]}' and arm in ARMS
         and config['protocol']==p.PROTOCOL and config['policy']==p.POLICY and config['seed']==24
         and config['source_sha256']==frozen==plan['source_sha256'] and all(summary.get(k)==v for k,v in config.items())
         and Path(config['plan_file']).resolve()==Path(plan_path).resolve() and config['plan_sha256']==sha(plan_path), 'Run/config/source/plan identity differs')
    for name,digest in frozen.items():bind(directory/'source'/name.replace('/','_'),digest)
    inherited=dict(source_sha256=plan['inherited_source_sha256'])
    need(config['inherited_source_sha256']==plan['inherited_source_sha256'] and read(directory/'inherited_sources.json')==inherited, 'Inherited source descriptor differs')
    for flag in ('passed','completed','computational_integrity_passed','checkpoint_roundtrip_passed','native_weights_unchanged','statistics_frozen','statistics_versions_unchanged','input_versions_unchanged','frozen_selector_exact',
                 'cached_first_query_only','no_pretrained_backbone_loaded','no_dev_or_test','no_native_or_whole_answer_claim'):
        need(summary[flag] is True,'Run invariant failed: '+flag)
    need(summary['phase']=='run' and summary['steps']==6000 and not any(k.startswith('permutation_') for k in summary),'Horizon/scope differs')
    need(config['hardware']['gpu']=='NVIDIA B200' and config['hardware']['matmul_allow_tf32'] is False
         and config['hardware']['torch_version']==str(torch.__version__),'Native hardware/precision differs')
    for key in ('native_identity','native_identity_sha256','native_module_identity','native_model_file','native_model_sha256',
                'features_file','features_sha256','feature_tensor','stats_file','stats_sha256','stats_tensors',
                'uniform_plan','oracle_report','conditioning_plan','conditioning_report','factor_plan','cache_report','confirmation_report','training_stage',
                'decision_step','evaluation_steps'):
        need(config[key]==plan[key],'Consumed native/input identity differs: '+key)
    need(config['reused_unfitted_initialization'] is plan['reused_unfitted_initialization'] is True
         and config['fresh_shared_initialization'] is plan['fresh_shared_initialization'] is False
         and config['initialized']==plan['initial_state'] and config['initial_checkpoint_sha256']==plan['initial_sha256'],
         'Original shared unfitted factor initialization differs')
    trainable=[k for k in initial if k not in FROZEN_NAMES]
    mask=dict(trainable_parameter_names=trainable,frozen_selector_parameter_names=list(FROZEN_NAMES),
        effective_trainable_parameters=1385760,retained_state_parameters=1385857)
    need(plan['trainability_mask']==mask and all(config[k]==v for k,v in mask.items()),'Six-parameter optimizer/fixed-selector scope differs')
    need(summary['interaction']==arm and summary['pairing_restored'] is True and summary['factor_derangement'] is False
         and not any(k.startswith('permutation_') for k in summary),'Only the original paired interaction is permitted')
    for key in ('training','captures','calls','evaluations'):bind(summary[key+'_file'],summary[key+'_sha256'])
    for key in ('rows','presentations','stats_inventory'):bind(config[key+'_file'],config[key+'_sha256'])
    need(read(config['rows_file'])==rows and read(config['presentations_file'])==order and config['order_sha256']==object_sha(order)
         and read(config['stats_inventory_file'])==read(plan['stats_inventory_file']),'Copied rows/order/statistic inventory differ')
    root=p.CKPT/directory.name;data=p.DATA/directory.name
    need(Path(config['checkpoint_directory'])==root and Path(config['data_directory'])==data and Path(config['initial_checkpoint'])==root/'initial.pt','Authorized artifact roots differ')
    bind(config['initial_checkpoint'],config['initial_checkpoint_sha256']);selected=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=True)
    need(selected['seed']==24 and all(torch.equal(selected['branch'][k],initial[k]) for k in initial),'Actual run initialization differs')
    denominators={fid:float(states[index[fid]].float().square().sum()+1e-6) for scene in scenes.values() for fid in scene['global_feature_ids']}
    logs,expected_calls,training=audit_training(directory,summary,scenes,order,denominators)
    need([len(r['target_ids']) for r in logs]==plan['head_rows_by_update'] and close(summary['training_seconds'],sum(r['seconds'] for r in logs))
         and summary['training_head_rows']==training['training_target_positions'] and summary['total_head_rows']==training['total_head_rows'],'Fixed training count/timing differs')
    counters=dict(core=6014,conditioning=6014,norm=6014,head=6014,vlm=0,vision=0);calls=read(summary['calls_file'])
    need(calls==expected_calls and summary['counters']==counters
         and read(directory/'final_counters.json')==dict(counters=counters,recorded_calls=calls,partial_outputs_retained=True),'Chronological actual call ledger differs')
    evaluations=summary['evaluations'];need(read(summary['evaluations_file'])==evaluations and Path(summary['evaluations_file'])==directory/'evaluations.json'
         and summary['final_head_rows']==216 and summary['earlier_endpoints_descriptive_only'] is False and summary['decision_step']==6000
         and set(evaluations)=={str(e) for e in ENDPOINTS},'Only the fixed6000 endpoint required')
    weights={};endpoint_audits={};packets={};optimizer_ids=None;native_table=dict(norm=plan['native_identity']['norm_weight'],head=plan['native_identity']['head_weight'])
    for e in ENDPOINTS:
        desc=evaluations[str(e)];checkpoint=checkpoint_path(root,e)
        need(desc['endpoint_step']==e and desc['checkpoint']==str(checkpoint) and desc['descriptive_only'] is (e!=6000),'Endpoint identity/decision scope differs')
        bind(checkpoint,desc['checkpoint_sha256']);packet=torch.load(checkpoint,map_location='cpu',weights_only=True)
        need(set(packet)=={'branch','step','config'} and packet['step']==e and packet['config']==config,'After-update checkpoint ownership differs')
        w=packet['branch'];check_weights(torch,w,initial);weights[e]=w;table={k:p.tensor_info(v) for k,v in w.items()}
        for key in ('endpoint','roundtrip','raw','predictions'):bind(desc[key+'_file'],desc[key+'_sha256'])
        endpoint=read(desc['endpoint_file']);roundtrip=read(desc['roundtrip_file'])
        need(endpoint['passed'] is True and endpoint['endpoint_step']==e and endpoint['before_evaluation']==endpoint['after_evaluation']==table
             and endpoint['parameter_sha256']==object_sha(table) and endpoint['native_before']==endpoint['native_after']==config['native_weight_identity']==native_table
             and endpoint['input_tensors_before']==endpoint['input_tensors_after']==dict(states=plan['feature_tensor'])
             and endpoint['input_versions_unchanged'] is True and endpoint['training_mode_restored'] is True
             and endpoint['optimizer_before']==endpoint['optimizer_after'] and endpoint['parameter_versions_before']==endpoint['parameter_versions_after'],
             'Final evaluation mutated readout/native/input/optimizer/trainability state')
        need(endpoint['working_statistics_before']==endpoint['working_statistics_after']=={k:plan['stats_tensors'][k] for k in ('means','global_mean','scale')}
             and endpoint['statistics_versions_unchanged'] is True and endpoint['frozen_selector_exact'] is True and endpoint['hooks_removed'] is True
             and endpoint['interaction']==arm and endpoint['pairing_restored'] is True and endpoint['factor_derangement'] is False,
             'Original statistics, fixed selector or factor pairing changed during endpoint evaluation')
        optimizer_identity=optimizer_audit(torch,endpoint['optimizer_before'],e,w)
        if optimizer_ids is None:optimizer_ids=optimizer_identity
        need(optimizer_ids==optimizer_identity and set(endpoint['parameter_versions_before'])==set(initial)
             and all(type(v) is int and v>=0 for v in endpoint['parameter_versions_before'].values()),'Parameter objects/version metadata changed across observations')
        count=e+14*sum(x<=e for x in ENDPOINTS)
        need(endpoint['counters']==dict(core=count,conditioning=count,norm=count,head=count,vlm=0,vision=0),'Endpoint cumulative call count differs')
        need(roundtrip['passed'] is True and roundtrip['step']==e and roundtrip['config_sha256']==sha(directory/'config.json')
             and roundtrip['selected']==roundtrip['serialized']==table and roundtrip['reset_performed'] is (e==6000)
             and roundtrip['initial']==(plan['initial_state'] if e==6000 else None),'Endpoint serialization/final reset identity differs')
        for value in (endpoint,roundtrip):need(value['checkpoint']==str(checkpoint) and value['checkpoint_sha256']==desc['checkpoint_sha256'],'Endpoint checkpoint hash differs')
        raw=torch.load(desc['raw_file'],map_location='cpu',weights_only=True);logits=raw['logits']
        need(raw['schema_version']==1 and raw['endpoint_step']==e and Path(desc['raw_file'])==data/f'logits_{e:04d}.pt'
             and Path(desc['predictions_file'])==directory/f'predictions_{e:04d}.json'
             and Path(desc['endpoint_file'])==directory/f'endpoint_{e:04d}.json' and Path(desc['roundtrip_file'])==directory/f'roundtrip_{e:04d}.json'
             and raw['sids']==[r['sid'] for r in rows] and logits.dtype==torch.float16 and logits.shape==(216,152064)
             and bool(torch.isfinite(logits).all()) and p.tensor_info(logits)==desc['raw_tensor']
             and raw['batches']==desc['raw_batches'] and len(raw['batches'])==14,'Endpoint full-vocabulary raw ownership differs')
        packets[e]=raw;endpoint_audits[str(e)]=dict(passed=True,endpoint_step=e,checkpoint=str(checkpoint),checkpoint_sha256=desc['checkpoint_sha256'],
            selected_parameter_sha256=object_sha(table),endpoint_file=desc['endpoint_file'],endpoint_sha256=desc['endpoint_sha256'],
            roundtrip_file=desc['roundtrip_file'],roundtrip_sha256=desc['roundtrip_sha256'],evaluation_preserved_training_state=True,descriptive_only=e!=6000)
    for key in ('checkpoint','checkpoint_sha256','endpoint_file','endpoint_sha256','roundtrip_file','roundtrip_sha256','raw_file','raw_sha256','raw_tensor','raw_batches','predictions_file','predictions_sha256','first_token_fit'):
        need(summary[key]==evaluations['6000'][key],'Ordinary summary alias must bind only step6000: '+key)
    need(summary['selected_parameter_sha256']==endpoint_audits['6000']['selected_parameter_sha256'],'Primary deployed endpoint identity differs')
    records=read(summary['captures_file']);expected=[]
    for step in range(1,6001):
        if step in DIAGNOSTIC_STEPS:expected.append(('training',step,None))
        if step in ENDPOINTS:expected.extend(('evaluation',batch,step) for batch in range(1,15))
    need([(r['phase'],r['step'],r['endpoint_step']) for r in records]==expected and all(r['arm']==arm and r['pairing']=='paired' for r in records),'Chronological22-capture inventory differs')
    need(sorted(f.name for f in root.glob('*.pt'))==sorted(['initial.pt','final.pt']+[f'training_core_{s:04d}.pt' for s in DIAGNOSTIC_STEPS]),'Fixed checkpoint inventory differs')
    need(sorted(f.name for f in directory.glob('progress_*.json'))==[f'progress_{s:04d}.json' for s in PROGRESS],'Four progress snapshots required')
    for step in PROGRESS:
        count=step+14*sum(e<=step for e in ENDPOINTS)
        prior=[r for r in records if (r['step'] if r['phase']=='training' else r['endpoint_step'])<=step]
        need(read(directory/f'progress_{step:04d}.json')==dict(step=step,training=logs[:step],captures=prior,
            counters=dict(core=count,conditioning=count,norm=count,head=count,vlm=0,vision=0),evaluations={str(e):evaluations[str(e)] for e in ENDPOINTS if e<=step}),'Progress snapshot/endpoint chronology differs')
    audits=[];gpu_nll={e:[] for e in ENDPOINTS};cursors={e:0 for e in ENDPOINTS}
    with torch.no_grad():
        for record in records:
            phase,step,e=record['phase'],record['step'],record['endpoint_step']
            expected_weight=root/f'training_core_{step:04d}.pt' if phase=='training' else checkpoint_path(root,e)
            expected_capture=data/(f'training_capture_{step:04d}.pt' if phase=='training' else f'evaluation_{e:04d}_capture_{step:04d}.pt')
            need(Path(record['weights_file'])==expected_weight and Path(record['file'])==expected_capture,'Capture/checkpoint paths differ')
            audit,actual=audit_capture(torch,out,record,arm,plan,rows,scenes,states,index,stats,initial,weights,logs,norm,head);audits.append(audit)
            if phase=='evaluation':
                batch=packets[e]['batches'][step-1];count=len(record['sids']);cursor=cursors[e]
                need(batch['capture_file']==record['file'] and batch['capture_sha256']==record['sha256'] and batch['sids']==record['sids']
                     and batch['logits']==record['tensors']['logits'] and len(batch['gpu_fp32_nll'])==count
                     and torch.equal(actual,packets[e]['logits'][cursor:cursor+count]), 'Endpoint batch/capture ownership differs')
                cursors[e]+=count;gpu_nll[e]+=batch['gpu_fp32_nll']
    need(set(cursors.values())=={216},'The final endpoint requires216 raw rows')
    endpoint_rows={};strata=[]
    for e in ENDPOINTS:
        desc=evaluations[str(e)];logits=packets[e]['logits'];ids=logits.argmax(-1).tolist();nll=fp64_nll(torch,logits,[r['first_token_id'] for r in rows])
        rescored=[dict(r,argmax_id=ids[i],first_token_correct=ids[i]==r['first_token_id'],nll=float(nll[i]),gpu_fp32_nll=gpu_nll[e][i]) for i,r in enumerate(rows)]
        precision=[dict(sid=r['sid'],fp64_nll=r['nll'],gpu_fp32_nll=r['gpu_fp32_nll'],absolute_difference=abs(r['nll']-r['gpu_fp32_nll']),
            passed=close(r['nll'],r['gpu_fp32_nll'])) for r in rescored]
        saved=read(desc['predictions_file']);save(out/f'{arm}_outcomes_{e:04d}.json',rescored);save(out/f'{arm}_nll_precision_{e:04d}.json',precision)
        need(len(saved)==216 and all(set(a)==set(b) and all(a[k]==b[k] for k in a if k!='nll') and close(a['nll'],b['nll']) for a,b in zip(rescored,saved))
             and all(r['passed'] for r in precision),'Endpoint raw argmax/NLL/metadata rescore differs')
        criterion=first_token_criterion(rescored);need(criterion==desc['first_token_fit'],'Endpoint screen counts differ')
        endpoint_rows[e]=rescored;endpoint_audits[str(e)].update(first_token_fit=criterion,mean_first_token_nll=sum(r['nll'] for r in rescored)/216,
            outcomes_file=str(out/f'{arm}_outcomes_{e:04d}.json'),outcomes_sha256=sha(out/f'{arm}_outcomes_{e:04d}.json'))
        for key in ('gold','question','n_frames','base_contrast_id','orientation_version'):
            groups=defaultdict(list)
            for row in rescored:groups[row[key]].append(row)
            for label,group in sorted(groups.items(),key=lambda x:str(x[0])):
                strata.append(dict(labels=dict(endpoint_step=e,**{key:label}),metrics=dict(correct=sum(r['first_token_correct'] for r in group),contexts=len(group),
                    mean_nll=sum(r['nll'] for r in group)/len(group),all_correct=all(r['first_token_correct'] for r in group))))
    final_criterion=primary_screen(endpoint_rows);need(final_criterion==summary['first_token_fit'],'Only final6000 screen may decide')
    save(out/f'{arm}_strata.json',strata);save(out/f'{arm}_training_loss_descriptions.json',training_loss_descriptions(logs,scenes))
    return dict(passed=True,completed=True,arm=arm,slurm_job_id=config['slurm_job_id'],run_directory=str(directory),run_summary_sha256=sha(directory/'summary.json'),
        first_token_fit=final_criterion,primary_endpoint_step=6000,endpoint_audits=endpoint_audits,training=training,counters=counters,capture_audits=audits,
        cpu_head_calls=14,cpu_norm_calls=14,cpu_head_rows=sum(r['head_rows'] for r in audits),
        maximum_native_replay_tv=max(m['full_vocabulary_tv'] for r in audits for m in r['native_head']),
        strata_file=str(out/f'{arm}_strata.json'),strata_sha256=sha(out/f'{arm}_strata.json'),
        training_loss_descriptions_file=str(out/f'{arm}_training_loss_descriptions.json'),training_loss_descriptions_sha256=sha(out/f'{arm}_training_loss_descriptions.json'))


def resources(out,runs):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'all_user_sacct.psv';file.write_text(raw)
    job_arms={name:arm for arm,name in driver().RUN_JOBS.items()};names=set(job_arms);records=allocation_rows(raw,names|{'identity_join_factor_orientation_training_run'})
    need(len(records)==2 and {r['name'] for r in records}==names,'Exactly one attempt per factor arm required; all failed/zero allocations retained')
    for row in records:
        arm=job_arms[row['name']]
        need(row['job_id']==runs[arm]['slurm_job_id'] and row['partition']=='gpu' and row['state']=='COMPLETED'
             and row['exit_code']=='0:0' and row['gpus']==1 and row['seconds']<=180,'Factor-binding allocation identity/cap differs')
    from datetime import datetime
    events=[]
    for row in records:
        a,b=map(datetime.fromisoformat,(row['start'],row['end']));need(b>=a,'Invalid allocation interval')
        if b>a:events.extend(((a,1),(b,-1)))
    current=peak=0
    for _,d in sorted(events):current+=d;need(current>=0,'Invalid allocation overlap');peak=max(peak,current)
    total=sum(r['gpu_seconds'] for r in records);need(current==0 and peak<=2 and total<=360,'Factor-binding campaign cap exceeded')
    return dict(passed=True,jobs=records,allocated_gpu_seconds=total,maximum_concurrent_gpus=peak,
        failed_and_zero_allocations_retained=True,file=str(file),sha256=sha(file),command=command)


def report(args,out,frozen):
    import torch
    torch.set_num_threads(4);p=driver();plan=p.verify_plan(args.plan,ancestors=True)
    need(frozen==plan['source_sha256']==p.sources() and len(frozen)==6 and len(args.runs)==2
         and plan['inherited_source_sha256']==p.inherited_sources(),'New/inherited source or two-run inventory differs')
    need(read(out/'inherited_sources.json')==dict(source_sha256=plan['inherited_source_sha256']),'Report source descriptor differs')
    need(plan['decision_step']==6000 and plan['evaluation_steps']==list(ENDPOINTS)
         and plan['reused_unfitted_initialization'] is True and plan['fresh_shared_initialization'] is False
         and plan['no_new_statistics'] is True and plan['orientation_coverage_only'] is True
         and plan['policy']['paired_only'] is True,'Registered orientation-only scope differs')
    rows,scenes,order,states,index,stats,initial=audit_inputs(torch,plan)
    exposures=Counter(r['orientation_version'] for r in order)
    save(out/'input_reconstruction_audit.json',dict(passed=True,factor_plan=plan['factor_plan'],cache_report=plan['cache_report'],
        confirmation_report=plan['confirmation_report'],training_stage=plan['training_stage'],contexts=216,local_occurrences=2592,
        original_statistic_contexts=108,original_statistic_occurrences=1296,pair_presentations=48000,
        base_pair_order_exact=True,orientation_exposure_counts=dict(exposures),training_target_positions=213330,endpoint_rows_per_arm=216,
        actual_native_bank_rows=len(states),all_native_row_hashes_exact=True,global_conditioning_independently_reconstructed=True,
        original_shared_unfitted_factor_bytes_exact=True,confirmation_manifest_bound_before_fit=True,no_new_architecture=True))
    need(p.oracle.native_module_identity(torch,plan['native_identity'])==plan['native_module_identity'],'Actual native norm/head source differs')
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True);norm,head=p.oracle.native_modules(torch,packet,plan['native_identity'],'cpu');del packet
    before=dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight));runs={}
    for directory in args.runs:
        value=audit_run(torch,out,directory,args.plan,plan,frozen,rows,scenes,order,states,index,stats,initial,norm,head)
        need(value['arm'] not in runs,'Duplicate factor arm');runs[value['arm']]=value
    need(set(runs)==set(ARMS) and before==dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight))
         and norm.weight.grad is head.weight.grad is None and not norm.weight.requires_grad and not head.weight.requires_grad,
         'Two-arm frozen native audit incomplete')
    need(all(r['cpu_head_calls']==r['cpu_norm_calls']==14 and r['cpu_head_rows']==216 and len(r['capture_audits'])==22 for r in runs.values()),
         'Both arms require22 captures/14 CPU head batches/216 rows')
    budget=resources(out,runs)
    outcomes={arm:read(runs[arm]['endpoint_audits']['6000']['outcomes_file']) for arm in ARMS}
    need([r['sid'] for r in outcomes['product']]==[r['sid'] for r in outcomes['additive']],'Matched endpoint order differs')
    transitions=Counter((a['first_token_correct'],b['first_token_correct']) for a,b in zip(outcomes['additive'],outcomes['product']))
    comparison=dict(endpoint_step=6000,contexts=216,all_comparisons_descriptive=True,
        additive_to_product_transitions={f'{int(a)}_to_{int(b)}':transitions[a,b] for a in (False,True) for b in (False,True)},
        product_minus_additive_correct=sum(r['first_token_correct'] for r in outcomes['product'])-sum(r['first_token_correct'] for r in outcomes['additive']),
        mean_first_token_nll={arm:runs[arm]['endpoint_audits']['6000']['mean_first_token_nll'] for arm in ARMS})
    result=dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),factor_plan=plan['factor_plan'],cache_report=plan['cache_report'],
        confirmation_report=plan['confirmation_report'],training_stage=plan['training_stage'],
        stats_file=plan['stats_file'],stats_sha256=plan['stats_sha256'],stats_tensors=plan['stats_tensors'],runs=runs,comparison=comparison,resources=budget,
        primary_endpoint_step=6000,first_token_screen={arm:runs[arm]['first_token_fit'] for arm in ARMS},
        cpu_head_calls=28,cpu_norm_calls=28,cpu_head_rows=432,gpu_head_calls=12028,gpu_norm_calls=12028,gpu_head_rows=427092,vlm_calls=0,vision_calls=0,
        paired_only=True,no_new_permutation=True,orientation_coverage_only=True,horizon_and_annealing_unchanged=True,
        no_factor_semantics_or_expressiveness_claim=True,cached_first_query_only=True,no_native_or_whole_answer_claim=True,
        no_generalization_or_reasoning_claim=True,no_new_fit_or_model_forward=True,all_outcomes_retained=True)
    save(out/'analysis.json',result)
    lines=['# Fixed-budget orientation-coverage training','',
        'Independent computation and provenance audits passed. Both locked interactions used the same original unfitted tensors, '
        'native feature bank and original global conditioning statistics. Each unchanged base-pair presentation alternated its '
        'original and flipped orientation; targets,6000 learning rates and full-name-plus-EOS CE weighting were unchanged.','',
        '| Interaction | Correct /216 | Original /108 | Flipped /108 | Complete families /18 | Screen |',
        '|---|---:|---:|---:|---:|---|']
    for arm in ARMS:
        c=runs[arm]['first_token_fit'];role='PASS' if c['passed'] else 'FAIL'
        lines.append(f"| {arm} | {c['first_correct']} | {c['orientation_correct']['original']} | {c['orientation_correct']['flipped']} | {c['complete_families']} | {role} |")
    lines+=['','Only the fixed step6000 endpoint determines acceptance:206/216 pooled,103/108 within each orientation and16/18 complete12-context families. '
        'All outcomes, orientation/length/name/question/family strata and training losses are retained. These are cached training first queries; '
        'neither a pass nor a failure establishes native whole-answer performance or fresh generalization. Native and fresh evaluation require separate release.','',
        'The final checkpoint was serialized, reset to the unfitted state and reloaded. Independent audits check optimizer state, '
        'frozen selectors, native weights, features, statistics, actual captured factor arithmetic, and all6000 scene-balanced CE reductions. '
        'They do not rerun optimization or certify convergence. Prior failed results are unchanged.','',
        f"Allocated GPU cost: {budget['allocated_gpu_seconds']} seconds. Each arm used6014 core/conditioning/norm/head calls and213546 head rows. "
        'VLM/vision calls were zero. CPU audit covered44 captures and28 endpoint head batches/432 rows, with exact argmax and TV<=0.02. '
        f"Maximum TV was {max(r['maximum_native_replay_tv'] for r in runs.values()):.9f}.\n"]
    (out/'REPORT.md').write_text('\n'.join(lines))
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        primary_endpoint_step=6000,first_token_screen=result['first_token_screen'],resources=budget,
        paired_only=True,cached_first_query_only=True,no_native_or_whole_answer_claim=True)


def main():
    import argparse,time
    p=driver();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs=2,required=True);args=parser.parse_args();p.native.require_slurm(gpu=False)
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only independent report')
    out=p.OUT/f'report_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    frozen=p.snapshot(out);save(out/'request.json',dict(plan=str(args.plan),runs=list(map(str,args.runs)),source_sha256=frozen))
    try:
        result=report(args,out,frozen);need(p.sources()==frozen and p.inherited_sources()==result['inherited_source_sha256'],'Source changed during report')
        result['elapsed_seconds']=time.perf_counter()-started;save(out/'summary.json',result)
        print(__import__('json').dumps(dict(passed=True,completed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

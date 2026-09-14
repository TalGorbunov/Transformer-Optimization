"""Independent cached-head factor-binding audit; no native-generation claim.

Reuses immutable scalar/layout/CE auditing helpers only. Product and additive
fits share all statistics and training data. Cyclic permutation is descriptive.
"""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from collections import Counter,defaultdict
import math
import os
import subprocess
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
from scripts import report_native_identity_join_conditioning_v2 as prior

OWN=('scripts/report_native_identity_join_factor_binding.py','slurm/native_identity_join_factor_binding_report.sbatch')
DIAGNOSTIC_STEPS=(1,2,32,128,300,600)
close=prior.close
expected_order=prior.expected_order
layout_for=prior.layout_for
first_token_criterion=prior.first_token_criterion
fp64_nll=prior.fp64_nll
rms=prior.rms
independent_statistics=prior.independent_statistics
tensor_error=prior.tensor_error
replay_metric=prior.replay_metric
gather=prior.gather
audit_training=prior.audit_training
allocation_rows=prior.allocation_rows


def driver():
    from scripts import diagnose_native_identity_join_factor_binding as module
    return module


def bind(file,digest):
    need(sha(Path(file))==digest,'Changed bound file: '+str(file))


def audit_inputs(torch,plan):
    p=driver();rows=read(plan['rows_file']);scenes=read(plan['scenes_file']);pairs=read(plan['pairs_file']);order=read(plan['order_file'])
    bind(plan['conditioning_plan']['file'],plan['conditioning_plan']['sha256']);conditioning=read(plan['conditioning_plan']['file'])
    bind(plan['uniform_plan']['file'],plan['uniform_plan']['sha256']);parent=read(plan['uniform_plan']['file'])
    need(plan['uniform_plan']==conditioning['uniform_plan'] and plan['oracle_report']==conditioning['oracle_report'], 'Conditioning ancestry differs')
    originals=read(parent['rows_file'])['train'];questions=sorted({r['sample']['question'] for r in originals});expected=[]
    for item in originals:
        s=item['sample'];need(s['split']=='train' and item['cell']=='train_N'+str(s['n_frames']),'Nontraining scene')
        expected.append({**{k:s[k] for k in ('sid','contrast_id','pair_id','variant','n_frames','question','gold','target_ids')},
            'first_token_id':s['target_ids'][0],'question_index':questions.index(s['question'])})
    need(rows==expected==read(conditioning['rows_file']) and scenes==read(parent['scenes_file'])==read(conditioning['scenes_file'])
         and pairs==read(parent['pairs_file']) and order==read(parent['order_file'])==expected_order(pairs), 'Original training inputs/order differ')
    need(len(rows)==len(scenes)==108 and len(questions)==6 and Counter(r['gold'] for r in rows)==Counter({name:12 for name in p.PEOPLE}), 'Balanced training subset differs')
    for r in rows:
        s=scenes[r['sid']];ids=r['target_ids']
        need(s['target_ids']==ids and ids[-1]==151645 and len(ids)==(3 if r['gold'] in ('Sandra','Noah') else 2)
             and s['target_prefixes']==[ids[:i] for i in range(len(ids))] and s['question']==r['question'], 'Full-name/EOS strict prefix differs')
    for key in ('features_file','features_sha256','feature_tensor','stats_file','stats_sha256','stats_tensors'):
        need(plan[key]==conditioning[key], 'Existing statistics/cache were not reused exactly: '+key)
    packet=torch.load(plan['features_file'],map_location='cpu',weights_only=True);states=packet['states'];ids=packet['feature_ids']
    need(states.dtype==torch.float16 and states.shape==(2144,3584) and len(set(ids))==len(ids)==2144
         and p.tensor_info(states)==plan['feature_tensor'] and bool(torch.isfinite(states).all()), 'Frozen feature packet differs')
    index={fid:i for i,fid in enumerate(ids)}
    for pair in pairs:
        a,b=(scenes[sid] for sid in pair['sids'])
        need([a['n_frames'],b['n_frames']]==[8,16] and a['target_ids']==b['target_ids'] and a['question']==b['question'], 'Paired scene mismatch')
        need(all(torch.equal(states[index[x]],states[index[y]]) for x,y in zip(a['global_feature_ids'],b['global_feature_ids'])), 'Paired global states differ')
    stats=torch.load(plan['stats_file'],map_location='cpu',weights_only=True)
    occurrences=[dict(sid=r['sid'],local_index=i,feature_id=fids[0],question_index=r['question_index']) for r in rows
                 for i,fids in enumerate(scenes[r['sid']]['local_feature_ids'])]
    need(stats['schema_version']==1 and stats['questions']==questions and stats['occurrences']==occurrences
         and len(occurrences)==1296 and Counter(x['question_index'] for x in occurrences)==Counter({i:216 for i in range(6)}), 'Statistic occurrence ownership differs')
    need(all(set(x)=={'sid','local_index','feature_id','question_index'} for x in stats['occurrences']), 'Labels entered statistic packet')
    x=rms(torch,states[torch.tensor([index[o['feature_id']] for o in occurrences])]);qi=torch.tensor([o['question_index'] for o in occurrences])
    need(torch.equal(stats['x'],x) and torch.equal(stats['question_indices'],qi), 'Cached RMS statistic input differs')
    for key,value in zip(('means_fp64','means','global_mean_fp64','global_mean','scale_fp64','scale'),independent_statistics(torch,x,qi,6)):
        need(torch.equal(stats[key],value), 'Frozen statistic arithmetic differs: '+key)
    need({k:p.tensor_info(stats[k]) for k in ('x','question_indices','means_fp64','means','global_mean_fp64','global_mean','scale_fp64','scale')}==plan['stats_tensors']
         and stats['scale'].ndim==0 and float(stats['scale'])>0, 'Statistic tensor identities/scale differ')
    return rows,scenes,pairs,order,states,index,stats


def check_weights(torch,weights,initial):
    shapes={'local_bias':(192,),'selection_weight':(96,),'selection_bias':(1,),'query.weight':(96,3584),
        'local.weight':(192,3584),'aggregate_projection.weight':(96,96),'aggregate_projection.bias':(96,),'up.weight':(3584,96)}
    need(set(weights)==set(initial)==set(shapes) and all(v.dtype==torch.float32 and tuple(v.shape)==shapes[k]
        and bool(torch.isfinite(v).all()) for k,v in weights.items()) and sum(v.numel() for v in weights.values())==1385857, 'Factor core parameter table differs')
    need(torch.equal(weights['selection_weight'],initial['selection_weight']) and bool((weights['selection_weight']==0).all())
         and torch.equal(weights['selection_bias'],initial['selection_bias']) and bool((weights['selection_bias']==.5).all()), 'Fixed selector changed')


def permutation_indices(torch,valid,cyclic):
    indices=torch.arange(len(valid),dtype=torch.int64).unsqueeze(1).expand_as(valid).clone()
    if cyclic:
        for j in range(valid.shape[1]):
            actual=torch.where(valid[:,j])[0]
            if actual.numel():indices[actual,j]=actual.roll(1)
    return indices


def functional_core(torch,h,g,valid,mean,scale,w,arm,cyclic=False):
    f=torch.nn.functional;rank=w['query.weight'].shape[0]
    original=rms(torch,h);conditioned=(original-mean.unsqueeze(0))/scale;q=f.linear(rms(torch,g),w['query.weight'])
    local=f.linear(conditioned,w['local.weight']);a=local[...,:rank]+q.unsqueeze(0)+w['local_bias'][:rank]
    b=local[...,rank:]+q.unsqueeze(0)+w['local_bias'][rank:];u=torch.tanh(a);v=torch.tanh(b)
    indices=permutation_indices(torch,valid,cyclic);used=v.gather(0,indices.unsqueeze(-1).expand_as(v))
    need(arm in ('product','additive'),'Unknown factor arm');payload=u*used if arm=='product' else .5*(u+used)
    scores=f.linear(payload,w['selection_weight'].unsqueeze(0)).squeeze(-1)+w['selection_bias']
    gates=torch.sigmoid(4*(scores-.5))*valid.float();messages=gates.unsqueeze(-1)*payload;aggregate=messages.sum(0)
    preactivation=f.linear(aggregate,w['aggregate_projection.weight'],w['aggregate_projection.bias'])+q
    delta=f.linear(f.silu(preactivation),w['up.weight'])
    return dict(original_local_rms_input=original,conditioned_local_rms_input=conditioned,query=q,
        factor_a=u,factor_b=v,factor_b_used=used,factor_preactivation_a=a,factor_preactivation_b=b,
        factor_permutation_indices=indices,payload=payload,scores=scores,gates=gates,messages=messages,
        aggregate=aggregate,preactivation=preactivation,delta=delta)


def additive_invariance(torch,cap,valid):
    a,b=cap['factor_a'],cap['factor_b'];indices=permutation_indices(torch,valid,True)
    shifted=b.gather(0,indices.unsqueeze(-1).expand_as(b));gate=valid.double().unsqueeze(-1)*.5
    paired=(gate*.5*(a.double()+b.double())).sum(0);permuted=(gate*.5*(a.double()+shifted.double())).sum(0)
    error=(paired-permuted).abs();passed=bool((error<=1e-10+1e-10*paired.abs()).all())
    fp32=(valid.float().unsqueeze(-1)*.5*(.5*(a+shifted))).sum(0)
    return dict(passed=passed,atol=1e-10,rtol=1e-10,fp64_max_absolute=float(error.max()),
        fp32_reassociated_max_absolute=float((fp32-cap['aggregate']).double().abs().max()),
        fp32_roundoff_descriptive_only=True,no_extra_head=True)


def audit_capture(torch,out,record,arm,plan,rows,scenes,states,index,stats,initial,final,logs,norm,head):
    p=driver();phase=record['phase'];step=record['step'];training=phase=='training';cyclic=phase=='permutation'
    need(phase in ('training','evaluation','permutation') and record['arm']==arm and (not cyclic or arm=='product'), 'Invalid capture phase/arm')
    bind(record['file'],record['sha256']);bind(record['weights_file'],record['weights_sha256'])
    cap=torch.load(record['file'],map_location='cpu',weights_only=True);wp=torch.load(record['weights_file'],map_location='cpu',weights_only=True);w=wp['branch']
    check_weights(torch,w,initial)
    if training:
        need(wp['arm']==arm and step in DIAGNOSTIC_STEPS and wp['step']==record['weights_step']==step-1
             and wp['optimizer_step']==step and wp['position']=='before_update' and wp['source_sha256']==plan['source_sha256'], 'Pre-update capture weight binding differs')
        sids=logs[step-1]['sids']
        if step==1:need(all(torch.equal(w[k],initial[k]) for k in initial),'First captured state differs from fresh initialization')
    else:
        need(1<=step<=7 and wp['step']==record['weights_step']==600 and all(torch.equal(w[k],final[k]) for k in final), 'Final capture does not use fixed endpoint')
        sids=[r['sid'] for r in rows[(step-1)*16:step*16]]
    need(cap['schema_version']==1 and cap['arm']==arm and cap['phase']==phase and cap['step']==step
         and cap['sids']==record['sids']==sids and cap['pairing']==record['pairing']==('cyclic' if cyclic else 'paired'), 'Capture scene/phase ownership differs')
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
    invariant=additive_invariance(torch,observed,valid) if arm=='additive' and not training else None
    result=dict(arm=arm,phase=phase,step=step,sids=sids,capture_file=record['file'],capture_sha256=record['sha256'],
        weights_file=record['weights_file'],weights_sha256=record['weights_sha256'],functional_core=metrics,
        native_head=head_metrics,normalized_max_absolute=normal_error,nll=nll_metrics,fp16_cast_before_add_exact=True,
        factor_pairing='cyclic' if cyclic else 'paired',additive_invariance=invariant,captured_rows=len(g),
        head_rows=0 if training else len(g),cpu_norm_calls=0 if training else 1,cpu_head_calls=0 if training else 1,
        passed=all(x['passed'] for x in metrics.values()) and all(x['passed'] for x in head_metrics+nll_metrics) and (invariant is None or invariant['passed']))
    save(out/f'{arm}_{phase}_{step:04d}_capture_audit.json',result)
    need(result['passed'],'Factor/core/native/NLL audit failed; metrics and raw evidence retained')
    small=None if training else {k:observed[k] for k in ('factor_a','factor_b','query','aggregate','delta')}
    return result,cap['logits'],small


def self_test(torch):
    valid=torch.tensor([[True,True],[False,True],[True,False]])
    indices=permutation_indices(torch,valid,True)
    need(torch.equal(indices,torch.tensor([[2,1],[1,0],[0,2]])) and torch.equal(permutation_indices(torch,valid,False),torch.arange(3)[:,None].expand(3,2)), 'Ragged cyclic ownership fixture failed')
    a=torch.tensor([[[1.,0.]],[[0.,1.]]]);b=a.clone();v=torch.ones((2,1),dtype=torch.bool)
    shifted=b.roll(1,0);product=(a*b).sum(0);changed=(a*shifted).sum(0)
    need(torch.equal(product,torch.ones((1,2))) and bool((changed==0).all()), 'Within-item product binding fixture failed')
    additive=dict(factor_a=a,factor_b=b,aggregate=(.5*(.5*(a+b))).sum(0))
    need(additive_invariance(torch,additive,v)['passed'],'Additive permutation invariance fixture failed')
    h=torch.tensor([[[.2,.7]],[[.4,-.8]]],dtype=torch.float16);g=torch.tensor([[.1,.4]],dtype=torch.float16)
    w={'query.weight':torch.eye(2),'local.weight':torch.cat((torch.eye(2),torch.eye(2).flip(0))),
        'local_bias':torch.zeros(4),'selection_weight':torch.zeros(2),'selection_bias':torch.tensor([.5]),
        'aggregate_projection.weight':torch.eye(2),'aggregate_projection.bias':torch.zeros(2),'up.weight':torch.eye(2)}
    valid=torch.tensor([[True],[False]]);mean=torch.tensor([[.3,-.2]]);scale=torch.tensor(2.)
    cap=functional_core(torch,h,g,valid,mean,scale,w,'product')
    need(torch.equal(cap['query'],rms(torch,g)) and torch.equal(cap['conditioned_local_rms_input'],(rms(torch,h)-mean)/scale)
         and bool((cap['messages'][1]==0).all()) and torch.equal(cap['factor_b_used'],cap['factor_b']), 'Local-only transform/query/padding fixture failed')
    rows=[dict(sid=f'{f}_{i}_{n}',contrast_id=f,variant=i,n_frames=n,first_token_correct=True) for f in range(18) for i in range(3) for n in (8,16)]
    for j in (0,6,12):rows[j]['first_token_correct']=False
    need(not first_token_criterion(rows)['passed'],'Complete-family screen fixture failed')
    return dict(passed=True,groups=5,ragged_cyclic_indices=True,product_pairing_sensitivity=True,
        additive_invariance=True,conditioned_global_and_padding=True,complete_family_screen=True)


def audit_run(torch,out,directory,plan_path,plan,frozen,rows,scenes,order,states,index,stats,initial,norm,head):
    p=driver();directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json');arm=config['arm']
    need(arm in ('product','additive') and directory.parent==p.OUT and directory.name==config['run_id']==f'run_{arm}_{config["slurm_job_id"]}'
         and config['protocol']==p.PROTOCOL and config['policy']==p.POLICY and config['seed']==24
         and config['source_sha256']==frozen==plan['source_sha256'] and all(summary.get(k)==v for k,v in config.items())
         and Path(config['plan_file']).resolve()==Path(plan_path).resolve() and config['plan_sha256']==sha(plan_path),'Run/config/source/plan join differs')
    for name,h in frozen.items():bind(directory/'source'/name.replace('/','_'),h)
    for flag in ('passed','completed','computational_integrity_passed','checkpoint_roundtrip_passed','native_weights_unchanged',
                 'statistics_frozen','cached_first_query_only','no_pretrained_backbone_loaded','no_dev_or_test','no_native_or_whole_answer_claim'):
        need(summary[flag] is True,'Run gate failed: '+flag)
    need(summary['phase']=='run' and summary['steps']==600 and summary['training_head_rows']==21334
         and summary['final_head_rows']==108 and summary['total_head_rows']==(21550 if arm=='product' else 21442),'Run endpoint/call rows differ')
    need(config['hardware']['gpu']=='NVIDIA B200' and config['hardware']['matmul_allow_tf32'] is False
         and config['hardware']['torch_version']==str(torch.__version__),'Native hardware/precision source differs')
    for key in ('native_identity','native_identity_sha256','native_module_identity','native_model_file','native_model_sha256',
                'stats_file','stats_sha256','stats_tensors','oracle_report','uniform_plan','conditioning_plan','conditioning_report'):
        need(config[key]==plan[key],'Fixed plan/native/statistics binding differs: '+key)
    need(config['fresh_shared_initialization'] is True and plan['fresh_shared_initialization'] is True and config['initialized']==plan['initial_state'] and config['initial_checkpoint_sha256']==plan['initial_sha256'],'Shared original initialization differs')
    bind(config['initial_checkpoint'],config['initial_checkpoint_sha256'])
    actual_initial=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=True)
    need(all(torch.equal(actual_initial['branch'][k],initial[k]) for k in initial),'Run actual initial tensors differ')
    mask=plan['trainability_mask'];trainable=[k for k in initial if k not in ('selection_weight','selection_bias')]
    need(mask==dict(trainable_parameter_names=trainable,frozen_selector_parameter_names=['selection_weight','selection_bias'],
        effective_trainable_parameters=1385760,retained_state_parameters=1385857)
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
         and endpoint['native_weights_unchanged'] is True and endpoint['passed'] is True and endpoint['hooks_removed'] is True
         and endpoint['interaction']==summary['interaction']==arm and endpoint['pairing_restored'] is summary['pairing_restored'] is True
         and endpoint['factor_derangement'] is summary['factor_derangement'] is False,'Final deployed/frozen endpoint differs')
    need(roundtrip['passed'] is True and roundtrip['initial']==plan['initial_state'] and roundtrip['final']==roundtrip['reloaded']==table
         and roundtrip['step']==600 and roundtrip['config_sha256']==sha(directory/'config.json'),'Actual reset/restricted reload differs')
    for value in (endpoint,roundtrip):
        need(value['checkpoint']==str(checkpoint_file) and value['checkpoint_sha256']==summary['checkpoint_sha256'],'Endpoint checkpoint binding differs')
    need(all(endpoint[k]==plan[k] for k in ('stats_file','stats_sha256','stats_tensors'))
         and endpoint['working_statistics_before']==endpoint['working_statistics_after']=={k:plan['stats_tensors'][k] for k in ('means','global_mean','scale')}
         and endpoint['statistics_versions_unchanged'] is True,'Actual deployed fixed statistics changed')
    denominators={fid:float(states[index[fid]].float().square().sum()+1e-6) for scene in scenes.values() for fid in scene['global_feature_ids']}
    logs,expected_calls,training=audit_training(directory,summary,scenes,order,denominators)
    training['permutation_head_rows']=108 if arm=='product' else 0;training['total_head_rows']=21550 if arm=='product' else 21442
    need([len(r['target_ids']) for r in logs]==plan['head_rows_by_update'] and close(summary['training_seconds'],sum(r['seconds'] for r in logs)), 'Training row/timing totals differ')
    total_calls=614 if arm=='product' else 607
    if arm=='product':
        expected_calls += [dict(row,phase='permutation') for row in expected_calls[-14:]]
    counters=dict(core=total_calls,conditioning=total_calls,norm=total_calls,head=total_calls,vlm=0,vision=0);calls=read(summary['calls_file'])
    need(calls==expected_calls and summary['counters']==endpoint['counters']==counters
         and read(directory/'final_counters.json')==dict(counters=counters,recorded_calls=calls,partial_outputs_retained=True), 'Actual work inventory differs')
    records=read(summary['captures_file']);expected_positions=[('training',step) for step in DIAGNOSTIC_STEPS]+[('evaluation',step) for step in range(1,8)]
    if arm=='product':expected_positions += [('permutation',step) for step in range(1,8)]
    need([(r['phase'],r['step']) for r in records]==expected_positions and all(r['arm']==arm for r in records), 'Captured batch coverage/arm differs')
    need(sorted(f.name for f in checkpoint_file.parent.glob('*.pt'))==sorted(['initial.pt','final.pt']+[f'training_core_{step:04d}.pt' for step in DIAGNOSTIC_STEPS]),'Checkpoint inventory differs')
    need(sorted(f.name for f in directory.glob('progress_*.json'))==[f'progress_{step:04d}.json' for step in range(100,601,100)],'Immutable progress coverage differs')
    for step in range(100,601,100):
        need(read(directory/f'progress_{step:04d}.json')==dict(step=step,training=logs[:step],captures=[r for r in records if r['phase']=='training' and r['step']<=step],
            counters=dict(core=step,conditioning=step,norm=step,head=step,vlm=0,vision=0)),'Immutable progress prefix changed')
    phases=('evaluation','permutation') if arm=='product' else ('evaluation',)
    if arm=='additive':need(not any(k.startswith('permutation_') for k in summary),'Additive must not run extra native permutation calls')
    else:need(summary['permutation_diagnostic']==dict(rule=p.POLICY['permutation'],contexts=108,head_calls=7,head_rows=108,
        descriptive_only=True,no_fit_or_checkpoint_selection=True), 'Prespecified permutation diagnostic differs')
    packets={};gpu_nll={phase:[] for phase in phases};phase_cursor={phase:0 for phase in phases};small={phase:{} for phase in phases}
    for phase in phases:
        prefix='' if phase=='evaluation' else 'permutation_'
        for key in ('raw','predictions'):bind(summary[prefix+key+'_file'],summary[prefix+key+'_sha256'])
        packet=torch.load(summary[prefix+'raw_file'],map_location='cpu',weights_only=True);logits=packet['logits']
        need(packet['schema_version']==1 and packet['sids']==[r['sid'] for r in rows] and logits.dtype==torch.float16 and logits.shape==(108,152064)
             and bool(torch.isfinite(logits).all()) and p.tensor_info(logits)==summary[prefix+'raw_tensor']
             and packet['batches']==summary[prefix+'raw_batches'] and len(packet['batches'])==7, 'Final raw output inventory differs')
        filename='final_logits.pt' if phase=='evaluation' else 'permutation_logits.pt'
        need(Path(summary[prefix+'raw_file'])==Path(config['data_directory'])/filename,'Raw storage root differs');packets[phase]=packet
    capture_audits=[]
    with torch.no_grad():
        for record in records:
            phase=record['phase'];step=record['step'];need(record['arm']==arm,'Capture arm differs')
            expected_weight=checkpoint_file.parent/f'training_core_{step:04d}.pt' if phase=='training' else checkpoint_file
            need(Path(record['weights_file'])==expected_weight and Path(record['file'])==Path(config['data_directory'])/f'{phase}_capture_{step:04d}.pt','Capture/checkpoint path differs')
            audit,captured_logits,values=audit_capture(torch,out,record,arm,plan,rows,scenes,states,index,stats,initial,final,logs,norm,head)
            capture_audits.append(audit)
            if phase!='training':
                count=len(record['sids']);batch=packets[phase]['batches'][step-1];cursor=phase_cursor[phase]
                need(batch['capture_file']==record['file'] and batch['capture_sha256']==record['sha256'] and batch['sids']==record['sids']
                     and batch['logits']==record['tensors']['logits'] and len(batch['gpu_fp32_nll'])==count
                     and torch.equal(captured_logits,packets[phase]['logits'][cursor:cursor+count]), 'Final batch raw/capture ownership differs')
                gpu_nll[phase]+=batch['gpu_fp32_nll'];phase_cursor[phase]+=count;small[phase][step]=values
    need(all(v==108 for v in phase_cursor.values()),'Final captured row coverage differs')
    outcomes={};phase_metrics={}
    for phase in phases:
        prefix='' if phase=='evaluation' else 'permutation_';logits=packets[phase]['logits'];ids=logits.argmax(-1).tolist()
        nll=fp64_nll(torch,logits,[r['first_token_id'] for r in rows]);saved=read(summary[prefix+'predictions_file'])
        need(len(saved)==108,'Prediction denominator differs')
        rescored=[dict(r,argmax_id=ids[i],first_token_correct=ids[i]==r['first_token_id'],nll=float(nll[i]),gpu_fp32_nll=gpu_nll[phase][i]) for i,r in enumerate(rows)]
        precision=[dict(sid=r['sid'],fp64_nll=r['nll'],gpu_fp32_nll=r['gpu_fp32_nll'],absolute_difference=abs(r['nll']-r['gpu_fp32_nll']),
            passed=close(r['nll'],r['gpu_fp32_nll'])) for r in rescored]
        save(out/f'{arm}_{phase}_outcomes.json',rescored);save(out/f'{arm}_{phase}_nll_precision.json',precision)
        need(all(set(a)==set(b) and all(a[k]==b[k] for k in a if k!='nll') and close(a['nll'],b['nll']) for a,b in zip(rescored,saved))
             and all(r['passed'] for r in precision), 'Independent final raw argmax/NLL/metadata rescore differs')
        outcomes[phase]=rescored
        counts=first_token_criterion(rescored)
        phase_metrics[phase]={k:counts[k] for k in ('correct','contexts','complete_families','families')}
        phase_metrics[phase]['mean_nll']=sum(r['nll'] for r in rescored)/108
    criterion=first_token_criterion(outcomes['evaluation']);need(criterion==summary['first_token_fit'],'Fixed paired first-token screen differs')
    permutation=None
    if arm=='product':
        changes=[]
        for step in range(1,8):
            before,after=small['evaluation'][step],small['permutation'][step]
            need(all(torch.equal(before[k],after[k]) for k in ('factor_a','factor_b','query')),'Product intervention changed original factor or query values')
            for j,row in enumerate(rows[(step-1)*16:step*16]):
                changes.append(dict(sid=row['sid'],aggregate_change_norm=float((after['aggregate'][j]-before['aggregate'][j]).double().norm()),
                    residual_change_norm=float((after['delta'][j]-before['delta'][j]).double().norm())))
        transitions=Counter((a['first_token_correct'],b['first_token_correct']) for a,b in zip(outcomes['evaluation'],outcomes['permutation']))
        permutation=dict(paired=phase_metrics['evaluation'],permuted=phase_metrics['permutation'],
            paired_to_permuted_transitions={f'{int(a)}_to_{int(b)}':transitions[a,b] for a in (False,True) for b in (False,True)},
            rows=changes,original_factors_and_query_exact=True,rule=p.POLICY['permutation'],descriptive_only=True,
            no_semantic_factor_or_disentanglement_claim=True,no_effect_on_first_token_fit=True)
        save(out/'product_permutation_comparison.json',permutation)
    strata=[]
    for phase,values in outcomes.items():
        for key in ('gold','question','n_frames','contrast_id'):
            groups=defaultdict(list)
            for row in values:groups[row[key]].append(row)
            for label,group in sorted(groups.items(),key=lambda x:str(x[0])):
                strata.append(dict(labels=dict(phase=phase,**{key:label}),metrics=dict(correct=sum(r['first_token_correct'] for r in group),contexts=len(group),
                    mean_nll=sum(r['nll'] for r in group)/len(group),all_correct=all(r['first_token_correct'] for r in group))))
    save(out/f'{arm}_strata.json',strata)
    return dict(passed=True,completed=True,arm=arm,slurm_job_id=config['slurm_job_id'],run_directory=str(directory),run_summary_sha256=sha(directory/'summary.json'),
        checkpoint=str(checkpoint_file),checkpoint_sha256=summary['checkpoint_sha256'],selected_parameter_sha256=object_sha(table),
        first_token_fit=criterion,phase_metrics=phase_metrics,training=training,counters=counters,capture_audits=capture_audits,
        cpu_head_calls=14 if arm=='product' else 7,cpu_norm_calls=14 if arm=='product' else 7,cpu_head_rows=sum(r['head_rows'] for r in capture_audits),
        maximum_native_replay_tv=max(m['full_vocabulary_tv'] for r in capture_audits for m in r['native_head']),
        outcomes={phase:dict(file=str(out/f'{arm}_{phase}_outcomes.json'),sha256=sha(out/f'{arm}_{phase}_outcomes.json')) for phase in phases},
        strata_file=str(out/f'{arm}_strata.json'),strata_sha256=sha(out/f'{arm}_strata.json'),
        permutation_diagnostic=permutation,additive_invariance_checks=sum(r['additive_invariance'] is not None for r in capture_audits))


def resources(out,runs):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'all_user_sacct.psv';file.write_text(raw)
    job_arms={name:arm for arm,name in driver().RUN_JOBS.items()};names=set(job_arms);records=allocation_rows(raw,names)
    need(len(records)==2 and {r['name'] for r in records}==names,'Exactly one attempt per factor arm required; all failed/zero allocations retained')
    for row in records:
        arm=job_arms[row['name']]
        need(row['job_id']==runs[arm]['slurm_job_id'] and row['partition']=='gpu' and row['state']=='COMPLETED'
             and row['exit_code']=='0:0' and row['gpus']==1 and row['seconds']<=90,'Factor-binding allocation identity/cap differs')
    from datetime import datetime
    events=[]
    for row in records:
        a,b=map(datetime.fromisoformat,(row['start'],row['end']));need(b>=a,'Invalid allocation interval')
        if b>a:events.extend(((a,1),(b,-1)))
    current=peak=0
    for _,d in sorted(events):current+=d;need(current>=0,'Invalid allocation overlap');peak=max(peak,current)
    total=sum(r['gpu_seconds'] for r in records);need(current==0 and peak<=2 and total<=180,'Factor-binding campaign cap exceeded')
    return dict(passed=True,jobs=records,allocated_gpu_seconds=total,maximum_concurrent_gpus=peak,
        failed_and_zero_allocations_retained=True,file=str(file),sha256=sha(file),command=command)



def report(args,out,frozen):
    import torch
    torch.set_num_threads(4);p=driver();plan=p.verify_plan(args.plan,ancestors=True)
    _,trigger=p.conditional_gate({});need(trigger==plan['conditioning_report'],'Factor-binding trigger changed')
    need(frozen==plan['source_sha256']==p.sources() and len(args.runs)==2,'Source or two-arm inventory differs')
    rows,scenes,pairs,order,states,index,stats=audit_inputs(torch,plan)
    need(read(plan['stats_inventory_file'])==dict(questions=stats['questions'],occurrences=stats['occurrences'],training_sids=[r['sid'] for r in rows],
        original_cached_empty_prefix_only=True,no_fitted_captures=True,no_local_labels=True),'Statistic inventory provenance differs')
    need(p.oracle.native_module_identity(torch,plan['native_identity'])==plan['native_module_identity'],'Actual installed native norm/head source differs')
    initial_packet=torch.load(plan['initial_file'],map_location='cpu',weights_only=True);initial=initial_packet['branch']
    need(set(initial_packet)=={'seed','branch'} and initial_packet['seed']==24 and {k:p.tensor_info(v) for k,v in initial.items()}==plan['initial_state']
         and bool((initial['up.weight']==0).all()), 'Fresh shared initialization tensor table differs')
    check_weights(torch,initial,initial)
    from gnnformer.parallel_local_factor_binding import ParallelLocalFactorBinding
    for arm in ('product','additive'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(24);unfitted=ParallelLocalFactorBinding(interaction=arm)
        need(p.base.v7.state_info(unfitted)==plan['initial_state'],'Independent matched seed24 initialization differs');del unfitted
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=p.oracle.native_modules(torch,packet,plan['native_identity'],'cpu');del packet
    native_before=dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight));runs={}
    for directory in args.runs:
        value=audit_run(torch,out,directory,args.plan,plan,frozen,rows,scenes,order,states,index,stats,initial,norm,head)
        need(value['arm'] not in runs,'Duplicate factor arm');runs[value['arm']]=value
    need(set(runs)=={'product','additive'} and native_before==dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight))
         and norm.weight.grad is head.weight.grad is None and not norm.weight.requires_grad and not head.weight.requires_grad, 'Two-arm frozen native audit incomplete')
    need(sum(r['cpu_head_calls'] for r in runs.values())==21 and sum(r['cpu_head_rows'] for r in runs.values())==324
         and runs['additive']['additive_invariance_checks']==7 and runs['product']['additive_invariance_checks']==0, 'Independent replay/invariance coverage differs')
    budget=resources(out,runs);outcomes={arm:read(runs[arm]['outcomes']['evaluation']['file']) for arm in runs}
    transitions=Counter((a['first_token_correct'],b['first_token_correct']) for a,b in zip(outcomes['additive'],outcomes['product']))
    comparison=dict(scope='paired cached training first queries; same fresh initialization/order/statistics and 96-coordinate output',contexts=108,
        additive_to_product_transitions={f'{int(a)}_to_{int(b)}':transitions[a,b] for a in (False,True) for b in (False,True)},
        product_minus_additive_correct=runs['product']['first_token_fit']['correct']-runs['additive']['first_token_fit']['correct'],
        all_comparisons_descriptive=True,no_factor_semantics_or_expressiveness_claim=True)
    result=dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),conditioning_report=trigger,conditioning_plan=plan['conditioning_plan'],
        oracle_report=plan['oracle_report'],uniform_plan=plan['uniform_plan'],stats_file=plan['stats_file'],stats_sha256=plan['stats_sha256'],
        stats_tensors=plan['stats_tensors'],runs=runs,comparison=comparison,resources=budget,
        first_token_screen={arm:runs[arm]['first_token_fit'] for arm in runs},cpu_head_calls=21,cpu_norm_calls=21,cpu_head_rows=324,
        gpu_head_calls=1221,gpu_head_rows=42992,vlm_calls=0,vision_calls=0,
        cached_first_query_only=True,no_native_or_whole_answer_claim=True,no_generalization_or_reasoning_claim=True,
        no_new_fit_or_model_forward=True,all_outcomes_retained=True)
    save(out/'analysis.json',result)
    lines=['# Matched factor-binding diagnostic','',
        'Independent computation and provenance audits passed. These are cached training first tokens, not native whole answers.','',
        '| Interaction | Correct /108 | Complete families /18 | Fixed paired screen |','|---|---:|---:|---|']
    for arm in ('product','additive'):
        c=runs[arm]['first_token_fit'];lines.append(f"| {arm} | {c['correct']} | {c['complete_families']} | {'PASS' if c['passed'] else 'FAIL'} |")
    perm=runs['product']['permutation_diagnostic']['permuted']
    lines+=['',f"Product after the single fixed cyclic factor permutation: {perm['correct']}/108 first tokens and {perm['complete_families']}/18 complete families. "
        'This intervention is descriptive and does not change either screen. Original factors and query were unchanged; only their pairing changed. '
        'Additive pooled invariance passed all7 CPU checks; FP32 reassociation differences are reported separately.','',
        'Both arms used identical fresh seed24 tensors,600 full-name-plus-EOS CE updates and the existing global conditioning statistics. '
        'Product/additive communicated the same96 coordinates. These learned factors have no assigned person/room roles. '
        'A passing screen permits preparation of separately audited native evaluation of that unchanged endpoint, without refitting.','',
        f"Allocated GPU cost: {budget['allocated_gpu_seconds']} seconds. GPU work:1221 norm/head batches and42992 rows; zero VLM/vision calls. "
        'Independent CPU head replay:21 batches/324 rows, fixed TV<=.02 and exact argmax.\n']
    (out/'REPORT.md').write_text('\n'.join(lines))
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),first_token_screen=result['first_token_screen'],
        resources=budget,cached_first_query_only=True,no_native_or_whole_answer_claim=True)


def main():
    import argparse,time
    p=driver();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs=2,required=True);args=parser.parse_args();p.native.require_slurm(gpu=False)
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

"""Independent CPU release and report for the conditional fixed-uniform control.

Reuses the immutable native outcome auditor; no test trajectory or model call.
"""
from pathlib import Path
import math,random
from collections import Counter
from scripts import report_native_identity_join_learned_v2 as old
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha,MODEL

def pilot():
    from scripts import diagnose_native_identity_join_uniform as module
    return module


def bind(path,digest):
    need(sha(Path(path))==digest,'Bound pilot artifact changed: '+str(path))


def expected_order(pairs):
    need(len(pairs)==54,'Training pair count differs')
    rng=random.Random(24);rows=[];cycle=0
    while len(rows)<4800:
        cycle+=1;slots=list(range(54));rng.shuffle(slots)
        for i in slots:rows.append(dict(cycle=cycle,pair_slot=i,pair_id=pairs[i]['pair_id'],sids=pairs[i]['sids']))
    return rows[:4800]


def rate(step):
    return .001*step/50 if step<=50 else .00001+(.001-.00001)*(1+math.cos(math.pi*(step-50)/550))/2


def check_data(plan):
    rows=read(plan['rows_file']);pairs=read(plan['pairs_file']);scenes=read(plan['scenes_file'])
    need(set(rows)=={'train','dev'} and len(rows['train'])==108 and len(rows['dev'])==54,'Diagnostic data inventory differs')
    training={x['sample']['sid']:x['sample'] for x in rows['train']};dev={x['sample']['sid']:x['sample'] for x in rows['dev']}
    need(len(training)==108 and len(dev)==54 and not set(training)&set(dev) and set(scenes)==set(training),'Training/dev ownership differs')
    need(all(x['cell']=='train_N'+str(x['sample']['n_frames']) and x['sample']['split']=='train' for x in rows['train'])
         and all(x['cell']=='dev_N16' and x['sample']['split']=='dev' and x['sample']['n_frames']==16 for x in rows['dev']),'Test or wrong-length scene in diagnostic')
    need(Counter(x['gold'] for x in training.values())==Counter({x:12 for x in old.NAMES})
         and Counter(x['gold'] for x in dev.values())==Counter({x:6 for x in old.NAMES}),'Name balance differs')
    grouping={}
    for sample in training.values():grouping.setdefault(sample['pair_id'],[]).append(sample)
    need(len(pairs)==len(grouping)==54 and len({p['pair_id'] for p in pairs})==54,'Pair coverage differs')
    for p in pairs:
        group=sorted(grouping[p['pair_id']],key=lambda r:r['n_frames'])
        need([x['sid'] for x in group]==p['sids'] and [x['n_frames'] for x in group]==[8,16]
             and all(group[0][k]==group[1][k] for k in ('gold','target_ids','contrast_id','variant','question')),'Paired target/context alignment differs')
        for sample in group:
            cached=scenes[sample['sid']]
            need(cached['target_ids']==sample['target_ids'] and cached['n_frames']==sample['n_frames'],'Cached target ownership differs')
    order=expected_order(pairs);need(read(plan['order_file'])==order and object_sha(order)==plan['order_object_sha256'],'Independent pair schedule differs')
    return rows,scenes,order


def config_and_endpoint(torch,directory,plan_path,profile):
    from scripts.stage_native_vision_v10_features import tensor_info
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    p=pilot();directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json');plan=p.verify_plan(plan_path,ancestors=False)
    mode,coefficient=p.arm_config(config['arm']);steps=32 if profile else 600
    need(directory.parent==p.OUT and directory.name==config['run_id'] and config['profile'] is profile and config['seed']==24
         and config['condition']=='uniform_half_ce' and config['selection_mode']==mode and config['consistency_coefficient']==coefficient
         and config['policy']==p.POLICY and config['protocol']==p.PROTOCOL,'Run identity/policy differs')
    need(all(summary.get(k)==v for k,v in config.items()) and summary['source_sha256']==p.sources()==plan['source_sha256']
         and Path(config['plan_file']).resolve()==Path(plan_path).resolve() and config['plan_sha256']==sha(plan_path),'Config/source/plan join differs')
    for n,h in config['source_sha256'].items():bind(directory/'source'/n.replace('/','_'),h)
    for k in ('passed','completed','computational_integrity_passed','checkpoint_roundtrip_passed','final_endpoint_unchanged','frozen_backbone_gradient_state_preserved'):
        need(summary[k] is True,'Incomplete run gate: '+k)
    need(config['native_identity']==plan['native_identity'] and config['native_identity_sha256']==plan['native_identity_sha256'],'Different native model')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);core=ParallelLocalLearnedSelection(mode=mode)
    expected={k:tensor_info(v) for k,v in core.state_dict().items()}
    need(config['initialized']==expected and config['initialized_sha256']==object_sha(expected) and plan['initial_state']==expected,'Initialization differs')
    bind(config['initial_checkpoint'],config['initial_checkpoint_sha256']);initial=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=True)
    need(initial['seed']==24 and {k:tensor_info(v) for k,v in initial['branch'].items()}==expected,'Actual initial checkpoint differs')
    bind(summary['final_endpoint_file'],summary['final_endpoint_sha256']);endpoint=read(summary['final_endpoint_file']);bind(summary['checkpoint'],summary['checkpoint_sha256'])
    packet=torch.load(summary['checkpoint'],map_location='cpu',weights_only=True)
    need(set(packet)=={'branch','step','config'} and packet['step']==summary['steps']==steps and packet['config']==config,'Final saved checkpoint differs')
    file=Path(summary['checkpoint']);need(file.parent==Path(config['checkpoint_directory'])==p.CKPT/config['run_id']
        and file.name==('profile.pt' if profile else 'final.pt'),'Checkpoint storage/endpoint differs')
    weights=packet['branch'];table={k:tensor_info(v) for k,v in weights.items()}
    need(set(table)==set(expected) and sum(v.numel() for v in weights.values())==1041697
         and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in weights.values()),'Malformed fitted core')
    need(endpoint['step']==steps and endpoint['checkpoint']==summary['checkpoint'] and endpoint['checkpoint_sha256']==summary['checkpoint_sha256']
         and endpoint['before_evaluation']==endpoint['after_evaluation']==table and endpoint['parameter_sha256']==summary['selected_parameter_sha256']==object_sha(table)
         and endpoint['native_weight_identity_before']==endpoint['native_weight_identity_after']==config['native_weight_identity'],'Final deployed endpoint differs')
    need(all(endpoint[k] is True for k in ('checkpoint_matches_deployed_endpoint','core_versions_unchanged','native_versions_unchanged')),'Endpoint changed during evaluation')
    bind(config['native_model_file'],config['native_model_sha256']);native_weights=torch.load(config['native_model_file'],map_location='cpu',weights_only=True)
    need(Path(config['native_model_file']).parent==p.CKPT/config['run_id']
         and tensor_info(native_weights['norm_weight'])==config['native_identity']['norm_weight']
         and tensor_info(native_weights['head_weight'])==config['native_identity']['head_weight'],'Native norm/head tensor copy differs')
    return directory,config,summary,plan,weights


def audit_training(torch,directory,config,summary,plan):
    p=pilot();_,scenes,order=check_data(plan);steps=32 if config['profile'] else 600;alpha=config['consistency_coefficient']
    bind(config['presentations_file'],config['presentations_sha256']);need(read(config['presentations_file'])==order and config['order_sha256']==object_sha(order),'Actual presentations differ')
    bind(plan['features_file'],plan['features_sha256']);features=torch.load(plan['features_file'],map_location='cpu',weights_only=True)
    ids=features['feature_ids'];states=features['states'];index={f:i for i,f in enumerate(ids)}
    need(len(index)==len(ids)==len(states) and p.base.v7.tensor_info(states)==plan['feature_tensor'] and states.dtype==torch.float16,'Actual selected features differ')
    denominators={f:float(states[index[f]].float().square().sum()+1e-6) for scene in scenes.values() for f in scene['global_feature_ids']}
    for pair in read(plan['pairs_file']):
        left,right=(scenes[sid] for sid in pair['sids'])
        need(all(torch.equal(states[index[a]],states[index[b]]) for a,b in zip(left['global_feature_ids'],right['global_feature_ids'])),'Paired global states differ')
    del features,states
    bind(summary['training_file'],summary['training_sha256']);logs=read(summary['training_file']);need(len(logs)==steps,'Update coverage differs');targets_total=0
    for step,row in enumerate(logs,1):
        batch=order[(step-1)*8:step*8];sids=[sid for pair in batch for sid in pair['sids']];targets=[scenes[sid]['target_ids'] for sid in sids]
        lengths=list(map(len,targets));offsets=[0]
        for length in lengths:offsets.append(offsets[-1]+length)
        flat=[t for ids_ in targets for t in ids_];prefixes=[ids_[:i] for ids_ in targets for i in range(len(ids_))]
        need(row['step']==step and row['sids']==sids and row['pair_ids']==[x['pair_id'] for x in batch]
             and row['cycles']==[x['cycle'] for x in batch] and row['target_ids']==flat and row['prefix_ids']==prefixes
             and old.close(row['lr'],rate(step),atol=1e-12,rtol=1e-12),'Training order/target/schedule differs')
        need(row['scene_lengths']==lengths and row['scene_offsets']==offsets and row['pair_lengths']==lengths[::2],'Loss sequence segmentation differs')
        ce=row['position_losses']['ce'];reg=row['position_losses']['consistency'];need(len(ce)==len(flat) and len(reg)==sum(lengths[::2]) and all(math.isfinite(x) and x>=0 for x in ce+reg),'Invalid position losses')
        means=[sum(ce[a:b])/(b-a) for a,b in zip(offsets,offsets[1:])];regularizers=[];cursor=0
        for length in lengths[::2]:regularizers.append(sum(reg[cursor:cursor+length])/length);cursor+=length
        need(len(row['per_scene_ce'])==16 and len(row['per_pair_consistency'])==8 and all(old.close(a,b) for a,b in zip(means,row['per_scene_ce']))
             and all(old.close(a,b) for a,b in zip(regularizers,row['per_pair_consistency'])) and old.close(sum(means)/16,row['ce_loss'])
             and old.close(sum(regularizers)/8,row['consistency_loss']) and row['consistency_coefficient']==alpha
             and row['weighted_consistency_loss']==alpha*row['consistency_loss'] and old.close(row['loss'],row['ce_loss']+alpha*row['consistency_loss']),'Native objective reduction/coefficient differs')
        den=[denominators[f] for sid in sids[::2] for f in scenes[sid]['global_feature_ids']]
        need(len(den)==len(row['pair_position_denominator'])==len(row['residual_difference_norms'])==len(reg)
             and all(old.close(a,b,atol=1e-4,rtol=1e-6) for a,b in zip(den,row['pair_position_denominator']))
             and all(old.close(a*a/b,c) for a,b,c in zip(row['residual_difference_norms'],row['pair_position_denominator'],reg)),'Actual global denominator/residual arithmetic differs')
        valid=[scenes[sid]['n_frames'] for sid,l in zip(sids,lengths) for _ in range(l)]
        need(row['valid_item_count_by_position']==valid and row['valid_gate_count']==sum(valid)
             and row['padding_messages_exact_zero'] and row['closed_message_coordinates_exact_zero']
             and 0<=row['gate_min']<=row['gate_mean']<=row['gate_max']<=1 and 0<=row['payload_max_abs']<=1
             and math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0 and row['clipped']==(row['gradient_norm']>1)
             and math.isfinite(row['seconds']) and row['seconds']>0,'Gate/padding/gradient validity differs')
        targets_total+=len(flat)
    need(logs[0]['consistency_loss']==0 and old.close(summary['training_seconds'],sum(r['seconds'] for r in logs))
         and summary['first_four_step_seconds']==sum(r['seconds'] for r in logs[:4])
         and summary['step_seconds_max_steady']==max(r['seconds'] for r in logs[4:]),'Training timing/initial residual differs')
    bind(summary['first_gradients_file'],summary['first_gradients_sha256']);grads=read(summary['first_gradients_file'])
    need([x['step'] for x in grads]==[1,2] and all(set(x['pre_clip'])==set(config['trainable_parameter_names']) for x in grads),'Initial gradient coverage differs')
    bind(summary['gradient_diagnostics_file'],summary['gradient_diagnostics_sha256']);diagnostics=read(summary['gradient_diagnostics_file'])
    need([x['step'] for x in diagnostics]==[x for x in p.POLICY['gradient_steps'] if x<=steps],'Objective gradient coverage differs')
    return dict(updates=steps,scene_presentations=16*steps,target_positions=targets_total,native_loss_reductions_verified=True,
                first_preclip_gradients=grads[0]['pre_clip'],gradient_diagnostics=diagnostics),logs


def audit_timing(torch,directory,config,summary,plan):
    timing=summary['native_timing'];head_rows=0
    need(timing['passed'] and timing['no_dev_or_test_evaluation'] and summary['no_dev_or_test_evaluation']
         and not (Path(directory)/'test.json').exists() and not (Path(directory)/'dev_final.json').exists(),
         'Timing profile cannot contain development/test outcomes')
    cases=read(plan['timing_cases_file'])['cases'];need(len(cases)==len(timing['rows'])==9,'Nine timing cases required')
    head_rows=0
    for row,case in zip(timing['rows'],cases):
        need(all(row[k]==case[k] for k in ('parent_sid','contrast_id','variant','n_frames')) and row['no_accuracy_scoring'],
             'Timing case selection differs')
        for kind in ('raw','natural_raw'):bind(row[kind+'_file'],row[kind+'_sha256'])
        archive=torch.load(row['raw_file'],map_location='cpu',weights_only=True);result=archive['result']
        original=torch.load(row['natural_raw_file'],map_location='cpu',weights_only=True)['result']
        need(original['generated_ids']==result['generated_ids'] and torch.equal(original['raw_logits'],result['raw_logits']),
             'Natural output changed before native replay')
        ids=result['generated_ids'];t=len(ids);raw=result['raw_logits'];replay=archive['replay_global_logits']
        need(t==row['generated_tokens']==row['standalone_head_calls'] and 1<=t<=4 and raw.shape==replay.shape==(t,152064)
             and raw.dtype==torch.float32 and replay.dtype==torch.float16 and torch.equal(raw,raw.half().float())
             and raw.argmax(-1).tolist()==ids and len(result['captures'])==t,'Timing native output/capture differs')
        need(result['metadata']['input_identity']==case['input_identity'] and result['metadata']['selection_mode']==config['selection_mode']
             and result['metadata']['native_identity_sha256']==plan['native_identity_sha256'],'Timing native ownership differs')
        need(row['counters']==result['counters']==dict(model=t,visual=1,language=t,norm=t,head=t,broadcast=t,selection=t,probe_head=0),
             'Timing work differs')
        need(not any(i in (151645,151643) for i in ids[:-1]) and (ids[-1] in (151645,151643) or t==4), 'Native profile termination differs')
        audit_uniform_captures(torch,result['captures'])
        for i,m in enumerate(row['native_head_replays']):
            a,b=raw[i].double(),replay[i].double();tv=float(.5*(a.softmax(-1)-b.softmax(-1)).abs().sum());top=int(a.argmax())==int(b.argmax())
            need(m['passed'] and m['top1_equal']==[top] and top and tv<=.02 and len(m['tv'])==1
                 and old.close(tv,m['tv'][0],atol=1e-12,rtol=1e-9),'Independent native head replay differs');head_rows+=1
        need(len(row['native_head_replays'])==t,'Timing head replay coverage differs')
        actual=row['preprocessing_seconds']+(row['generation_seconds']+row['replay_seconds']+row['archive_seconds'])*4/t
        need(old.close(actual,row['four_token_seconds_bound'],atol=1e-12,rtol=1e-12),'Four-token full-work bound differs')
    need(timing['natural_generations']==timing['vision_calls']==9 and timing['origin_probe_calls']==0
         and timing['native_model_calls']==timing['standalone_head_calls']==head_rows<=36,'Profile invocation inventory differs')
    for n in (16,32,64):need(timing['T'+str(n)]==max(r['four_token_seconds_bound'] for r in timing['rows'] if r['n_frames']==n),'Native length timing bound differs')
    return dict(head_rows=head_rows,timing_inputs=dict(setup=summary['setup_before_training_seconds'],first4=summary['first_four_step_seconds'],steady=summary['step_seconds_max_steady'],T16=timing['T16']))


def audit_queries(row,scenes):
    q=row['query_diagnostics'];offsets=row['scene_offsets'];ce=row['position_losses']['ce']
    first=offsets[:-1];continuation=[i for a,b in zip(offsets,offsets[1:]) for i in range(a+1,b-1)];eos=[b-1 for b in offsets[1:]]
    for label,indices,losslabel in [('first',first,'first_token_nll'),('continuation',continuation,'continuation_token_nll'),('eos',eos,'eos_token_nll')]:
        need(q[label+'_indices']==indices and q[losslabel]==[ce[i] for i in indices],'First/continuation/EOS loss split differs')
    keys=['first_gate_sum','first_gate_min','first_gate_max','first_gate_zero_count','first_gate_valid_count','first_all_gates_closed']
    need(all(len(q[k])==16 for k in keys),'First-query gate coverage differs')
    for i,sid in enumerate(row['sids']):
        n=scenes[sid]['n_frames'];lo,hi,total=q['first_gate_min'][i],q['first_gate_max'][i],q['first_gate_sum'][i]
        zero,closed=q['first_gate_zero_count'][i],q['first_all_gates_closed'][i]
        need(q['first_gate_valid_count'][i]==n and type(zero) is int and 0<=zero<=n and type(closed) is bool
             and all(math.isfinite(x) for x in (lo,hi,total)) and 0<=lo<=hi<=1 and 0<=total<=n
             and closed==(zero==n)==(hi==0)==(total==0),'First-query gate statistics invalid')
    return q


def audit_gradients(torch,diagnostics,logs,config):
    names=config['trainable_parameter_names'];expected_groups=dict(all=names,selector=[n for n in names if n.startswith('selection_')],
        local=[n for n in names if n=='local_bias' or n.startswith('local.')],up=[n for n in names if n.startswith('up.')])
    # Parameter order in a state table is preserved by JSON; set membership is
    # the contract, while the recorded order determines reduction order.
    for item in diagnostics:
        row=logs[item['step']-1];need(item['sids']==row['sids'] and item['consistency_coefficient']==config['consistency_coefficient']
             and item['gradient_buffers_unchanged'] and item['ce_and_consistency_both_live'],'Gradient/optimizer observation differs')
        bind(item['raw_file'],item['raw_sha256']);raw=torch.load(item['raw_file'],map_location='cpu',weights_only=True)
        need(set(raw)=={'ce','consistency','score_ce','score_consistency','payload_ce','payload_consistency'}
             and set(raw['ce'])==set(raw['consistency'])==set(names),'Gradient packet coverage differs')
        for objective in ('ce','consistency'):
            for name,v in raw[objective].items():
                need(list(v.shape)==config['initialized'][name]['shape'] and v.dtype==torch.float32 and bool(torch.isfinite(v).all()),'Native core gradient tensor invalid')
        need(set(item['groups'])==set(expected_groups),'Gradient group coverage differs')
        for group,allowed in expected_groups.items():
            value=item['groups'][group];order=value['parameter_names'];need(set(order)==set(allowed) and len(order)==len(allowed),'Gradient group membership differs')
            a=sum(float(raw['ce'][n].double().square().sum()) for n in order);b=sum(float(raw['consistency'][n].double().square().sum()) for n in order)
            dot=sum(float((raw['ce'][n].double()*raw['consistency'][n].double()).sum()) for n in order);alpha=config['consistency_coefficient']
            expected=dict(ce_norm=math.sqrt(a),consistency_norm=math.sqrt(b),dot=dot,
                cosine=None if a==0 or b==0 else dot/math.sqrt(a*b),combined_norm=math.sqrt(max(0,a+alpha*alpha*b+2*alpha*dot)))
            for k,v in expected.items():need(value[k] is None if v is None else old.close(value[k],v,atol=1e-10,rtol=1e-6),'Independent objective gradient statistic differs')
        for name in ('score_ce','score_consistency','payload_ce','payload_consistency'):
            v=raw[name];shape=(16,len(row['target_ids']))+((96,) if name.startswith('payload') else ())
            need(tuple(v.shape)==shape and v.dtype==torch.float32 and bool(torch.isfinite(v).all()),'Intermediate gradient shape/precision differs')
            need(old.close(float(v.double().norm()),item['intermediate_gradient_norms'][name],atol=1e-10,rtol=1e-6),'Intermediate gradient norm differs')
            for j,n in enumerate(row['valid_item_count_by_position']):need(not bool(v[n:,j].any()),'Padding acquired objective gradient')
        if item['step']==1:
            need(all(not bool(v.any()) for v in raw['consistency'].values()) and not bool(raw['score_consistency'].any())
                 and not bool(raw['payload_consistency'].any()),'Zero-initial residual has nonzero gradients')
        del raw
    return dict(passed=True,records=len(diagnostics),both_objectives_from_saved_fp32_gradients=True,no_optimizer_effect_claim_bound_to_source=True)


def audit_run(torch,directory,plan_path,profile):
    directory,config,summary,plan,weights=config_and_endpoint(torch,directory,plan_path,profile)
    training,logs=audit_training(torch,directory,config,summary,plan)
    scenes=read(plan['scenes_file'])
    for row in logs:audit_queries(row,scenes)
    training['gradient_audit']=audit_gradients(torch,training['gradient_diagnostics'],logs,config)
    training['terminal_query_diagnostics']=logs[-1]['query_diagnostics']
    training['immutable_progress_snapshots']=verify_progress_snapshots(directory,logs,training['gradient_diagnostics'],profile)
    training['fixed_uniform_contract']=audit_uniform_contract(torch,directory,config,summary,plan,weights,training,logs)
    return directory,config,summary,plan,weights,training,logs



def verify_progress_snapshots(directory, logs, diagnostics, profile):
    expected=[] if profile else list(range(100,601,100))
    need(sorted(x.name for x in directory.glob('training_step_*.json'))==[f'training_step_{s:04d}.json' for s in expected]
         and sorted(x.name for x in directory.glob('gradient_diagnostics_step_*.json'))==[f'gradient_diagnostics_step_{s:04d}.json' for s in expected],
         'Immutable progress snapshot inventory differs')
    records=[]
    for step in expected:
        a=directory/f'training_step_{step:04d}.json';b=directory/f'gradient_diagnostics_step_{step:04d}.json'
        need(read(a)==logs[:step] and read(b)==[d for d in diagnostics if d['step']<=step],
             'Progress snapshots differ from final log prefixes')
        records.append(dict(step=step,training_sha256=sha(a),gradient_diagnostics_sha256=sha(b)))
    return records


def verify_jobs(budget,records,name):
    jobs={r['job_id']:r for r in budget['jobs']}
    for r in records:
        j=jobs[r['job_id']];need(j['state']=='COMPLETED' and j['exit_code']=='0:0' and j['gpus']==1 and j['name']==name,'Missing successful exact pilot allocation')


def release(args,out,frozen):
    import torch
    p=pilot();torch.set_num_threads(4);plan=p.verify_plan(args.plan,ancestors=True);check_data(plan)
    need(len(args.profiles)==1,'Exactly one fixed-uniform profile required');profiles={};records=[];initialized=None;first=None
    for directory in args.profiles:
        directory,config,summary,_,weights,training,logs=audit_run(torch,directory,args.plan,True)
        arm=config['arm'];need(arm not in profiles,'Duplicate profile arm')
        bind(summary['native_timing_file'],summary['native_timing_sha256']);need(read(summary['native_timing_file'])==summary['native_timing'],'Native timing file differs')
        timing=audit_timing(torch,directory,config,summary,plan)
        profiles[arm]=dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'))
        records.append(dict(arm=arm,job_id=config['slurm_job_id'],training=training,**timing))
        if initialized is None:initialized=config['initialized'];first=training['first_preclip_gradients']
        need(initialized==config['initialized'] and first==training['first_preclip_gradients'],'Matched profile initial tensors/gradients differ')
    need(set(profiles)==set(p.ARMS),'Fixed-uniform profile missing')
    pooled={k:max(r['timing_inputs'][k] for r in records) for k in ('setup','first4','steady','T16')}
    seconds=pooled['setup']+1.25*(pooled['first4']+596*pooled['steady']+162*pooled['T16'])+120
    budget=p.resources(out);verify_jobs(budget,records,p.PROFILE_JOB)
    need(not any(r['name']==p.MAIN_JOB for r in budget['jobs']),'Uniform main attempted before release')
    budget.update(reserved_main_gpu_seconds=840,projected_total_with_main_reservations=budget['allocated_gpu_seconds']+840)
    passed=seconds<=840 and budget['projected_total_with_main_reservations']<=990;budget['passed']=passed
    projection=dict(passed=seconds<=840,projected_seconds=seconds,pooled_inputs=pooled,
        formula='maxsetup+1.25*(maxfirst4+596*maxsteady+162*maxT16)+120',
        scope='conservative measured engineering projection; N16 timing also used for smaller N8; not a guaranteed runtime bound')
    value=dict(passed=passed,completed=True,protocol='identity_join_uniform_main_release',source_sha256=frozen,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),profiles=profiles,profile_audits=records,
        pooled_projection=projection,campaign_budget=budget,per_main_seconds_cap=840,campaign_gpu_seconds_cap=990,
        no_outcome_dependent_release=True,all_failed_allocations_retained=True)
    save(out/'release.json',value);need(passed,'Fixed pilot timing/resource gate failed; evidence retained')
    return dict(passed=True,completed=True,phase='release',protocol=p.PROTOCOL,source_sha256=frozen,
        release_file=str(out/'release.json'),release_sha256=sha(out/'release.json'),pooled_projection=projection,campaign_budget=budget)


def report(args,out,frozen):
    import torch
    from transformers import AutoTokenizer
    p=pilot();torch.set_num_threads(4);plan=p.verify_plan(args.plan,ancestors=True);rows,_,_=check_data(plan)
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=True)
    states=read(plan['semantic_states_file']);need(set(states)=={x['sample']['sid'] for group in rows.values() for x in group},'Offline subset semantic coverage differs')
    need(len(args.runs)==1,'Exactly one fixed-uniform endpoint required');records=[];outcomes={};gate_diagnostics={};seen=set();release_binding=None;initial=None;first=None
    for directory in args.runs:
        directory,config,summary,_,weights,training,logs=audit_run(torch,directory,args.plan,False)
        arm=config['arm'];need(arm not in seen,'Duplicate final arm');seen.add(arm)
        rb=config['main_release'];bind(rb['file'],rb['sha256']);p.verify_release(rb['file'],args.plan,plan)
        need(Path(rb['file']).resolve()==Path(args.main_release).resolve(),'Requested main release differs from fitted release')
        release_summary=read(Path(rb['file']).parent/'summary.json')
        need(release_summary['passed'] and release_summary['completed'] and release_summary['phase']=='release'
             and release_summary['release_sha256']==rb['sha256'],'Independent release did not complete')
        if release_binding is None:release_binding=rb
        need(rb==release_binding,'Different final-fit release')
        if initial is None:initial=config['initialized'];first=training['first_preclip_gradients']
        need(initial==config['initialized'] and first==training['first_preclip_gradients'],'Matched final initialization/first gradients differ')
        evaluated={}
        for label in ('train','dev'):
            bind(summary[label+'_file'],summary[label+'_sha256']);expected=[(x['cell'],x['sample']) for x in rows[label]]
            evaluated[label]=old.audit_eval(torch,summary[label+'_file'],expected,config['selection_mode'],24,tokenizer,
                plan['native_identity_sha256'],weights,states)
        train=p.criteria(evaluated['train']['rows'],True);dev=p.criteria(evaluated['dev']['rows'],False)
        need(train==summary['trainability'] and dev==summary['id_generalization'] and summary['native_train_count']==108
             and summary['native_dev_count']==54 and summary['native_test_count']==0 and summary['fixed_final_checkpoint_only']
             and summary['dev_evaluated_once'],'Pilot criteria/evaluation inventory differs')
        outcomes[arm]={k:v['rows'] for k,v in evaluated.items()}
        gate_diagnostics[arm]={k:final_gate_diagnostics(torch,v['rows'],states) for k,v in evaluated.items()}
        records.append(dict(arm=arm,job_id=config['slurm_job_id'],directory=str(directory),summary_sha256=sha(directory/'summary.json'),
            checkpoint=summary['checkpoint'],checkpoint_sha256=summary['checkpoint_sha256'],training=training,
            trainability=train,id_generalization=dev,gate_summary={k:v['summary'] for k,v in gate_diagnostics[arm].items()},native_inventory={k:v['inventory'] for k,v in evaluated.items()}))
    need(seen==set(p.ARMS),'Incomplete pilot matrix');budget=p.resources(out);verify_jobs(budget,records,p.MAIN_JOB)
    save(out/'outcomes.json',outcomes);save(out/'selector_diagnostics.json',gate_diagnostics)
    analysis=dict(passed=True,completed=True,protocol=p.PROTOCOL,source_sha256=frozen,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
        main_release=release_binding,runs=records,resources=budget,outcomes_file=str(out/'outcomes.json'),outcomes_sha256=sha(out/'outcomes.json'),
        selector_diagnostics_file=str(out/'selector_diagnostics.json'),selector_diagnostics_sha256=sha(out/'selector_diagnostics.json'),
        all_arms_retained=True,one_fixed_seed=True,no_test_evaluation=True,no_extrapolation_or_reasoning_claim=True)
    save(out/'analysis.json',analysis)
    lines=['# Fixed-uniform join control','','One fixed seed24 endpoint; no test evaluation or confirmed efficacy.','',
        '| Arm | Train first | Train whole | Train families | Dev first | Dev whole | Dev triples | Train/ID pass |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in records:
        a,b=r['trainability'],r['id_generalization'];lines.append(f"| {r['arm']} | {a['first_correct']}/108 | {a['whole_correct']}/108 | {a['complete_families']}/18 | {b['first_correct']}/54 | {b['whole_correct']}/54 | {b['complete_families']}/18 | {a['passed']}/{b['passed']} |")
    lines+=['',f"Allocated GPU-seconds including failures: {budget['allocated_gpu_seconds']}.",'','[Analysis](analysis.json) · [All outcomes](outcomes.json)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),resources=budget,
        criteria={r['arm']:dict(trainability=r['trainability'],id_generalization=r['id_generalization']) for r in records},no_efficacy_confirmation=True)


def final_gate_diagnostics(torch,rows,states):
    """Descriptive scalar gates at every actual native prefix; no model calls."""
    records=[]
    for row in rows:
        bind(row['raw_file'],row['raw_sha256']);raw=torch.load(row['raw_file'],map_location='cpu',weights_only=True)
        relevant=[any(room in row['room_pair'] and people for room,people in state['rooms'].items()) for state in states[row['sid']]]
        need(len(relevant)==row['n_frames'] and sum(relevant)==6,'Offline included-frame count differs')
        for step,cap in enumerate(raw['captures']):
            audit_uniform_captures(torch,[cap])
            gates=cap['gates'][:,0].double();mask=torch.tensor(relevant,dtype=torch.bool)
            need(len(gates)==len(relevant) and bool(torch.isfinite(gates).all()) and bool(((gates>=0)&(gates<=1)).all()),'Final native scalar gate malformed')
            records.append(dict(cell=row['cell'],sid=row['sid'],n_frames=row['n_frames'],token_index=step,prefix_ids=raw['generated_ids'][:step],
                exact=row['exact'],first_token_correct=row['first_token_correct'],gate_sum=float(gates.sum()),gate_min=float(gates.min()),gate_max=float(gates.max()),
                zero_count=int((gates==0).sum()),all_closed=not bool(gates.any()),relevant_gate_sum=float(gates[mask].sum()),
                irrelevant_gate_sum=float(gates[~mask].sum()),closed_relevant_count=int((gates[mask]==0).sum()),offline_labels_only=True))
        del raw
    first=[r for r in records if r['token_index']==0]
    need(len(first)==len(rows),'First-query gate coverage differs')
    return dict(summary=dict(contexts=len(first),prefixes=len(records),first_all_closed=sum(r['all_closed'] for r in first),
        all_prefixes_closed=sum(r['all_closed'] for r in records),first_all_relevant_closed=sum(r['closed_relevant_count']==6 for r in first),
        first_mean_gate_sum=sum(r['gate_sum'] for r in first)/len(first),all_actual_prefixes_retained=True,descriptive_only=True),rows=records)


def audit_uniform_captures(torch,captures):
    for cap in captures:
        need(torch.equal(cap['scores'],torch.full_like(cap['scores'],.5))
             and torch.equal(cap['gates'],torch.full_like(cap['gates'],.5)),
             'Native uniform score/gate is not exactly one half')


def audit_uniform_contract(torch,directory,config,summary,plan,weights,training,logs):
    p=pilot();condition=plan['conditional_report'];parent_binding=plan['parent_plan']
    need(config['conditional_report']==condition and config['reference_first_gradients']==plan['reference_first_gradients']
         and condition['summary_file']==str(p.CONDITIONAL) and condition['summary_sha256']==p.CONDITIONAL_SHA
         and condition['analysis_sha256']==p.CONDITIONAL_ANALYSIS_SHA and condition['all_four_valid_training_fit_failures'] is True,
         'Uniform conditional release provenance differs')
    bind(condition['summary_file'],condition['summary_sha256']);bind(condition['analysis_file'],condition['analysis_sha256'])
    proof=read(condition['summary_file']);analysis=read(condition['analysis_file'])
    need(proof['passed'] is True and proof['completed'] is True and proof['analysis_sha256']==condition['analysis_sha256']
         and analysis['passed'] is True and analysis['completed'] is True
         and len(analysis['runs'])==4 and {r['arm'] for r in analysis['runs']}==set(p.pilot.ARMS)
         and all(r['trainability']['passed'] is False for r in analysis['runs']),
         'Completed four-arm training failures do not satisfy conditional control trigger')
    need(parent_binding==dict(file=str(p.PARENT),sha256=p.PARENT_SHA),'Different parent subset plan')
    bind(parent_binding['file'],parent_binding['sha256']);parent=read(parent_binding['file'])
    need(config['initialized']==parent['initial_state'] and all(plan[k]==parent[k] for k in
         ('rows_file','pairs_file','prepared_file','scenes_file','semantic_states_file','order_file','order_object_sha256',
          'features_file','features_sha256','feature_tensor','initial_file','initial_sha256','timing_cases_file','native_identity_sha256')),
         'Fixed-uniform inputs/order/initialization differ from original pilot')
    frozen=['selection_weight','selection_bias']
    expected=[n for n in config['initialized'] if n not in frozen]
    need(len(expected)==6 and config['trainable_parameter_names']==plan['trainable_parameter_names']==expected
         and config['frozen_selector_parameter_names']==plan['frozen_selector_parameter_names']==frozen
         and config['effective_trainable_parameters']==plan['effective_trainable_parameters']==1041600
         and config['retained_state_parameters']==plan['retained_state_parameters']==1041697,
         'Uniform active/frozen parameter contract differs')
    need(config['selector_initial_state']==plan['selector_initial_state']=={n:config['initialized'][n] for n in frozen}
         and sum(weights[n].numel() for n in expected)==1041600
         and torch.equal(weights['selection_weight'],torch.zeros_like(weights['selection_weight']))
         and torch.equal(weights['selection_bias'],torch.full_like(weights['selection_bias'],.5)),
         'Frozen selector bytes or effective count changed')
    reference=plan['reference_first_gradients'];bind(reference['file'],reference['sha256'])
    before=read(reference['file'])[0]
    need(before['step']==1 and training['first_preclip_gradients']=={n:before['pre_clip'][n] for n in expected},
         'Initial active gradients differ from the original CE-only initialization')
    for row in logs:
        need(row['arm']==row['condition']=='uniform_half_ce' and row['frozen_selector_exact'] is True
             and row['consistency_coefficient']==0 and row['gate_min']==row['gate_max']==row['gate_mean']==.5,
             'Training adaptive gate/objective or fixed selector drift detected')
        q=row['query_diagnostics']
        need(all(v==.5 for key in ('first_gate_min','first_gate_max') for v in q[key])
             and q['first_gate_sum']==[n*.5 for n in q['first_gate_valid_count']]
             and q['first_gate_zero_count']==[0]*16 and q['first_all_gates_closed']==[False]*16,
             'Training first-query uniform weights differ')
    return dict(passed=True,effective_trainable_parameters=1041600,retained_state_parameters=1041697,
        fixed_selector_parameters=97,actual_initial_active_gradients_match_reference=True,
        all_training_steps_fixed_half=True,no_adaptive_selection=True)

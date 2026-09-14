"""Independent fixed-6000-update privileged joint-code optimization audit."""
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
from scripts import report_native_identity_join_joint_code as old
OWN=('scripts/report_native_identity_join_joint_code_long.py',)
DIAGNOSTIC_STEPS=(1,2,32,128,300,600,2000,6000)
ENDPOINTS=(600,2000,6000)
PROGRESS=(600,2000,4000,6000)
WEIGHT_NAMES=old.WEIGHT_NAMES
close=old.close
layout_for=old.layout_for
first_token_criterion=old.first_token_criterion
fp64_nll=old.fp64_nll
tensor_error=old.tensor_error
replay_metric=old.replay_metric
allocation_rows=old.allocation_rows
check_weights=old.check_weights
gather=old.gather
functional_core=old.functional_core
bind=old.bind


def driver():
    from scripts import diagnose_native_identity_join_joint_code_long as module
    return module


def expected_order(pairs):
    need(len(pairs)==54 and len({x['pair_id'] for x in pairs})==54,'Original pair inventory differs')
    rng=random.Random(24);result=[];cycle=0
    while len(result)<48000:
        cycle+=1;slots=list(range(54));rng.shuffle(slots)
        result.extend(dict(cycle=cycle,pair_slot=i,pair_id=pairs[i]['pair_id'],sids=pairs[i]['sids']) for i in slots)
    return result[:48000]


def learning_rate(step):
    need(type(step) is int and 1<=step<=6000,'Update outside fixed horizon')
    return .001*step/50 if step<=50 else .00001+(.001-.00001)*(1+math.cos(math.pi*(step-50)/5950))/2


def primary_screen(endpoint_rows):
    need(set(endpoint_rows)==set(ENDPOINTS),'All three fixed endpoints required')
    return first_token_criterion(endpoint_rows[6000])


def audit_inputs(torch,plan):
    p=driver();reference=plan['joint_code_plan'];bind(reference['file'],reference['sha256'])
    parent=old.driver().verify_plan(reference['file'],ancestors=True)
    rows,scenes,short_order,states,index,by_sid,initial=old.audit_inputs(torch,parent)
    need(read(plan['rows_file'])==rows and read(plan['scenes_file'])==scenes
         and read(plan['pairs_file'])==read(parent['pairs_file']) and read(plan['training_samples_file'])==read(parent['training_samples_file']), 'Original semantic/scene inputs changed')
    order=read(plan['order_file']);expected=expected_order(read(parent['pairs_file']))
    need(order==expected and len(order)==48000 and order[:4800]==short_order,'Persistent extended pair order or original prefix differs')
    for key in ('features_sha256','feature_tensor','global_feature_ids','codes_sha256','code_tensor','code_inventory_sha256',
                'initial_sha256','initial_state','native_identity','native_identity_sha256','native_module_identity','native_model_sha256',
                'parent_initial_file','parent_initial_sha256','factor_plan','uniform_plan','trainability_mask'):
        need(plan[key]==parent[key], 'Original model/code/native input differs: '+key)
    for field,digest_field in (('features_file','features_sha256'),('codes_file','codes_sha256'),('initial_file','initial_sha256'),
        ('code_inventory_file','code_inventory_sha256'),('native_model_file','native_model_sha256')):
        bind(plan[field],plan[digest_field]);need(sha(plan[field])==sha(parent[field]),'Consumed artifact bytes differ: '+field)
    need(read(plan['code_inventory_file'])==read(parent['code_inventory_file']),'Original semantic code inventory differs')
    counts=[sum(len(scenes[sid]['target_ids']) for pair in order[i:i+8] for sid in pair['sids']) for i in range(0,48000,8)]
    need(len(counts)==6000 and counts==plan['head_rows_by_update'] and counts[:600]==parent['head_rows_by_update']
         and sum(counts)==plan['training_head_rows'] and sum(counts)+324==plan['total_head_rows']
         and read(plan['head_rows_file'])==dict(by_update=counts,total=sum(counts),maximum=max(counts),endpoint_rows=324,all_head_rows=sum(counts)+324), 'Extended target-position inventory differs')
    return rows,scenes,order,states,index,by_sid,initial


def self_test(torch):
    pairs=[dict(pair_id=f'p{i}',sids=[f'a{i}',f'b{i}']) for i in range(54)]
    order=expected_order(pairs);counts=Counter(x['cycle'] for x in order)
    need(len(order)==48000 and order[:4800]==old.expected_order(pairs) and counts[889]==48
         and all(counts[i]==54 for i in range(1,889)),'Carried shuffle/cycle/prefix fixture failed')
    need(close(learning_rate(1),.00002,atol=1e-15,rtol=1e-15) and learning_rate(50)==.001
         and learning_rate(6000)==.00001 and learning_rate(600)>.0009,'Warmup and extended cosine horizon differ')
    rows=[dict(sid=f'{f}_{i}_{n}',contrast_id=f,variant=i,n_frames=n,first_token_correct=True) for f in range(18) for i in range(3) for n in (8,16)]
    failed=[dict(r,first_token_correct=False) for r in rows]
    need(not primary_screen({600:rows,2000:rows,6000:failed})['passed']
         and primary_screen({600:failed,2000:failed,6000:rows})['passed'],'Early endpoints must never select final acceptance')
    parameters={k:torch.nn.Parameter(torch.ones(2,1)) for k in WEIGHT_NAMES}
    optimizer=torch.optim.AdamW(list(parameters.values()),lr=learning_rate(3),weight_decay=0.)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        for value in parameters.values():value.grad=torch.ones_like(value)
        optimizer.step()
    info=driver().tensor_info
    snapshot=dict(parameter_ids={k:id(v) for k,v in parameters.items()},optimizer_parameter_ids=[id(v) for v in parameters.values()],
        requires_grad={k:True for k in parameters},state_tensor_info={k:{name:info(value) for name,value in optimizer.state[v].items()} for k,v in parameters.items()},
        param_group_hyperparameters=[{k:v for k,v in optimizer.param_groups[0].items() if k!='params'}])
    snapshot=__import__('json').loads(__import__('json').dumps(snapshot));optimizer_audit(torch,snapshot,3,parameters)
    snapshot['state_tensor_info'][WEIGHT_NAMES[0]]['step']=info(torch.tensor(2.))
    try:optimizer_audit(torch,snapshot,3,parameters)
    except Exception as exc:need('Adam update counter' in str(exc),'Unexpected optimizer-reset fixture failure')
    else:raise AssertionError('A reset Adam counter must fail the audit')
    return dict(passed=True,groups=4,persistent_order_original_prefix=True,extended_cosine_schedule=True,final6000_only_screen=True,actual_Adam_counter_and_reset_rejection=True)

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
        need(all(row[k] is True for k in ('native_frozen','codes_frozen'))
             and row['code_input_shape']==[16,len(layout['targets']),96]
             and math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0 and row['clipped']==(row['gradient_norm']>1)
             and math.isfinite(row['seconds']) and row['seconds']>0, 'Gradient/timing metadata invalid')
        q=len(layout['targets']);need(q<=48,'Per-call target row cap exceeded');total+=q;maximum=max(maximum,q)
        for module in ('norm','head'):
            expected_calls.append(dict(module=module,phase='training',step=step,endpoint_step=None,input_shape=[1,q,3584],
                output_shape=[1,q,3584 if module=='norm' else 152064],input_dtype='torch.float16',output_dtype='torch.float16'))
        if step in ENDPOINTS:
            for batch in range(1,8):
                q=12 if batch==7 else 16
                for module in ('norm','head'):
                    expected_calls.append(dict(module=module,phase='evaluation',step=batch,endpoint_step=step,input_shape=[1,q,3584],
                        output_shape=[1,q,3584 if module=='norm' else 152064],input_dtype='torch.float16',output_dtype='torch.float16'))
    independent_total=sum(len(scenes[sid]['target_ids']) for pair in order for sid in pair['sids'])
    need(total==independent_total and maximum==max(len(r['target_ids']) for r in logs) and logs[0]['consistency_loss']==0,
         'Full training call-row inventory differs')
    return logs,expected_calls,dict(passed=True,updates=6000,pair_presentations=48000,scene_presentations=96000,
        training_target_positions=total,maximum_training_positions_per_call=maximum,endpoint_first_query_positions=324,total_head_rows=total+324,
        terminal_ce=logs[-1]['ce_loss'],terminal_residual_diagnostic=logs[-1]['consistency_loss'],consistency_coefficient=0.,
        full_sequence_loss_reductions_verified=True,no_independent_optimizer_trajectory_replay=True)


def audit_capture(torch,out,record,plan,scenes,states,index,by_sid,rows,initial,endpoints,logs,norm,head):
    p=driver();phase=record['phase'];step=record['step'];training=phase=='training';endpoint_step=record['endpoint_step']
    need(endpoint_step is None if training else endpoint_step in ENDPOINTS,'Invalid capture endpoint association')
    need(phase in ('training','evaluation') and record['arm']==p.ARM,'Invalid joint-code capture phase/arm')
    bind(record['file'],record['sha256']);bind(record['weights_file'],record['weights_sha256'])
    cap=torch.load(record['file'],map_location='cpu',weights_only=True);packet=torch.load(record['weights_file'],map_location='cpu',weights_only=True);w=packet['branch']
    check_weights(torch,w,initial)
    if training:
        need(packet['arm']==p.ARM and step in DIAGNOSTIC_STEPS and packet['step']==record['weights_step']==step-1
             and packet['optimizer_step']==step and packet['position']=='before_update' and packet['source_sha256']==plan['source_sha256'], 'Pre-update weight identity differs')
        sids=logs[step-1]['sids']
        if step==1:need(all(torch.equal(w[k],initial[k]) for k in initial),'First captured state differs from selected unfitted initialization')
    else:
        need(1<=step<=7 and packet['step']==record['weights_step']==endpoint_step and all(torch.equal(w[k],endpoints[endpoint_step][k]) for k in initial),'Evaluation capture changed its prespecified endpoint')
        sids=[r['sid'] for r in rows[(step-1)*16:step*16]]
    need(cap['schema_version']==1 and cap['arm']==p.ARM and cap['phase']==phase and cap['step']==step
         and cap['sids']==record['sids']==sids and cap['endpoint_step']==endpoint_step, 'Capture scene/phase ownership differs')
    codes,g,valid,layout=gather(torch,scenes,states,index,by_sid,sids,first_only=not training)
    need(cap['layout']==layout,'Full-target or empty-prefix capture layout differs')
    for key,value in (('codes',codes),('global_states',g),('valid_mask',valid)):
        need(cap[key].dtype==value.dtype and torch.equal(cap[key],value),'Captured fixed code/global ownership differs: '+key)
    observed=cap['capture'];reference=functional_core(torch,codes,g,valid,w)
    need(set(observed)==set(reference),'Readout capture coverage differs')
    metrics={key:tensor_error(torch,observed[key],value) for key,value in reference.items()}
    need(torch.equal(observed['payload'],codes) and torch.equal(observed['gates'],valid.float()*.5)
         and bool((observed['scores']==.5).all()) and torch.equal(observed['messages'],codes*observed['gates'].unsqueeze(-1))
         and torch.equal(observed['aggregate'],.5*codes.sum(0)) and bool((codes[~valid]==0).all()), 'Raw code/half-SUM/padding arithmetic differs')
    need(observed['delta'].dtype==torch.float32 and cap['fused_global'].dtype==cap['normalized'].dtype==cap['logits'].dtype==torch.float16
         and cap['fused_global'].shape==cap['normalized'].shape==g.shape and cap['logits'].shape==(len(g),152064)
         and all(bool(torch.isfinite(cap[k]).all()) for k in ('fused_global','normalized','logits')), 'Native output dtype/shape differs')
    need(torch.equal(g+observed['delta'].half(),cap['fused_global'])
         and {k:p.tensor_info(cap[k]) for k in ('fused_global','normalized','logits')}==record['tensors'],'Native cast-before-add/tensor hash differs')
    if training and step==1:need(bool((observed['delta']==0).all()) and torch.equal(cap['fused_global'],g),'Zero-up identity differs')
    native_metrics=[];normalized_error=None
    if not training:
        normalized=norm(cap['fused_global'].unsqueeze(0))[0];replayed=head(normalized.unsqueeze(0))[0]
        native_metrics=replay_metric(torch,cap['logits'],replayed);normalized_error=float((normalized.float()-cap['normalized'].float()).abs().max())
    targets=layout['targets'] if training else layout['target_ids'];nll=fp64_nll(torch,cap['logits'],targets)
    precision=[dict(position=i,fp64=float(a),gpu_fp32=b,passed=close(float(a),b)) for i,(a,b) in
        enumerate(zip(nll,logs[step-1]['position_losses']['ce']))] if training else []
    audit=dict(passed=all(v['passed'] for v in metrics.values()) and all(v['passed'] for v in native_metrics+precision),
        arm=p.ARM,phase=phase,step=step,endpoint_step=endpoint_step,sids=sids,capture_file=record['file'],capture_sha256=record['sha256'],
        weights_file=record['weights_file'],weights_sha256=record['weights_sha256'],functional_core=metrics,
        fixed_local_codes_exact=True,fp16_cast_before_add_exact=True,native_head=native_metrics,normalized_max_absolute=normalized_error,
        nll=precision,captured_rows=len(g),head_rows=0 if training else len(g),cpu_norm_calls=0 if training else 1,cpu_head_calls=0 if training else 1)
    save(out/(f'training_{step:04d}_capture_audit.json' if training else f'evaluation_{endpoint_step:04d}_{step:04d}_capture_audit.json'),audit);need(audit['passed'],'Captured readout/native/NLL check failed; raw evidence retained')
    return audit,cap['logits']



def resources(out,runs):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'all_user_sacct.psv';file.write_text(raw)
    job_arms={name:arm for arm,name in driver().RUN_JOBS.items()};names=set(job_arms);records=allocation_rows(raw,names)
    need(len(records)==1 and {r['name'] for r in records}==names,'Exactly one joint-code attempt required; failed/zero allocations retained')
    for row in records:
        arm=job_arms[row['name']]
        need(row['job_id']==runs[arm]['slurm_job_id'] and row['partition']=='gpu' and row['state']=='COMPLETED'
             and row['exit_code']=='0:0' and row['gpus']==1 and row['seconds']<=180,'Joint-code allocation identity/cap differs')
    from datetime import datetime
    events=[]
    for row in records:
        a,b=map(datetime.fromisoformat,(row['start'],row['end']));need(b>=a,'Invalid allocation interval')
        if b>a:events.extend(((a,1),(b,-1)))
    current=peak=0
    for _,d in sorted(events):current+=d;need(current>=0,'Invalid allocation overlap');peak=max(peak,current)
    total=sum(r['gpu_seconds'] for r in records);need(current==0 and peak<=1 and total<=180,'Long-horizon joint-code cap exceeded')
    return dict(passed=True,jobs=records,allocated_gpu_seconds=total,maximum_concurrent_gpus=peak,
        failed_and_zero_allocations_retained=True,file=str(file),sha256=sha(file),command=command)
def optimizer_audit(torch,value,step,weights):
    need(set(value)=={'parameter_ids','optimizer_parameter_ids','requires_grad','state_tensor_info','param_group_hyperparameters'}, 'Optimizer snapshot schema differs')
    ids=value['parameter_ids'];need(set(ids)==set(WEIGHT_NAMES) and all(type(v) is int and v>0 for v in ids.values())
         and len(set(ids.values()))==4 and value['optimizer_parameter_ids']==[ids[k] for k in WEIGHT_NAMES]
         and value['requires_grad']=={k:True for k in WEIGHT_NAMES}, 'Optimizer must retain the same four live parameter objects')
    need(set(value['state_tensor_info'])==set(WEIGHT_NAMES),'Adam state coverage differs')
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


def audit_run(torch,out,directory,plan_path,plan,frozen,rows,scenes,order,states,index,by_sid,initial,norm,head):
    p=driver();directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json')
    need(directory.parent==p.OUT and directory.name==config['run_id']==f'run_{config["slurm_job_id"]}' and config['arm']==p.ARM
         and config['protocol']==p.PROTOCOL and config['policy']==p.POLICY and config['seed']==24
         and config['source_sha256']==frozen==plan['source_sha256'] and all(summary.get(k)==v for k,v in config.items())
         and Path(config['plan_file']).resolve()==Path(plan_path).resolve() and config['plan_sha256']==sha(plan_path), 'Run/config/source/plan identity differs')
    for name,digest in frozen.items():bind(directory/'source'/name.replace('/','_'),digest)
    inherited=dict(plan_file=plan['joint_code_plan']['file'],plan_sha256=plan['joint_code_plan']['sha256'],source_sha256=plan['inherited_source_sha256'])
    need(config['inherited_source_sha256']==plan['inherited_source_sha256'] and read(directory/'inherited_sources.json')==inherited, 'Inherited source descriptor differs')
    for flag in ('passed','completed','computational_integrity_passed','checkpoint_roundtrip_passed','native_weights_unchanged','codes_frozen',
                 'cached_first_query_only','no_pretrained_backbone_loaded','no_dev_or_test','no_native_or_whole_answer_claim','oracle_local_semantics','no_answer_or_intersection_code'):
        need(summary[flag] is True,'Run invariant failed: '+flag)
    need(summary['phase']=='run' and summary['steps']==6000 and not any(k.startswith('permutation_') for k in summary),'Horizon/scope differs')
    need(config['hardware']['gpu']=='NVIDIA B200' and config['hardware']['matmul_allow_tf32'] is False
         and config['hardware']['torch_version']==str(torch.__version__),'Native hardware/precision differs')
    for key in ('native_identity','native_identity_sha256','native_module_identity','native_model_file','native_model_sha256',
                'codes_file','codes_sha256','code_tensor','code_inventory_sha256','features_file','features_sha256','feature_tensor',
                'parent_initial_file','parent_initial_sha256','joint_code_plan','short_control_report','factor_plan','uniform_plan','architecture_stop_report','decision_step','evaluation_steps'):
        need(config[key]==plan[key],'Consumed native/input identity differs: '+key)
    bind(config['code_inventory_file'],config['code_inventory_sha256'])
    need(Path(config['code_inventory_file'])==directory/'code_inventory.json' and read(config['code_inventory_file'])==read(plan['code_inventory_file']), 'Copied semantic code inventory differs')
    need(config['reused_unfitted_readout'] is plan['reused_unfitted_readout'] is True and config['initialized']==plan['initial_state']
         and config['initial_checkpoint_sha256']==plan['initial_sha256'],'Original unfitted readout differs')
    mask=dict(trainable_parameter_names=list(WEIGHT_NAMES),effective_trainable_parameters=697440,retained_state_parameters=697440)
    need(plan['trainability_mask']==mask and all(config[k]==v for k,v in mask.items()),'Four-parameter optimizer scope differs')
    for key in ('training','captures','calls','evaluations'):bind(summary[key+'_file'],summary[key+'_sha256'])
    for key in ('rows','presentations'):bind(config[key+'_file'],config[key+'_sha256'])
    need(read(config['rows_file'])==rows and read(config['presentations_file'])==order and config['order_sha256']==object_sha(order),'Copied rows/order differ')
    root=p.CKPT/directory.name;data=p.DATA/directory.name
    need(Path(config['checkpoint_directory'])==root and Path(config['data_directory'])==data and Path(config['initial_checkpoint'])==root/'initial.pt','Authorized artifact roots differ')
    bind(config['initial_checkpoint'],config['initial_checkpoint_sha256']);selected=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=True)
    need(selected['seed']==24 and all(torch.equal(selected['branch'][k],initial[k]) for k in WEIGHT_NAMES),'Actual run initialization differs')
    denominators={fid:float(states[index[fid]].float().square().sum()+1e-6) for scene in scenes.values() for fid in scene['global_feature_ids']}
    logs,expected_calls,training=audit_training(directory,summary,scenes,order,denominators)
    need([len(r['target_ids']) for r in logs]==plan['head_rows_by_update'] and close(summary['training_seconds'],sum(r['seconds'] for r in logs))
         and summary['training_head_rows']==training['training_target_positions'] and summary['total_head_rows']==training['total_head_rows'],'Extended training count/timing differs')
    counters=dict(core=6021,norm=6021,head=6021,vlm=0,vision=0);calls=read(summary['calls_file'])
    need(calls==expected_calls and summary['counters']==counters
         and read(directory/'final_counters.json')==dict(counters=counters,recorded_calls=calls,partial_outputs_retained=True),'Chronological actual call ledger differs')
    evaluations=summary['evaluations'];need(read(summary['evaluations_file'])==evaluations and Path(summary['evaluations_file'])==directory/'evaluations.json'
         and summary['final_head_rows']==324 and summary['earlier_endpoints_descriptive_only'] is True and summary['decision_step']==6000
         and set(evaluations)=={str(e) for e in ENDPOINTS},'All three prespecified endpoints required')
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
             and endpoint['parameter_sha256']==object_sha(table) and endpoint['native_before']==endpoint['native_after']==native_table
             and endpoint['input_tensors_before']==endpoint['input_tensors_after']==dict(codes=plan['code_tensor'],globals=plan['feature_tensor'])
             and endpoint['input_versions_unchanged'] is True and endpoint['training_mode_restored'] is True
             and endpoint['optimizer_before']==endpoint['optimizer_after'] and endpoint['parameter_versions_before']==endpoint['parameter_versions_after'],
             'Intermediate evaluation mutated readout/native/input/optimizer/trainability state')
        optimizer_identity=optimizer_audit(torch,endpoint['optimizer_before'],e,w)
        if optimizer_ids is None:optimizer_ids=optimizer_identity
        need(optimizer_ids==optimizer_identity and set(endpoint['parameter_versions_before'])==set(WEIGHT_NAMES)
             and all(type(v) is int and v>=0 for v in endpoint['parameter_versions_before'].values()),'Parameter objects/version metadata changed across observations')
        count=e+7*sum(x<=e for x in ENDPOINTS)
        need(endpoint['counters']==dict(core=count,norm=count,head=count,vlm=0,vision=0),'Endpoint cumulative call count differs')
        need(roundtrip['passed'] is True and roundtrip['step']==e and roundtrip['config_sha256']==sha(directory/'config.json')
             and roundtrip['selected']==roundtrip['serialized']==table and roundtrip['reset_performed'] is (e==6000)
             and roundtrip['initial']==(plan['initial_state'] if e==6000 else None),'Endpoint serialization/final reset identity differs')
        for value in (endpoint,roundtrip):need(value['checkpoint']==str(checkpoint) and value['checkpoint_sha256']==desc['checkpoint_sha256'],'Endpoint checkpoint hash differs')
        raw=torch.load(desc['raw_file'],map_location='cpu',weights_only=True);logits=raw['logits']
        need(raw['schema_version']==1 and raw['endpoint_step']==e and Path(desc['raw_file'])==data/f'logits_{e:04d}.pt'
             and Path(desc['predictions_file'])==directory/f'predictions_{e:04d}.json'
             and Path(desc['endpoint_file'])==directory/f'endpoint_{e:04d}.json' and Path(desc['roundtrip_file'])==directory/f'roundtrip_{e:04d}.json'
             and raw['sids']==[r['sid'] for r in rows] and logits.dtype==torch.float16 and logits.shape==(108,152064)
             and bool(torch.isfinite(logits).all()) and p.tensor_info(logits)==desc['raw_tensor']
             and raw['batches']==desc['raw_batches'] and len(raw['batches'])==7,'Endpoint full-vocabulary raw ownership differs')
        packets[e]=raw;endpoint_audits[str(e)]=dict(passed=True,endpoint_step=e,checkpoint=str(checkpoint),checkpoint_sha256=desc['checkpoint_sha256'],
            selected_parameter_sha256=object_sha(table),endpoint_file=desc['endpoint_file'],endpoint_sha256=desc['endpoint_sha256'],
            roundtrip_file=desc['roundtrip_file'],roundtrip_sha256=desc['roundtrip_sha256'],evaluation_preserved_training_state=True,descriptive_only=e!=6000)
    for key in ('checkpoint','checkpoint_sha256','endpoint_file','endpoint_sha256','roundtrip_file','roundtrip_sha256','raw_file','raw_sha256','raw_tensor','raw_batches','predictions_file','predictions_sha256','first_token_fit'):
        need(summary[key]==evaluations['6000'][key],'Ordinary summary alias must bind only step6000: '+key)
    need(summary['selected_parameter_sha256']==endpoint_audits['6000']['selected_parameter_sha256'],'Primary deployed endpoint identity differs')
    records=read(summary['captures_file']);expected=[]
    for step in range(1,6001):
        if step in DIAGNOSTIC_STEPS:expected.append(('training',step,None))
        if step in ENDPOINTS:expected.extend(('evaluation',batch,step) for batch in range(1,8))
    need([(r['phase'],r['step'],r['endpoint_step']) for r in records]==expected and all(r['arm']==p.ARM for r in records),'Chronological29-capture inventory differs')
    need(sorted(f.name for f in root.glob('*.pt'))==sorted(['initial.pt','step_0600.pt','step_2000.pt','final.pt']+[f'training_core_{s:04d}.pt' for s in DIAGNOSTIC_STEPS]),'Fixed checkpoint inventory differs')
    need(sorted(f.name for f in directory.glob('progress_*.json'))==[f'progress_{s:04d}.json' for s in PROGRESS],'Four progress snapshots required')
    for step in PROGRESS:
        count=step+7*sum(e<=step for e in ENDPOINTS)
        prior=[r for r in records if (r['step'] if r['phase']=='training' else r['endpoint_step'])<=step]
        need(read(directory/f'progress_{step:04d}.json')==dict(step=step,training=logs[:step],captures=prior,
            counters=dict(core=count,norm=count,head=count,vlm=0,vision=0),evaluations={str(e):evaluations[str(e)] for e in ENDPOINTS if e<=step}),'Progress snapshot/endpoint chronology differs')
    audits=[];gpu_nll={e:[] for e in ENDPOINTS};cursors={e:0 for e in ENDPOINTS}
    with torch.no_grad():
        for record in records:
            phase,step,e=record['phase'],record['step'],record['endpoint_step']
            expected_weight=root/f'training_core_{step:04d}.pt' if phase=='training' else checkpoint_path(root,e)
            expected_capture=data/(f'training_capture_{step:04d}.pt' if phase=='training' else f'evaluation_{e:04d}_capture_{step:04d}.pt')
            need(Path(record['weights_file'])==expected_weight and Path(record['file'])==expected_capture,'Capture/checkpoint paths differ')
            audit,actual=audit_capture(torch,out,record,plan,scenes,states,index,by_sid,rows,initial,weights,logs,norm,head);audits.append(audit)
            if phase=='evaluation':
                batch=packets[e]['batches'][step-1];count=len(record['sids']);cursor=cursors[e]
                need(batch['capture_file']==record['file'] and batch['capture_sha256']==record['sha256'] and batch['sids']==record['sids']
                     and batch['logits']==record['tensors']['logits'] and len(batch['gpu_fp32_nll'])==count
                     and torch.equal(actual,packets[e]['logits'][cursor:cursor+count]), 'Endpoint batch/capture ownership differs')
                cursors[e]+=count;gpu_nll[e]+=batch['gpu_fp32_nll']
    need(set(cursors.values())=={108},'Every endpoint requires108 raw rows')
    endpoint_rows={};strata=[]
    for e in ENDPOINTS:
        desc=evaluations[str(e)];logits=packets[e]['logits'];ids=logits.argmax(-1).tolist();nll=fp64_nll(torch,logits,[r['first_token_id'] for r in rows])
        rescored=[dict(r,argmax_id=ids[i],first_token_correct=ids[i]==r['first_token_id'],nll=float(nll[i]),gpu_fp32_nll=gpu_nll[e][i]) for i,r in enumerate(rows)]
        precision=[dict(sid=r['sid'],fp64_nll=r['nll'],gpu_fp32_nll=r['gpu_fp32_nll'],absolute_difference=abs(r['nll']-r['gpu_fp32_nll']),
            passed=close(r['nll'],r['gpu_fp32_nll'])) for r in rescored]
        saved=read(desc['predictions_file']);save(out/f'outcomes_{e:04d}.json',rescored);save(out/f'nll_precision_{e:04d}.json',precision)
        need(len(saved)==108 and all(set(a)==set(b) and all(a[k]==b[k] for k in a if k!='nll') and close(a['nll'],b['nll']) for a,b in zip(rescored,saved))
             and all(r['passed'] for r in precision),'Endpoint raw argmax/NLL/metadata rescore differs')
        criterion=first_token_criterion(rescored);need(criterion==desc['first_token_fit'],'Endpoint screen counts differ')
        endpoint_rows[e]=rescored;endpoint_audits[str(e)].update(first_token_fit=criterion,mean_first_token_nll=sum(r['nll'] for r in rescored)/108,
            outcomes_file=str(out/f'outcomes_{e:04d}.json'),outcomes_sha256=sha(out/f'outcomes_{e:04d}.json'))
        for key in ('gold','question','n_frames','contrast_id'):
            groups=defaultdict(list)
            for row in rescored:groups[row[key]].append(row)
            for label,group in sorted(groups.items(),key=lambda x:str(x[0])):
                strata.append(dict(labels=dict(endpoint_step=e,**{key:label}),metrics=dict(correct=sum(r['first_token_correct'] for r in group),contexts=len(group),
                    mean_nll=sum(r['nll'] for r in group)/len(group),all_correct=all(r['first_token_correct'] for r in group))))
    final_criterion=primary_screen(endpoint_rows);need(final_criterion==summary['first_token_fit'],'Only final6000 screen may decide')
    save(out/'strata.json',strata);save(out/'training_loss_descriptions.json',training_loss_descriptions(logs,scenes))
    return dict(passed=True,completed=True,arm=p.ARM,slurm_job_id=config['slurm_job_id'],run_directory=str(directory),run_summary_sha256=sha(directory/'summary.json'),
        first_token_fit=final_criterion,primary_endpoint_step=6000,endpoint_audits=endpoint_audits,training=training,counters=counters,capture_audits=audits,
        cpu_head_calls=21,cpu_norm_calls=21,cpu_head_rows=sum(r['head_rows'] for r in audits),
        maximum_native_replay_tv=max(m['full_vocabulary_tv'] for r in audits for m in r['native_head']),
        strata_file=str(out/'strata.json'),strata_sha256=sha(out/'strata.json'),
        training_loss_descriptions_file=str(out/'training_loss_descriptions.json'),training_loss_descriptions_sha256=sha(out/'training_loss_descriptions.json'))

def training_loss_descriptions(logs,scenes):
    result=old.training_loss_descriptions(logs,scenes);by_cycle=defaultdict(lambda:defaultdict(list))
    for row in logs:
        cursor=0
        for i,sid in enumerate(row['sids']):
            count=len(scenes[sid]['target_ids']);values=row['position_losses']['ce'][cursor:cursor+count];cursor+=count
            group=by_cycle[row['cycles'][i//2]];group['first_token'].append(values[0]);group['EOS'].append(values[-1]);group['name_continuation'].extend(values[1:-1])
    for cycle in result['cycles']:
        cycle['token_roles']=[dict(labels=dict(role=role),metrics=dict(positions=len(values),mean_nll=None if not values else sum(values)/len(values)))
            for role,values in sorted(by_cycle[cycle['cycle']].items())]
    result.update(horizon_and_annealing_both_changed=True,all_complete_cycles_and_final_partial_retained=True,no_convergence_certificate=True)
    return result


def report(args,out,frozen):
    import torch
    torch.set_num_threads(4);p=driver();plan=p.verify_plan(args.plan,ancestors=True)
    need(frozen==plan['source_sha256']==p.sources() and len(frozen)==6 and len(args.runs)==1
         and plan['inherited_source_sha256']==p.inherited_sources() and len(plan['inherited_source_sha256'])==106,'New/inherited source or one-run inventory differs')
    inherited=dict(plan_file=plan['joint_code_plan']['file'],plan_sha256=plan['joint_code_plan']['sha256'],source_sha256=plan['inherited_source_sha256'])
    need(read(out/'inherited_sources.json')==inherited,'Report inherited-source descriptor differs')
    need(plan['decision_step']==6000 and plan['evaluation_steps']==list(ENDPOINTS) and plan['original_order_prefix_equal'] is True
         and plan['reused_unfitted_readout'] is plan['oracle_local_semantics'] is plan['no_answer_or_intersection_code'] is True,'Registered horizon/scope differs')
    trigger=plan['short_control_report'];bind(trigger['file'],trigger['sha256']);short=read(trigger['file'])
    need(short['passed'] is short['completed'] is True and short['phase']=='report' and short['first_token_screen'][p.ARM]['passed'] is False,'Original short-control failure must remain preserved')
    bind(short['analysis_file'],short['analysis_sha256'])
    rows,scenes,order,states,index,by_sid,initial=audit_inputs(torch,plan)
    save(out/'input_reconstruction_audit.json',dict(passed=True,joint_code_plan=plan['joint_code_plan'],contexts=108,local_occurrences=1296,
        pair_presentations=48000,original4800_prefix_exact=True,training_target_positions=plan['training_head_rows'],endpoint_rows=324,
        semantic_codes_independently_reconstructed=True,selected_unfitted_readout_exact=True,consumed_input_bytes_unchanged=True,
        oracle_local_semantics=True,no_new_data_or_representation=True))
    need(p.oracle.native_module_identity(torch,plan['native_identity'])==plan['native_module_identity'],'Actual native norm/head source differs')
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True);norm,head=p.oracle.native_modules(torch,packet,plan['native_identity'],'cpu');del packet
    before=dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight))
    run=audit_run(torch,out,args.runs[0],args.plan,plan,frozen,rows,scenes,order,states,index,by_sid,initial,norm,head)
    need(before==dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight)) and norm.weight.grad is head.weight.grad is None
         and not norm.weight.requires_grad and not head.weight.requires_grad,'Independent native weights changed')
    need(run['cpu_head_calls']==run['cpu_norm_calls']==21 and run['cpu_head_rows']==324 and len(run['capture_audits'])==29,'All29 captures/21 head batches required')
    runs={p.ARM:run};budget=resources(out,runs)
    result=dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),joint_code_plan=plan['joint_code_plan'],short_control_report=trigger,
        runs=runs,resources=budget,primary_endpoint_step=6000,first_token_screen={p.ARM:run['first_token_fit']},
        cpu_head_calls=21,cpu_norm_calls=21,cpu_head_rows=324,gpu_head_calls=6021,gpu_norm_calls=6021,gpu_head_rows=plan['total_head_rows'],vlm_calls=0,vision_calls=0,
        horizon_and_annealing_both_changed=True,earlier_endpoints_descriptive_only=True,oracle_local_semantics=True,no_answer_or_intersection_code=True,
        no_runtime_method_claim=True,cached_first_query_only=True,no_native_or_whole_answer_claim=True,no_generalization_or_reasoning_claim=True,
        no_new_fit_or_model_forward=True,all_outcomes_retained=True)
    save(out/'analysis.json',result);criterion=run['first_token_fit']
    lines=['# Extended optimization of the unchanged local joint-code oracle','',
        'Independent computation and provenance audits passed. The same privileged per-image semantic codes and four unfitted readout tensors were used. '
        'Both the optimization horizon and cosine annealing horizon changed from 600 to 6,000 updates.','',
        '| Endpoint | Correct /108 | Complete families /18 | Mean first-token NLL | Decision role |','|---|---:|---:|---:|---|']
    for e in ENDPOINTS:
        value=run['endpoint_audits'][str(e)];c=value['first_token_fit'];role='Descriptive only' if e!=6000 else ('PASS' if c['passed'] else 'FAIL')
        lines.append(f"| {e} | {c['correct']} | {c['complete_families']} | {value['mean_first_token_nll']:.9f} | {role} |")
    lines+=['','Only step 6,000 controls the unchanged 103/108 and 16/18-family screen. Every endpoint includes all 108 training contexts; all three raw archives are retained. '
        'Intermediate evaluations preserved parameters, optimizer state, parameter versions, native weights, inputs and training mode. No early score selected a checkpoint, restart or schedule change.','',
        ('The longer horizon/schedule fits this privileged training screen, so the shorter recipe was insufficient relative to this control. This does not show that native visual states can supply the codes or that the join generalizes.' if criterion['passed'] else
         'The fixed longer horizon/schedule still fails the privileged training screen. This remains an incomplete fit under a bounded recipe, not an information or capacity impossibility result. The budget-extension route stops.'),'',
        'Full-name-plus-EOS CE uses the original mean-scene weighting and all 48,000 paired presentations. First-token, continuation and EOS losses are reported separately over the full trajectory and each complete cycle; the partial tail is retained. '
        'Those trends are descriptive, not convergence certificates. Prior failures and closed architecture branches remain unchanged; no subsequent fit or native/benchmark evaluation is automatically released.','',
        f"Allocated GPU cost: {budget['allocated_gpu_seconds']} seconds. GPU work: 6,021 core/norm/head calls each and {plan['total_head_rows']:,} head rows, with zero VLM/vision calls. "
        f"CPU replay covered 21 endpoint batches / 324 rows; maximum full-vocabulary TV {run['maximum_native_replay_tv']:.9f}, fixed TV <= 0.02 and exact argmax. "
        'All 29 captured computations and semantic-code provenance were independently audited.\n']
    (out/'REPORT.md').write_text('\n'.join(lines))
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),primary_endpoint_step=6000,
        first_token_screen=result['first_token_screen'],resources=budget,oracle_local_semantics=True,no_runtime_method_claim=True,cached_first_query_only=True)


def main():
    import argparse,time
    p=driver();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs=1,required=True);args=parser.parse_args();p.native.require_slurm(gpu=False)
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

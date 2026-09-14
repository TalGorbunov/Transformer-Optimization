"""Independent first-query supervision audit with unchanged native forwards.

Only the fixed6000 endpoint is scored. This diagnostic never authorizes native
or fresh evaluation, even when the cached first-query training screen passes.
"""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from collections import Counter,defaultdict
import ast
import math
import os
import re
import subprocess
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
from scripts import report_native_identity_join_joint_code as old
from scripts import report_native_identity_join_orientation_paired_code as previous
from scripts import report_native_identity_join_factor_orientation_training as orientation
OWN=('scripts/report_native_identity_join_orientation_first_query_code.py',)
DIAGNOSTIC_STEPS=(1,2,32,128,300,600,2000,6000)
ENDPOINTS=(6000,)
PROGRESS=(600,2000,4000,6000)
WEIGHT_NAMES=old.WEIGHT_NAMES
close=old.close
layout_for=old.layout_for
first_token_criterion=orientation.first_token_criterion
expected_order=orientation.expected_order
learning_rate=orientation.learning_rate
fp64_nll=old.fp64_nll
tensor_error=old.tensor_error
replay_metric=old.replay_metric
allocation_rows=old.allocation_rows
check_weights=old.check_weights
gather=old.gather
functional_core=old.functional_core
local_code=old.local_code
bind=old.bind


def driver():
    from scripts import diagnose_native_identity_join_orientation_first_query_code as module
    return module


def primary_screen(endpoint_rows):
    need(set(endpoint_rows)=={6000},'Only the fixed6000 diagnostic endpoint is authorized')
    return first_token_criterion(endpoint_rows[6000])


def first_query_terms(layout):
    sequences=layout['target_sequences'];offsets=layout['offsets']
    need(len(sequences)==16 and len(offsets)==17 and offsets[0]==0
         and all(b-a==len(ids) for a,b,ids in zip(offsets,offsets[1:],sequences)),
         'Exactly16 complete scene sequences are required')
    indices=offsets[:-1];weights=[1/(16*len(ids)) for ids in sequences]
    return indices,weights,[i in indices for i in range(offsets[-1])]


def objective_fixture(torch):
    from gnnformer.parallel_local_joint_code_oracle import ParallelLocalJointCodeOracle
    from gnnformer.paired_sequence_objectives import sequence_layout
    p=driver();sequences=[list(ids) for pair in range(8) for ids in ([[1,6]]*2 if pair%2==0 else [[2,3,6]]*2)]
    layout=sequence_layout(sequences);indices,weights,mask=first_query_terms(layout);q=len(layout['targets'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);core=ParallelLocalJointCodeOracle(4,rank=18)
        head=torch.nn.Linear(4,7,bias=False,dtype=torch.float16).requires_grad_(False)
    norm=torch.nn.Identity();codes=torch.zeros(2,q,18);codes[0,:,0]=1;codes[1,:,1]=1
    valid=torch.ones(2,q,dtype=torch.bool);g=torch.tensor([.2,-.3,.4,.1],dtype=torch.float16).repeat(q,1)
    with torch.no_grad():core.up.weight.copy_(.03*torch.arange(72,dtype=torch.float32).cos().reshape(4,18))
    observed=[];handle=head.register_forward_hook(lambda module,args,output:observed.append(output))
    try:
        objective,full,residual,parts,cap=p.losses(torch,core,codes,g,layout,valid,norm,head)
        _,old_full,old_residual,old_parts,old_cap=p.base.losses(torch,core,codes,g,layout,valid,norm,head)
    finally:handle.remove()
    need(len(observed)==2 and torch.equal(observed[0],observed[1])
         and torch.equal(full,old_full) and torch.equal(residual,old_residual)
         and all(torch.equal(cap[k],old_cap[k]) for k in cap)
         and all(parts[k]==v for k,v in old_parts.items()),'The first-only loss changed a full-prefix forward or existing diagnostic')
    need(parts['first_query_indices']==indices and parts['first_query_weights']==weights
         and parts['first_query_position_mask']==mask and parts['full_name_EOS_diagnostic_only'] is True,
         'First-query metadata or original non-renormalized weights differ')
    m=torch.tensor(mask);first_grad=torch.autograd.grad(objective,observed[0],retain_graph=True)[0][0]
    full_grad=torch.autograd.grad(full,observed[0],retain_graph=True)[0][0]
    delta_grad=torch.autograd.grad(objective,cap['delta'],retain_graph=True)[0]
    need(torch.equal(first_grad[m],full_grad[m]) and bool((first_grad[~m]==0).all())
         and bool((first_grad[m].abs().sum(-1)>0).all()) and bool((delta_grad[~m]==0).all())
         and bool((delta_grad[m].abs().sum(-1)>0).all()),'Actual loss must preserve first gradients and zero every later logit/delta gradient')
    gradients=torch.autograd.grad(objective,tuple(core.parameters()),allow_unused=True)
    need(all(v is not None and bool(torch.isfinite(v).all()) for v in gradients)
         and head.weight.grad is None and not g.requires_grad and not codes.requires_grad,
         'First-query graph or frozen fixture inputs differ')
    expected=sum(parts['position_losses']['ce'][i]*w for i,w in zip(indices,weights))
    need(close(float(objective),expected) and close(parts['first_query_objective'],expected)
         and sum(weights)<1 and set(weights)=={1/32,1/48},'Original mixed-length first-position coefficients changed')
    return dict(passed=True,actual_loss_helper=True,full_prefix_forward_and_diagnostics_exact=True,
        first_logit_gradients_match_full_CE=True,later_logit_and_delta_gradients_exact_zero=True,
        nonzero_first_gradients=True,original_mixed_length_weights=True,no_pretrained_head_forward=True)


def self_test(torch):
    return dict(passed=True,parent=previous.self_test(torch),first_query_objective=objective_fixture(torch))


def audit_inputs(torch,plan):
    p=driver();ref=plan['paired_code_plan'];bind(ref['file'],ref['sha256'])
    need(ref==dict(file=str(p.PARENT),sha256=p.PARENT_SHA),'Exact passed paired-code preparation required')
    parent=previous.driver().verify_plan(ref['file'],ancestors=True)
    rows,scenes,order,states,index,by_sid,initial=previous.audit_inputs(torch,parent)
    for key in ('rows_file','scenes_file','pairs_file','training_samples_file','code_inventory_file','order_file','head_rows_file'):
        need(sha(plan[key])==sha(parent[key]),'Unchanged original216 input bytes differ: '+key)
    for key in ('codes_sha256','code_tensor','features_sha256','feature_tensor','global_feature_ids','initial_sha256','initial_state',
                'native_identity','native_identity_sha256','native_module_identity','native_model_file','native_model_sha256',
                'parent_initial_file','parent_initial_sha256','factor_plan','uniform_plan','orientation_plan','joint_code_plan',
                'joint_code_orientation_plan','paired_order_stage','negative_evidence','capacity_witness','training_stage','trainability_mask',
                'order_object_sha256','head_rows_by_update','training_head_rows','total_head_rows'):
        need(plan[key]==parent[key],'Original code/readout/native/order identity differs: '+key)
    for key in ('codes','features','initial'):
        bind(plan[key+'_file'],plan[key+'_sha256']);need(sha(plan[key+'_file'])==sha(parent[key+'_file']),'Copied frozen artifact differs: '+key)
    need(plan['paired_minibatch_only'] is False and plan['unchanged_presentation_multiset'] is True
         and plan['paired_minibatch_order_retained'] is plan['first_query_supervision_only'] is plan['full_target_forward_unchanged'] is True,
         'Only backward supervision may change')
    control=plan['paired_code_report'];bind(control['file'],control['sha256']);bind(control['analysis_file'],control['analysis_sha256'])
    need(control['file']==str(p.CONTROL) and control['sha256']==p.CONTROL_SHA
         and control['analysis_sha256']==p.CONTROL_ANALYSIS_SHA,'Exact passed full-sequence comparison required')
    proof=read(control['file']);analysis=read(control['analysis_file']);run=analysis['runs'][p.ARM]
    need(proof['passed'] is proof['completed'] is analysis['passed'] is analysis['completed'] is analysis['all_numerical_audits_collected'] is True
         and proof['analysis_file']==control['analysis_file'] and proof['analysis_sha256']==control['analysis_sha256']
         and proof['source_sha256']==analysis['source_sha256']==parent['source_sha256']
         and proof['inherited_source_sha256']==analysis['inherited_source_sha256']==parent['inherited_source_sha256']
         and analysis['plan_file']==ref['file'] and analysis['plan_sha256']==ref['sha256']
         and run['passed'] is True and len(run['capture_audits'])==22 and run['cpu_head_calls']==14 and run['cpu_head_rows']==216,
         'Completed same-parent numerical comparison audit differs')
    endpoint=run['endpoint_audits']['6000'];bind(endpoint['outcomes_file'],endpoint['outcomes_sha256']);outcomes=read(endpoint['outcomes_file'])
    need(len(outcomes)==216 and all({k:a[k] for k in b}==b for a,b in zip(outcomes,rows))
         and all(r['first_token_correct']==(r['argmax_id']==r['first_token_id']) for r in outcomes),'Parent full-CE outcome ownership differs')
    criterion=first_token_criterion(outcomes)
    need(criterion==run['first_token_fit']==proof['first_token_screen'][p.ARM]
         and criterion['first_correct']==68 and criterion['complete_families']==0 and criterion['passed'] is False,
         'Registered negative full-sequence control differs')
    return rows,scenes,order,states,index,by_sid,initial


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
             and row['consistency_coefficient']==0. and row['weighted_consistency_loss']==0., 'Full-name mean-scene CE diagnostic differs')
        indices,weights,mask=first_query_terms(layout);objective=sum(ce[i]*w for i,w in zip(indices,weights))
        need(row['first_query_indices']==indices and row['first_query_weights']==weights and row['first_query_position_mask']==mask
             and row['full_name_EOS_diagnostic_only'] is True and close(row['loss'],objective) and close(row['first_query_objective'],objective),
             'Original-weight first-query-only backward objective differs')
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
        terminal_first_query_objective=logs[-1]['loss'],terminal_full_CE_diagnostic=logs[-1]['ce_loss'],terminal_residual_diagnostic=logs[-1]['consistency_loss'],consistency_coefficient=0.,
        first_query_loss_reductions_verified=True,full_sequence_CE_diagnostic_reductions_verified=True,
        original_first_query_weights_unchanged=True,later_CE_excluded_from_backward=True,no_independent_optimizer_trajectory_replay=True)


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
        need(1<=step<=14 and packet['step']==record['weights_step']==endpoint_step and all(torch.equal(w[k],endpoints[endpoint_step][k]) for k in initial),'Evaluation capture changed its prespecified endpoint')
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
    native_metrics=[];normalized_error=None;cpu_replay=None
    if not training:
        normalized=norm(cap['fused_global'].unsqueeze(0))[0];replayed=head(normalized.unsqueeze(0))[0]
        replay_file=p.DATA/out.name/f'evaluation_{endpoint_step:04d}_batch_{step:04d}_native_replay.pt'
        torch.save(dict(capture_file=record['file'],capture_sha256=record['sha256'],sids=sids,
            normalized=normalized,logits=replayed,native_input_shape=[1,len(g),3584]),replay_file)
        cpu_replay=dict(file=str(replay_file),sha256=sha(replay_file),normalized=p.tensor_info(normalized),logits=p.tensor_info(replayed))
        native_metrics=replay_metric(torch,cap['logits'],replayed);normalized_error=float((normalized.float()-cap['normalized'].float()).abs().max())
    targets=layout['targets'] if training else layout['target_ids'];nll=fp64_nll(torch,cap['logits'],targets)
    precision=[dict(position=i,fp64=float(a),gpu_fp32=b,passed=close(float(a),b)) for i,(a,b) in
        enumerate(zip(nll,logs[step-1]['position_losses']['ce']))] if training else []
    first_objective=None
    if training:
        first_indices,first_weights,_=first_query_terms(layout)
        expected_first=sum(float(nll[i])*weight for i,weight in zip(first_indices,first_weights))
        first_objective=dict(fp64_from_saved_logits=expected_first,gpu_fp32=logs[step-1]['loss'],passed=close(expected_first,logs[step-1]['loss']))
    audit=dict(passed=all(v['passed'] for v in metrics.values()) and all(v['passed'] for v in native_metrics+precision)
        and (first_objective is None or first_objective['passed']),
        arm=p.ARM,phase=phase,step=step,endpoint_step=endpoint_step,sids=sids,capture_file=record['file'],capture_sha256=record['sha256'],
        weights_file=record['weights_file'],weights_sha256=record['weights_sha256'],functional_core=metrics,
        fixed_local_codes_exact=True,fp16_cast_before_add_exact=True,native_head=native_metrics,normalized_max_absolute=normalized_error,
        nll=precision,first_query_objective=first_objective,cpu_replay=cpu_replay,captured_rows=len(g),head_rows=0 if training else len(g),cpu_norm_calls=0 if training else 1,cpu_head_calls=0 if training else 1)
    save(out/(f'training_{step:04d}_capture_audit.json' if training else f'evaluation_{endpoint_step:04d}_{step:04d}_capture_audit.json'),audit)
    return audit,cap['logits']


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
    result.update(orientation_schedule_unchanged=True,all_complete_cycles_and_final_partial_retained=True,no_convergence_certificate=True)
    return result


def checkpoint_path(root,step):
    need(step==6000,'No intermediate checkpoint is authorized');return root/'final.pt'


def audit_run(torch,out,directory,plan_path,plan,frozen,rows,scenes,order,states,index,by_sid,initial,norm,head):
    p=driver();directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json')
    need(directory.parent==p.OUT and directory.name==config['run_id']==f'run_{config["slurm_job_id"]}' and config['arm']==p.ARM
         and config['protocol']==p.PROTOCOL and config['policy']==p.POLICY and config['seed']==24
         and config['source_sha256']==frozen==plan['source_sha256'] and all(summary.get(k)==v for k,v in config.items())
         and Path(config['plan_file']).resolve()==Path(plan_path).resolve() and config['plan_sha256']==sha(plan_path), 'Run/config/source/plan identity differs')
    for name,digest in frozen.items():bind(directory/'source'/name.replace('/','_'),digest)
    inherited=dict(plan_file=plan['paired_code_plan']['file'],plan_sha256=plan['paired_code_plan']['sha256'],source_sha256=plan['inherited_source_sha256'])
    need(config['inherited_source_sha256']==plan['inherited_source_sha256'] and read(directory/'inherited_sources.json')==inherited, 'Inherited source descriptor differs')
    for flag in ('passed','completed','computational_integrity_passed','checkpoint_roundtrip_passed','native_weights_unchanged','codes_frozen',
                 'cached_first_query_only','no_pretrained_backbone_loaded','no_dev_or_test','no_native_or_whole_answer_claim','oracle_local_semantics','no_answer_or_intersection_code'):
        need(summary[flag] is True,'Run invariant failed: '+flag)
    need(summary['phase']=='run' and summary['steps']==6000 and not any(k.startswith('permutation_') for k in summary),'Horizon/scope differs')
    need(config['hardware']['gpu']=='NVIDIA B200' and config['hardware']['matmul_allow_tf32'] is False
         and config['hardware']['torch_version']==str(torch.__version__),'Native hardware/precision differs')
    for key in ('native_identity','native_identity_sha256','native_module_identity','native_model_file','native_model_sha256',
                'codes_file','codes_sha256','code_tensor','code_inventory_sha256','features_file','features_sha256','feature_tensor',
                'parent_initial_file','parent_initial_sha256','paired_code_plan','paired_code_report','first_query_supervision_only','full_target_forward_unchanged','joint_code_plan','joint_code_orientation_plan','paired_order_stage','negative_evidence','orientation_plan','capacity_witness','training_stage','factor_plan','uniform_plan','architecture_stop_report','decision_step','evaluation_steps'):
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
    counters=dict(core=6014,norm=6014,head=6014,vlm=0,vision=0);calls=read(summary['calls_file'])
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
             and endpoint['parameter_sha256']==object_sha(table) and endpoint['native_before']==endpoint['native_after']==native_table
             and endpoint['input_tensors_before']==endpoint['input_tensors_after']==dict(codes=plan['code_tensor'],globals=plan['feature_tensor'])
             and endpoint['input_versions_unchanged'] is True and endpoint['training_mode_restored'] is True
             and endpoint['optimizer_before']==endpoint['optimizer_after'] and endpoint['parameter_versions_before']==endpoint['parameter_versions_after'],
             'Intermediate evaluation mutated readout/native/input/optimizer/trainability state')
        optimizer_identity=optimizer_audit(torch,endpoint['optimizer_before'],e,w)
        if optimizer_ids is None:optimizer_ids=optimizer_identity
        need(optimizer_ids==optimizer_identity and set(endpoint['parameter_versions_before'])==set(WEIGHT_NAMES)
             and all(type(v) is int and v>=0 for v in endpoint['parameter_versions_before'].values()),'Parameter objects/version metadata changed across observations')
        count=e+14*sum(x<=e for x in ENDPOINTS)
        need(endpoint['counters']==dict(core=count,norm=count,head=count,vlm=0,vision=0),'Endpoint cumulative call count differs')
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
    need([(r['phase'],r['step'],r['endpoint_step']) for r in records]==expected and all(r['arm']==p.ARM for r in records),'Chronological22-capture inventory differs')
    need(sorted(f.name for f in root.glob('*.pt'))==sorted(['initial.pt','final.pt']+[f'training_core_{s:04d}.pt' for s in DIAGNOSTIC_STEPS]),'Fixed checkpoint inventory differs')
    need(sorted(f.name for f in directory.glob('progress_*.json'))==[f'progress_{s:04d}.json' for s in PROGRESS],'Four progress snapshots required')
    for step in PROGRESS:
        count=step+14*sum(e<=step for e in ENDPOINTS)
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
    need(set(cursors.values())=={216},'The final endpoint requires216 raw rows')
    endpoint_rows={};strata=[]
    for e in ENDPOINTS:
        desc=evaluations[str(e)];logits=packets[e]['logits'];ids=logits.argmax(-1).tolist();nll=fp64_nll(torch,logits,[r['first_token_id'] for r in rows])
        rescored=[dict(r,argmax_id=ids[i],first_token_correct=ids[i]==r['first_token_id'],nll=float(nll[i]),gpu_fp32_nll=gpu_nll[e][i]) for i,r in enumerate(rows)]
        saved=read(desc['predictions_file'])
        need(len(saved)==216 and all(set(a)==set(b) and all(a[k]==b[k] for k in a if k!='nll') for a,b in zip(rescored,saved)),
             'Endpoint actual argmax/scene ownership differs')
        precision=[dict(sid=r['sid'],fp64_nll=r['nll'],gpu_fp32_nll=r['gpu_fp32_nll'],absolute_difference=abs(r['nll']-r['gpu_fp32_nll']),
            saved_fp64_nll=b['nll'],saved_fp64_passed=close(r['nll'],b['nll']),
            passed=close(r['nll'],r['gpu_fp32_nll']) and close(r['nll'],b['nll'])) for r,b in zip(rescored,saved)]
        save(out/f'outcomes_{e:04d}.json',rescored);save(out/f'nll_precision_{e:04d}.json',precision)
        criterion=first_token_criterion(rescored);need(criterion==desc['first_token_fit'],'Endpoint screen counts differ')
        endpoint_rows[e]=rescored;endpoint_audits[str(e)].update(nll_precision_passed=all(r['passed'] for r in precision),first_token_fit=criterion,mean_first_token_nll=sum(r['nll'] for r in rescored)/216,
            outcomes_file=str(out/f'outcomes_{e:04d}.json'),outcomes_sha256=sha(out/f'outcomes_{e:04d}.json'))
        for key in ('gold','question','n_frames','base_contrast_id','orientation_version'):
            groups=defaultdict(list)
            for row in rescored:groups[row[key]].append(row)
            for label,group in sorted(groups.items(),key=lambda x:str(x[0])):
                strata.append(dict(labels=dict(endpoint_step=e,**{key:label}),metrics=dict(correct=sum(r['first_token_correct'] for r in group),contexts=len(group),
                    mean_nll=sum(r['nll'] for r in group)/len(group),all_correct=all(r['first_token_correct'] for r in group))))
    final_criterion=primary_screen(endpoint_rows);need(final_criterion==summary['first_token_fit'],'Only final6000 screen may decide')
    save(out/'strata.json',strata);save(out/'training_loss_descriptions.json',training_loss_descriptions(logs,scenes))
    return dict(passed=all(r['passed'] for r in audits) and all(r['nll_precision_passed'] for r in endpoint_audits.values()),
        completed=True,all_numerical_audits_collected=True,arm=p.ARM,slurm_job_id=config['slurm_job_id'],run_directory=str(directory),run_summary_sha256=sha(directory/'summary.json'),
        first_token_fit=final_criterion,primary_endpoint_step=6000,endpoint_audits=endpoint_audits,training=training,counters=counters,capture_audits=audits,
        cpu_head_calls=14,cpu_norm_calls=14,cpu_head_rows=sum(r['head_rows'] for r in audits),
        maximum_native_replay_tv=max(m['full_vocabulary_tv'] for r in audits for m in r['native_head']),
        strata_file=str(out/'strata.json'),strata_sha256=sha(out/'strata.json'),
        training_loss_descriptions_file=str(out/'training_loss_descriptions.json'),training_loss_descriptions_sha256=sha(out/'training_loss_descriptions.json'))


def resources(out,runs):
    p=driver();command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'all_user_sacct.psv';file.write_text(raw)
    names=set(p.RUN_JOBS.values());records=allocation_rows(raw,names)
    need(len(records)==1 and {r['name'] for r in records}==names,'Exactly one orientation joint-code GPU attempt required; failed/zero attempts retained')
    row=records[0];run=runs[p.ARM]
    need(row['job_id']==run['slurm_job_id'] and row['partition']=='gpu' and row['state']=='COMPLETED' and row['exit_code']=='0:0'
         and row['gpus']==1 and row['seconds']<=120 and row['gpu_seconds']<=120,'Single120-second GPU diagnostic allocation differs')
    from datetime import datetime
    need(datetime.fromisoformat(row['end'])>=datetime.fromisoformat(row['start']),'Invalid allocation time interval')
    return dict(passed=True,jobs=records,allocated_gpu_seconds=row['gpu_seconds'],maximum_concurrent_gpus=1,
        failed_and_zero_allocations_retained=True,file=str(file),sha256=sha(file),command=command)


def report(args,out,frozen):
    import torch
    torch.set_num_threads(4);p=driver();plan=p.verify_plan(args.plan,ancestors=True)
    need(frozen==plan['source_sha256']==p.sources() and len(frozen)==6 and len(args.runs)==1
         and plan['inherited_source_sha256']==p.inherited_sources(),'Source or one-run inventory differs')
    inherited=dict(plan_file=plan['paired_code_plan']['file'],plan_sha256=plan['paired_code_plan']['sha256'],source_sha256=plan['inherited_source_sha256'])
    need(read(out/'inherited_sources.json')==inherited and plan['paired_code_plan']==dict(file=str(p.PARENT),sha256=p.PARENT_SHA)
         and plan['joint_code_plan']==dict(file=str(p.JOINT_CODE_PLAN),sha256=p.JOINT_CODE_PLAN_SHA),'Exact original plan/source descriptor differs')
    need(plan['decision_step']==6000 and plan['evaluation_steps']==[6000]
         and plan['reused_unfitted_readout'] is plan['oracle_local_semantics'] is plan['no_answer_or_intersection_code'] is True,'Diagnostic scope or selected initialization differs')
    gate=plan['capacity_witness'];bind(gate['file'],gate['sha256']);bind(gate['analysis_file'],gate['analysis_sha256'])
    summary=read(gate['file']);need(summary['analysis_file']==gate['analysis_file'] and summary['analysis_sha256']==gate['analysis_sha256'],'Capacity-witness analysis descriptor differs')
    witness=p.witness.verify_witness(gate['file'])
    need(witness['capacity_witness_passed'] is True and witness['orientation_plan']==plan['orientation_plan']
         and witness['native_identity']==plan['native_identity'] and witness['native_identity_sha256']==plan['native_identity_sha256'], 'Passed same-data/native capacity witness required')
    rows,scenes,order,states,index,by_sid,initial=audit_inputs(torch,plan)
    (p.DATA/out.name).mkdir(parents=True,exist_ok=False)
    save(out/'input_reconstruction_audit.json',dict(passed=True,orientation_plan=plan['orientation_plan'],joint_code_plan=plan['joint_code_plan'],capacity_witness=gate,
        contexts=216,local_occurrences=2592,pair_presentations=48000,shared_paired_minibatch_order_exact=True,training_target_positions=213330,
        endpoint_rows=216,semantic_codes_independently_reconstructed=True,selected_unfitted_readout_exact=True,global_only_native_inputs_exact=True,
        oracle_local_semantics=True,no_answer_or_intersection_code=True,no_new_data_or_representation=True))
    need(p.oracle.native_module_identity(torch,plan['native_identity'])==plan['native_module_identity'],'Actual frozen native module source differs')
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True);norm,head=p.oracle.native_modules(torch,packet,plan['native_identity'],'cpu');del packet
    before=dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight));versions=(norm.weight._version,head.weight._version)
    run=audit_run(torch,out,args.runs[0],args.plan,plan,frozen,rows,scenes,order,states,index,by_sid,initial,norm,head)
    need(before==dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight)) and versions==(norm.weight._version,head.weight._version)
         and norm.weight.grad is head.weight.grad is None and not norm.weight.requires_grad and not head.weight.requires_grad,'Independent frozen native weights changed')
    need(run['cpu_head_calls']==run['cpu_norm_calls']==14 and run['cpu_head_rows']==216 and len(run['capture_audits'])==22,'All22 captures/14 final head batches required')
    runs={p.ARM:run};budget=resources(out,runs)
    result=dict(passed=run['passed'],completed=True,all_numerical_audits_collected=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),orientation_plan=plan['orientation_plan'],joint_code_plan=plan['joint_code_plan'],
        capacity_witness=gate,paired_code_plan=plan['paired_code_plan'],paired_code_report=plan['paired_code_report'],
        joint_code_orientation_plan=plan['joint_code_orientation_plan'],paired_order_stage=plan['paired_order_stage'],negative_evidence=plan['negative_evidence'],runs=runs,resources=budget,primary_endpoint_step=6000,first_token_screen={p.ARM:run['first_token_fit']},
        cpu_head_calls=14,cpu_norm_calls=14,cpu_head_rows=216,gpu_head_calls=6014,gpu_norm_calls=6014,gpu_head_rows=213546,vlm_calls=0,vision_calls=0,
        optimizer_settings_and_exposure_unchanged=True,paired_minibatch_order_retained=True,
        first_query_supervision_only=True,original_first_query_weights_unchanged=True,full_target_forward_unchanged=True,oracle_local_semantics=True,no_answer_or_intersection_code=True,
        no_runtime_method_claim=True,cached_first_query_only=True,no_native_or_whole_answer_claim=True,no_generalization_or_reasoning_claim=True,
        no_new_fit_or_backbone_forward=True,no_automatic_native_or_fresh_release=True,all_outcomes_retained=True)
    save(out/'analysis.json',result);criterion=run['first_token_fit']
    lines=['# First-query supervision local joint-code diagnostic','',
        ('All independent computation and provenance audits passed. ' if run['passed'] else 'The unchanged numerical audit failed; all prespecified comparisons and raw replays are retained. ')+
        'This privileged control uses raw per-image person/room codes and the exact four original unfitted readout tensors. '
        'Only backward supervision changes from the passed paired full-sequence control: first-query CE retains its original1/(16*Lscene) coefficients; continuation and EOS CE remain diagnostic. The6000-update optimizer settings,48000 paired presentations, all full-prefix forward shapes and final-only decision remain fixed.','',
        '| Endpoint | Correct /216 | Original /108 | Flipped /108 | Complete families /18 | Saved GPU screen |',
        '|---|---:|---:|---:|---:|---|',
        f"|6000|{criterion['first_correct']}|{criterion['orientation_correct']['original']}|{criterion['orientation_correct']['flipped']}|{criterion['complete_families']}|{'PASS' if criterion['passed'] else 'FAIL'}|",'',
        'The registered screen requires206 pooled correct,103 in each orientation and16 complete twelve-context families. '
        'All216 first-query full-vocabulary logits and all22 captured computations are retained. CPU replay covers14 native-shape batches/216 rows, using the unchanged TV<=0.02 and exact-argmax rule.','',
        ('With continuation/EOS supervision removed, this fixed optimizer fits the privileged first-query screen. This comparison implicates the omitted supervision under this recipe; it does not establish visual feature learning, whole-answer generation or a general optimization cause.' if run['passed'] and criterion['passed'] else
         'The original numerical fidelity gate failed. No training-screen conclusion is accepted from this report.' if not run['passed'] else
         'Removing continuation/EOS supervision was insufficient for this fixed privileged first-query training screen. The capacity witness remains distinct from a gradient-descent learnability guarantee; this result does not isolate a unique remaining optimization cause.'),'',
        'No result here releases native generation, fresh confirmation, continuation, an architecture change or a new optimization budget. '
        'Prior failed studies remain unchanged. Logged first-token/continuation/EOS losses and cycle trends are descriptive observations, not convergence certificates.','',
        f"Allocated GPU cost: {budget['allocated_gpu_seconds']} seconds. The run used6014 core/norm/head calls each and213546 head rows, with no VLM or vision calls. "
        f"Maximum CPU native replay TV: {run['maximum_native_replay_tv']:.9f}.\n"]
    (out/'REPORT.md').write_text('\n'.join(lines))
    need(run['passed'],'Unchanged numerical core/native/NLL gate failed; all22 capture audits and14 native replay batches retained')
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),
        primary_endpoint_step=6000,first_token_screen=result['first_token_screen'],resources=budget,capacity_witness=gate,
        paired_code_plan=plan['paired_code_plan'],paired_code_report=plan['paired_code_report'],first_query_supervision_only=True,full_target_forward_unchanged=True,
        oracle_local_semantics=True,no_runtime_method_claim=True,cached_first_query_only=True,no_automatic_native_or_fresh_release=True)


def main():
    import argparse,time
    p=driver();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs=1,required=True);args=parser.parse_args();p.native.require_slurm(gpu=False)
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only independent report')
    out=p.OUT/f'report_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    frozen=p.snapshot(out);save(out/'request.json',dict(plan=str(args.plan),runs=list(map(str,args.runs)),source_sha256=frozen))
    try:
        result=report(args,out,frozen);need(p.sources()==frozen and p.inherited_sources()==result['inherited_source_sha256'],'Source changed during report')
        result['elapsed_seconds']=time.perf_counter()-started;need(result['elapsed_seconds']<=300,'Fixed CPU report cap exceeded');save(out/'summary.json',result)
        print(__import__('json').dumps(dict(passed=True,completed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True,
            analysis_file=str(out/'analysis.json') if (out/'analysis.json').exists() else None,
            analysis_sha256=sha(out/'analysis.json') if (out/'analysis.json').exists() else None));raise


if __name__=='__main__':main()

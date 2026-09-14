"""First-query supervision with unchanged full-prefix native forwards.

Only the loss sent to backward changes. All later-prefix CE remains diagnostic.
"""
from __future__ import annotations
import argparse
import ast
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_joint_code_long as previous
from scripts import diagnose_native_identity_join_factor_orientation_training as orientation
from scripts import witness_native_identity_join_orientation_capacity as witness
from scripts import diagnose_native_identity_join_orientation_paired_code as parent_code
from scripts import stage_native_identity_join_orientation_paired_order as paired
original=previous.original
from scripts.stage_native_vision_v6_teacher import need,read,sha,save,object_sha
uniform=original.uniform;oracle=original.oracle;base=original.base;native=original.native
tensor_info=original.tensor_info;bind=original.bind;cpu_tree=original.cpu_tree
make_core=original.make_core;trainability=original.trainability;batch_states=original.batch_states
PEOPLE=original.PEOPLE;ROOMS=original.ROOMS;READOUT_KEYS=original.READOUT_KEYS;code_indices=original.code_indices
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_orientation_first_query_code'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_orientation_first_query_code')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_orientation_first_query_code')
PROTOCOL='identity_join_orientation_first_query_local_joint_code_oracle'
ARM='joint_code';ARMS=(ARM,);RUN_JOBS={ARM:'identity_join_orientation_first_query_code_run'}
PARENT=parent_code.OUT/'check_443728/plan.json'
PARENT_SHA='07ff5b435a0f1bda1791a05289ec7c07916f6bc7b186e9ddceaacbceedd0f62f'
CONTROL=parent_code.OUT/'report_443742/summary.json'
CONTROL_SHA='951130b304f5f21b7750d42ddad9b87289514a04fb26bf72acd2236b8d505ad0'
CONTROL_ANALYSIS_SHA='32391ac1e17084d915b209e28b0d448a4789f29f043423019a7b6b9dd5b8d502'
JOINT_CODE_PLAN=original.OUT/'check_443472/plan.json'
JOINT_CODE_PLAN_SHA='f8d11406cbf1e2766bbde03859b45ba15148b5429804257b060f28c4a5bdce63'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_ORIENTATION_FIRST_QUERY_CODE_PROPOSAL.md'
PROPOSAL_SHA='26d8e84c35baf926f8e0fd21301b2358ef02edbae12414268889b3d0e5d79a24'
DIAGNOSTIC_STEPS=(1,2,32,128,300,600,2000,6000)
EVALUATION_STEPS=(6000,)
PROGRESS_STEPS=(600,2000,4000,6000)
POLICY=dict(parent_code.POLICY,protocol=PROTOCOL,
    loss='first query CE only with original1/(16*Lscene) weights; no renormalization',
    paired_minibatch_only=False,paired_minibatch_order_retained=True,first_query_supervision_only=True,
    full_target_forward_unchanged=True,full_name_EOS_CE_diagnostic_only=True,
    full_prefix_native_head_shapes_unchanged=True,no_new_initialization=True)

OWN=('scripts/diagnose_native_identity_join_orientation_first_query_code.py','scripts/report_native_identity_join_orientation_first_query_code.py',
     'slurm/native_identity_join_orientation_first_query_code_check.sbatch','slurm/native_identity_join_orientation_first_query_code_run.sbatch',
     'slurm/native_identity_join_orientation_first_query_code_report.sbatch',PROPOSAL)


def inherited_sources():
    result={}
    for group in (parent_code.sources(),parent_code.inherited_sources()):
        for name,digest in group.items():
            need(name not in result or result[name]==digest,'Inherited source conflict');result[name]=digest
    for name,digest in result.items():need(sha(REPO/name)==digest,'Inherited source changed: '+name)
    return result


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,digest in frozen.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Source copy changed')
    save(out/'source_hashes.json',frozen)
    save(out/'inherited_sources.json',dict(plan_file=str(PARENT),plan_sha256=PARENT_SHA,source_sha256=inherited_sources()))
    return frozen


def learning_rate(step):
    need(type(step) is int and 1<=step<=6000,'Unregistered training step')
    return .001*step/50 if step<=50 else 1e-5+(.001-1e-5)*(1+math.cos(math.pi*(step-50)/5950))/2


def first_query_objective(torch,ce_positions,layout):
    indices=layout['offsets'][:-1]
    weights=torch.tensor(layout['scene_weights'],device=ce_positions.device)
    mask=torch.zeros_like(ce_positions,dtype=torch.bool);mask[indices]=True
    objective=(ce_positions*weights*mask).sum()
    return objective,indices,[layout['scene_weights'][i] for i in indices],mask


def losses(torch, core, local, g, layout, valid, norm, head):
    from gnnformer.paired_sequence_objectives import sequence_layout
    import torch.nn.functional as F
    need(layout==sequence_layout(layout['target_sequences']) and not local.requires_grad and not g.requires_grad,
         'Changed layout or trainable frozen features')
    delta,cap=core(local,g,valid_mask=valid,output_dtype=torch.float32,capture=True)
    native_hidden=g+delta.to(g.dtype)
    need(bool(torch.isfinite(native_hidden).all()), 'Native training addition overflowed')
    logits=head(norm(native_hidden.unsqueeze(0)))[0]
    left=torch.tensor(layout['left'],device=g.device); right=torch.tensor(layout['right'],device=g.device)
    need(torch.equal(g[left],g[right]), 'Paired frozen strict-prefix states differ')
    with torch.autocast(device_type=g.device.type,enabled=False):
        ce_positions=F.cross_entropy(logits.float(),torch.tensor(layout['targets'],device=g.device),reduction='none')
        denominator=g[left].detach().float().square().sum(-1)+1e-6
        residual_positions=(delta[right]-delta[left]).square().sum(-1)/denominator
        ce=(ce_positions*torch.tensor(layout['scene_weights'],device=g.device)).sum()
        residual=(residual_positions*torch.tensor(layout['pair_weights'],device=g.device)).sum()
        objective,first_indices,first_weights,first_mask=first_query_objective(torch,ce_positions,layout)
    with torch.no_grad():
        offsets=layout['offsets']; lengths=[b-a for a,b in zip(offsets,offsets[1:])]
        pair_losses=[];cursor=0
        for length in lengths[::2]:
            pair_losses.append(float(residual_positions[cursor:cursor+length].mean()));cursor+=length
        selected=cap['gates'][valid]
        parts=dict(first_query_indices=first_indices,first_query_weights=first_weights,first_query_position_mask=first_mask.tolist(),
            first_query_objective=float(objective),full_name_EOS_diagnostic_only=True,
            scene_lengths=lengths,scene_offsets=offsets,pair_lengths=lengths[::2],
            prefix_ids=[prefix for row in layout['prefixes'] for prefix in row],
            per_scene_ce=[float(ce_positions[a:b].mean()) for a,b in zip(offsets,offsets[1:])],
            per_pair_consistency=pair_losses,
            position_losses=dict(ce=ce_positions.tolist(),consistency=residual_positions.tolist()),
            pair_position_denominator=denominator.tolist(),
            residual_difference_norms=torch.linalg.vector_norm(delta[right]-delta[left],dim=-1).tolist(),
            valid_item_count_by_position=valid.sum(0).tolist(), gate_sum_by_position=cap['gates'].sum(0).tolist(),
            gate_min=float(selected.min()),gate_max=float(selected.max()),gate_mean=float(selected.mean()),
            gate_zero_count=int((selected==0).sum()),gate_one_count=int((selected==1).sum()),
            valid_gate_count=selected.numel(),payload_max_abs=float(cap['payload'].abs().max()),
            padding_messages_exact_zero=bool((cap['messages'][~valid]==0).all()),
            closed_message_coordinates_exact_zero=bool((cap['messages'][cap['gates']==0]==0).all()))
    need(bool(torch.isfinite(objective)) and bool(torch.isfinite(ce)) and bool(torch.isfinite(residual)) and parts['padding_messages_exact_zero']
         and parts['closed_message_coordinates_exact_zero'], 'Nonfinite objective or invalid message path')
    return objective,ce,residual,parts,cap



def self_test(torch):
    from gnnformer.paired_sequence_objectives import sequence_layout
    layout=sequence_layout([[1,2],[1,2],[3,4,5],[3,4,5]])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);logits=torch.randn(10,7,requires_grad=True)
    targets=torch.tensor(layout['targets']);ce=torch.nn.functional.cross_entropy(logits,targets,reduction='none')
    objective,indices,weights,mask=first_query_objective(torch,ce,layout)
    full=(ce*torch.tensor(layout['scene_weights'])).sum()
    gradient=torch.autograd.grad(objective,logits,retain_graph=True)[0]
    reference=torch.autograd.grad(full,logits)[0]
    need(indices==[0,2,4,7] and weights==[1/8,1/8,1/12,1/12] and int(mask.sum())==4
         and torch.equal(gradient[mask],reference[mask]) and bool((gradient[~mask]==0).all())
         and bool((gradient[mask].abs().sum(-1)>0).all()),'Original first weights/nonzero first/zero later gradient fixture failed')
    changed=logits.detach().clone();changed[~mask]=changed[~mask].flip(-1)*3+5
    changed_ce=torch.nn.functional.cross_entropy(changed,targets,reduction='none')
    changed_objective=first_query_objective(torch,changed_ce,layout)[0]
    need(torch.equal(objective.detach(),changed_objective) and not torch.equal(full.detach(),(changed_ce*torch.tensor(layout['scene_weights'])).sum()),
         'Later-position logits must affect only the full-CE diagnostic')
    return dict(passed=True,parent=parent_code.self_test(torch),original_first_query_weights_exact=True,
        first_position_gradients_unchanged=True,later_position_gradients_exact_zero=True,later_logits_objective_invariant=True)

def criteria(predictions):return orientation.criteria(predictions)


def control_report(bindings):
    bind(CONTROL,bindings,CONTROL_SHA);proof=read(CONTROL)
    bind(proof['analysis_file'],bindings,CONTROL_ANALYSIS_SHA);analysis=read(proof['analysis_file'])
    need(proof['passed'] is proof['completed'] is True and proof['analysis_sha256']==CONTROL_ANALYSIS_SHA
         and proof['source_sha256']==parent_code.sources() and proof['inherited_source_sha256']==parent_code.inherited_sources()
         and analysis['passed'] is analysis['completed'] is analysis['all_numerical_audits_collected'] is True
         and analysis['plan_file']==str(PARENT) and analysis['plan_sha256']==PARENT_SHA,'Completed paired-code numerical audit required')
    run=analysis['runs'][ARM];fit=run['first_token_fit']
    need(run['passed'] is True and len(run['capture_audits'])==22 and run['cpu_head_calls']==14 and run['cpu_head_rows']==216
         and fit['first_correct']==68 and fit['complete_families']==0 and fit['passed'] is False,
         'Registered paired full-sequence control outcome differs')
    summary_file=Path(run['run_directory'])/'summary.json';bind(summary_file,bindings,run['run_summary_sha256']);summary=read(summary_file)
    need(summary['plan_sha256']==PARENT_SHA and summary['first_token_fit']==fit,'Paired-code run ownership differs')
    for name,digest in proof['source_sha256'].items():bind(CONTROL.parent/'source'/name.replace('/','_'),bindings,digest)
    for key in ('raw','predictions'):bind(summary[key+'_file'],bindings,summary[key+'_sha256'])
    return dict(file=str(CONTROL),sha256=CONTROL_SHA,analysis_file=proof['analysis_file'],analysis_sha256=CONTROL_ANALYSIS_SHA)


def check(args,out,frozen):
    import torch
    torch.set_num_threads(4);bindings={};bind(PARENT,bindings,PARENT_SHA);parent=parent_code.verify_plan(PARENT,ancestors=True)
    reference=control_report(bindings);bind(REPO/PROPOSAL,bindings,PROPOSAL_SHA)
    for file,digest in parent['runtime_bindings'].items():bind(file,bindings,digest)
    witness.verify_witness(parent['capacity_witness']['file']);paired.verify_stage(parent['paired_order_stage']['file'])
    rows=read(parent['rows_file']);scenes=read(parent['scenes_file']);order=read(parent['order_file'])
    need(len(rows)==len(scenes)==216 and len(order)==48000,'Complete fixed cohort/presentation inventory required')
    counts=[sum(len(scenes[sid]['target_ids']) for pair in order[i:i+8] for sid in pair['sids']) for i in range(0,48000,8)]
    need(counts==parent['head_rows_by_update'] and len(counts)==6000 and sum(counts)==213330 and max(counts)<=48,
         'Every full-prefix native forward shape must remain unchanged')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    copied={}
    for key,destination in (('codes',data/'codes.pt'),('features',data/'global_features.pt'),('initial',ckpt/'initial.pt')):
        shutil.copyfile(parent[key+'_file'],destination);need(sha(destination)==parent[key+'_sha256'],'Original unfitted/input bytes changed: '+key);copied[key]=destination
    for key,name in (('rows_file','rows.json'),('scenes_file','scenes.json'),('pairs_file','pairs.json'),('order_file','order.json'),
                     ('head_rows_file','head_rows.json'),('code_inventory_file','code_inventory.json'),('training_samples_file','training_samples.json')):
        shutil.copyfile(parent[key],out/name);need(sha(out/name)==sha(parent[key]),'Fixed copied JSON changed: '+key)
    packet=torch.load(copied['codes'],map_location='cpu',weights_only=True)
    need(tensor_info(packet['codes'])==parent['code_tensor'] and packet['sids']==[r['sid'] for r in rows],'Original per-image codes differ')
    packet=torch.load(copied['features'],map_location='cpu',weights_only=True)
    need(tensor_info(packet['states'])==parent['feature_tensor'] and packet['feature_ids']==parent['global_feature_ids'],'Original native globals differ')
    packet=torch.load(copied['initial'],map_location='cpu',weights_only=True);core=make_core(torch);core.load_state_dict(packet['branch']);mask=trainability(core)
    need(base.v7.state_info(core)==parent['initial_state'] and mask==parent['trainability_mask'] and bool((packet['branch']['up.weight']==0).all()),'Original unfitted readout differs')
    tests=self_test(torch)
    from scripts.report_native_identity_join_orientation_first_query_code import self_test as reporter_self_test
    tests['reporter']=reporter_self_test(torch)
    module_identity=oracle.native_module_identity(torch,parent['native_identity']);need(module_identity==parent['native_module_identity'],'Actual native module changed')
    runtime_bindings={}
    for file in (out/'rows.json',out/'scenes.json',out/'pairs.json',out/'order.json',out/'head_rows.json',out/'training_samples.json',out/'code_inventory.json',
                 *copied.values(),Path(parent['native_model_file']),CONTROL,Path(reference['analysis_file'])):bind(file,runtime_bindings)
    for file,digest in parent['runtime_bindings'].items():bind(file,runtime_bindings,digest)
    plan=dict(parent,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),tests=tests,
        input_bindings=bindings,runtime_bindings=runtime_bindings,paired_code_plan=dict(file=str(PARENT),sha256=PARENT_SHA),paired_code_report=reference,
        rows_file=str(out/'rows.json'),scenes_file=str(out/'scenes.json'),pairs_file=str(out/'pairs.json'),order_file=str(out/'order.json'),
        training_samples_file=str(out/'training_samples.json'),head_rows_file=str(out/'head_rows.json'),
        code_inventory_file=str(out/'code_inventory.json'),code_inventory_sha256=sha(out/'code_inventory.json'),
        codes_file=str(copied['codes']),codes_sha256=sha(copied['codes']),features_file=str(copied['features']),features_sha256=sha(copied['features']),
        initial_file=str(copied['initial']),initial_sha256=sha(copied['initial']),paired_minibatch_only=False,paired_minibatch_order_retained=True,
        first_query_supervision_only=True,full_target_forward_unchanged=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,training_contexts=216,image_occurrences=2592,
        training_head_rows=213330,final_head_rows=216,total_head_rows=213546,paired_code_plan=plan['paired_code_plan'],paired_code_report=reference,
        oracle_local_semantics=True,no_pretrained_head_forward=True,first_query_supervision_only=True,full_target_forward_unchanged=True)

def verify_plan(path,ancestors=False):
    path=Path(path).resolve();plan=read(path)
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Long joint-code plan/source changed')
    for name,digest in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'New source snapshot changed')
    need(read(path.parent/'inherited_sources.json')==dict(plan_file=str(PARENT),plan_sha256=PARENT_SHA,source_sha256=plan['inherited_source_sha256']),'Inherited source descriptor changed')
    for file,digest in plan['runtime_bindings'].items():need(sha(file)==digest,'Consumed input changed')
    if ancestors:
        for file,digest in plan['input_bindings'].items():need(sha(file)==digest,'Bound parent input changed')
    proof=read(path.parent/'summary.json')
    need(proof['passed'] is True and proof['completed'] is True and proof['phase']=='check' and proof['plan_sha256']==sha(path)
         and object_sha(read(plan['order_file']))==plan['order_object_sha256']
         and object_sha(plan['native_identity'])==plan['native_identity_sha256'],'Passed CPU/native gate required')
    return plan


def single_run_guard(out):
    need(os.environ.get('SLURM_JOB_NAME')==RUN_JOBS[ARM],'Exact registered single-attempt job name required')
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'launch_sacct.psv';file.write_text(raw)
    save(out/'launch_accounting.json',dict(command=command,file=str(file),sha256=sha(file)))
    for line in raw.splitlines():
        fields=line.split('|');need(len(fields)==9,'Unexpected allocation schema')
        if fields[1]==RUN_JOBS[ARM] and fields[2]=='gpu':need(fields[0]==os.environ['SLURM_JOB_ID'],'One longer-oracle GPU attempt only')


def run(args,out,frozen,started):
    import torch
    torch.set_num_threads(4);args.plan=args.plan.resolve();plan=verify_plan(args.plan);single_run_guard(out)
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200'
         and not torch.backends.cuda.matmul.allow_tf32,'One B200 with TF32 disabled required')
    hardware=dict(gpu=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    torch.manual_seed(24);torch.cuda.manual_seed_all(24);device=torch.device('cuda')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    rows=read(plan['rows_file']);scenes=read(plan['scenes_file']);order=read(plan['order_file'])
    save(out/'rows.json',rows);save(out/'presentations.json',order);save(out/'code_inventory.json',read(plan['code_inventory_file']))
    packet=torch.load(plan['features_file'],map_location='cpu',weights_only=True)
    need(tensor_info(packet['states'])==plan['feature_tensor'] and packet['feature_ids']==plan['global_feature_ids'],'Frozen global features changed')
    states=packet['states'].to(device);index={fid:i for i,fid in enumerate(packet['feature_ids'])};del packet
    packet=torch.load(plan['codes_file'],map_location='cpu',weights_only=True)
    inventory=read(plan['code_inventory_file']);offsets=[0]+[s['code_stop'] for s in inventory['scenes']]
    need(set(packet)=={'schema_version','sids','scene_offsets','codes'} and packet['schema_version']==1
         and packet['sids']==[r['sid'] for r in rows] and packet['scene_offsets']==offsets
         and tensor_info(packet['codes'])==plan['code_tensor'],'Fixed local-code packet changed')
    codes=packet['codes'].to(device);code_offsets={sid:(a,b) for sid,a,b in zip(packet['sids'],offsets,offsets[1:])};del packet
    input_versions=(codes._version,states._version);input_before=dict(codes=tensor_info(codes),globals=tensor_info(states))
    need(input_before==dict(codes=plan['code_tensor'],globals=plan['feature_tensor']),'Deployed code/global tensor identity differs')
    core=make_core(torch).to(device);initial=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    core.load_state_dict(initial['branch']);mask=trainability(core)
    need(base.v7.state_info(core)==plan['initial_state'] and mask==plan['trainability_mask'],'Selected original unfitted readout differs')
    initial_file=ckpt/'initial.pt';shutil.copyfile(plan['initial_file'],initial_file)
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=oracle.native_modules(torch,packet,plan['native_identity'],device);del packet
    module_identity=oracle.native_module_identity(torch,plan['native_identity']);need(module_identity==plan['native_module_identity'],'Native module source changed')
    native_initial=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight));versions=(norm.weight._version,head.weight._version)
    counters=dict(core=0,norm=0,head=0,vlm=0,vision=0);calls=[];captures=[];current_phase='training';current_step=0;current_endpoint=None;capture_current=False;io={}
    def observer(name):
        def hook(module,arguments,output):
            counters[name]+=1
            calls.append(dict(module=name,phase=current_phase,step=current_step,endpoint_step=current_endpoint,input_shape=list(arguments[0].shape),
                output_shape=list(output.shape),input_dtype=str(arguments[0].dtype),output_dtype=str(output.dtype)))
            if capture_current:
                if name=='norm':io.update(fused_global=arguments[0][0].detach().cpu().clone(),normalized=output[0].detach().cpu().clone())
                else:io['logits']=output[0].detach().cpu().clone()
        return hook
    def core_observer(module,args,output):counters['core']+=1
    handles=[norm.register_forward_hook(observer('norm')),head.register_forward_hook(observer('head')),core.register_forward_hook(core_observer)]
    config=dict(protocol=PROTOCOL,run_id=out.name,slurm_job_id=os.environ['SLURM_JOB_ID'],arm=ARM,seed=24,policy=POLICY,
        source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],plan_file=str(args.plan),plan_sha256=sha(args.plan),
        factor_plan=plan['factor_plan'],uniform_plan=plan['uniform_plan'],architecture_stop_report=plan['architecture_stop_report'],
        joint_code_plan=plan['joint_code_plan'],orientation_plan=plan['orientation_plan'],capacity_witness=plan['capacity_witness'],training_stage=plan['training_stage'],decision_step=6000,evaluation_steps=list(EVALUATION_STEPS),
        joint_code_orientation_plan=plan['joint_code_orientation_plan'],paired_order_stage=plan['paired_order_stage'],negative_evidence=plan['negative_evidence'],
        paired_code_plan=plan['paired_code_plan'],paired_code_report=plan['paired_code_report'],first_query_supervision_only=True,full_target_forward_unchanged=True,
        reused_unfitted_readout=True,oracle_local_semantics=True,no_answer_or_intersection_code=True,
        hardware=hardware,native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        native_module_identity=module_identity,native_model_file=plan['native_model_file'],native_model_sha256=plan['native_model_sha256'],
        native_weight_identity=native_initial,initialized=base.v7.state_info(core),initial_checkpoint=str(initial_file),initial_checkpoint_sha256=sha(initial_file),
        parent_initial_file=plan['parent_initial_file'],parent_initial_sha256=plan['parent_initial_sha256'],
        features_file=plan['features_file'],features_sha256=plan['features_sha256'],feature_tensor=plan['feature_tensor'],
        codes_file=plan['codes_file'],codes_sha256=plan['codes_sha256'],code_tensor=plan['code_tensor'],
        rows_file=str(out/'rows.json'),rows_sha256=sha(out/'rows.json'),presentations_file=str(out/'presentations.json'),
        presentations_sha256=sha(out/'presentations.json'),order_sha256=object_sha(order),
        code_inventory_file=str(out/'code_inventory.json'),code_inventory_sha256=sha(out/'code_inventory.json'),
        data_directory=str(data),checkpoint_directory=str(ckpt),**mask)
    save(out/'config.json',config)
    parameters=list(core.parameters());optimizer=torch.optim.AdamW(parameters,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    torch.cuda.synchronize();setup=time.perf_counter()-started;logs=[];evaluations={}
    def optimizer_snapshot():
        return dict(parameter_ids={name:id(p) for name,p in core.named_parameters()},
            optimizer_parameter_ids=[id(p) for group in optimizer.param_groups for p in group['params']],
            requires_grad={name:p.requires_grad for name,p in core.named_parameters()},
            state_tensor_info={name:{key:tensor_info(value) if isinstance(value,torch.Tensor) else value
                for key,value in optimizer.state[p].items()} for name,p in core.named_parameters()},
            param_group_hyperparameters=[{key:value for key,value in group.items() if key!='params'} for group in optimizer.param_groups])
    def save_capture(local,g,valid,cap,layout,sids,weights_file,weights_step):
        name=f'training_capture_{current_step:04d}.pt' if current_phase=='training' else f'evaluation_{current_endpoint:04d}_capture_{current_step:04d}.pt'
        blob=dict(schema_version=1,arm=ARM,phase=current_phase,step=current_step,endpoint_step=current_endpoint,sids=sids,layout=layout,
            codes=local,global_states=g,valid_mask=valid,capture=cap,**io)
        path=data/name;torch.save(cpu_tree(torch,blob),path)
        record=dict(arm=ARM,phase=current_phase,step=current_step,endpoint_step=current_endpoint,sids=sids,file=str(path),sha256=sha(path),
            weights_file=str(weights_file),weights_sha256=sha(weights_file),weights_step=weights_step,tensors={k:tensor_info(v) for k,v in io.items()})
        captures.append(record);return record
    def evaluate(update):
        nonlocal current_phase,current_step,current_endpoint,capture_current,io
        optimizer.zero_grad(set_to_none=True);table=base.v7.state_info(core)
        checkpoint=ckpt/('final.pt' if update==6000 else f'step_{update:04d}.pt')
        torch.save(dict(branch=cpu_tree(torch,core.state_dict()),step=update,config=config),checkpoint)
        packet=torch.load(checkpoint,map_location='cpu',weights_only=True)
        need(packet['step']==update and packet['config']==config,'Fixed endpoint serialization metadata differs')
        serialized={name:tensor_info(value) for name,value in packet['branch'].items()}
        need(serialized==table,'Serialized endpoint tensor identities differ')
        if update==6000:
            core.load_state_dict(initial['branch']);need(base.v7.state_info(core)==plan['initial_state'],'Final reset failed')
            core.load_state_dict(packet['branch']);need(base.v7.state_info(core)==table,'Final restricted reload differs')
        roundtrip_file=out/f'roundtrip_{update:04d}.json'
        save(roundtrip_file,dict(passed=True,step=update,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),config_sha256=sha(out/'config.json'),
            selected=table,serialized=serialized,reset_performed=update==6000,initial=plan['initial_state'] if update==6000 else None))
        del packet
        before_optimizer=optimizer_snapshot();before_versions={name:p._version for name,p in core.named_parameters()}
        before_input=dict(codes=tensor_info(codes),globals=tensor_info(states));before_native=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight))
        was_training=core.training;core.eval();current_phase='evaluation';current_endpoint=update;capture_current=True;raw_batches=[]
        try:
            with torch.no_grad():
                for offset in range(0,216,16):
                    current_step=offset//16+1;io={};chosen=rows[offset:offset+16];sids=[r['sid'] for r in chosen]
                    h,g,layout,valid=batch_states(torch,scenes,states,index,codes,code_offsets,sids)
                    first=torch.tensor(layout['offsets'][:-1],device=device);h=h[:,first];g=g[first];valid=valid[:,first]
                    final_layout=dict(first_query_only=True,target_ids=[r['first_token_id'] for r in chosen],full_target_ids=[r['target_ids'] for r in chosen],
                        first_indices=layout['offsets'][:-1],original_full_layout=layout)
                    delta,cap=core(h,g,valid_mask=valid,output_dtype=torch.float32,capture=True)
                    logits=head(norm((g+delta.half()).unsqueeze(0)))[0]
                    record=save_capture(h,g,valid,cap,final_layout,sids,checkpoint,update)
                    gpu_nll=torch.nn.functional.cross_entropy(logits.float(),torch.tensor(final_layout['target_ids'],device=device),reduction='none')
                    raw_batches.append(dict(capture_file=record['file'],capture_sha256=record['sha256'],sids=sids,
                        gpu_fp32_nll=gpu_nll.tolist(),logits=tensor_info(io['logits'])))
        finally:core.train(was_training)
        after_optimizer=optimizer_snapshot();after_versions={name:p._version for name,p in core.named_parameters()}
        after_input=dict(codes=tensor_info(codes),globals=tensor_info(states));after_native=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight))
        need(before_optimizer==after_optimizer and before_versions==after_versions and base.v7.state_info(core)==table
             and core.training==was_training and all(p.requires_grad and p.grad is None for p in core.parameters())
             and before_input==after_input==input_before and before_native==after_native==native_initial
             and input_versions==(codes._version,states._version) and versions==(norm.weight._version,head.weight._version),
             'Endpoint evaluation changed optimizer/parameters/mode/native/frozen inputs')
        packets=[torch.load(r['capture_file'],map_location='cpu',weights_only=True) for r in raw_batches]
        raw_logits=torch.cat([p['logits'] for p in packets]);del packets
        raw_file=data/f'logits_{update:04d}.pt';torch.save(dict(schema_version=1,sids=[r['sid'] for r in rows],logits=raw_logits,batches=raw_batches,endpoint_step=update),raw_file)
        targets=torch.tensor([r['first_token_id'] for r in rows]);ids=raw_logits.argmax(-1)
        promoted=raw_logits.double();shift=promoted-promoted.amax(-1,keepdim=True)
        nll=torch.logsumexp(shift,-1)-shift[torch.arange(216),targets];gpu_losses=[v for batch in raw_batches for v in batch['gpu_fp32_nll']]
        predictions=[dict(r,argmax_id=int(ids[i]),first_token_correct=int(ids[i])==r['first_token_id'],gpu_fp32_nll=gpu_losses[i],nll=float(nll[i])) for i,r in enumerate(rows)]
        prediction_file=out/f'predictions_{update:04d}.json';save(prediction_file,predictions)
        criterion=criteria(predictions)
        endpoint_file=out/f'endpoint_{update:04d}.json'
        save(endpoint_file,dict(passed=True,endpoint_step=update,before_evaluation=table,after_evaluation=base.v7.state_info(core),parameter_sha256=object_sha(table),
            native_before=before_native,native_after=after_native,input_tensors_before=before_input,input_tensors_after=after_input,
            input_versions_unchanged=True,optimizer_before=before_optimizer,optimizer_after=after_optimizer,
            parameter_versions_before=before_versions,parameter_versions_after=after_versions,training_mode_restored=True,
            counters=dict(counters),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
        return dict(endpoint_step=update,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),endpoint_file=str(endpoint_file),endpoint_sha256=sha(endpoint_file),
            roundtrip_file=str(roundtrip_file),roundtrip_sha256=sha(roundtrip_file),raw_file=str(raw_file),raw_sha256=sha(raw_file),raw_tensor=tensor_info(raw_logits),raw_batches=raw_batches,
            predictions_file=str(prediction_file),predictions_sha256=sha(prediction_file),first_token_fit=criterion,descriptive_only=update!=6000)
    try:
        for step in range(1,6001):
            current_phase='training';current_step=step;current_endpoint=None;capture_current=step in DIAGNOSTIC_STEPS;io={};tick=time.perf_counter()
            batch=order[(step-1)*8:step*8];sids=[sid for pair in batch for sid in pair['sids']]
            h,g,layout,valid=batch_states(torch,scenes,states,index,codes,code_offsets,sids)
            need(g.shape[0]==plan['head_rows_by_update'][step-1]<=48,'Fixed full-target row budget changed')
            optimizer.zero_grad(set_to_none=True);rate=learning_rate(step);optimizer.param_groups[0]['lr']=rate
            weights_file=None
            if capture_current:
                weights_file=ckpt/f'training_core_{step:04d}.pt'
                torch.save(dict(arm=ARM,branch=cpu_tree(torch,core.state_dict()),step=step-1,optimizer_step=step,position='before_update',source_sha256=frozen),weights_file)
            objective,ce,residual,parts,cap=losses(torch,core,h,g,layout,valid,norm,head)
            need(torch.equal(cap['payload'],h) and torch.equal(cap['gates'],valid.float()*.5) and bool((cap['scores']==.5).all()),'Raw-code half-SUM changed')
            if step==1:need(bool((cap['delta']==0).all()),'Initial U zero identity failed')
            if capture_current:save_capture(h,g,valid,cap,layout,sids,weights_file,step-1)
            objective.backward();need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),'Readout gradient missing/nonfinite')
            gradnorm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            need(all(bool(torch.isfinite(p).all()) for p in core.parameters()) and norm.weight.grad is head.weight.grad is None
                 and versions==(norm.weight._version,head.weight._version) and input_versions==(codes._version,states._version),
                 'Readout/native/frozen-input invariant failed')
            torch.cuda.synchronize()
            logs.append(dict(step=step,sids=sids,pair_ids=[p['pair_id'] for p in batch],cycles=[p['cycle'] for p in batch],target_ids=layout['targets'],
                lr=rate,loss=float(objective),ce_loss=float(ce),consistency_loss=float(residual),weighted_consistency_loss=0.,consistency_coefficient=0.,
                gradient_norm=float(gradnorm),clipped=float(gradnorm)>1.,native_frozen=True,codes_frozen=True,
                code_input_shape=list(h.shape),seconds=time.perf_counter()-tick,**parts))
            del objective,ce,residual,cap
            if step in EVALUATION_STEPS:evaluations[str(step)]=evaluate(step)
            if step in PROGRESS_STEPS:
                save(out/f'progress_{step:04d}.json',dict(step=step,training=logs,captures=captures,counters=counters,evaluations=evaluations))
                print(json.dumps(dict(step=step,ce=logs[-1]['ce_loss'])),flush=True)
        save(out/'training.json',logs);optimizer.zero_grad(set_to_none=True);del optimizer
        core.eval().requires_grad_(False);final=base.v7.state_info(core);last=evaluations['6000']
        save(out/'captures.json',captures);save(out/'calls.json',calls);save(out/'evaluations.json',evaluations)
        expected=dict(core=6014,norm=6014,head=6014,vlm=0,vision=0)
        need(counters==expected and input_versions==(codes._version,states._version) and versions==(norm.weight._version,head.weight._version)
             and dict(codes=tensor_info(codes),globals=tensor_info(states))==input_before
             and dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight))==native_initial
             and all(not p.requires_grad and p.grad is None for p in core.parameters()),'Final endpoint/work invariant failed')
        result=dict(**config,passed=True,completed=True,phase='run',computational_integrity_passed=True,steps=6000,
            **{key:value for key,value in last.items() if key not in ('endpoint_step','descriptive_only')},selected_parameter_sha256=object_sha(final),checkpoint_roundtrip_passed=True,
            training_file=str(out/'training.json'),training_sha256=sha(out/'training.json'),captures_file=str(out/'captures.json'),captures_sha256=sha(out/'captures.json'),
            calls_file=str(out/'calls.json'),calls_sha256=sha(out/'calls.json'),counters=counters,evaluations=evaluations,
            evaluations_file=str(out/'evaluations.json'),evaluations_sha256=sha(out/'evaluations.json'),earlier_endpoints_descriptive_only=False,
            training_head_rows=plan['training_head_rows'],final_head_rows=216,total_head_rows=plan['total_head_rows'],
            setup_before_training_seconds=setup,training_seconds=sum(r['seconds'] for r in logs),native_weights_unchanged=True,
            codes_frozen=True,input_versions_unchanged=True,cached_first_query_only=True,no_pretrained_backbone_loaded=True,no_dev_or_test=True,no_native_or_whole_answer_claim=True)
        need(time.perf_counter()-started<=120,'Single orientation-oracle job exceeded120 seconds');return result
    finally:
        for handle in handles:handle.remove()
        save(out/'final_counters.json',dict(counters=counters,recorded_calls=calls,partial_outputs_retained=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true');action.add_argument('--run',action='store_true');parser.add_argument('--plan',type=Path)
    args=parser.parse_args();native.require_slurm(gpu=args.run)
    if args.check:need(args.plan is None and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'Fixed CPU-only preparation required')
    else:need(args.plan is not None,'GPU consumes only passed CPU plan')
    phase='check' if args.check else 'run';out=OUT/f'{phase}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();frozen=snapshot(out);save(out/'request.json',dict(phase=phase,plan=None if args.plan is None else str(args.plan),source_sha256=frozen))
    try:
        result=check(args,out,frozen) if args.check else run(args,out,frozen,started)
        need(sources()==frozen and inherited_sources()==read(out/'inherited_sources.json')['source_sha256'],'Source changed during execution')
        result['elapsed_seconds']=time.perf_counter()-started;need(result['elapsed_seconds']<=POLICY['cpu_check_seconds' if args.check else 'gpu_seconds_cap'],'Fixed phase cap exceeded');save(out/'summary.json',result)
        print(json.dumps(dict(passed=result['passed'],directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

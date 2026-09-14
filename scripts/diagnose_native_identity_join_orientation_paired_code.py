"""Held paired-minibatch control of the unchanged orientation local-code oracle.

Only the fixed48000-entry order changes. No extra fit, head, or native release.
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
from scripts import diagnose_native_identity_join_orientation_joint_code as parent_code
from scripts import stage_native_identity_join_orientation_paired_order as paired
original=previous.original
from scripts.stage_native_vision_v6_teacher import need,read,sha,save,object_sha
uniform=original.uniform;oracle=original.oracle;base=original.base;native=original.native
tensor_info=original.tensor_info;bind=original.bind;cpu_tree=original.cpu_tree
make_core=original.make_core;trainability=original.trainability;batch_states=original.batch_states
PEOPLE=original.PEOPLE;ROOMS=original.ROOMS;READOUT_KEYS=original.READOUT_KEYS;code_indices=original.code_indices
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_orientation_paired_code'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_orientation_paired_code')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_orientation_paired_code')
PROTOCOL='identity_join_orientation_paired_local_joint_code_oracle'
ARM='joint_code';ARMS=(ARM,);RUN_JOBS={ARM:'identity_join_orientation_paired_code_run'}
PARENT=parent_code.OUT/'check_443711/plan.json'
PARENT_SHA='91798a0fcbbe9357445506f0a98ae99c2790b7927f0053e85416622558b3201e'
JOINT_CODE_PLAN=original.OUT/'check_443472/plan.json'
JOINT_CODE_PLAN_SHA='f8d11406cbf1e2766bbde03859b45ba15148b5429804257b060f28c4a5bdce63'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_ORIENTATION_PAIRED_CODE_PROPOSAL.md'
PROPOSAL_SHA='e2c3ffe7ddb39094804cddc4136a816089913d76ed31a31a9445c219433e0367'
DIAGNOSTIC_STEPS=(1,2,32,128,300,600,2000,6000)
EVALUATION_STEPS=(6000,)
PROGRESS_STEPS=(600,2000,4000,6000)
POLICY=dict(parent_code.POLICY,protocol=PROTOCOL,
    schedule='444 paired complete-cycle groups; original/flipped counterparts adjacent; unchanged48-entry tail',
    maximum_training_head_rows=48,orientation_coverage_only=False,paired_minibatch_only=True,
    unchanged_presentation_multiset=True,no_new_initialization=True)

OWN=('scripts/diagnose_native_identity_join_orientation_paired_code.py','scripts/report_native_identity_join_orientation_paired_code.py',
     'slurm/native_identity_join_orientation_paired_code_check.sbatch','slurm/native_identity_join_orientation_paired_code_run.sbatch',
     'slurm/native_identity_join_orientation_paired_code_report.sbatch',PROPOSAL)


def inherited_sources():
    result={}
    for group in (parent_code.sources(),parent_code.inherited_sources(),paired.sources(),paired.inherited_sources()):
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


def self_test(torch):
    need(learning_rate(1)==.00002 and learning_rate(50)==.001 and learning_rate(6000)==.00001
         and 6000+math.ceil(216/16)==6014 and len(DIAGNOSTIC_STEPS)+math.ceil(216/16)==22,'Fixed work/schedule differs')
    return dict(passed=True,unchanged_oracle=parent_code.self_test(torch),unchanged_horizon_and_endpoint=True)

def criteria(predictions):return orientation.criteria(predictions)


def check(args,out,frozen):
    import torch
    torch.set_num_threads(4);bindings={};bind(PARENT,bindings,PARENT_SHA);parent=parent_code.verify_plan(PARENT,ancestors=True)
    stage_path=args.order_stage.resolve();stage_path=stage_path/'summary.json' if stage_path.is_dir() else stage_path
    stage=paired.verify_stage(stage_path);proof=read(stage_path)
    stage_ref=dict(file=str(stage_path),sha256=bind(stage_path,bindings),plan_file=proof['plan_file'],plan_sha256=proof['plan_sha256'])
    bind(stage_ref['plan_file'],bindings,stage_ref['plan_sha256']);bind(REPO/PROPOSAL,bindings,PROPOSAL_SHA)
    need(stage['orientation_plan']==parent['orientation_plan']
         and stage['joint_code_orientation_plan']==dict(file=str(PARENT),sha256=PARENT_SHA),
         'Shared order must consume the same frozen216-scene/code plans')
    for file,digest in parent['runtime_bindings'].items():bind(file,bindings,digest)
    for file,digest in stage['runtime_bindings'].items():bind(file,bindings,digest)
    rows=read(parent['rows_file']);scenes=read(parent['scenes_file']);pairs=read(parent['pairs_file']);old_order=read(parent['order_file'])
    order=read(stage['order_file']);permutation=read(stage['permutation_file'])
    need(len(rows)==len(scenes)==216 and len(pairs)==108 and len(order)==len(old_order)==48000,'Complete fixed cohort/presentation inventory required')
    need(len(permutation)==48000 and sorted(permutation)==list(range(48000)) and order==[old_order[i] for i in permutation],
         'Shared paired order must be only a permutation of the original48000 entries')
    counts=[sum(len(scenes[sid]['target_ids']) for pair in order[i:i+8] for sid in pair['sids']) for i in range(0,48000,8)]
    need(counts==stage['head_rows_by_update'] and sum(counts)==stage['total_training_head_rows']==213330
         and max(counts)==stage['max_training_head_rows']<=48,'Paired full-target row inventory differs')
    need(order[-48:]==old_order[-48:] and len(counts)==6000,'Original incomplete tail/horizon changed')
    witness.verify_witness(parent['capacity_witness']['file'])
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    for name,value in (('rows',rows),('scenes',scenes),('pairs',pairs),('order',order),('code_inventory',read(parent['code_inventory_file'])),
                       ('training_samples',read(parent['training_samples_file']))):save(out/(name+'.json'),value)
    save(out/'head_rows.json',dict(by_update=counts,total=sum(counts),maximum=max(counts),endpoint_rows=216,all_head_rows=213546))
    copied={}
    for key,destination in (('codes',data/'codes.pt'),('features',data/'global_features.pt'),('initial',ckpt/'initial.pt')):
        shutil.copyfile(parent[key+'_file'],destination);need(sha(destination)==parent[key+'_sha256'],'Original unfitted/input bytes changed: '+key);copied[key]=destination
    packet=torch.load(copied['codes'],map_location='cpu',weights_only=True)
    need(tensor_info(packet['codes'])==parent['code_tensor'] and packet['sids']==[r['sid'] for r in rows],'Original per-image codes differ')
    packet=torch.load(copied['features'],map_location='cpu',weights_only=True)
    need(tensor_info(packet['states'])==parent['feature_tensor'] and packet['feature_ids']==parent['global_feature_ids'],'Original native globals differ')
    packet=torch.load(copied['initial'],map_location='cpu',weights_only=True)
    core=make_core(torch);core.load_state_dict(packet['branch']);mask=trainability(core)
    need(base.v7.state_info(core)==parent['initial_state'] and mask==parent['trainability_mask']
         and bool((packet['branch']['up.weight']==0).all()),'Original unfitted readout differs')
    tests=self_test(torch)
    from scripts.report_native_identity_join_orientation_paired_code import self_test as reporter_self_test
    tests['reporter']=reporter_self_test(torch)
    module_identity=oracle.native_module_identity(torch,parent['native_identity']);need(module_identity==parent['native_module_identity'],'Actual native module changed')
    runtime_bindings={}
    for file in (out/'rows.json',out/'scenes.json',out/'pairs.json',out/'order.json',out/'head_rows.json',out/'training_samples.json',out/'code_inventory.json',
                 *copied.values(),Path(parent['native_model_file']),stage_path,Path(stage_ref['plan_file'])):bind(file,runtime_bindings)
    for file,digest in stage['runtime_bindings'].items():bind(file,runtime_bindings,digest)
    plan=dict(parent,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),tests=tests,
        input_bindings=bindings,runtime_bindings=runtime_bindings,joint_code_orientation_plan=dict(file=str(PARENT),sha256=PARENT_SHA),
        paired_order_stage=stage_ref,negative_evidence=stage['negative_evidence'],trainability_mask=mask,
        rows_file=str(out/'rows.json'),scenes_file=str(out/'scenes.json'),pairs_file=str(out/'pairs.json'),order_file=str(out/'order.json'),
        training_samples_file=str(out/'training_samples.json'),order_object_sha256=object_sha(order),head_rows_by_update=counts,
        head_rows_file=str(out/'head_rows.json'),training_head_rows=213330,total_head_rows=213546,
        code_inventory_file=str(out/'code_inventory.json'),code_inventory_sha256=sha(out/'code_inventory.json'),
        codes_file=str(copied['codes']),codes_sha256=sha(copied['codes']),features_file=str(copied['features']),features_sha256=sha(copied['features']),
        initial_file=str(copied['initial']),initial_sha256=sha(copied['initial']),paired_minibatch_only=True,unchanged_presentation_multiset=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,training_contexts=216,image_occurrences=2592,
        global_feature_count=len(parent['global_feature_ids']),training_head_rows=213330,final_head_rows=216,total_head_rows=213546,
        capacity_witness=parent['capacity_witness'],paired_order_stage=stage_ref,negative_evidence=stage['negative_evidence'],oracle_local_semantics=True,no_pretrained_head_forward=True)

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
            _,ce,residual,parts,cap=base.losses(torch,core,h,g,layout,valid,norm,head)
            need(torch.equal(cap['payload'],h) and torch.equal(cap['gates'],valid.float()*.5) and bool((cap['scores']==.5).all()),'Raw-code half-SUM changed')
            if step==1:need(bool((cap['delta']==0).all()),'Initial U zero identity failed')
            if capture_current:save_capture(h,g,valid,cap,layout,sids,weights_file,step-1)
            ce.backward();need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),'Readout gradient missing/nonfinite')
            gradnorm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            need(all(bool(torch.isfinite(p).all()) for p in core.parameters()) and norm.weight.grad is head.weight.grad is None
                 and versions==(norm.weight._version,head.weight._version) and input_versions==(codes._version,states._version),
                 'Readout/native/frozen-input invariant failed')
            torch.cuda.synchronize()
            logs.append(dict(step=step,sids=sids,pair_ids=[p['pair_id'] for p in batch],cycles=[p['cycle'] for p in batch],target_ids=layout['targets'],
                lr=rate,loss=float(ce),ce_loss=float(ce),consistency_loss=float(residual),weighted_consistency_loss=0.,consistency_coefficient=0.,
                gradient_norm=float(gradnorm),clipped=float(gradnorm)>1.,native_frozen=True,codes_frozen=True,
                code_input_shape=list(h.shape),seconds=time.perf_counter()-tick,**parts))
            del ce,residual,cap
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
    action.add_argument('--check',action='store_true');action.add_argument('--run',action='store_true');parser.add_argument('--plan',type=Path);parser.add_argument('--order-stage',type=Path)
    args=parser.parse_args();native.require_slurm(gpu=args.run)
    if args.check:need(args.order_stage is not None and args.plan is None and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only preparation with passed shared order stage required')
    else:need(args.plan is not None and args.order_stage is None,'GPU consumes only passed CPU plan')
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

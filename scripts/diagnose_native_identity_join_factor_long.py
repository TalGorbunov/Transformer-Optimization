"""Held 6000-update CE calibration of the locked product/additive factor pair.

Only horizon and cosine annealing change. All endpoints use cached first queries;
600/2000 are descriptive and only6000 controls the fixed training-fit decision.
"""
from __future__ import annotations
import argparse
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
from scripts import diagnose_native_identity_join_factor_binding as original
from scripts.stage_native_vision_v6_teacher import need,read,sha,save,object_sha
uniform=original.uniform;oracle=original.oracle;base=original.base;native=original.native
make_core=original.make_core;freeze_selector=original.freeze_selector;query_means=original.query_means
FixedLocalConditioning=original.FixedLocalConditioning;tensor_info=original.tensor_info;bind=original.bind;cpu_tree=original.cpu_tree
PEOPLE=original.PEOPLE;ARMS=('product','additive')
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_factor_long'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_factor_long')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_factor_long')
PROTOCOL='identity_join_locked_factor_long';RUN_JOBS={arm:'identity_join_factor_long_'+arm for arm in ARMS}
PARENT=original.OUT/'check_443373/plan.json'
PARENT_SHA='aa9a44067f86ce6340d56419b5efe9411ccd84cb52115f0b31d10745b4e68a07'
CONDITIONAL=REPO/'outputs/native_aggregation_vlm/identity_join_joint_code_long/report_443511/summary.json'
CONDITIONAL_SHA='b702fa30ab354708f3d82192c7e54ba83af491ebf5010ca880b4df6996718fed'
CONDITIONAL_ANALYSIS_SHA='d0938deeb7212851b6f044ee88d7d09803c17535894c77db4951c2a8ef636ef5'
LONG_PLAN=REPO/'outputs/native_aggregation_vlm/identity_join_joint_code_long/check_443496/plan.json'
LONG_PLAN_SHA='3e692f8fe3f834fecb59407807f99b7c8b533ece25497c08a1f2c2853ac6f308'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_FACTOR_LONG_PROPOSAL_V2.md'
PROPOSAL_SHA='3c4af1bd12ae328c83764bd7fa949d1453d84dade61ca612451bbc7a8c14c875'
DIAGNOSTIC_STEPS=(1,2,32,128,300,600,2000,6000)
EVALUATION_STEPS=(600,2000,6000);PROGRESS_STEPS=(600,2000,4000,6000)
POLICY={key:value for key,value in original.POLICY.items() if key not in ('permutation_head_rows','permutation','permutation_is_descriptive')}
POLICY.update(protocol=PROTOCOL,steps=6000,cosine_end_step=6000,
    schedule='persistent Random(24) shuffled54-pair cycles;48000 pairs; exact original4800-prefix',
    horizon_and_annealing_calibration=True,additional_steps_only=False,fresh_shared_initialization=False,reused_unfitted_initialization=True,
    training_head_rows=None,training_head_rows_rule='CPU exact sum over6000 fixed batches',max_training_head_rows=44,
    final_head_rows=324,total_head_rows={arm:213654 for arm in ARMS},core_calls={arm:6021 for arm in ARMS},
    norm_calls={arm:6021 for arm in ARMS},head_calls={arm:6021 for arm in ARMS},cpu_final_head_calls=21,cpu_final_head_rows=324,
    diagnostic_steps=list(DIAGNOSTIC_STEPS),evaluation_steps=list(EVALUATION_STEPS),decision_step=6000,
    earlier_endpoints_descriptive_only=True,progress_steps=list(PROGRESS_STEPS),final_checkpoint_only=False,
    final_checkpoint_only_for_decision=True,paired_only=True,gpu_seconds_cap=240,campaign_gpu_seconds_cap=480)
OWN=('scripts/diagnose_native_identity_join_factor_long.py','scripts/report_native_identity_join_factor_long.py',
     'slurm/native_identity_join_factor_long_check.sbatch','slurm/native_identity_join_factor_long_run.sbatch',
     'slurm/native_identity_join_factor_long_report.sbatch',PROPOSAL)


def inherited_sources():
    need(sha(PARENT)==PARENT_SHA,'Original factor plan changed');parent=read(PARENT)
    inherited=parent['source_sha256'];need(len(inherited)==99 and inherited==original.sources(),'Original factor helper map changed')
    for name,digest in inherited.items():
        need(sha(REPO/name)==digest and sha(PARENT.parent/'source'/name.replace('/','_'))==digest,'Inherited frozen source changed')
    return inherited


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,digest in frozen.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Source copy changed')
    save(out/'source_hashes.json',frozen)
    save(out/'inherited_sources.json',dict(plan_file=str(PARENT),plan_sha256=PARENT_SHA,source_sha256=inherited_sources()))
    return frozen


def order_pairs(pairs):
    need(len(pairs)==54,'Exactly54 pairs required');rng=random.Random(24);order=[];cycle=0
    while len(order)<48000:
        cycle+=1;indices=list(range(54));rng.shuffle(indices)
        order.extend(dict(cycle=cycle,pair_slot=i,pair_id=pairs[i]['pair_id'],sids=pairs[i]['sids']) for i in indices)
    return order[:48000]


def learning_rate(step):
    need(type(step) is int and 1<=step<=6000,'Unregistered training step')
    return .001*step/50 if step<=50 else 1e-5+(.001-1e-5)*(1+math.cos(math.pi*(step-50)/5950))/2


def self_test(torch):
    pairs=[dict(pair_id=str(i),sids=[f'{i}a',f'{i}b']) for i in range(54)];order=order_pairs(pairs)
    need(order[:4800]==uniform.order_pairs(pairs) and len(order)==48000,'Original order prefix changed')
    need(all({r['pair_slot'] for r in order[start:start+54]}==set(range(54)) for start in range(0,47952,54))
         and len(order[47952:])==48,'Complete cycle/partial tail differs')
    need(learning_rate(1)==.00002 and learning_rate(50)==.001 and learning_rate(6000)==.00001
         and learning_rate(600)>original.learning_rate(600)
         and all(learning_rate(i)>learning_rate(i+1) for i in (50,600,2000,5999)),'Fixed longer cosine differs')
    need(6000+len(EVALUATION_STEPS)*7==6021 and len(DIAGNOSTIC_STEPS)+len(EVALUATION_STEPS)*7==29,'Fixed call/capture count differs')
    return dict(passed=True,extended_order_prefix=True,complete_cycles=888,partial_cycle_pairs=48,
        horizon_and_annealing_changed=True,call_inventory=True,unchanged_factors=original.self_test(torch))


def check(out,frozen):
    import torch
    torch.set_num_threads(4);bindings={};bind(PARENT,bindings,PARENT_SHA);parent=original.verify_plan(PARENT,ancestors=True)
    bind(CONDITIONAL,bindings,CONDITIONAL_SHA);proof=read(CONDITIONAL)
    bind(proof['analysis_file'],bindings,CONDITIONAL_ANALYSIS_SHA);analysis=read(proof['analysis_file'])
    bind(LONG_PLAN,bindings,LONG_PLAN_SHA);long_plan=read(LONG_PLAN)
    need(proof['passed'] is proof['completed'] is analysis['passed'] is analysis['completed'] is True
         and proof['analysis_sha256']==CONDITIONAL_ANALYSIS_SHA and analysis['plan_file']==str(LONG_PLAN)
         and analysis['plan_sha256']==LONG_PLAN_SHA and proof['first_token_screen']==analysis['first_token_screen']
         and proof['first_token_screen']['joint_code']['passed'] is True,'Completed successful long-oracle calibration required')
    bind(REPO/PROPOSAL,bindings,PROPOSAL_SHA)
    for file,digest in parent['runtime_bindings'].items():bind(file,bindings,digest)
    bind(long_plan['order_file'],bindings,long_plan['runtime_bindings'][long_plan['order_file']])
    pairs=read(parent['pairs_file']);rows=read(parent['rows_file']);scenes=read(parent['scenes_file']);order=order_pairs(pairs)
    need(len(rows)==len(scenes)==108 and order[:4800]==read(parent['order_file']) and order==read(long_plan['order_file']),
         'Original input/order prefix or exact audited longer schedule differs')
    counts=[sum(len(scenes[sid]['target_ids']) for pair in order[i:i+8] for sid in pair['sids']) for i in range(0,48000,8)]
    need(len(counts)==6000 and counts[:600]==parent['head_rows_by_update'] and counts==long_plan['head_rows_by_update']
         and sum(counts)==213330 and max(counts)==44,'Exact full-target position schedule differs')
    save(out/'order.json',order);save(out/'head_rows.json',dict(by_update=counts,total=sum(counts),maximum=max(counts),endpoint_rows=324,all_head_rows=sum(counts)+324))
    stats=torch.load(parent['stats_file'],map_location='cpu',weights_only=True)
    need({key:tensor_info(v) for key,v in stats.items() if isinstance(v,torch.Tensor)}==parent['stats_tensors']
         and len(stats['occurrences'])==1296 and stats['occurrences']==read(parent['stats_inventory_file'])['occurrences'],'Original conditioning statistics differ')
    packet=torch.load(parent['features_file'],map_location='cpu',weights_only=True)
    need(tensor_info(packet['states'])==parent['feature_tensor'],'Original cached native features differ');del packet,stats
    ckpt=CKPT/out.name;ckpt.mkdir(parents=True,exist_ok=False);initial_file=ckpt/'initial.pt';shutil.copyfile(parent['initial_file'],initial_file)
    need(sha(initial_file)==parent['initial_sha256'],'Exact original unfitted full checkpoint required')
    packet=torch.load(initial_file,map_location='cpu',weights_only=True)
    for arm in ARMS:
        core=make_core(torch,arm);core.load_state_dict(packet['branch']);mask=freeze_selector(torch,core)
        need(base.v7.state_info(core)==parent['initial_state'] and mask==parent['trainability_mask']
             and core.interaction==arm and core.factor_derangement is False,'Original unfitted factor state/mask differs')
    tests=self_test(torch)
    from scripts.report_native_identity_join_factor_long import self_test as reporter_self_test
    tests['reporter']=reporter_self_test(torch)
    identity=oracle.native_module_identity(torch,parent['native_identity']);need(identity==parent['native_module_identity'],'Native implementation changed')
    runtime_bindings={file:digest for file,digest in parent['runtime_bindings'].items() if file not in (parent['order_file'],parent['initial_file'])}
    bind(initial_file,runtime_bindings);bind(out/'order.json',runtime_bindings);bind(out/'head_rows.json',runtime_bindings)
    plan=dict(parent,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),tests=tests,
        input_bindings=bindings,runtime_bindings=runtime_bindings,factor_plan=dict(file=str(PARENT),sha256=PARENT_SHA),
        long_oracle_report=dict(file=str(CONDITIONAL),sha256=CONDITIONAL_SHA,analysis_file=proof['analysis_file'],analysis_sha256=CONDITIONAL_ANALYSIS_SHA),
        long_oracle_plan=dict(file=str(LONG_PLAN),sha256=LONG_PLAN_SHA),order_file=str(out/'order.json'),order_object_sha256=object_sha(order),
        head_rows_by_update=counts,head_rows_file=str(out/'head_rows.json'),training_head_rows=sum(counts),total_head_rows=sum(counts)+324,
        initial_file=str(initial_file),initial_sha256=sha(initial_file),fresh_shared_initialization=False,reused_unfitted_initialization=True,
        original_order_prefix_equal=True,decision_step=6000,evaluation_steps=list(EVALUATION_STEPS),no_new_statistics_or_features=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,training_contexts=108,pair_presentations=48000,
        training_head_rows=sum(counts),maximum_training_head_rows=max(counts),evaluation_head_rows=324,total_head_rows=sum(counts)+324,no_pretrained_head_forward=True)


def verify_plan(path,ancestors=False):
    path=Path(path).resolve();plan=read(path)
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Long factor plan/source changed')
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


def single_run_guard(out,arm):
    need(os.environ.get('SLURM_JOB_NAME')==RUN_JOBS[arm],'Exact registered per-arm single-attempt job name required')
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'launch_sacct.psv';file.write_text(raw)
    save(out/'launch_accounting.json',dict(command=command,file=str(file),sha256=sha(file)))
    for line in raw.splitlines():
        fields=line.split('|');need(len(fields)==9,'Unexpected allocation schema')
        if fields[1]==RUN_JOBS[arm] and fields[2]=='gpu':need(fields[0]==os.environ['SLURM_JOB_ID'],'One longer-factor GPU attempt per arm only')


def run(args,out,frozen,started):
    import torch
    torch.set_num_threads(4);args.plan=args.plan.resolve();plan=verify_plan(args.plan);single_run_guard(out,args.arm)
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200'
         and not torch.backends.cuda.matmul.allow_tf32,'One B200 with TF32 disabled required')
    hardware=dict(gpu=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    torch.manual_seed(24);torch.cuda.manual_seed_all(24);device=torch.device('cuda')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    rows=read(plan['rows_file']);scenes=read(plan['scenes_file']);cache=dict(scenes=scenes);order=read(plan['order_file'])
    save(out/'rows.json',rows);save(out/'presentations.json',order);save(out/'stats_inventory.json',read(plan['stats_inventory_file']))
    packet=torch.load(plan['features_file'],map_location='cpu',weights_only=True)
    need(tensor_info(packet['states'])==plan['feature_tensor'],'Frozen features changed')
    states=packet['states'].to(device);index={fid:i for i,fid in enumerate(packet['feature_ids'])};del packet
    stats=torch.load(plan['stats_file'],map_location='cpu',weights_only=True)
    need({k:tensor_info(v) for k,v in stats.items() if isinstance(v,torch.Tensor)}==plan['stats_tensors']
         and stats['occurrences']==read(plan['stats_inventory_file'])['occurrences'],'Fixed statistics changed')
    working_stats=dict(questions=stats['questions'],means=stats['means'].to(device),global_mean=stats['global_mean'].to(device),scale=stats['scale'].to(device))
    statistic_keys=('means','global_mean','scale')
    statistic_versions={k:working_stats[k]._version for k in statistic_keys}
    working_before={k:tensor_info(working_stats[k]) for k in statistic_keys}
    need(working_before=={k:plan['stats_tensors'][k] for k in statistic_keys},'Deployed fixed statistics differ')
    input_version=states._version;input_before=dict(states=tensor_info(states))
    core=make_core(torch,args.arm).to(device);initial=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    core.load_state_dict(initial['branch']);mask=freeze_selector(torch,core)
    need(base.v7.state_info(core)==plan['initial_state'] and mask==plan['trainability_mask']
         and core.interaction==args.arm and core.factor_derangement is False,'Exact original unfitted core/mask/paired mode required')
    initial_file=ckpt/'initial.pt';shutil.copyfile(plan['initial_file'],initial_file)
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=oracle.native_modules(torch,packet,plan['native_identity'],device);del packet
    module_identity=oracle.native_module_identity(torch,plan['native_identity']);need(module_identity==plan['native_module_identity'],'Native modules changed')
    native_initial=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight));versions=(norm.weight._version,head.weight._version)
    counters=dict(core=0,conditioning=0,norm=0,head=0,vlm=0,vision=0);calls=[];captures=[];current_phase='training';current_step=0;current_endpoint=None;capture_current=False;io={}
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
    config=dict(protocol=PROTOCOL,run_id=out.name,slurm_job_id=os.environ['SLURM_JOB_ID'],arm=args.arm,seed=24,policy=POLICY,
        source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],plan_file=str(args.plan),plan_sha256=sha(args.plan),oracle_report=plan['oracle_report'],uniform_plan=plan['uniform_plan'],
        conditioning_plan=plan['conditioning_plan'],conditioning_report=plan['conditioning_report'],fresh_shared_initialization=False,reused_unfitted_initialization=True,
        factor_plan=plan['factor_plan'],long_oracle_report=plan['long_oracle_report'],long_oracle_plan=plan['long_oracle_plan'],
        decision_step=6000,evaluation_steps=list(EVALUATION_STEPS),
        features_file=plan['features_file'],features_sha256=plan['features_sha256'],feature_tensor=plan['feature_tensor'],
        hardware=hardware,native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        native_module_identity=module_identity,native_model_file=plan['native_model_file'],native_model_sha256=plan['native_model_sha256'],
        native_weight_identity=native_initial,initialized=base.v7.state_info(core),initial_checkpoint=str(initial_file),initial_checkpoint_sha256=sha(initial_file),
        stats_file=plan['stats_file'],stats_sha256=plan['stats_sha256'],stats_tensors=plan['stats_tensors'],
        rows_file=str(out/'rows.json'),rows_sha256=sha(out/'rows.json'),presentations_file=str(out/'presentations.json'),
        presentations_sha256=sha(out/'presentations.json'),order_sha256=object_sha(order),
        stats_inventory_file=str(out/'stats_inventory.json'),stats_inventory_sha256=sha(out/'stats_inventory.json'),
        data_directory=str(data),checkpoint_directory=str(ckpt),**mask)
    save(out/'config.json',config)
    parameters=[p for p in core.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    torch.cuda.synchronize();setup=time.perf_counter()-started;logs=[];evaluations={}
    def optimizer_snapshot():
        return dict(parameter_ids={name:id(p) for name,p in core.named_parameters()},
            optimizer_parameter_ids=[id(p) for group in optimizer.param_groups for p in group['params']],
            requires_grad={name:p.requires_grad for name,p in core.named_parameters()},
            state_tensor_info={name:{key:tensor_info(value) if isinstance(value,torch.Tensor) else value
                for key,value in optimizer.state[p].items()} for name,p in core.named_parameters() if p.requires_grad},
            param_group_hyperparameters=[{key:value for key,value in group.items() if key!='params'} for group in optimizer.param_groups])
    def save_capture(local,g,valid,means,cap,layout,sids,weights_file,weights_step):
        need(core.interaction==args.arm and core.factor_derangement is False,'Only ordinary paired factors are authorized')
        name=f'training_capture_{current_step:04d}.pt' if current_phase=='training' else f'evaluation_{current_endpoint:04d}_capture_{current_step:04d}.pt'
        blob=dict(schema_version=1,arm=args.arm,phase=current_phase,step=current_step,endpoint_step=current_endpoint,pairing='paired',sids=sids,layout=layout,
            local_states=local,global_states=g,valid_mask=valid,question_means=means,capture=cap,**io)
        path=data/name;torch.save(cpu_tree(torch,blob),path)
        record=dict(arm=args.arm,phase=current_phase,step=current_step,endpoint_step=current_endpoint,pairing='paired',sids=sids,file=str(path),sha256=sha(path),
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
        before_input=dict(states=tensor_info(states));before_statistics={k:tensor_info(working_stats[k]) for k in statistic_keys};before_native=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight))
        was_training=core.training;core.eval();current_phase='evaluation';current_endpoint=update;capture_current=True;raw_batches=[]
        try:
            with torch.no_grad():
                for offset in range(0,108,16):
                    current_step=offset//16+1;io={};chosen=rows[offset:offset+16];sids=[r['sid'] for r in chosen]
                    h,g,layout,valid=base.batch_states(torch,cache,states,index,sids)
                    first=torch.tensor(layout['offsets'][:-1],device=device);h=h[:,first];g=g[first];valid=valid[:,first]
                    final_layout=dict(first_query_only=True,target_ids=[r['first_token_id'] for r in chosen],full_target_ids=[r['target_ids'] for r in chosen],
                        first_indices=layout['offsets'][:-1],original_full_layout=layout)
                    means=query_means(torch,working_stats,scenes,sids,arm=args.arm)
                    with FixedLocalConditioning(core,means,working_stats['scale']) as hook:
                        delta,cap=core(h,g,valid_mask=valid,output_dtype=torch.float32,capture=True);cap=hook.add_capture(cap)
                    counters['conditioning']+=hook.calls
                    logits=head(norm((g+delta.half()).unsqueeze(0)))[0]
                    record=save_capture(h,g,valid,means,cap,final_layout,sids,checkpoint,update)
                    gpu_nll=torch.nn.functional.cross_entropy(logits.float(),torch.tensor(final_layout['target_ids'],device=device),reduction='none')
                    raw_batches.append(dict(capture_file=record['file'],capture_sha256=record['sha256'],sids=sids,
                        gpu_fp32_nll=gpu_nll.tolist(),logits=tensor_info(io['logits'])))
        finally:core.train(was_training)
        after_optimizer=optimizer_snapshot();after_versions={name:p._version for name,p in core.named_parameters()}
        after_input=dict(states=tensor_info(states));after_statistics={k:tensor_info(working_stats[k]) for k in statistic_keys};after_native=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight))
        need(before_optimizer==after_optimizer and before_versions==after_versions and base.v7.state_info(core)==table
             and core.training==was_training and all(p.requires_grad==(name not in uniform.FROZEN_SELECTOR) and p.grad is None for name,p in core.named_parameters())
             and core.interaction==args.arm and core.factor_derangement is False
             and before_input==after_input==input_before and before_native==after_native==native_initial
             and input_version==states._version and versions==(norm.weight._version,head.weight._version)
             and before_statistics==after_statistics==working_before and statistic_versions=={k:working_stats[k]._version for k in statistic_keys}
             and not core.local._forward_pre_hooks and not hasattr(core.local,'_fixed_conditioning_active'),
             'Endpoint evaluation changed optimizer/parameters/mode/native/frozen inputs')
        packets=[torch.load(r['capture_file'],map_location='cpu',weights_only=True) for r in raw_batches]
        raw_logits=torch.cat([p['logits'] for p in packets]);del packets
        raw_file=data/f'logits_{update:04d}.pt';torch.save(dict(schema_version=1,sids=[r['sid'] for r in rows],logits=raw_logits,batches=raw_batches,endpoint_step=update),raw_file)
        targets=torch.tensor([r['first_token_id'] for r in rows]);ids=raw_logits.argmax(-1)
        promoted=raw_logits.double();shift=promoted-promoted.amax(-1,keepdim=True)
        nll=torch.logsumexp(shift,-1)-shift[torch.arange(108),targets];gpu_losses=[v for batch in raw_batches for v in batch['gpu_fp32_nll']]
        predictions=[dict(r,argmax_id=int(ids[i]),first_token_correct=int(ids[i])==r['first_token_id'],gpu_fp32_nll=gpu_losses[i],nll=float(nll[i])) for i,r in enumerate(rows)]
        prediction_file=out/f'predictions_{update:04d}.json';save(prediction_file,predictions)
        criterion=oracle.criteria(predictions);criterion.pop('oracle_answer_code_supplied');criterion['cached_first_query_only']=True
        endpoint_file=out/f'endpoint_{update:04d}.json'
        save(endpoint_file,dict(passed=True,endpoint_step=update,before_evaluation=table,after_evaluation=base.v7.state_info(core),parameter_sha256=object_sha(table),
            native_before=before_native,native_after=after_native,input_tensors_before=before_input,input_tensors_after=after_input,
            input_versions_unchanged=True,working_statistics_before=before_statistics,working_statistics_after=after_statistics,
            statistics_versions_unchanged=True,frozen_selector_exact=True,interaction=args.arm,pairing_restored=True,factor_derangement=False,hooks_removed=True,
            optimizer_before=before_optimizer,optimizer_after=after_optimizer,
            parameter_versions_before=before_versions,parameter_versions_after=after_versions,training_mode_restored=True,
            counters=dict(counters),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
        return dict(endpoint_step=update,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),endpoint_file=str(endpoint_file),endpoint_sha256=sha(endpoint_file),
            roundtrip_file=str(roundtrip_file),roundtrip_sha256=sha(roundtrip_file),raw_file=str(raw_file),raw_sha256=sha(raw_file),raw_tensor=tensor_info(raw_logits),raw_batches=raw_batches,
            predictions_file=str(prediction_file),predictions_sha256=sha(prediction_file),first_token_fit=criterion,descriptive_only=update!=6000)
    try:
        for step in range(1,6001):
            current_phase='training';current_step=step;current_endpoint=None;capture_current=step in DIAGNOSTIC_STEPS;io={};tick=time.perf_counter()
            batch=order[(step-1)*8:step*8];sids=[sid for pair in batch for sid in pair['sids']]
            h,g,layout,valid=base.batch_states(torch,cache,states,index,sids)
            need(g.shape[0]==plan['head_rows_by_update'][step-1]<=44,'Fixed full-target row budget changed')
            optimizer.zero_grad(set_to_none=True);rate=learning_rate(step);optimizer.param_groups[0]['lr']=rate
            weights_file=None
            if capture_current:
                weights_file=ckpt/f'training_core_{step:04d}.pt'
                torch.save(dict(arm=args.arm,branch=cpu_tree(torch,core.state_dict()),step=step-1,optimizer_step=step,position='before_update',source_sha256=frozen),weights_file)
            means=query_means(torch,working_stats,scenes,sids,layout,arm=args.arm)
            with FixedLocalConditioning(core,means,working_stats['scale']) as hook:
                _,ce,residual,parts,cap=base.losses(torch,core,h,g,layout,valid,norm,head);cap=hook.add_capture(cap)
            counters['conditioning']+=hook.calls
            uniform.check_selector(torch,core)
            need(core.interaction==args.arm and core.factor_derangement is False
                 and torch.equal(cap['gates'],valid.float()*.5) and bool((cap['scores']==.5).all()),'Locked paired factors/selection changed')
            if step==1:need(bool((cap['delta']==0).all()),'Initial U zero identity failed')
            if capture_current:save_capture(h,g,valid,means,cap,layout,sids,weights_file,step-1)
            ce.backward();need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),'Readout gradient missing/nonfinite')
            gradnorm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();uniform.check_selector(torch,core)
            need(all(bool(torch.isfinite(p).all()) for p in core.parameters()) and norm.weight.grad is head.weight.grad is None
                 and versions==(norm.weight._version,head.weight._version) and input_version==states._version
                 and statistic_versions=={k:working_stats[k]._version for k in statistic_keys},
                 'Factor/native/frozen-input invariant failed')
            torch.cuda.synchronize()
            logs.append(dict(step=step,sids=sids,pair_ids=[p['pair_id'] for p in batch],cycles=[p['cycle'] for p in batch],target_ids=layout['targets'],
                lr=rate,loss=float(ce),ce_loss=float(ce),consistency_loss=float(residual),weighted_consistency_loss=0.,consistency_coefficient=0.,
                gradient_norm=float(gradnorm),clipped=float(gradnorm)>1.,native_frozen=True,statistics_frozen=True,frozen_selector_exact=True,
                conditioned_input_shape=list(cap['conditioned_local_rms_input'].shape),seconds=time.perf_counter()-tick,**parts))
            del ce,residual,cap
            if step in EVALUATION_STEPS:evaluations[str(step)]=evaluate(step)
            if step in PROGRESS_STEPS:
                save(out/f'progress_{step:04d}.json',dict(step=step,training=logs,captures=captures,counters=counters,evaluations=evaluations))
                print(json.dumps(dict(step=step,ce=logs[-1]['ce_loss'])),flush=True)
        save(out/'training.json',logs);optimizer.zero_grad(set_to_none=True);del optimizer
        core.eval().requires_grad_(False);final=base.v7.state_info(core);last=evaluations['6000']
        save(out/'captures.json',captures);save(out/'calls.json',calls);save(out/'evaluations.json',evaluations)
        expected=dict(core=6021,conditioning=6021,norm=6021,head=6021,vlm=0,vision=0)
        need(counters==expected and input_version==states._version and versions==(norm.weight._version,head.weight._version)
             and dict(states=tensor_info(states))==input_before
             and {k:tensor_info(working_stats[k]) for k in statistic_keys}==working_before
             and statistic_versions=={k:working_stats[k]._version for k in statistic_keys}
             and core.interaction==args.arm and core.factor_derangement is False
             and not core.local._forward_pre_hooks and not hasattr(core.local,'_fixed_conditioning_active')
             and dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight))==native_initial
             and all(not p.requires_grad and p.grad is None for p in core.parameters()),'Final endpoint/work invariant failed')
        result=dict(**config,passed=True,completed=True,phase='run',computational_integrity_passed=True,steps=6000,
            **{key:value for key,value in last.items() if key not in ('endpoint_step','descriptive_only')},selected_parameter_sha256=object_sha(final),checkpoint_roundtrip_passed=True,
            training_file=str(out/'training.json'),training_sha256=sha(out/'training.json'),captures_file=str(out/'captures.json'),captures_sha256=sha(out/'captures.json'),
            calls_file=str(out/'calls.json'),calls_sha256=sha(out/'calls.json'),counters=counters,evaluations=evaluations,
            evaluations_file=str(out/'evaluations.json'),evaluations_sha256=sha(out/'evaluations.json'),earlier_endpoints_descriptive_only=True,
            training_head_rows=plan['training_head_rows'],final_head_rows=324,total_head_rows=plan['total_head_rows'],
            setup_before_training_seconds=setup,training_seconds=sum(r['seconds'] for r in logs),native_weights_unchanged=True,
            statistics_frozen=True,statistics_versions_unchanged=True,frozen_selector_exact=True,
            interaction=args.arm,pairing_restored=True,factor_derangement=False,input_versions_unchanged=True,cached_first_query_only=True,no_pretrained_backbone_loaded=True,no_dev_or_test=True,no_native_or_whole_answer_claim=True)
        need(time.perf_counter()-started<=240,'Single longer-factor job exceeded240 seconds');return result
    finally:
        for handle in handles:handle.remove()
        save(out/'final_counters.json',dict(counters=counters,recorded_calls=calls,partial_outputs_retained=True))



def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true');action.add_argument('--run',action='store_true');parser.add_argument('--plan',type=Path)
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();native.require_slurm(gpu=args.run)
    if args.check:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only source/statistics check')
    else:need(args.plan is not None and args.arm in ARMS,'Passed frozen CPU plan and prespecified arm required')
    phase='check' if args.check else 'run';suffix='' if args.check else args.arm+'_'
    out=OUT/f'{phase}_{suffix}{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();frozen=snapshot(out);save(out/'request.json',dict(phase=phase,arm=args.arm,plan=None if args.plan is None else str(args.plan),source_sha256=frozen))
    try:
        result=check(out,frozen) if args.check else run(args,out,frozen,started)
        need(sources()==frozen and inherited_sources()==read(out/'inherited_sources.json')['source_sha256'],'Source changed during longer-factor execution')
        result['elapsed_seconds']=time.perf_counter()-started;save(out/'summary.json',result)
        print(json.dumps(dict(passed=result['passed'],directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

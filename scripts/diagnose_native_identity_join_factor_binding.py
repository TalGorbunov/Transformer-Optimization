"""Held matched product/additive factors; cached native CE, no model generation.

Both fresh cores share all initial parameter bytes and frozen global conditioning.
The product-only final cyclic pairing diagnostic never changes fitted parameters.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_conditioning_v2 as original
from scripts.stage_native_vision_v6_teacher import need,read,sha,save,object_sha
uniform=original.uniform;oracle=original.oracle;base=original.base;native=original.native
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_factor_binding'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_factor_binding')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_factor_binding')
PROTOCOL='identity_join_matched_factor_binding'
ARMS=('product','additive')
RUN_JOBS={arm:'identity_join_factor_binding_'+arm for arm in ARMS}
PEOPLE=oracle.PEOPLE
PROPOSAL='docs/paper/NATIVE_AGGREGATION_FACTOR_BINDING_PROPOSAL.md'
PROPOSAL_SHA='5932b65f4e80ebb878be28e5dcf1ce7cc1a991b47b8319ca53598031a335aa93'
PARENT=original.OUT/'check_443335/plan.json'
PARENT_SHA='5fd647817756f643ccd2b265de2b44dbc2c43d90c43d6902ecd4d14b73fc644d'
CONDITIONAL=original.OUT/'report_443351/summary.json'
CONDITIONAL_SHA='134af48d0e3be92d8fd542fa01c6eea0da4cc9fd85bd03e688ab87cf16b69012'
CONDITIONAL_ANALYSIS_SHA='37904d35314ab4cf04afdf41241c9cf03fc0bff91e220966d565ae624c65c77c'
DIAGNOSTIC_STEPS=(1,2,32,128,300,600)
POLICY=dict(protocol=PROTOCOL,arms=list(ARMS),seed=24,steps=600,batch_pairs=8,batch_scenes=16,
    training_contexts=108,pairs=54,families=18,questions=6,origin_occurrences=1296,hidden_size=3584,rank=96,
    retained_state_parameters=1385857,effective_trainable_parameters=1385760,frozen_selector_parameters=97,
    fresh_shared_initialization=True,original_uniform_initialization=False,mode='sigmoid',
    fixed_selector_weight=0.,fixed_selector_bias=.5,consistency_coefficient=0.,
    factor_rule=dict(product='tanh(a)*tanh(b)',additive='.5*(tanh(a)+tanh(b))'),
    shared_query_both_factors=True,merge='sum',post_activation='silu',
    loss='unchanged mean_scene mean_all_name_plus_EOS_token native CE',lr=.001,warmup=50,final_lr=1e-5,
    weight_decay=0.,clip_norm=1.,optimizer='AdamW',betas=[.9,.999],adam_epsilon=1e-8,
    conditioning='(existing_FP32_local_RMS - frozen_global_mean_FP32)/shared_pooled_scale_FP32 before local Linear; global query unchanged',
    statistics='exact conditioning check443335 tensors; no new estimate',epsilon=1e-6,
    same_statistics_every_prefix=True,projection_tf32=False,diagnostic_steps=list(DIAGNOSTIC_STEPS),
    native_dtype='torch.float16',core_dtype='torch.float32',training_head_rows=21334,max_training_head_rows=44,
    final_head_rows=108,permutation_head_rows=108,
    norm_calls=dict(product=614,additive=607),head_calls=dict(product=614,additive=607),
    core_calls=dict(product=614,additive=607),total_head_rows=dict(product=21550,additive=21442),
    permutation='product only after paired final screen: cyclic factor_b within actual valid items independently per query; no labels',
    permutation_is_descriptive=True,first_token_fit=dict(correct=103,contexts=108,complete_families=16,families=18),
    gpu_seconds_cap=90,campaign_gpu_seconds_cap=180,cpu_check_seconds=90,maximum_gpus=2,no_profile=True,
    no_dev_or_test=True,final_checkpoint_only=True,cached_first_query_only=True,no_native_or_whole_answer_claim=True)
OWN=('gnnformer/parallel_local_factor_binding.py','scripts/diagnose_native_identity_join_factor_binding.py',
     'scripts/report_native_identity_join_factor_binding.py','slurm/native_identity_join_factor_binding_check.sbatch',
     'slurm/native_identity_join_factor_binding_run.sbatch','slurm/native_identity_join_factor_binding_report.sbatch',PROPOSAL)


def sources():return {**original.sources(),**{n:sha(REPO/n) for n in OWN}}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for n,h in frozen.items():
        dst=out/'source'/n.replace('/','_');dst.write_bytes((REPO/n).read_bytes());need(sha(dst)==h,'Source copy changed')
    save(out/'source_hashes.json',frozen);return frozen


tensor_info=original.tensor_info
bind=original.bind
FixedLocalConditioning=original.FixedLocalConditioning
cpu_tree=original.cpu_tree
learning_rate=original.learning_rate


def conditional_gate(bindings):
    bind(PARENT,bindings,PARENT_SHA);parent=original.verify_plan(PARENT,ancestors=True)
    bind(CONDITIONAL,bindings,CONDITIONAL_SHA);proof=read(CONDITIONAL)
    bind(proof['analysis_file'],bindings,CONDITIONAL_ANALYSIS_SHA);analysis=read(proof['analysis_file'])
    need(proof['passed'] is True and proof['completed'] is True and proof['phase']=='report'
         and proof['source_sha256']==original.sources() and proof['analysis_sha256']==CONDITIONAL_ANALYSIS_SHA
         and analysis['passed'] is True and analysis['completed'] is True
         and analysis['source_sha256']==original.sources() and analysis['plan_file']==str(PARENT)
         and analysis['plan_sha256']==PARENT_SHA and proof['first_token_screen']==analysis['first_token_screen'],
         'Independent completed conditioning report required')
    need(set(proof['first_token_screen'])=={'global','question'}
         and all(c['passed'] is False and c['contexts']==108 and c['families']==18 for c in proof['first_token_screen'].values()),
         'Both fixed conditioning screens must fail before this separate comparison')
    for n,h in proof['source_sha256'].items():need(sha(CONDITIONAL.parent/'source'/n.replace('/','_'))==h,'Conditioning report source snapshot changed')
    bind(REPO/PROPOSAL,bindings,PROPOSAL_SHA)
    return parent,dict(summary_file=str(CONDITIONAL),summary_sha256=CONDITIONAL_SHA,
        analysis_file=proof['analysis_file'],analysis_sha256=CONDITIONAL_ANALYSIS_SHA,
        first_token_screen=proof['first_token_screen'])


def make_core(torch,arm):
    from gnnformer.parallel_local_factor_binding import ParallelLocalFactorBinding
    need(arm in ARMS,'Unknown factor-binding arm')
    return ParallelLocalFactorBinding(interaction=arm)


def freeze_selector(torch,core):
    for name,p in core.named_parameters():
        need(p.grad is None,'Freeze selector before any gradient computation')
        p.requires_grad_(name not in uniform.FROZEN_SELECTOR)
    uniform.check_selector(torch,core)
    result=dict(trainable_parameter_names=[n for n,p in core.named_parameters() if p.requires_grad],
        frozen_selector_parameter_names=list(uniform.FROZEN_SELECTOR),
        effective_trainable_parameters=sum(p.numel() for p in core.parameters() if p.requires_grad),
        retained_state_parameters=sum(p.numel() for p in core.parameters()))
    need(result['effective_trainable_parameters']==1385760 and result['retained_state_parameters']==1385857,
         'Matched two-factor trainable/state counts differ')
    return result


def query_means(torch,stats,scenes,sids,layout=None,*,arm):
    need(arm in ARMS,'Unknown factor-binding arm')
    return original.query_means(torch,stats,scenes,sids,layout,arm='global')


def self_test(torch):
    from gnnformer.parallel_local_factor_binding import self_test as core_self_test
    result=core_self_test(torch)
    need(result['passed'] is True,'Factor core fixtures failed')
    from gnnformer.parallel_local_factor_binding import ParallelLocalFactorBinding
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);tiny=ParallelLocalFactorBinding(hidden_size=4,rank=2,interaction='product')
    h=torch.arange(24,dtype=torch.float32).reshape(3,2,4).half()/10
    g=torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.]],dtype=torch.float16)
    valid=torch.tensor([[True,True],[True,True],[False,True]]);h[~valid]=0
    means=torch.full((2,4),.25);scale=torch.tensor(.5)
    with FixedLocalConditioning(tiny,means,scale) as hook:
        delta,cap=tiny(h,g,valid_mask=valid,output_dtype=torch.float32,capture=True);cap=hook.add_capture(cap)
    need(hook.calls==1 and torch.equal(cap['conditioned_local_rms_input'],(tiny.rms(h)-means.unsqueeze(0))/scale)
         and torch.equal(cap['payload'],cap['factor_a']*cap['factor_b']) and bool((cap['messages'][~valid]==0).all())
         and bool((delta==0).all()) and not tiny.local._forward_pre_hooks,'Conditioned two-factor scope integration differs')
    return dict(passed=True,core=result,conditioned_factor_scope=True,
        conditioning=original.self_test(torch),scalar_tensor_info=original.tensor_info_self_test(torch))


def check(out,frozen):
    import torch
    torch.set_num_threads(4);bindings={};parent,report=conditional_gate(bindings)
    for file,h in parent['runtime_bindings'].items():bind(file,bindings,h)
    rows=read(parent['rows_file']);scenes=read(parent['scenes_file']);pairs=read(parent['pairs_file']);order=read(parent['order_file'])
    need(len(rows)==len(scenes)==108 and len(pairs)==54 and order==uniform.order_pairs(pairs),'Exact original training/order inventory differs')
    need(Counter(r['gold'] for r in rows)==Counter({name:12 for name in PEOPLE}),'Balanced training target inventory differs')
    head_counts=parent['head_rows_by_update'];need(len(head_counts)==600 and sum(head_counts)==21334 and max(head_counts)==44,'Full-CE head inventory differs')
    for name,value in (('rows',rows),('scenes',scenes),('pairs',pairs),('order',order),('stats_inventory',read(parent['stats_inventory_file']))):save(out/(name+'.json'),value)
    stats=torch.load(parent['stats_file'],map_location='cpu',weights_only=True)
    need({k:tensor_info(v) for k,v in stats.items() if isinstance(v,torch.Tensor)}==parent['stats_tensors']
         and len(stats['occurrences'])==1296 and stats['occurrences']==read(parent['stats_inventory_file'])['occurrences'],
         'Exact existing global statistics required; no re-estimation')
    packet=torch.load(parent['features_file'],map_location='cpu',weights_only=True)
    need(tensor_info(packet['states'])==parent['feature_tensor'] and packet['feature_ids']==sorted(packet['feature_ids']), 'Original frozen feature packet differs')
    del packet,stats
    ckpt=CKPT/out.name;ckpt.mkdir(parents=True,exist_ok=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);core=make_core(torch,'product')
    mask=freeze_selector(torch,core);initial_state=base.v7.state_info(core)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);other=make_core(torch,'additive')
    need(base.v7.state_info(other)==initial_state and freeze_selector(torch,other)==mask,'All fresh initialization bytes must match across factor arms')
    initial_file=ckpt/'initial.pt';torch.save(dict(seed=24,branch=cpu_tree(torch,core.state_dict())),initial_file)
    tests=self_test(torch)
    from scripts.report_native_identity_join_factor_binding import self_test as reporter_self_test
    tests['reporter']=reporter_self_test(torch)
    module_identity=oracle.native_module_identity(torch,parent['native_identity'])
    native_packet=torch.load(parent['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=oracle.native_modules(torch,native_packet,parent['native_identity'],'cpu');del native_packet,norm,head
    runtime_bindings={}
    for path in (out/'rows.json',out/'scenes.json',out/'pairs.json',out/'order.json',out/'stats_inventory.json',
                 initial_file,Path(parent['stats_file']),Path(parent['features_file']),Path(parent['native_model_file'])):bind(path,runtime_bindings)
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,tests=tests,trainability_mask=mask,
        input_bindings=bindings,runtime_bindings=runtime_bindings,
        conditioning_plan=dict(file=str(PARENT),sha256=PARENT_SHA),conditioning_report=report,
        uniform_plan=parent['uniform_plan'],oracle_report=parent['oracle_report'],
        rows_file=str(out/'rows.json'),scenes_file=str(out/'scenes.json'),pairs_file=str(out/'pairs.json'),order_file=str(out/'order.json'),
        order_object_sha256=object_sha(order),head_rows_by_update=head_counts,stats_inventory_file=str(out/'stats_inventory.json'),
        stats_file=parent['stats_file'],stats_sha256=parent['stats_sha256'],stats_tensors=parent['stats_tensors'],
        features_file=parent['features_file'],features_sha256=parent['features_sha256'],feature_tensor=parent['feature_tensor'],
        initial_file=str(initial_file),initial_sha256=sha(initial_file),initial_state=initial_state,
        native_model_file=parent['native_model_file'],native_model_sha256=parent['native_model_sha256'],native_identity=parent['native_identity'],
        native_identity_sha256=parent['native_identity_sha256'],native_module_identity=module_identity,
        fresh_shared_initialization=True,no_pretrained_backbone_loaded=True,no_pretrained_head_forward=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,conditioning_report=report,
        occurrences=1296,training_contexts=108,training_head_rows=21334,final_head_rows=108,no_pretrained_head_forward=True)

def verify_plan(path,ancestors=False):
    path=Path(path).resolve();plan=read(path)
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Factor-binding plan/source changed')
    for n,h in plan['source_sha256'].items():need(sha(path.parent/'source'/n.replace('/','_'))==h,'Source snapshot changed')
    for file,h in plan['runtime_bindings'].items():need(sha(file)==h,'Consumed factor-binding input changed')
    if ancestors:
        for file,h in plan['input_bindings'].items():need(sha(file)==h,'Factor-binding ancestor changed')
    proof=read(path.parent/'summary.json')
    need(proof['passed'] is True and proof['completed'] is True and proof['phase']=='check' and proof['plan_sha256']==sha(path)
         and object_sha(read(plan['order_file']))==plan['order_object_sha256']
         and object_sha(plan['native_identity'])==plan['native_identity_sha256'],'Passed CPU gate/identity required')
    return plan


def single_run_guard(out,arm):
    need(os.environ.get('SLURM_JOB_NAME')==RUN_JOBS[arm],'Submit the exact per-arm Slurm job name before any tensor/model loading')
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;path=out/'launch_sacct.psv';path.write_text(raw)
    save(out/'launch_accounting.json',dict(command=command,file=str(path),sha256=sha(path)))
    for line in raw.splitlines():
        fields=line.split('|');need(len(fields)==9,'Unexpected allocation schema')
        if fields[1]==RUN_JOBS[arm] and fields[2]=='gpu':need(fields[0]==os.environ['SLURM_JOB_ID'],'Single factor-binding GPU attempt per arm only')


def run(args,out,frozen,started):
    import torch
    torch.set_num_threads(4);args.plan=args.plan.resolve();plan=verify_plan(args.plan);single_run_guard(out,args.arm)
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200'
         and not torch.backends.cuda.matmul.allow_tf32,'One B200 with FP32 projection/TF32 disabled required')
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
    core=make_core(torch,args.arm).to(device);initial=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    core.load_state_dict(initial['branch']);mask=freeze_selector(torch,core)
    need(base.v7.state_info(core)==plan['initial_state'] and mask==plan['trainability_mask']
         and core.interaction==args.arm and core.factor_derangement is False,'Identical fresh initial core/mask/paired mode required')
    initial_file=ckpt/'initial.pt';shutil.copyfile(plan['initial_file'],initial_file)
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=oracle.native_modules(torch,packet,plan['native_identity'],device);del packet
    module_identity=oracle.native_module_identity(torch,plan['native_identity']);need(module_identity==plan['native_module_identity'],'Native modules changed')
    native_initial=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight));versions=(norm.weight._version,head.weight._version)
    counters=dict(core=0,conditioning=0,norm=0,head=0,vlm=0,vision=0);calls=[];captures=[];current_phase='training';current_step=0;capture_current=False;io={}
    def observer(name):
        def hook(module,arguments,output):
            counters[name]+=1
            calls.append(dict(module=name,phase=current_phase,step=current_step,input_shape=list(arguments[0].shape),
                output_shape=list(output.shape),input_dtype=str(arguments[0].dtype),output_dtype=str(output.dtype)))
            if capture_current:
                if name=='norm':io.update(fused_global=arguments[0][0].detach().cpu().clone(),normalized=output[0].detach().cpu().clone())
                else:io['logits']=output[0].detach().cpu().clone()
        return hook
    def core_observer(module,args,output):counters['core']+=1
    handles=[norm.register_forward_hook(observer('norm')),head.register_forward_hook(observer('head')),core.register_forward_hook(core_observer)]
    config=dict(protocol=PROTOCOL,run_id=out.name,slurm_job_id=os.environ['SLURM_JOB_ID'],arm=args.arm,seed=24,policy=POLICY,
        source_sha256=frozen,plan_file=str(args.plan),plan_sha256=sha(args.plan),oracle_report=plan['oracle_report'],uniform_plan=plan['uniform_plan'],
        conditioning_plan=plan['conditioning_plan'],conditioning_report=plan['conditioning_report'],fresh_shared_initialization=True,
        hardware=hardware,native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        native_module_identity=module_identity,native_model_file=plan['native_model_file'],native_model_sha256=plan['native_model_sha256'],
        native_weight_identity=native_initial,initialized=base.v7.state_info(core),initial_checkpoint=str(initial_file),initial_checkpoint_sha256=sha(initial_file),
        stats_file=plan['stats_file'],stats_sha256=plan['stats_sha256'],stats_tensors=plan['stats_tensors'],
        rows_file=str(out/'rows.json'),rows_sha256=sha(out/'rows.json'),presentations_file=str(out/'presentations.json'),
        presentations_sha256=sha(out/'presentations.json'),order_sha256=object_sha(order),
        stats_inventory_file=str(out/'stats_inventory.json'),stats_inventory_sha256=sha(out/'stats_inventory.json'),
        data_directory=str(data),checkpoint_directory=str(ckpt),**mask)
    save(out/'config.json',config)
    parameters=[p for p in core.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    torch.cuda.synchronize();setup=time.perf_counter()-started;logs=[];raw_batches=[]
    def save_capture(local,g,valid,means,cap,layout,sids,weights_file,weights_step):
        pairing='cyclic' if current_phase=='permutation' else 'paired'
        need(core.interaction==args.arm and core.factor_derangement is (pairing=='cyclic'),'Actual factor pairing mode differs')
        blob=dict(schema_version=1,arm=args.arm,phase=current_phase,step=current_step,pairing=pairing,sids=sids,layout=layout,
            local_states=local,global_states=g,valid_mask=valid,question_means=means,capture=cap,**io)
        path=data/f'{current_phase}_capture_{current_step:04d}.pt';torch.save(cpu_tree(torch,blob),path)
        record=dict(arm=args.arm,phase=current_phase,step=current_step,pairing=pairing,sids=sids,file=str(path),sha256=sha(path),
            weights_file=str(weights_file),weights_sha256=sha(weights_file),weights_step=weights_step,
            tensors={k:tensor_info(v) for k,v in io.items()})
        captures.append(record);return record
    try:
        for step in range(1,601):
            current_step=step;capture_current=step in DIAGNOSTIC_STEPS;io={};tick=time.perf_counter()
            batch=order[(step-1)*8:step*8];sids=[sid for pair in batch for sid in pair['sids']]
            h,g,layout,valid=base.batch_states(torch,cache,states,index,sids)
            need(g.shape[0]==plan['head_rows_by_update'][step-1]<=44,'Fixed full-target row budget changed')
            means=query_means(torch,working_stats,scenes,sids,layout,arm=args.arm)
            optimizer.zero_grad(set_to_none=True);rate=learning_rate(step);optimizer.param_groups[0]['lr']=rate
            weights_file=None
            if capture_current:
                weights_file=ckpt/f'training_core_{step:04d}.pt'
                torch.save(dict(arm=args.arm,branch=cpu_tree(torch,core.state_dict()),step=step-1,optimizer_step=step,position='before_update',source_sha256=frozen),weights_file)
            with FixedLocalConditioning(core,means,working_stats['scale']) as hook:
                _,ce,residual,parts,cap=base.losses(torch,core,h,g,layout,valid,norm,head);cap=hook.add_capture(cap)
            counters['conditioning']+=hook.calls
            uniform.check_selector(torch,core)
            need(torch.equal(cap['gates'],valid.float()*.5) and bool((cap['scores']==.5).all()),'Uniform selection changed')
            if step==1:need(bool((cap['delta']==0).all()),'Initial U zero identity failed')
            if capture_current:save_capture(h,g,valid,means,cap,layout,sids,weights_file,step-1)
            ce.backward();need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),'Core gradient missing/nonfinite')
            gradnorm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            uniform.check_selector(torch,core)
            need(all(bool(torch.isfinite(p).all()) for p in core.parameters()) and norm.weight.grad is head.weight.grad is None
                 and versions==(norm.weight._version,head.weight._version)
                 and statistic_versions=={k:working_stats[k]._version for k in statistic_keys},'Core/native/fixed-statistic invariant failed')
            torch.cuda.synchronize()
            logs.append(dict(step=step,sids=sids,pair_ids=[p['pair_id'] for p in batch],cycles=[p['cycle'] for p in batch],target_ids=layout['targets'],
                lr=rate,loss=float(ce),ce_loss=float(ce),consistency_loss=float(residual),weighted_consistency_loss=0.,consistency_coefficient=0.,
                gradient_norm=float(gradnorm),clipped=float(gradnorm)>1.,native_frozen=True,statistics_frozen=True,frozen_selector_exact=True,
                conditioned_input_shape=list(cap['conditioned_local_rms_input'].shape),seconds=time.perf_counter()-tick,**parts))
            if step%100==0:
                save(out/f'progress_{step:04d}.json',dict(step=step,training=logs,captures=captures,counters=counters))
                print(json.dumps(dict(step=step,ce=float(ce))),flush=True)
        save(out/'training.json',logs);optimizer.zero_grad(set_to_none=True);del optimizer,ce,residual,cap
        core.eval().requires_grad_(False);final=base.v7.state_info(core);checkpoint=ckpt/'final.pt'
        torch.save(dict(branch=cpu_tree(torch,core.state_dict()),step=600,config=config),checkpoint)
        core.load_state_dict(initial['branch']);need(base.v7.state_info(core)==plan['initial_state'],'Initial reset failed')
        packet=torch.load(checkpoint,map_location='cpu',weights_only=True)
        need(packet['step']==600 and packet['config']==config,'Restricted checkpoint metadata mismatch')
        core.load_state_dict(packet['branch']);need(base.v7.state_info(core)==final,'Final checkpoint roundtrip failed')
        save(out/'checkpoint_roundtrip.json',dict(passed=True,initial=plan['initial_state'],final=final,reloaded=base.v7.state_info(core),
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),step=600,config_sha256=sha(out/'config.json')))
        capture_current=True;evaluations={}
        phases=['evaluation']+(['permutation'] if args.arm=='product' else [])
        for phase in phases:
            current_phase=phase;raw_batches=[]
            with core.factor_pairing('cyclic' if phase=='permutation' else 'paired'),torch.no_grad():
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
                    record=save_capture(h,g,valid,means,cap,final_layout,sids,checkpoint,600)
                    gpu_nll=torch.nn.functional.cross_entropy(logits.float(),torch.tensor(final_layout['target_ids'],device=device),reduction='none')
                    raw_batches.append(dict(capture_file=record['file'],capture_sha256=record['sha256'],sids=sids,
                        gpu_fp32_nll=gpu_nll.tolist(),logits=tensor_info(io['logits'])))
            final_packets=[torch.load(r['capture_file'],map_location='cpu',weights_only=True) for r in raw_batches]
            raw_logits=torch.cat([p['logits'] for p in final_packets]);del final_packets
            raw_file=data/('final_logits.pt' if phase=='evaluation' else 'permutation_logits.pt')
            torch.save(dict(schema_version=1,sids=[r['sid'] for r in rows],logits=raw_logits,batches=raw_batches),raw_file)
            targets=torch.tensor([r['first_token_id'] for r in rows]);ids=raw_logits.argmax(-1)
            promoted=raw_logits.double();shift=promoted-promoted.amax(-1,keepdim=True)
            nll=torch.logsumexp(shift,-1)-shift[torch.arange(108),targets];gpu_losses=[v for batch in raw_batches for v in batch['gpu_fp32_nll']]
            predictions=[dict(r,argmax_id=int(ids[i]),first_token_correct=int(ids[i])==r['first_token_id'],gpu_fp32_nll=gpu_losses[i],nll=float(nll[i])) for i,r in enumerate(rows)]
            prediction_file=out/('predictions.json' if phase=='evaluation' else 'permutation_predictions.json')
            save(prediction_file,predictions)
            evaluations[phase]=dict(raw_file=str(raw_file),raw_sha256=sha(raw_file),raw_tensor=tensor_info(raw_logits),raw_batches=raw_batches,
                predictions_file=str(prediction_file),predictions_sha256=sha(prediction_file),predictions=predictions)
        paired=evaluations['evaluation'];predictions=paired['predictions']
        save(out/'captures.json',captures);save(out/'calls.json',calls)
        native_final=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight))
        total_calls=POLICY['core_calls'][args.arm];expected=dict(core=total_calls,conditioning=total_calls,norm=total_calls,head=total_calls,vlm=0,vision=0)
        working_after={k:tensor_info(working_stats[k]) for k in statistic_keys}
        need(working_after==working_before and statistic_versions=={k:working_stats[k]._version for k in statistic_keys},'Final fixed statistics changed')
        need(counters==expected and native_final==native_initial and base.v7.state_info(core)==final
             and core.interaction==args.arm and core.factor_derangement is False
             and not core.local._forward_pre_hooks and not hasattr(core.local,'_fixed_conditioning_active')
             and all(not p.requires_grad and p.grad is None for p in core.parameters()),'Final endpoint/calls/hooks differ')
        endpoint=dict(passed=True,interaction=core.interaction,pairing_restored=True,factor_derangement=core.factor_derangement,before_evaluation=final,after_evaluation=base.v7.state_info(core),parameter_sha256=object_sha(final),
            native_before=native_initial,native_after=native_final,native_weights_unchanged=True,counters=counters,
            stats_tensors=plan['stats_tensors'],stats_file=plan['stats_file'],stats_sha256=plan['stats_sha256'],
            working_statistics_before=working_before,working_statistics_after=working_after,statistics_versions_unchanged=True,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),hooks_removed=True)
        save(out/'final_endpoint.json',endpoint)
        result=dict(**config,passed=True,completed=True,phase='run',computational_integrity_passed=True,steps=600,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),selected_parameter_sha256=object_sha(final),checkpoint_roundtrip_passed=True,
            training_file=str(out/'training.json'),training_sha256=sha(out/'training.json'),captures_file=str(out/'captures.json'),captures_sha256=sha(out/'captures.json'),
            **{k:v for k,v in paired.items() if k!='predictions'},
            calls_file=str(out/'calls.json'),calls_sha256=sha(out/'calls.json'),counters=counters,
            endpoint_file=str(out/'final_endpoint.json'),endpoint_sha256=sha(out/'final_endpoint.json'),
            roundtrip_file=str(out/'checkpoint_roundtrip.json'),roundtrip_sha256=sha(out/'checkpoint_roundtrip.json'),
            first_token_fit=oracle.criteria(predictions),training_head_rows=21334,final_head_rows=108,total_head_rows=POLICY['total_head_rows'][args.arm],
            setup_before_training_seconds=setup,training_seconds=sum(r['seconds'] for r in logs),
            native_weights_unchanged=True,statistics_frozen=True,statistics_versions_unchanged=True,cached_first_query_only=True,no_pretrained_backbone_loaded=True,
            no_dev_or_test=True,no_native_or_whole_answer_claim=True,interaction=core.interaction,pairing_restored=True,factor_derangement=False)
        if args.arm=='product':
            result.update({'permutation_'+k:v for k,v in evaluations['permutation'].items() if k!='predictions'})
            result['permutation_diagnostic']=dict(rule=POLICY['permutation'],contexts=108,head_calls=7,head_rows=108,
                descriptive_only=True,no_fit_or_checkpoint_selection=True)
        # This is ordinary cached supervision; remove the oracle-specific label from the reused counting utility.
        result['first_token_fit'].pop('oracle_answer_code_supplied');result['first_token_fit']['cached_first_query_only']=True
        need(time.perf_counter()-started<=90,'Single head-only factor-binding job exceeded90 seconds')
        return result
    finally:
        for h in handles:h.remove()
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
        need(sources()==frozen,'Source changed during factor-binding execution')
        result['elapsed_seconds']=time.perf_counter()-started;save(out/'summary.json',result)
        print(json.dumps(dict(passed=result['passed'],directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

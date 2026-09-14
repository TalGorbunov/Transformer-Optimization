"""Held fixed training-only local conditioning; no VLM or local-label calls.

Same uniform core, initialization/order and complete name+EOS CE. A scoped local
Linear pre-hook transforms only already-RMSed local inputs with frozen statistics.
Final cached first-token screening cannot establish native whole-answer behavior.
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
from scripts import diagnose_native_identity_join_uniform as uniform
from scripts import diagnose_native_identity_join_readout as oracle
from scripts import report_native_identity_join_readout_precision as precision
from scripts.stage_native_vision_v6_teacher import need,read,sha,save,object_sha
base=uniform.base;native=uniform.native
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_conditioning'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_conditioning')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_conditioning')
PROTOCOL='identity_join_fixed_local_conditioning'
ARMS=('global','question')
RUN_JOBS={arm:'identity_join_conditioning_'+arm for arm in ARMS}
PEOPLE=oracle.PEOPLE
PROPOSAL='docs/paper/NATIVE_AGGREGATION_CONDITIONING_PROPOSAL.md'
PROPOSAL_SHA='4e638f3473b4f4eab55291493a1ebd835014657ee5c3bd1175c27d291998f49d'
ORACLE=oracle.OUT/'report_precision_443304/summary.json'
ORACLE_SHA='cb280b2bf6d769885ee20f395a6adaeed0aaec09ef00a5899258d6049b5490c7'
ORACLE_ANALYSIS_SHA='936c5a9fee35bda4f67df0c6e20e5ef64055dbaea6ed94cd33edfcae1feabe82'
DIAGNOSTIC_STEPS=(1,2,32,128,300,600)
POLICY=dict(protocol=PROTOCOL,arms=list(ARMS),seed=24,steps=600,batch_pairs=8,batch_scenes=16,training_contexts=108,pairs=54,families=18,
    questions=6,origin_occurrences=1296,hidden_size=3584,rank=96,retained_state_parameters=1041697,effective_trainable_parameters=1041600,
    mode='sigmoid',fixed_selector_weight=0.,fixed_selector_bias=.5,consistency_coefficient=0.,
    loss='unchanged mean_scene mean_all_name_plus_EOS_token native CE',lr=.001,warmup=50,final_lr=1e-5,
    weight_decay=0.,clip_norm=1.,optimizer='AdamW',betas=[.9,.999],adam_epsilon=1e-8,
    conditioning='(existing_FP32_local_RMS - chosen_global_or_question_mean_FP32)/shared_pooled_scale_FP32 before local Linear; global query unchanged',
    means='FP64 occurrence mean of original TRAIN cached empty-prefix local RMS, then FP32',
    scale='shared sqrt(mean((x.double()-global_mean_FP64)^2)+1e-6), then FP32',
    epsilon=1e-6,same_statistics_every_prefix=True,projection_tf32=False,
    diagnostic_steps=list(DIAGNOSTIC_STEPS),native_dtype='torch.float16',core_dtype='torch.float32',
    norm_calls=607,head_calls=607,core_calls=607,training_head_rows=21334,final_head_rows=108,total_head_rows=21442,max_training_head_rows=44,
    first_token_fit=dict(correct=103,contexts=108,complete_families=16,families=18),
    gpu_seconds_cap=90,campaign_gpu_seconds_cap=180,cpu_check_seconds=90,maximum_gpus=2,no_profile=True,no_dev_or_test=True,
    final_checkpoint_only=True,cached_first_query_only=True,no_native_or_whole_answer_claim=True,no_new_trainable_parameters=True)
OWN=('scripts/diagnose_native_identity_join_conditioning.py','scripts/report_native_identity_join_conditioning.py',
     'slurm/native_identity_join_conditioning_check.sbatch','slurm/native_identity_join_conditioning_run.sbatch',
     'slurm/native_identity_join_conditioning_report.sbatch',PROPOSAL)


def sources():return {**precision.reporting_sources(),**{n:sha(REPO/n) for n in OWN}}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for n,h in frozen.items():
        dst=out/'source'/n.replace('/','_');dst.write_bytes((REPO/n).read_bytes());need(sha(dst)==h,'Source copy changed')
    save(out/'source_hashes.json',frozen);return frozen


def bind(path,bindings,expected=None):return oracle.bind(path,bindings,expected)


def conditional_gate(bindings):
    parent,uniform_failure=oracle.conditional_gate(bindings)
    bind(REPO/PROPOSAL,bindings,PROPOSAL_SHA);bind(ORACLE,bindings,ORACLE_SHA);proof=read(ORACLE)
    bind(proof['analysis_file'],bindings,ORACLE_ANALYSIS_SHA);analysis=read(proof['analysis_file'])
    need(proof['passed'] is True and proof['completed'] is True and proof['analysis_sha256']==ORACLE_ANALYSIS_SHA
         and proof['reporting_source_sha256']==precision.reporting_sources() and analysis['passed'] is True
         and analysis['completed'] is True and proof['first_token_fit']==analysis['first_token_fit']
         and proof['first_token_fit']['passed'] is True and proof['first_token_fit']['contexts']==108
         and proof['first_token_fit']['families']==18 and proof['original_report_remains_failed'] is True,
         'Completed independently repaired oracle positive control required')
    for n,h in proof['reporting_source_sha256'].items():need(sha(ORACLE.parent/'source'/n.replace('/','_'))==h,'Oracle report source snapshot changed')
    return parent,dict(summary_file=str(ORACLE),summary_sha256=ORACLE_SHA,analysis_file=proof['analysis_file'],
        analysis_sha256=ORACLE_ANALYSIS_SHA,first_token_fit=proof['first_token_fit'],uniform_failure=uniform_failure)


def learning_rate(step):return uniform.learning_rate(step)


def compute_statistics(torch,x,question_indices,n_questions):
    need(x.dtype==torch.float32 and x.ndim==2 and not x.requires_grad and bool(torch.isfinite(x).all())
         and question_indices.dtype==torch.int64 and question_indices.shape==(len(x),)
         and set(question_indices.tolist())==set(range(n_questions)),'Statistics input/coverage differs')
    means64=torch.stack([x[question_indices==i].double().mean(0) for i in range(n_questions)])
    scale64=((x.double()-x.double().mean(0)).square().mean()+1e-6).sqrt()
    return means64,means64.float(),scale64,scale64.float()


class FixedLocalConditioning:
    """Temporary hook; means are flat current-query rows, reused at every prefix."""
    def __init__(self,core,means,scale):
        import torch
        need(means.dtype==scale.dtype==torch.float32 and means.ndim==2 and means.shape[1]==core.hidden_size
             and scale.shape==() and not means.requires_grad and not scale.requires_grad
             and means.device==scale.device==core.local.weight.device
             and bool(torch.isfinite(means).all()) and bool(torch.isfinite(scale)) and float(scale)>0,
             'Fixed conditioning constants/layout differ')
        self.core=core;self.means=means;self.scale=scale;self.calls=0;self.handle=None
        self.original_local_rms_input=None;self.conditioned_local_rms_input=None
    def __enter__(self):
        need(self.handle is None and not getattr(self.core.local,'_fixed_conditioning_active',False),'Nested/reused conditioning hook')
        def hook(module,args):
            import torch
            need(len(args)==1 and args[0].dtype==torch.float32 and args[0].ndim==3
                 and args[0].shape[1:]==self.means.shape,'Actual local RMS query layout differs')
            self.original_local_rms_input=args[0]
            value=(args[0]-self.means.unsqueeze(0))/self.scale
            need(bool(torch.isfinite(value).all()),'Conditioned local input overflow')
            self.conditioned_local_rms_input=value;self.calls+=1
            return (value,)
        self.handle=self.core.local.register_forward_pre_hook(hook)
        self.core.local._fixed_conditioning_active=True
        return self
    def __exit__(self,kind,value,tb):
        if self.handle is not None:self.handle.remove();self.handle=None
        if hasattr(self.core.local,'_fixed_conditioning_active'):del self.core.local._fixed_conditioning_active
        return False
    def add_capture(self,cap):
        need(self.calls==1,'Exactly one unchanged core local projection required per scoped batch')
        return dict(cap,original_local_rms_input=self.original_local_rms_input,
                    conditioned_local_rms_input=self.conditioned_local_rms_input)


def query_means(torch,stats,scenes,sids,layout=None,*,arm):
    need(arm in ARMS,'Unknown centering arm')
    if arm=='global':
        count=len(sids) if layout is None else layout['offsets'][-1]
        return stats['global_mean'].unsqueeze(0).expand(count,-1)
    indices=[]
    for i,sid in enumerate(sids):
        question=scenes[sid]['question'];need(question in stats['questions'],'Unknown original question')
        count=1 if layout is None else layout['offsets'][i+1]-layout['offsets'][i]
        indices.extend([stats['questions'].index(question)]*count)
    return stats['means'][torch.tensor(indices,dtype=torch.int64,device=stats['means'].device)]


def cpu_tree(torch,value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(torch,v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(torch,v) for v in value]
    return value


def self_test(torch):
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    x=torch.tensor([[0.,0.],[1.,1.],[1.,2.],[4.,5.]],dtype=torch.float32);q=torch.tensor([0,0,0,1])
    m64,m,s64,s=compute_statistics(torch,x,q,2)
    expected_means=torch.stack((x[:3].double().mean(0),x[3].double()))
    need(torch.equal(m64,expected_means) and torch.equal(m,m64.float())
         and torch.equal(s64,((x.double()-x.double().mean(0)).square().mean()+1e-6).sqrt()) and torch.equal(s,s64.float()),
         'Unrounded FP64 mean/scale rule failed')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);core=ParallelLocalLearnedSelection(hidden_size=4,rank=2,mode='sigmoid')
    for name,param in core.named_parameters():param.requires_grad_(name not in uniform.FROZEN_SELECTOR)
    h=torch.arange(24,dtype=torch.float32).reshape(3,2,4).half()/10
    g=torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.]],dtype=torch.float16)
    valid=torch.tensor([[True,True],[True,True],[False,True]]);h[~valid]=0
    table=base.v7.state_info(core);delta,plain=core(h,g,valid_mask=valid,capture=True,output_dtype=torch.float32)
    with FixedLocalConditioning(core,torch.zeros((2,4)),torch.tensor(1.)) as hook:
        same,cap=core(h,g,valid_mask=valid,capture=True,output_dtype=torch.float32);cap=hook.add_capture(cap)
    need(torch.equal(same,delta) and torch.equal(cap['payload'],plain['payload']) and base.v7.state_info(core)==table,
         'Neutral hook changed core values/state')
    means=torch.full((2,4),.25);scale=torch.tensor(.5)
    with FixedLocalConditioning(core,means,scale) as hook:
        delta,cap=core(h,g,valid_mask=valid,capture=True,output_dtype=torch.float32);cap=hook.add_capture(cap)
    need(torch.equal(cap['conditioned_local_rms_input'],(core.rms(h)-means.unsqueeze(0))/.5)
         and torch.equal(cap['query'],plain['query']) and torch.equal(cap['gates'],valid.float()*.5)
         and bool((cap['messages'][~valid]==0).all()) and bool((delta==0).all()),
         'Actual local-only transformation/global query/padding/zero-U differs')
    delta.sum().backward();need(core.local.weight.grad is not None and bool((core.local.weight.grad==0).all())
         and core.up.weight.grad is not None and bool((core.up.weight.grad!=0).any())
         and means.grad is scale.grad is None,'Fixed-statistics/zero-U gradient contract failed')
    try:
        with FixedLocalConditioning(core,means,scale):raise RuntimeError('fixture')
    except RuntimeError:pass
    need(not core.local._forward_pre_hooks and not hasattr(core.local,'_fixed_conditioning_active'),'Exception left local hook installed')
    with FixedLocalConditioning(core,means,scale):
        try:
            with FixedLocalConditioning(core,means,scale):pass
        except ValueError:pass
        else:raise ValueError('Nested conditioning accepted')
    need(not core.local._forward_pre_hooks,'Nested fixture left hooks')
    scenes={'a':dict(question='q')};st=dict(questions=['q'],means=means[:1])
    repeated=query_means(torch,st,scenes,['a'],dict(offsets=[0,3]),arm='question')
    need(torch.equal(repeated,means[:1].expand(3,4)),'Same original mean must cover all target prefixes')
    global_only=query_means(torch,dict(global_mean=means[0]),{},['unknown'],dict(offsets=[0,3]),arm='global')
    need(torch.equal(global_only,means[:1].expand(3,4)),'Global arm must not read a question lookup')
    return dict(passed=True,checks=6,statistics_unrounded_fp64=True,neutral_hook_identity=True,
        local_only_transform_global_unchanged=True,padding_and_zero_U_gradient=True,exception_nesting_cleanup=True,same_mean_all_prefixes=True)


def check(out,frozen):
    import torch
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    torch.set_num_threads(4);bindings={};parent,oracle_report=conditional_gate(bindings)
    for key in ('rows_file','scenes_file','pairs_file','order_file','features_file','initial_file','native_model_file'):
        bind(parent[key],bindings,parent['runtime_bindings'][str(Path(parent[key]).resolve())])
    originals=read(parent['rows_file'])['train'];scenes=read(parent['scenes_file']);pairs=read(parent['pairs_file']);order=read(parent['order_file'])
    need(len(originals)==len(scenes)==108 and order==uniform.order_pairs(pairs),'Exact uniform training inventory/order differs')
    questions=sorted({r['sample']['question'] for r in originals});need(len(questions)==6,'Exactly six original questions required')
    rows=[{**{k:r['sample'][k] for k in ('sid','contrast_id','pair_id','variant','n_frames','question','gold','target_ids')},
        'first_token_id':r['sample']['target_ids'][0],'question_index':questions.index(r['sample']['question'])} for r in originals]
    need(Counter(r['gold'] for r in rows)==Counter({name:12 for name in oracle.PEOPLE})
         and all(r['sample']['split']=='train' for r in originals),'Balanced TRAIN-only label inventory differs')
    row_map={r['sid']:r for r in rows};head_counts=[sum(len(row_map[sid]['target_ids']) for pair in order[i:i+8] for sid in pair['sids']) for i in range(0,4800,8)]
    need(len(head_counts)==600 and sum(head_counts)==21334 and max(head_counts)==44,'Exact registered full-CE head row inventory differs')
    save(out/'rows.json',rows);save(out/'scenes.json',scenes);save(out/'pairs.json',pairs);save(out/'order.json',order)
    occurrences=[]
    for row in rows:
        for i,ids in enumerate(scenes[row['sid']]['local_feature_ids']):
            need(scenes[row['sid']]['target_prefixes'][0]==[],'Statistics must use original empty-prefix features')
            occurrences.append(dict(sid=row['sid'],local_index=i,feature_id=ids[0],question_index=row['question_index']))
    need(len(occurrences)==1296 and Counter(o['question_index'] for o in occurrences)==Counter({i:216 for i in range(6)}),
         'Question/occurrence statistic coverage differs')
    save(out/'stats_inventory.json',dict(questions=questions,occurrences=occurrences,training_sids=[r['sid'] for r in rows],
        original_cached_empty_prefix_only=True,no_fitted_captures=True,no_local_labels=True))
    packet=torch.load(parent['features_file'],map_location='cpu',weights_only=True)
    need(base.v7.tensor_info(packet['states'])==parent['feature_tensor'] and packet['feature_ids']==sorted(packet['feature_ids']),
         'Original training feature tensor differs')
    index={fid:i for i,fid in enumerate(packet['feature_ids'])}
    h=torch.stack([packet['states'][index[o['feature_id']]] for o in occurrences])
    x=h.float()*torch.rsqrt(h.float().square().mean(-1,keepdim=True)+1e-6)
    qi=torch.tensor([o['question_index'] for o in occurrences],dtype=torch.int64)
    m64,m,s64,s=compute_statistics(torch,x,qi,6)
    stats=dict(schema_version=1,questions=questions,occurrences=occurrences,x=x,question_indices=qi,
        global_mean_fp64=x.double().mean(0),global_mean=x.double().mean(0).float(),
        means_fp64=m64,means=m,scale_fp64=s64,scale=s)
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    stats_file=data/'statistics.pt';torch.save(stats,stats_file)
    initial_file=ckpt/'initial.pt';shutil.copyfile(parent['initial_file'],initial_file)
    initial=torch.load(initial_file,map_location='cpu',weights_only=True)
    with torch.random.fork_rng(devices=[]):torch.manual_seed(24);core=ParallelLocalLearnedSelection(mode='sigmoid')
    need(initial['seed']==24 and base.v7.state_info(core)==parent['initial_state'],'Exact uniform random initialization differs')
    core.load_state_dict(initial['branch']);mask=uniform.freeze_selector(torch,core)
    need(base.v7.state_info(core)==parent['initial_state'],'Freezing selector changed initial state')
    tests=self_test(torch)
    from scripts.report_native_identity_join_conditioning import self_test as report_self_test
    tests['reporter']=report_self_test(torch)
    module_identity=oracle.native_module_identity(torch,parent['native_identity'])
    native_packet=torch.load(parent['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=oracle.native_modules(torch,native_packet,parent['native_identity'],'cpu');del native_packet,norm,head
    runtime_bindings={}
    for path in (out/'rows.json',out/'scenes.json',out/'pairs.json',out/'order.json',out/'stats_inventory.json',
                 stats_file,initial_file,Path(parent['features_file']),Path(parent['native_model_file'])):bind(path,runtime_bindings)
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,tests=tests,trainability_mask=mask,
        input_bindings=bindings,runtime_bindings=runtime_bindings,uniform_plan=dict(file=str(oracle.PARENT),sha256=oracle.PARENT_SHA),
        oracle_report=oracle_report,rows_file=str(out/'rows.json'),scenes_file=str(out/'scenes.json'),pairs_file=str(out/'pairs.json'),
        order_file=str(out/'order.json'),order_object_sha256=object_sha(order),head_rows_by_update=head_counts,
        stats_inventory_file=str(out/'stats_inventory.json'),stats_file=str(stats_file),stats_sha256=sha(stats_file),
        stats_tensors={k:base.v7.tensor_info(v) for k,v in stats.items() if isinstance(v,torch.Tensor)},
        features_file=parent['features_file'],features_sha256=parent['features_sha256'],feature_tensor=parent['feature_tensor'],
        initial_file=str(initial_file),initial_sha256=sha(initial_file),initial_state=parent['initial_state'],
        native_model_file=parent['native_model_file'],native_model_sha256=parent['native_model_sha256'],
        native_identity=parent['native_identity'],native_identity_sha256=parent['native_identity_sha256'],native_module_identity=module_identity,
        no_pretrained_backbone_loaded=True,no_pretrained_head_forward=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,oracle_report=oracle_report,
        occurrences=1296,training_contexts=108,training_head_rows=21334,final_head_rows=108,no_pretrained_head_forward=True)


def verify_plan(path,ancestors=False):
    path=Path(path).resolve();plan=read(path)
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Conditioning plan/source changed')
    for n,h in plan['source_sha256'].items():need(sha(path.parent/'source'/n.replace('/','_'))==h,'Source snapshot changed')
    for file,h in plan['runtime_bindings'].items():need(sha(file)==h,'Consumed conditioning input changed')
    if ancestors:
        for file,h in plan['input_bindings'].items():need(sha(file)==h,'Conditioning ancestor changed')
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
        if fields[1]==RUN_JOBS[arm] and fields[2]=='gpu':need(fields[0]==os.environ['SLURM_JOB_ID'],'Single conditioning GPU attempt per arm only')


def run(args,out,frozen,started):
    import torch
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    torch.set_num_threads(4);args.plan=args.plan.resolve();plan=verify_plan(args.plan);single_run_guard(out,args.arm)
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200'
         and not torch.backends.cuda.matmul.allow_tf32,'One B200 with FP32 projection/TF32 disabled required')
    hardware=dict(gpu=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    torch.manual_seed(24);torch.cuda.manual_seed_all(24);device=torch.device('cuda')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    rows=read(plan['rows_file']);scenes=read(plan['scenes_file']);cache=dict(scenes=scenes);order=read(plan['order_file'])
    save(out/'rows.json',rows);save(out/'presentations.json',order);save(out/'stats_inventory.json',read(plan['stats_inventory_file']))
    packet=torch.load(plan['features_file'],map_location='cpu',weights_only=True)
    need(base.v7.tensor_info(packet['states'])==plan['feature_tensor'],'Frozen features changed')
    states=packet['states'].to(device);index={fid:i for i,fid in enumerate(packet['feature_ids'])};del packet
    stats=torch.load(plan['stats_file'],map_location='cpu',weights_only=True)
    need({k:base.v7.tensor_info(v) for k,v in stats.items() if isinstance(v,torch.Tensor)}==plan['stats_tensors']
         and stats['occurrences']==read(plan['stats_inventory_file'])['occurrences'],'Fixed statistics changed')
    working_stats=dict(questions=stats['questions'],means=stats['means'].to(device),global_mean=stats['global_mean'].to(device),scale=stats['scale'].to(device))
    statistic_keys=('means','global_mean','scale')
    statistic_versions={k:working_stats[k]._version for k in statistic_keys}
    working_before={k:base.v7.tensor_info(working_stats[k]) for k in statistic_keys}
    need(working_before=={k:plan['stats_tensors'][k] for k in statistic_keys},'Deployed fixed statistics differ')
    core=ParallelLocalLearnedSelection(mode='sigmoid').to(device);initial=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    core.load_state_dict(initial['branch']);mask=uniform.freeze_selector(torch,core)
    need(base.v7.state_info(core)==plan['initial_state'] and mask==plan['trainability_mask'],'Identical initial core/mask required')
    initial_file=ckpt/'initial.pt';shutil.copyfile(plan['initial_file'],initial_file)
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=oracle.native_modules(torch,packet,plan['native_identity'],device);del packet
    module_identity=oracle.native_module_identity(torch,plan['native_identity']);need(module_identity==plan['native_module_identity'],'Native modules changed')
    native_initial=dict(norm=base.v7.tensor_info(norm.weight),head=base.v7.tensor_info(head.weight));versions=(norm.weight._version,head.weight._version)
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
        blob=dict(schema_version=1,arm=args.arm,phase=current_phase,step=current_step,sids=sids,layout=layout,
            local_states=local,global_states=g,valid_mask=valid,question_means=means,capture=cap,**io)
        path=data/f'{current_phase}_capture_{current_step:04d}.pt';torch.save(cpu_tree(torch,blob),path)
        record=dict(arm=args.arm,phase=current_phase,step=current_step,sids=sids,file=str(path),sha256=sha(path),
            weights_file=str(weights_file),weights_sha256=sha(weights_file),weights_step=weights_step,
            tensors={k:base.v7.tensor_info(v) for k,v in io.items()})
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
        current_phase='evaluation';capture_current=True
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
                record=save_capture(h,g,valid,means,cap,final_layout,sids,checkpoint,600)
                gpu_nll=torch.nn.functional.cross_entropy(logits.float(),torch.tensor(final_layout['target_ids'],device=device),reduction='none')
                raw_batches.append(dict(capture_file=record['file'],capture_sha256=record['sha256'],sids=sids,
                    gpu_fp32_nll=gpu_nll.tolist(),logits=base.v7.tensor_info(io['logits'])))
        final_packets=[torch.load(r['capture_file'],map_location='cpu',weights_only=True) for r in raw_batches]
        raw_logits=torch.cat([p['logits'] for p in final_packets]);del final_packets
        raw_file=data/'final_logits.pt';torch.save(dict(schema_version=1,sids=[r['sid'] for r in rows],logits=raw_logits,batches=raw_batches),raw_file)
        targets=torch.tensor([r['first_token_id'] for r in rows]);ids=raw_logits.argmax(-1)
        promoted=raw_logits.double();shift=promoted-promoted.amax(-1,keepdim=True)
        nll=torch.logsumexp(shift,-1)-shift[torch.arange(108),targets];gpu_losses=[v for batch in raw_batches for v in batch['gpu_fp32_nll']]
        predictions=[dict(r,argmax_id=int(ids[i]),first_token_correct=int(ids[i])==r['first_token_id'],gpu_fp32_nll=gpu_losses[i],nll=float(nll[i])) for i,r in enumerate(rows)]
        save(out/'predictions.json',predictions);save(out/'captures.json',captures);save(out/'calls.json',calls)
        native_final=dict(norm=base.v7.tensor_info(norm.weight),head=base.v7.tensor_info(head.weight))
        expected=dict(core=607,conditioning=607,norm=607,head=607,vlm=0,vision=0)
        working_after={k:base.v7.tensor_info(working_stats[k]) for k in statistic_keys}
        need(working_after==working_before and statistic_versions=={k:working_stats[k]._version for k in statistic_keys},'Final fixed statistics changed')
        need(counters==expected and native_final==native_initial and base.v7.state_info(core)==final
             and not core.local._forward_pre_hooks and not hasattr(core.local,'_fixed_conditioning_active')
             and all(not p.requires_grad and p.grad is None for p in core.parameters()),'Final endpoint/calls/hooks differ')
        endpoint=dict(passed=True,before_evaluation=final,after_evaluation=base.v7.state_info(core),parameter_sha256=object_sha(final),
            native_before=native_initial,native_after=native_final,native_weights_unchanged=True,counters=counters,
            stats_tensors=plan['stats_tensors'],stats_file=plan['stats_file'],stats_sha256=plan['stats_sha256'],
            working_statistics_before=working_before,working_statistics_after=working_after,statistics_versions_unchanged=True,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),hooks_removed=True)
        save(out/'final_endpoint.json',endpoint)
        result=dict(**config,passed=True,completed=True,phase='run',computational_integrity_passed=True,steps=600,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),selected_parameter_sha256=object_sha(final),checkpoint_roundtrip_passed=True,
            training_file=str(out/'training.json'),training_sha256=sha(out/'training.json'),captures_file=str(out/'captures.json'),captures_sha256=sha(out/'captures.json'),
            raw_file=str(raw_file),raw_sha256=sha(raw_file),raw_tensor=base.v7.tensor_info(raw_logits),raw_batches=raw_batches,
            predictions_file=str(out/'predictions.json'),predictions_sha256=sha(out/'predictions.json'),
            calls_file=str(out/'calls.json'),calls_sha256=sha(out/'calls.json'),counters=counters,
            endpoint_file=str(out/'final_endpoint.json'),endpoint_sha256=sha(out/'final_endpoint.json'),
            roundtrip_file=str(out/'checkpoint_roundtrip.json'),roundtrip_sha256=sha(out/'checkpoint_roundtrip.json'),
            first_token_fit=oracle.criteria(predictions),training_head_rows=21334,final_head_rows=108,total_head_rows=21442,
            setup_before_training_seconds=setup,training_seconds=sum(r['seconds'] for r in logs),
            native_weights_unchanged=True,statistics_frozen=True,statistics_versions_unchanged=True,cached_first_query_only=True,no_pretrained_backbone_loaded=True,
            no_dev_or_test=True,no_native_or_whole_answer_claim=True)
        # This is ordinary cached supervision; remove the oracle-specific label from the reused counting utility.
        result['first_token_fit'].pop('oracle_answer_code_supplied');result['first_token_fit']['cached_first_query_only']=True
        need(time.perf_counter()-started<=90,'Single head-only conditioning job exceeded90 seconds')
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
        need(sources()==frozen,'Source changed during conditioning execution')
        result['elapsed_seconds']=time.perf_counter()-started;save(out/'summary.json',result)
        print(json.dumps(dict(passed=result['passed'],directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

"""Held natural native trainability evaluation of both fixed factor endpoints.

CPU preparation binds already audited checkpoints, original image bundles and
fixed statistics. Profiles precede a separate CPU release. No fitting, local
labels, answer mask, teacher-forced answer prefix, or fresh evaluation occurs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_factor_long as cached
from scripts import report_native_identity_join_factor_binding as factor_audit
from scripts import native_factor_binding_runtime as runtime
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,object_sha
native=runtime.native;oracle=cached.oracle;software=cached.uniform.software;tensor_info=cached.tensor_info
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_factor_native'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_factor_native')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_factor_native')
PROTOCOL='identity_join_fixed_factor_native_trainability';ARMS=('product','additive')
JOBS={phase:{arm:f'identity_join_factor_native_{phase}_{arm}' for arm in ARMS} for phase in ('profile','main')}
PROPOSAL='docs/paper/NATIVE_AGGREGATION_FACTOR_NATIVE_PROPOSAL.md'
PROPOSAL_SHA='e228fc4a52f8409bfc7443beede0730908751132840a1cb039d2db53c43338e5'
CACHED_REPORT=REPO/'outputs/native_aggregation_vlm/identity_join_factor_long/report_443543/summary.json'
CACHED_REPORT_SHA='705fef511c80c11bfd54fa0ce2adda426462c395345f15ab7a676fc49a6f8a43'
CACHED_ANALYSIS_SHA='c6f8db9921ab30d6200ef5fd362ee277d82c6ea8f9130ec807dfd0c333336b98'
CACHED_PLAN_SHA='3442ba95f15bcb833291f71d078832e7e95b7e0137e423696ff6ea535fd018e2'
POLICY=dict(protocol=PROTOCOL,arms=list(ARMS),checkpoint_step=6000,maximum_new_tokens=4,
    profile_trajectories=4,profile_model_cap=16,profile_vision_calls=4,main_trajectories=108,main_model_cap=432,main_vision_calls=108,
    profile_order='N8 zero,fitted; N16 zero,fitted; first Sandra pair in frozen training row order',
    zero='fixed fitted state with only up.weight zeroed',paired_only=True,no_fitting=True,training_only=True,
    profile_seconds=120,main_seconds=480,campaign_seconds=1200,max_concurrent_gpus=2,
    cpu_check_seconds=90,cpu_release_seconds=300,cpu_report_seconds=300,
    historical_four_token_floor=2.30838355794549,projection_multiplier=1.25,projection_reserve=60.,
    cpu_native_head_replay='all profile prefixes at actual N+1 head batch shape; none for main',
    head_tv_max=.02,head_argmax_exact=True,functional_atol=1e-4,functional_rtol=1e-4,
    whole_answer_threshold=103,complete_family_threshold=16,families=18,contexts=108,
    no_output_filter=True,no_answer_prefix=True,no_fresh_or_dev=True)
OWN=('scripts/diagnose_native_identity_join_factor_native.py','scripts/report_native_identity_join_factor_native.py',
     'scripts/native_factor_binding_runtime.py','tests/test_native_factor_binding_runtime.py',
     'slurm/native_identity_join_factor_native_check.sbatch','slurm/native_identity_join_factor_native_profile.sbatch',
     'slurm/native_identity_join_factor_native_main.sbatch','slurm/native_identity_join_factor_native_release.sbatch',
     'slurm/native_identity_join_factor_native_report.sbatch',PROPOSAL)


def inherited_sources():
    values={**cached.inherited_sources(),**cached.sources()};need(len(values)==105,'Locked cached-source closure differs')
    for name,digest in values.items():need(sha(REPO/name)==digest,'Inherited source changed: '+name)
    return values


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,digest in frozen.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'New source snapshot changed')
    save(out/'source_hashes.json',frozen);save(out/'inherited_sources.json',dict(source_sha256=inherited_sources()))
    return frozen


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path);need(expected is None or digest==expected,'Bound artifact changed: '+str(path));bindings[str(path)]=digest;return digest


def model_view(sample):return {key:sample[key] for key in ('sid','n_frames','question','image_files')}


def criterion(rows):
    need(len(rows)==len({r['sid'] for r in rows})==108,'Native training denominator differs')
    groups={}
    for row in rows:groups.setdefault(row['contrast_id'],[]).append(row)
    need(len(groups)==18 and all({(r['variant'],r['n_frames']) for r in values}=={(i,n) for i in range(3) for n in (8,16)}
         and len(values)==6 for values in groups.values()),'Complete native family inventory differs')
    exact=sum(r['exact'] for r in rows);complete=sum(all(r['exact'] for r in values) for values in groups.values())
    return dict(passed=exact>=103 and complete>=16,whole_correct=exact,first_correct=sum(r['first_token_correct'] for r in rows),
        contexts=108,complete_families=complete,families=18,thresholds=dict(whole_correct=103,complete_families=16),
        natural_whole_answer_and_eos=True,training_only=True)


def projection(setups,timings):
    need(len(setups)==2 and len(timings)==8 and all(math.isfinite(x) and x>0 for x in setups+timings),'Incomplete positive timing inventory')
    setup=max(setups);observed=max(timings);bound=max(POLICY['historical_four_token_floor'],observed)
    seconds=setup+1.25*108*bound+60.
    return dict(passed=seconds<=480,pooled_setup_seconds=setup,maximum_observed_four_token_seconds=observed,
        historical_floor_seconds=POLICY['historical_four_token_floor'],pooled_scene_seconds=bound,multiplier=1.25,
        contexts=108,reserve_seconds=60.,projected_seconds=seconds,cap_seconds=480.)


def check(args,out,frozen):
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(4);bindings={};path=args.cached_report.resolve();need(path==CACHED_REPORT,'Only the fixed registered cached report is eligible');bind(path,bindings,CACHED_REPORT_SHA);proof=read(path)
    need(proof['passed'] is proof['completed'] is True and proof['phase']=='report' and proof['protocol']==cached.PROTOCOL,
         'Passed independent cached report required')
    need(proof['analysis_sha256']==CACHED_ANALYSIS_SHA,'Registered cached analysis changed');bind(proof['analysis_file'],bindings,CACHED_ANALYSIS_SHA);analysis=read(proof['analysis_file'])
    need(analysis['passed'] is analysis['completed'] is True and set(analysis['runs'])==set(ARMS)
         and proof['first_token_screen']==analysis['first_token_screen'] and any(proof['first_token_screen'][a]['passed'] for a in ARMS),
         'Either fixed cached screen must pass; both arms must remain present')
    need(analysis['plan_sha256']==CACHED_PLAN_SHA,'Registered cached CPU plan changed');bind(analysis['plan_file'],bindings,CACHED_PLAN_SHA);parent=cached.verify_plan(analysis['plan_file'],ancestors=True)
    need(proof['source_sha256']==parent['source_sha256'] and proof['inherited_source_sha256']==parent['inherited_source_sha256'],
         'Cached reporting/source closure differs')
    bind(REPO/PROPOSAL,bindings,PROPOSAL_SHA)
    uniform_ref=parent['uniform_plan'];bind(uniform_ref['file'],bindings,uniform_ref['sha256']);uniform=read(uniform_ref['file'])
    for key in ('rows_file','prepared_file','scenes_file'):
        bind(uniform[key],bindings,uniform['runtime_bindings'][uniform[key]])
    old_rows=read(uniform['rows_file'])['train'];rows=[dict(item['sample']) for item in old_rows]
    compact=read(parent['rows_file']);need(len(rows)==108 and [r['sid'] for r in rows]==[r['sid'] for r in compact], 'Exact original training row order differs')
    prepared_all=read(uniform['prepared_file']);prepared={r['sid']:prepared_all[r['sid']] for r in rows}
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    _,rope,api=software.backend.native_api(processor);need(api==parent['native_identity']['native_api'],'Installed native position/API source differs');layouts={}
    for sample,row in zip(rows,compact):
        need(sample['split']=='train' and all(sample[k]==row[k] for k in ('sid','contrast_id','pair_id','variant','n_frames','question','gold','target_ids')),
             'Training semantic/label ownership differs')
        need(processor.tokenizer.encode(sample['gold'],add_special_tokens=False)+[151645]==sample['target_ids'],'Native whole-name target differs')
        for image in sample['image_files']:bind(image['path'],bindings,image['sha256'])
        item=prepared[sample['sid']];bind(item['file'],bindings,item['sha256']);bundle=torch.load(item['file'],map_location='cpu',weights_only=True)
        runtime.validate_bundle(bundle);meta=bundle['metadata'];need(meta==item['metadata'] and meta['prefix_ids']==[]
            and meta['sid']==sample['sid'] and meta['question']==sample['question'] and meta['n_frames']==sample['n_frames']
            and meta['image_paths']==[x['path'] for x in sample['image_files']] and meta['image_sha256']==[x['sha256'] for x in sample['image_files']],
            'Original prepared image/question bundle differs')
        layout=native.audit_layout(rope,bundle);prepared[sample['sid']]=dict(item,layout=layout['metadata']);layouts[sample['sid']]=layout
    first=next(r for r in rows if r['gold']=='Sandra');pair=[r for r in rows if r['pair_id']==first['pair_id']]
    need([r['n_frames'] for r in pair]==[8,16] and all(r['gold']=='Sandra' for r in pair),'First frozen-row Sandra pair differs')
    cases=[dict(sid=r['sid'],n_frames=r['n_frames'],state=state) for r in pair for state in ('zero','fitted')]
    ckpt=CKPT/out.name;ckpt.mkdir(parents=True,exist_ok=False);checkpoints={}
    initial=torch.load(parent['initial_file'],map_location='cpu',weights_only=True)['branch']
    for arm in ARMS:
        run=analysis['runs'][arm];endpoint=run['endpoint_audits']['6000'];source=Path(endpoint['checkpoint'])
        bind(source,bindings,endpoint['checkpoint_sha256']);packet=torch.load(source,map_location='cpu',weights_only=True)
        need(packet['step']==6000 and packet['config']['arm']==arm and packet['config']['plan_sha256']==analysis['plan_sha256'], 'Fixed cached endpoint ownership differs')
        weights=packet['branch'];factor_audit.check_weights(torch,weights,initial);table={k:tensor_info(v) for k,v in weights.items()}
        need(object_sha(table)==endpoint['selected_parameter_sha256'],'Audited final factor tensor identities differ')
        fitted=ckpt/f'{arm}_fitted.pt';shutil.copyfile(source,fitted);need(sha(fitted)==endpoint['checkpoint_sha256'],'Fitted endpoint copy differs')
        zero={k:v.detach().clone() for k,v in weights.items()};zero['up.weight'].zero_();zero_file=ckpt/f'{arm}_zero.pt'
        torch.save(dict(branch=zero,source_checkpoint=str(source),source_checkpoint_sha256=sha(source),intervention='zero_up_only'),zero_file)
        need(all(torch.equal(zero[k],weights[k]) for k in weights if k!='up.weight') and bool((zero['up.weight']==0).all()),'Zero-U identity copy changed other tensors')
        checkpoints[arm]=dict(fitted=dict(file=str(fitted),sha256=sha(fitted),state=table),
            zero=dict(file=str(zero_file),sha256=sha(zero_file),state={k:tensor_info(v) for k,v in zero.items()}),
            cached_checkpoint=str(source),cached_checkpoint_sha256=sha(source),cached_endpoint=endpoint)
    stats=torch.load(parent['stats_file'],map_location='cpu',weights_only=True)
    conditioning={k:tensor_info(stats[k]) for k in ('global_mean','scale')};need(conditioning=={k:parent['stats_tensors'][k] for k in conditioning},'Frozen global conditioning changed')
    native_plan=uniform['native_software_plan'];bind(native_plan['file'],bindings,native_plan['sha256'])
    need(read(native_plan['file'])['native_identity']==parent['native_identity'],'Existing native software identity differs')
    tests=dict(runtime=runtime.self_test(torch))
    from scripts.report_native_identity_join_factor_native import self_test
    tests['reporter']=self_test(torch)
    save(out/'rows.json',rows);save(out/'prepared.json',prepared)
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);layout_file=data/'layouts.pt';torch.save(layouts,layout_file);bind(layout_file,bindings)
    runtime_bindings={}
    for file in (out/'rows.json',out/'prepared.json',Path(parent['stats_file']),Path(parent['native_model_file']),Path(native_plan['file'])):bind(file,runtime_bindings)
    for item in prepared.values():bind(item['file'],runtime_bindings,item['sha256'])
    for states_by_arm in checkpoints.values():
        for state in ('fitted','zero'):bind(states_by_arm[state]['file'],runtime_bindings,states_by_arm[state]['sha256'])
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),tests=tests,
        cached_report=dict(file=str(path),sha256=sha(path),analysis_file=proof['analysis_file'],analysis_sha256=proof['analysis_sha256']),
        cached_plan=dict(file=analysis['plan_file'],sha256=analysis['plan_sha256']),uniform_plan=uniform_ref,
        rows_file=str(out/'rows.json'),prepared_file=str(out/'prepared.json'),layout_file=str(layout_file),layout_sha256=sha(layout_file),profile_cases=cases,checkpoints=checkpoints,
        stats_file=parent['stats_file'],stats_sha256=parent['stats_sha256'],conditioning_identity=conditioning,
        native_model_file=parent['native_model_file'],native_model_sha256=parent['native_model_sha256'],
        native_identity=parent['native_identity'],native_identity_sha256=parent['native_identity_sha256'],native_module_identity=parent['native_module_identity'],
        native_software_plan=native_plan,input_bindings=bindings,runtime_bindings=runtime_bindings,no_new_preparation=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,training_contexts=108,profile_cases_per_arm=4,no_model_loaded=True)


def verify_plan(path,ancestors=False):
    path=Path(path).resolve();plan=read(path)
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Frozen native preparation changed')
    for name,digest in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Source snapshot changed')
    need(read(path.parent/'inherited_sources.json')==dict(source_sha256=plan['inherited_source_sha256']),'Inherited snapshot map changed')
    for file,digest in plan['runtime_bindings'].items():need(sha(file)==digest,'Consumed native input changed: '+file)
    if ancestors:
        for file,digest in plan['input_bindings'].items():need(sha(file)==digest,'Bound ancestry changed: '+file)
    proof=read(path.parent/'summary.json');need(proof['passed'] is proof['completed'] is True and proof['phase']=='check'
        and proof['plan_sha256']==sha(path) and proof['source_sha256']==plan['source_sha256'],'Passed CPU preparation required')
    return plan


def verify_release(path,plan_path,plan):
    path=Path(path).resolve();value=read(path);summary=read(path.parent/'summary.json')
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='release' and summary['release_file']==str(path)
         and summary['release_sha256']==sha(path) and value['source_sha256']==plan['source_sha256'] and value['policy']==POLICY
         and value['plan_file']==str(Path(plan_path).resolve()) and value['plan_sha256']==sha(plan_path)
         and value['projection']['passed'] is True and value['projection']['projected_seconds']<=480
         and value['main_reservation_seconds']==960 and set(value['profiles'])==set(ARMS),'Passed measured two-arm release required')
    for item in value['profiles'].values():need(sha(Path(item['directory'])/'summary.json')==item['summary_sha256'],'Audited profile summary changed')
    return dict(file=str(path),sha256=sha(path),summary_file=str(path.parent/'summary.json'),summary_sha256=sha(path.parent/'summary.json'))


def single_attempt_guard(out,phase,arm):
    need(os.environ.get('SLURM_JOB_NAME')==JOBS[phase][arm],'Exact registered per-arm phase job name required')
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'launch_sacct.psv';file.write_text(raw)
    save(out/'launch_accounting.json',dict(command=command,file=str(file),sha256=sha(file)))
    names={name for group in JOBS.values() for name in group.values()};seen=Counter()
    for line in raw.splitlines():
        fields=line.split('|');need(len(fields)==9,'Unexpected scheduler schema')
        if fields[1] in names:
            seen[fields[1]]+=1;need(fields[2]=='gpu' and seen[fields[1]]==1,'All failed/zero attempts count; no duplicate phase/arm attempt')
            if fields[1]==JOBS[phase][arm]:need(fields[0]==os.environ['SLURM_JOB_ID'],'A previous attempt forbids this launch')


def cpu_tree(torch,value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(torch,v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(torch,v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(torch,v) for v in value)
    return value


def basic_result(torch,result,bundle,arm,identity,condition):
    ids=result['generated_ids'];steps=len(ids);raw=result['raw_logits'];meta=result['metadata'];n=bundle['metadata']['n_frames']
    need(1<=steps<=4 and raw.dtype==torch.float32 and raw.shape==(steps,152064) and bool(torch.isfinite(raw).all())
         and torch.equal(raw,raw.half().float()) and raw.argmax(-1).tolist()==ids,'Natural raw FP16 logits/argmax differ')
    need(result['interaction']==arm and meta['native_identity_sha256']==identity and meta['scene_input_identity']==runtime.input_identity(bundle,identity)
         and len(result['captures'])==steps and result['capture_call_indices']==list(range(1,steps+1)),'Native source/capture ownership differs')
    expected=dict(model=steps,visual=1,language=steps,norm=steps,head=steps,broadcast=steps,fusion=steps,conditioning=steps,probe_head=0)
    need(result['counters']==expected,'Exactly one native call and one factor/conditioning call per token required')
    eos=meta['generation']['native_eos_token_ids'];complete=ids[-1] in eos
    need(not any(i in eos for i in ids[:-1]) and (complete or steps==4) and result['completed'] is complete
         and result['truncated'] is (not complete),'Natural EOS stopping differs')
    for cap in result['captures']:
        need(cap['native_query_hidden'].shape==cap['fused_query_hidden'].shape==(n+1,1,3584)
             and torch.equal(cap['native_delta'],cap['delta'].half()) and torch.equal(cap['fused_global'],cap['global_states']+cap['native_delta'])
             and torch.equal(cap['fused_query_hidden'][:-1],cap['native_query_hidden'][:-1])
             and torch.equal(cap['fused_query_hidden'][-1],cap['fused_global']),'Native FP16 cast/global-only write differs')
        if condition=='zero':need(bool((cap['delta']==0).all()) and torch.equal(cap['native_query_hidden'],cap['fused_query_hidden']),'Zero-U must leave every current native row unchanged')
    return dict(passed=True,emitted_tokens=steps,counters=expected,zero_up_identity=condition=='zero',raw_unmasked_argmax=True)


def run(args,out,frozen,started):
    import torch
    from gnnformer.runtime import load_runtime
    torch.set_num_threads(4);phase='profile' if args.profile else 'main';single_attempt_guard(out,phase,args.arm)
    args.plan=args.plan.resolve();plan=verify_plan(args.plan);release=None if args.profile else verify_release(args.main_release,args.plan,plan)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
    loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);native.native_contract(model,None)
    hardware=json.loads(json.dumps(software.live_identity(torch,loaded,read(plan['native_software_plan']['file']))))
    need(hardware['gpu']=='NVIDIA B200','Native hardware identity differs');norm=model.model.language_model.norm;head=model.lm_head
    native_table=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight))
    need(native_table==dict(norm=plan['native_identity']['norm_weight'],head=plan['native_identity']['head_weight']),'Actual native head/norm differs')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    states={}
    for state in ('fitted','zero') if args.profile else ('fitted',):
        descriptor=plan['checkpoints'][args.arm][state];file=ckpt/f'{state}.pt';shutil.copyfile(descriptor['file'],file)
        need(sha(file)==descriptor['sha256'],'Actual deployed checkpoint copy differs')
        packet=torch.load(file,map_location='cpu',weights_only=True);states[state]=packet['branch']
        need({k:tensor_info(v) for k,v in states[state].items()}==descriptor['state'],'Actual deployed tensor identities differ')
    stats=torch.load(plan['stats_file'],map_location='cpu',weights_only=True)
    mean=stats['global_mean'].to(loaded.device);scale=stats['scale'].to(loaded.device);del stats
    stats_before=dict(global_mean=tensor_info(mean),scale=tensor_info(scale));stats_versions=(mean._version,scale._version)
    need(stats_before==plan['conditioning_identity'],'Actual fixed global statistics differ')
    core=cached.make_core(torch,args.arm).to(loaded.device);core.load_state_dict(states['fitted']);core.eval().requires_grad_(False)
    model_versions={k:v._version for k,v in model.named_parameters()};rows=read(plan['rows_file']);lookup={r['sid']:r for r in rows};prepared=read(plan['prepared_file'])
    cases=plan['profile_cases'] if args.profile else [dict(sid=r['sid'],n_frames=r['n_frames'],state='fitted') for r in rows]
    config=dict(protocol=PROTOCOL,policy=POLICY,phase=phase,arm=args.arm,run_id=out.name,slurm_job_id=os.environ['SLURM_JOB_ID'],
        source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],plan_file=str(args.plan),plan_sha256=sha(args.plan),
        cached_report=plan['cached_report'],cached_plan=plan['cached_plan'],native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        native_weight_identity=native_table,hardware=hardware,conditioning_identity=stats_before,stats_file=plan['stats_file'],stats_sha256=plan['stats_sha256'],
        checkpoints={state:dict(file=str(ckpt/f'{state}.pt'),sha256=sha(ckpt/f'{state}.pt'),state=plan['checkpoints'][args.arm][state]['state']) for state in states},
        cases=cases,data_directory=str(data),checkpoint_directory=str(ckpt),main_release=release)
    save(out/'config.json',config);torch.cuda.synchronize();setup=time.perf_counter()-started;records=[];total=Counter()
    for number,case in enumerate(cases):
        tick=time.perf_counter();descriptor=plan['checkpoints'][args.arm][case['state']];core.load_state_dict(states[case['state']]);core.eval().requires_grad_(False)
        table={k:tensor_info(v) for k,v in core.state_dict().items()};need(table==descriptor['state'],'Selected per-trajectory state differs')
        versions={k:v._version for k,v in core.named_parameters()};item=prepared[case['sid']]
        need(sha(item['file'])==item['sha256'],'Prepared original image bundle changed');bundle=torch.load(item['file'],map_location='cpu',weights_only=True)
        runtime.validate_bundle(bundle);prep=time.perf_counter()-tick;observations=[];latest={}
        def normalized(module,arguments,output):
            latest.update(normalized=output[:,-1:,:].detach().cpu().clone(),norm_input_shape=list(arguments[0].shape),norm_output_shape=list(output.shape))
        def projected(module,arguments,output):
            observations.append(dict(normalized=latest['normalized'],head_input=arguments[0].detach().cpu().clone(),native_logits=output.detach().cpu().clone(),
                norm_input_shape=latest['norm_input_shape'],norm_output_shape=latest['norm_output_shape'],head_input_shape=list(arguments[0].shape),head_output_shape=list(output.shape)))
        work=time.perf_counter();partial=[];controller=[None];native_inputs=[];native_positions=[]
        def observed_input(module,arguments,kwargs):
            cache=kwargs.get('past_key_values')
            native_inputs.append(dict(input_ids=kwargs['input_ids'].detach().cpu().clone(),
                attention_mask=kwargs['attention_mask'].detach().cpu().clone(),has_pixels=kwargs.get('pixel_values') is not None,
                past_length=0 if cache is None else cache.get_seq_length()))
        def observed_positions(module,arguments,kwargs):
            native_positions.append(kwargs['position_ids'].detach().cpu().clone())
        def observe_controller(value):controller[0]=value
        def observed_forward(module,arguments,output):
            item=dict(raw_global_logits=output.logits[-1,-1].detach().cpu().clone())
            if controller[0] is not None:item['capture']=controller[0].export_last_capture(cpu=True)
            partial.append(item)
        try:
            with ExitStack() as stack:
                stack.callback(model.register_forward_hook(observed_forward).remove)
                stack.callback(model.register_forward_pre_hook(observed_input,with_kwargs=True).remove)
                stack.callback(model.model.language_model.register_forward_pre_hook(observed_positions,with_kwargs=True).remove)
                if args.profile:
                    stack.callback(norm.register_forward_hook(normalized).remove);stack.callback(head.register_forward_hook(projected).remove)
                result=runtime.generate_native(model,loaded.processor,core,bundle,interaction=args.arm,global_mean=mean,scale=scale,
                    native_identity_sha256=plan['native_identity_sha256'],max_new_tokens=4,capture=True,controller_observer=observe_controller)
        except BaseException:
            failure_file=data/f'trajectory_{number:03d}_partial.pt'
            torch.save(cpu_tree(torch,dict(case=case,prefixes=partial,profile_native=observations,core_state=table,
                conditioning_identity=stats_before,native_inputs=native_inputs,native_positions=native_positions,partial_outputs_retained=True)),failure_file)
            save(out/f'trajectory_{number:03d}_failure.json',dict(case=case,file=str(failure_file),sha256=sha(failure_file),partial_outputs_retained=True))
            raise
        raw_file=data/f'trajectory_{number:03d}.pt';packet=dict(schema_version=1,case=case,result=result,profile_native=observations if args.profile else None,
            core_state=table,conditioning_identity=stats_before,native_inputs=native_inputs,native_positions=native_positions)
        torch.save(cpu_tree(torch,packet),raw_file);digest=sha(raw_file)
        save(raw_file.with_suffix('.json'),dict(case=case,raw_file=str(raw_file),raw_sha256=digest,validation_pending=True))
        need(result['metadata']['layout']==item['layout'],'Actual native positions differ from CPU-bound original layout')
        audit=basic_result(torch,result,bundle,args.arm,plan['native_identity_sha256'],case['state']);steps=len(result['generated_ids'])
        need({k:v._version for k,v in core.named_parameters()}==versions and {k:tensor_info(v) for k,v in core.state_dict().items()}==table,
             'Native trajectory changed the deployed checkpoint')
        need(len(observations)==steps if args.profile else not observations,'Profile observation count differs')
        total.update(result['counters'])
        save(out/f'progress_{number+1:03d}.json',dict(processed=number+1,case=case,raw_file=str(raw_file),raw_sha256=digest,accuracy_not_scored=True))
        work_seconds=time.perf_counter()-work
        records.append(dict(index=number,case=case,raw_file=str(raw_file),raw_sha256=digest,prepared_file=item['file'],prepared_sha256=item['sha256'],
            preprocessing_seconds=prep,work_seconds=work_seconds,four_token_seconds=prep+work_seconds*4/steps,generated_tokens=steps,audit=audit))
    need(model_versions=={k:v._version for k,v in model.named_parameters()} and all(not v.requires_grad and v.grad is None for v in model.parameters())
         and native_table==dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight))
         and stats_versions==(mean._version,scale._version) and stats_before==dict(global_mean=tensor_info(mean),scale=tensor_info(scale)),
         'Native model or fixed conditioning changed')
    save(out/'raw_manifest.json',dict(completed=True,rows=records,n=len(records),all_raw_retained_before_scoring=True))
    expected_scenes=4 if args.profile else 108;cap=16 if args.profile else 432
    need(len(records)==total['visual']==expected_scenes and 1<=total['model']<=cap and total['probe_head']==0,
         'Native phase call inventory exceeds its bound')
    endpoint=dict(passed=True,native_weights_before=native_table,native_weights_after=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight)),
        native_parameter_versions_unchanged=True,conditioning_before=stats_before,conditioning_after=dict(global_mean=tensor_info(mean),scale=tensor_info(scale)),
        statistics_versions_unchanged=True,all_trajectory_checkpoint_versions_unchanged=True,final_core_state={k:tensor_info(v) for k,v in core.state_dict().items()},
        counters=dict(total),no_optimizer_or_training=True)
    save(out/'endpoint.json',endpoint)
    need(time.perf_counter()-started<=POLICY[phase+'_seconds'],'Native phase exceeded its fixed allocation cap')
    return dict(**config,passed=True,completed=True,raw_manifest_file=str(out/'raw_manifest.json'),raw_manifest_sha256=sha(out/'raw_manifest.json'),
        endpoint_file=str(out/'endpoint.json'),endpoint_sha256=sha(out/'endpoint.json'),counters=dict(total),setup_seconds=setup,
        trajectories=len(records),all_raw_retained_before_scoring=True,no_accuracy_selection=True,extra_gpu_head_calls=0,no_fitting=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true');action.add_argument('--profile',action='store_true');action.add_argument('--main',action='store_true')
    parser.add_argument('--cached-report',type=Path,default=CACHED_REPORT);parser.add_argument('--plan',type=Path);parser.add_argument('--arm',choices=ARMS);parser.add_argument('--main-release',type=Path)
    args=parser.parse_args();native.require_slurm(gpu=not args.check)
    if args.check:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only native preparation')
    else:need(args.plan is not None and args.arm in ARMS and (args.profile or args.main_release is not None),'Frozen plan/arm and measured main release required')
    phase='check' if args.check else ('profile' if args.profile else 'main');suffix='' if args.check else args.arm+'_'
    out=OUT/f'{phase}_{suffix}{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen=snapshot(out)
    save(out/'request.json',dict(phase=phase,arm=args.arm,plan=None if args.plan is None else str(args.plan),source_sha256=frozen))
    try:
        result=check(args,out,frozen) if args.check else run(args,out,frozen,started)
        need(sources()==frozen and inherited_sources()==result['inherited_source_sha256'],'Source changed during native stage')
        result['elapsed_seconds']=time.perf_counter()-started;save(out/'summary.json',result);print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

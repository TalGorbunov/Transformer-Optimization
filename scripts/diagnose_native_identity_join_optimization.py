"""Bounded 2x2 training-only identity-join trainability pilot.

Clip/sigmoid x native CE/CE+same-pair residual consistency. Fixed seed24,
600 updates on108 existing balanced training contexts; final108train+54dev
native greedy evaluations only. No extrapolation/test trajectory is consumed.
Four32-update profiles precede pooled CPU resource release. No fitted-parent
checkpoint, local labels, gate supervision, output masking or parameter search.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import train_native_identity_join_learned_v2 as base
from scripts import profile_native_learned_selection as software
from scripts import native_learned_selection_runtime as runtime
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,sha,save,object_sha
native=runtime.native
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_optimization'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_optimization')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_optimization')
PARENT=REPO/'outputs/native_aggregation_vlm/identity_join_learned/training/train_check_443056/plan.json'
PARENT_SHA='739477cc4a9b5e8278d8e632f0691543fd02ce42673585a349015970c251d79f'
ARMS=('clip_ce','clip_consistency','sigmoid_ce','sigmoid_consistency')
PROTOCOL='identity_join_fixed_trainability_2x2'
PROFILE_JOB='identity_join_optimization_profile';MAIN_JOB='identity_join_optimization_main'
POLICY=dict(protocol=PROTOCOL,arms=list(ARMS),seed=24,steps=600,profile_steps=32,pairs=54,training_contexts=108,
    dev_contexts=54,families=18,batch_pairs=8,batch_scenes=16,parameters=1041697,rank=96,
    lr=.001,warmup=50,final_lr=.00001,weight_decay=0.,clip_norm=1.,optimizer='AdamW',
    loss='original mean-scene mean-token native CE + coefficient * original mean-pair mean-prefix residual consistency',
    coefficients=[0.,1.],epsilon=1e-6,all_native_name_tokens_and_eos=True,maximum_new_tokens=4,
    subset='lexicographically first six allowed pairs with every room degree2; canonical three disjoint person trios; all3variants and N8/N16',
    schedule='persistent Random(24) shuffled54-pair cycles concatenated;600 batches of8 intact pairs; cycles may cross batch boundaries',
    gradient_steps=[1,2,32,128,300,600],gradient_diagnostics='separate CE/R norms and cosines before common clipping; no optimizer effect',
    training_fit=dict(first_correct=103,whole_correct=103,all_six_correct_families=16),
    id_generalization=dict(first_correct=49,whole_correct=49,all_three_correct_families=16),
    profile_generations=9,profile_max_model_calls=36,profile_vision_calls=9,profile_max_extra_heads=36,
    main_generations=162,main_max_model_calls=648,main_vision_calls=162,profile_seconds=150,main_seconds=600,
    campaign_gpu_seconds=3000,maximum_project_gpus=4,final_checkpoint_only=True,dev_evaluated_once=True,
    no_test_trajectory=True,no_fitted_parent_weights=True,no_local_supervision=True,
    inference_local_prompt='unaltered complete global question',native_dtype='torch.float16',core_dtype='torch.float32')
OWN=('scripts/diagnose_native_identity_join_optimization.py','scripts/report_native_identity_join_optimization.py',
     'scripts/report_native_identity_join_learned_v2.py','scripts/analyze_native_identity_join_learned.py',
     'tests/test_native_identity_join_learned_analysis.py','tests/test_native_identity_join_learned_report_v2.py',
     'slurm/native_identity_join_optimization_check.sbatch','slurm/native_identity_join_optimization_profile.sbatch',
     'slurm/native_identity_join_optimization_main.sbatch','slurm/native_identity_join_optimization_report.sbatch')


def sources():return {**base.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,h in frozen.items():
        dest=out/'source'/name.replace('/','_');dest.write_bytes((REPO/name).read_bytes());need(sha(dest)==h,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# Fixed training-only optimization pilot\n\n[Summary](summary.json) · [Sources](source_hashes.json). No test evaluation or efficacy confirmation.\n')
    return frozen


def arm_config(arm):
    need(arm in ARMS,'Unknown fixed arm');return arm.split('_')[0],float(arm.endswith('_consistency'))


def learning_rate(step):
    need(type(step) is int and 1<=step<=600,'Unregistered training step')
    return .001*step/50 if step<=50 else 1e-5+(.001-1e-5)*(1+math.cos(math.pi*(step-50)/550))/2


def subset(manifest):
    available=sorted({tuple(r['room_pair']) for r in manifest['splits']['train_N16']['samples']})
    rooms={x for p in available for x in p};pairs=next(p for p in itertools.combinations(available,6)
        if Counter(x for pair in p for x in pair)==Counter({x:2 for x in rooms}))
    need(pairs==(('Bathroom','Bedroom'),('Bathroom','Garden'),('Garden','Bedroom'),
        ('Kitchen','Office'),('Kitchen','Park'),('Office','Park')),'Registered six room pairs differ')
    chars=tuple(base.data_stage.CHARS);trios=tuple(tuple(chars[i:i+3]) for i in (0,3,6));train=[]
    for cell in ('train_N8','train_N16'):
        train += [(cell,r) for r in manifest['splits'][cell]['samples'] if tuple(r['room_pair']) in pairs and tuple(r['trio']) in trios]
    dev=[('dev_N16',r) for r in manifest['splits']['dev_N16']['samples'] if tuple(r['room_pair']) in pairs]
    train.sort(key=lambda x:(x[1]['contrast_id'],x[1]['variant'],x[1]['n_frames']));dev.sort(key=lambda x:(x[1]['contrast_id'],x[1]['variant']))
    need(len(train)==108 and len(dev)==54 and len({r['contrast_id'] for _,r in train})==18
         and Counter(r['gold'] for _,r in train)==Counter({c:12 for c in chars})
         and Counter(r['gold'] for _,r in dev)==Counter({c:6 for c in chars})
         and all(tuple(r['trio']) in trios for _,r in dev),'Exact balanced training/dev subset differs')
    groups={}
    for _,r in train:groups.setdefault(r['contrast_id'],[]).append(r)
    need(all({(r['variant'],r['n_frames']) for r in values}==set(itertools.product(range(3),(8,16))) for values in groups.values()),'Training family incomplete')
    return train,dev,dict(room_pairs=[list(p) for p in pairs],person_trios=[list(t) for t in trios],training_family_ids=sorted(groups),
        each_room_degree=2,training_answers_per_name=12,dev_answers_per_name=6)


def order_pairs(pairs):
    need(len(pairs)==54,'Exactly54 training pairs required');rng=random.Random(24);out=[];cycle=0
    while len(out)<4800:
        cycle+=1;indices=list(range(54));rng.shuffle(indices)
        out += [dict(cycle=cycle,pair_slot=i,pair_id=pairs[i]['pair_id'],sids=pairs[i]['sids']) for i in indices]
    return out[:4800]


def bind(path,bindings,expected=None):
    path=Path(path).resolve();h=sha(path);need(expected is None or h==expected,'Bound artifact differs: '+str(path));bindings[str(path)]=h;return h


def criteria(rows,training):
    required=108 if training else 54;need(len(rows)==required and len({r['sid'] for r in rows})==required,'Diagnostic denominator differs')
    first=sum(r['first_token_correct'] for r in rows);whole=sum(r['exact'] for r in rows);families={}
    for r in rows:families.setdefault(r['contrast_id'],[]).append(r)
    size=6 if training else 3;need(len(families)==18 and all(len(x)==size for x in families.values()),'Complete-family denominator differs')
    complete=sum(all(r['exact'] for r in x) for x in families.values());limit=POLICY['training_fit' if training else 'id_generalization']
    passed=first>=limit['first_correct'] and whole>=limit['whole_correct'] and complete>=16
    return dict(passed=passed,first_correct=first,whole_correct=whole,contexts=required,complete_families=complete,families=18,
        thresholds=limit,scope='fixed-seed trainability diagnostic' if training else 'same-question/trio new-context ID diagnostic')


def parse_resources(raw):
    jobs=[];events=[]
    for line in raw.splitlines():
        if not line.strip():continue
        f=line.split('|');need(len(f)==9,'Unexpected scheduler row');job,name,partition,state,code,seconds,tres,start,end=f
        if name not in (PROFILE_JOB,MAIN_JOB) or partition!='gpu':continue
        state=state.split()[0].rstrip('+');need(job.isdigit() and state in software.TERMINAL,'Nonterminal pilot allocation in ledger')
        values=dict(x.split('=',1) for x in tres.split(',') if '=' in x);typed=[int(v) for k,v in values.items() if k.startswith('gres/gpu:')]
        gpus=int(values['gres/gpu']) if 'gres/gpu' in values else sum(typed);seconds=int(seconds);cap=150 if name==PROFILE_JOB else 600
        need(gpus in (0,1) and (not typed or sum(typed)==gpus) and 0<=seconds and (not gpus or seconds<=cap),'Pilot allocation cap/TRES differs')
        jobs.append(dict(job_id=job,name=name,state=state,exit_code=code,gpus=gpus,seconds=seconds,gpu_seconds=gpus*seconds))
        if gpus and start!=end:events.extend(((datetime.fromisoformat(start),1),(datetime.fromisoformat(end),-1)))
    need(len(jobs)==len({r['job_id'] for r in jobs}),'Duplicate pilot allocation');total=sum(r['gpu_seconds'] for r in jobs);now=peak=0
    for _,d in sorted(events):now+=d;need(now>=0,'Invalid pilot allocation interval');peak=max(peak,now)
    need(now==0 and peak<=4 and total<=3000,'Pilot cumulative budget/concurrency exceeded')
    return dict(passed=True,jobs=jobs,allocated_gpu_seconds=total,maximum_concurrent_gpus=peak,failed_and_zero_allocations_retained=True)


def resources(out):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(command,capture_output=True,text=True,check=True);path=out/'all_user_sacct.psv';path.write_text(result.stdout)
    return dict(parse_resources(result.stdout),file=str(path),sha256=sha(path),command=command)


def self_test():
    fake=[dict(pair_id=str(i),sids=[str(2*i),str(2*i+1)]) for i in range(54)];order=order_pairs(fake)
    need(len(order)==4800 and set(Counter(r['pair_slot'] for r in order).values())=={88,89}
         and order==order_pairs(fake),'Fixed crossing-cycle schedule differs')
    need(learning_rate(1)==.00002 and learning_rate(50)==.001 and learning_rate(600)==.00001,'Fixed learning-rate schedule differs')
    rows=[dict(sid=str(i),contrast_id=str(i//6),first_token_correct=True,exact=True) for i in range(108)]
    need(criteria(rows,True)['passed'],'Perfect training criterion failed')
    for i in range(6):rows[i*6]['exact']=False
    need(not criteria(rows,True)['passed'],'Incomplete-family criterion ignored')
    row='1|'+PROFILE_JOB+'|gpu|FAILED|1:0|7|gres/gpu=1,gres/gpu:b200=1|2026-09-11T10:00:00|2026-09-11T10:00:07'
    zero='2|'+MAIN_JOB+'|gpu|CANCELLED|0:0|0||Unknown|Unknown'
    need(parse_resources(row+'\n'+zero)['allocated_gpu_seconds']==7,'Failed/zero accounting differs')
    for bad in (row+'\n'+row,row.replace('|7|','|151|'),row.replace('gpu:b200=1','gpu:b200=2')):
        try:parse_resources(bad)
        except ValueError:pass
        else:raise ValueError('Invalid scheduler fixture accepted')
    return dict(passed=True,schedule_threshold_resource_checks=True,no_model_loaded=True)


def check(out,frozen):
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(4);tests=dict(self_test(),tensor_algebra=tensor_self_test(torch));bindings={};bind(PARENT,bindings,PARENT_SHA)
    parent=base.verify_plan(PARENT,ancestors=False);proof=read(PARENT.parent/'summary.json');need(proof['passed'] and proof['completed'],'Parent CPU cache/source proof incomplete')
    bind(PARENT.parent/'summary.json',bindings);bind(parent['manifest']['file'],bindings,parent['manifest']['sha256'])
    manifest=read(parent['manifest']['file']);training,dev,description=subset(manifest)
    bind(parent['pairing_file'],bindings,parent['pairing_sha256']);all_pairs=read(parent['pairing_file'])['pairs'];sids={r['sid'] for _,r in training}
    pairs=[r for r in all_pairs if set(r['sids'])<=sids];need(len(pairs)==54,'Selected complete length pairs differ')
    by_sid={r['sid']:r for _,r in training}
    for pair in pairs:
        rows_in_pair=[by_sid[sid] for sid in pair['sids']]
        need(pair['n_frames']==[8,16] and [r['n_frames'] for r in rows_in_pair]==[8,16]
             and all(r['pair_id']==pair['pair_id'] and r['contrast_id']==pair['contrast_id']
                     and r['variant']==pair['variant'] and r['question']==pair['question']
                     and r['gold']==pair['gold'] and r['target_ids']==pair['target_ids'] for r in rows_in_pair),
             'Selected pair lost native N8/N16 ordering or same-variant ownership')
    need({sid for p in pairs for sid in p['sids']}==sids and len([sid for p in pairs for sid in p['sids']])==108,
         'Selected pair ownership differs')
    cache=read(parent['cache_binding']['file']);bind(parent['cache_binding']['file'],bindings,parent['cache_binding']['sha256'])
    bind(cache['native_model_file'],bindings,cache['native_model_sha256'])
    scenes={sid:cache['scenes'][sid] for sid in sorted(sids)}
    need(all(scenes[p['sids'][0]]['global_feature_ids']==scenes[p['sids'][1]]['global_feature_ids'] for p in pairs),
         'Selected paired global strict-prefix features differ')
    wanted=sorted({f for row in scenes.values() for f in row['global_feature_ids']}
        |{f for row in scenes.values() for per_item in row['local_feature_ids'] for f in per_item})
    narrowed=dict(features={f:cache['features'][f] for f in wanted});states,index=base.v7.load_features(torch,narrowed,'cpu')
    need(index=={f:i for i,f in enumerate(wanted)},'Subset feature row order differs')
    for f in narrowed['features'].values():bind(f['file'],bindings,f['file_sha256'])
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    dataout=DATA/out.name;dataout.mkdir(parents=True,exist_ok=False);ckpt=CKPT/out.name;ckpt.mkdir(parents=True,exist_ok=False)
    bind(manifest['samples_file'],bindings,manifest['samples_sha256'])
    wanted_sids={r['sid'] for _,r in training+dev}
    semantic={r['sid']:r['states'] for r in read(manifest['samples_file']) if r['sid'] in wanted_sids}
    need(set(semantic)==wanted_sids and len(semantic)==162,'Selected semantic audit metadata incomplete')
    semantic_file=dataout/'semantic_states.json';save(semantic_file,semantic)
    rows={};prepared={}
    for name,items in (('train',training),('dev',dev)):
        rows[name]=[]
        for cell,row in items:
            need(base.name_target(processor.tokenizer,row['gold'])==row['target_ids'],'Native name target differs')
            for image in row['image_files']:bind(image['path'],bindings,image['sha256'])
            bundle=runtime.prepare_scene(processor,base.data_stage.runtime_view(row),verify_processor_parity=True)
            path=dataout/(row['sid']+'.pt');torch.save(base.cpu_tree(torch,bundle),path)
            prepared[row['sid']]=dict(file=str(path),sha256=sha(path),metadata=bundle['metadata'])
            rows[name].append(dict(cell=cell,sample=row))
    feature_file=dataout/'features.pt';torch.save(dict(states=states,feature_ids=wanted),feature_file)
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    with torch.random.fork_rng(devices=[]):torch.manual_seed(24);core=ParallelLocalLearnedSelection(mode='clip')
    initial=ckpt/'initial.pt';torch.save(dict(branch=core.state_dict(),seed=24),initial)
    timing=read(parent['timing_cases_file']);need(len(timing['cases'])==9 and timing['derivation']['no_dev_or_test_scene_used'],'Only existing training-derived timings permitted')
    save(out/'timing_cases.json',timing);bind(parent['timing_cases_file'],bindings,parent['timing_cases_sha256'])
    for item in timing['cases']:
        for image in item['sample']['image_files']:bind(image['path'],bindings,image['sha256'])
    save(out/'rows.json',rows);save(out/'pairs.json',pairs);save(out/'prepared.json',prepared);save(out/'scenes.json',scenes)
    order=order_pairs(pairs);save(out/'order.json',order)
    runtime_bindings={}
    for path in [feature_file,initial,semantic_file,out/'rows.json',out/'pairs.json',out/'prepared.json',out/'scenes.json',out/'order.json',out/'timing_cases.json',Path(parent['native_software_plan']['file']),Path(cache['native_model_file'])]:bind(path,runtime_bindings)
    for item in prepared.values():bind(item['file'],runtime_bindings,item['sha256'])
    # Image hashes are checked during preparation/generation, but no test metadata
    # or trajectory file is in the runtime plan's inputs.
    for items in rows.values():
        for item in items:
            for image in item['sample']['image_files']:bind(image['path'],runtime_bindings,image['sha256'])
    for item in timing['cases']:
        for image in item['sample']['image_files']:bind(image['path'],runtime_bindings,image['sha256'])
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,runtime_bindings=runtime_bindings,
        subset=description,rows_file=str(out/'rows.json'),pairs_file=str(out/'pairs.json'),prepared_file=str(out/'prepared.json'),scenes_file=str(out/'scenes.json'),
        semantic_states_file=str(semantic_file),semantic_states_sha256=sha(semantic_file),
        order_file=str(out/'order.json'),order_object_sha256=object_sha(order),features_file=str(feature_file),features_sha256=sha(feature_file),
        feature_tensor=base.v7.tensor_info(states),initial_file=str(initial),initial_sha256=sha(initial),initial_state=base.v7.state_info(core),
        timing_cases_file=str(out/'timing_cases.json'),native_software_plan=parent['native_software_plan'],
        native_model_file=cache['native_model_file'],native_model_sha256=cache['native_model_sha256'],
        native_identity=parent['native_identity'],native_identity_sha256=parent['native_identity_sha256'],tests=tests,no_model_loaded=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),subset=description,tests=tests,no_model_loaded=True)


def verify_plan(path,ancestors=False):
    path=Path(path).resolve();plan=read(path);need(sha(path)==path.with_suffix('.sha256').read_text().strip()
        and plan['policy']==POLICY and plan['source_sha256']==sources(),'Frozen diagnostic plan/source differs')
    for n,h in plan['source_sha256'].items():need(sha(path.parent/'source'/n.replace('/','_'))==h,'Frozen source copy differs')
    for file,h in plan['runtime_bindings'].items():need(sha(file)==h,'Diagnostic consumed artifact differs')
    if ancestors:
        for file,h in plan['artifact_bindings'].items():need(sha(file)==h,'CPU parent binding differs')
    need(object_sha(read(plan['order_file']))==plan['order_object_sha256'] and object_sha(plan['native_identity'])==plan['native_identity_sha256'],'Order/native identity differs')
    proof=read(path.parent/'summary.json');need(proof['passed'] and proof['completed'] and proof['plan_sha256']==sha(path),'Complete CPU source freeze required')
    return plan


def gradient_geometry(torch,core,ce,residual,coefficient,cap):
    """Separate live losses without writing optimizer .grad buffers or changing RNG."""
    names,parameters=zip(*core.named_parameters())
    need(all(p.grad is None for p in parameters),'Diagnostic requires untouched zeroed gradient buffers')
    inputs=parameters+(cap['scores'],cap['payload'])
    cg=torch.autograd.grad(ce,inputs,retain_graph=True,allow_unused=True)
    rg=torch.autograd.grad(residual,inputs,retain_graph=True,allow_unused=True)
    need(all(x is not None and bool(torch.isfinite(x).all()) for x in cg+rg)
         and all(p.grad is None for p in parameters),'Diagnostic detached graph or changed optimizer gradient buffers')
    groups=dict(all=list(range(len(names))),selector=[i for i,n in enumerate(names) if n.startswith('selection_')],
        local=[i for i,n in enumerate(names) if n=='local_bias' or n.startswith('local.')],
        up=[i for i,n in enumerate(names) if n.startswith('up.')])
    geometry={}
    for group,indices in groups.items():
        a=sum(float(cg[i].double().square().sum()) for i in indices)
        b=sum(float(rg[i].double().square().sum()) for i in indices)
        dot=sum(float((cg[i].double()*rg[i].double()).sum()) for i in indices)
        geometry[group]=dict(ce_norm=math.sqrt(a),consistency_norm=math.sqrt(b),dot=dot,
            cosine=None if a==0 or b==0 else dot/math.sqrt(a*b),
            combined_norm=math.sqrt(max(0.,a+coefficient**2*b+2*coefficient*dot)),parameter_names=[names[i] for i in indices])
    raw=dict(ce={n:g.detach().cpu() for n,g in zip(names,cg)},
        consistency={n:g.detach().cpu() for n,g in zip(names,rg)},
        score_ce=cg[-2].detach().cpu(),score_consistency=rg[-2].detach().cpu(),
        payload_ce=cg[-1].detach().cpu(),payload_consistency=rg[-1].detach().cpu())
    extra={name:float(torch.linalg.vector_norm(value.double())) for name,value in raw.items() if not isinstance(value,dict)}
    return raw,dict(groups=geometry,intermediate_gradient_norms=extra,gradient_buffers_unchanged=True,
        ce_and_consistency_both_live=True,statistics_dtype='FP64 reductions of saved native FP32 gradients')


def query_diagnostics(torch,parts,cap,valid,layout):
    """Keep origin association loss separate from teacher-forced suffix loss."""
    offsets=layout['offsets'];losses=parts['position_losses']['ce'];first=offsets[:-1]
    continuation=[i for a,b in zip(offsets,offsets[1:]) for i in range(a+1,b-1)];eos=[b-1 for b in offsets[1:]]
    gates=cap['gates'].detach();selected=[gates[:,i][valid[:,i]] for i in first]
    return dict(first_indices=first,continuation_indices=continuation,eos_indices=eos,
        first_token_nll=[losses[i] for i in first],continuation_token_nll=[losses[i] for i in continuation],
        eos_token_nll=[losses[i] for i in eos],first_gate_sum=[float(x.sum()) for x in selected],
        first_gate_min=[float(x.min()) for x in selected],first_gate_max=[float(x.max()) for x in selected],
        first_gate_zero_count=[int((x==0).sum()) for x in selected],first_gate_valid_count=[x.numel() for x in selected],
        first_all_gates_closed=[bool((x==0).all()) for x in selected])


def verify_release(path,plan_path,plan):
    need(path is not None,'Fixed mains require a measured independent CPU release')
    path=Path(path).resolve();release=read(path)
    need(release['protocol']=='identity_join_optimization_main_release' and release['passed'] is True
         and release['completed'] is True and release['plan_sha256']==sha(plan_path)
         and release['source_sha256']==sources() and release['per_main_seconds_cap']==600
         and release['campaign_gpu_seconds_cap']==3000 and set(release['profiles'])==set(ARMS)
         and release['pooled_projection']['passed'] and 0<release['pooled_projection']['projected_seconds']<=600
         and release['campaign_budget']['passed'],'Independent pilot release/source/resource contract differs')
    for arm,record in release['profiles'].items():
        sp=Path(record['directory'])/'summary.json';need(sha(sp)==record['summary_sha256'],'Released training profile changed')
        result=read(sp);mode,coef=arm_config(arm)
        need(result['passed'] and result['completed'] and result['profile'] and result['arm']==arm
             and result['condition']==mode and result['consistency_coefficient']==coef and result['seed']==24
             and result['steps']==32 and result['source_sha256']==sources() and result['plan_sha256']==sha(plan_path)
             and result['native_timing']['passed'] and result['checkpoint_roundtrip_passed']
             and result['no_dev_or_test_evaluation'],'Released four-arm profile is invalid')
    return dict(file=str(path),sha256=sha(path))


def run(args,out,frozen,started):
    import shutil
    import torch
    from gnnformer.runtime import load_runtime
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    torch.set_num_threads(4);args.plan=args.plan.resolve();plan=verify_plan(args.plan)
    mode,coefficient=arm_config(args.arm);release=None if args.profile else verify_release(args.main_release,args.plan,plan)
    dataout=DATA/out.name;dataout.mkdir(parents=True,exist_ok=False);ckpt=CKPT/out.name;ckpt.mkdir(parents=True,exist_ok=False)
    load_start=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);native.native_contract(model)
    hardware=json.loads(json.dumps(software.live_identity(torch,loaded,read(plan['native_software_plan']['file']))))
    need(hardware['gpu']=='NVIDIA B200' and torch.cuda.device_count()==1,'One frozen-identity B200 is required')
    norm=model.model.language_model.norm;head=model.lm_head
    weights=dict(norm=base.v7.tensor_info(norm.weight),head=base.v7.tensor_info(head.weight))
    need(weights['norm']==plan['native_identity']['norm_weight'] and weights['head']==plan['native_identity']['head_weight'],
         'Actual frozen native head/norm identity differs')
    native_file=ckpt/'native_norm_head.pt';shutil.copyfile(plan['native_model_file'],native_file)
    need(sha(native_file)==plan['native_model_sha256'],'Frozen per-run head/norm copy differs')
    packet=torch.load(plan['features_file'],map_location='cpu',weights_only=True)
    need(base.v7.tensor_info(packet['states'])==plan['feature_tensor'] and packet['feature_ids']==sorted(packet['feature_ids'])
         and len(set(packet['feature_ids']))==len(packet['feature_ids']),'Subset feature packet differs')
    states=packet['states'].to(loaded.device);index={f:i for i,f in enumerate(packet['feature_ids'])};del packet
    cache=dict(scenes=read(plan['scenes_file']));initial=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(initial['seed']==24,'Initial shared seed differs');torch.manual_seed(24);torch.cuda.manual_seed_all(24)
    core=ParallelLocalLearnedSelection(mode=mode).to(loaded.device);core.load_state_dict(initial['branch'])
    initialized=base.v7.state_info(core);need(initialized==plan['initial_state'] and sum(p.numel() for p in core.parameters())==1041697,
         'Matched initialized tensors differ')
    initial_file=ckpt/'initial.pt';shutil.copyfile(plan['initial_file'],initial_file)
    need(sha(initial_file)==plan['initial_sha256'],'Per-run initial checkpoint differs')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    order=read(plan['order_file']);need(order==order_pairs(read(plan['pairs_file'])),'Fixed crossing-cycle pair order differs')
    save(out/'presentations.json',order)
    config=dict(protocol=PROTOCOL,run_id=out.name,arm=args.arm,condition=mode,selection_mode=mode,
        consistency_coefficient=coefficient,seed=24,profile=args.profile,policy=POLICY,
        slurm_job_id=os.environ['SLURM_JOB_ID'],plan_file=str(args.plan),plan_sha256=sha(args.plan),source_sha256=frozen,
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        native_weight_identity=weights,native_model_file=str(native_file),native_model_sha256=sha(native_file),hardware=hardware,
        initialized=initialized,initialized_sha256=object_sha(initialized),initial_checkpoint=str(initial_file),
        initial_checkpoint_sha256=sha(initial_file),presentations_file=str(out/'presentations.json'),
        presentations_sha256=sha(out/'presentations.json'),order_sha256=object_sha(order),main_release=release,
        checkpoint_directory=str(ckpt),data_directory=str(dataout),model_and_features_load_seconds=load_seconds,
        native_software_plan=plan['native_software_plan'])
    save(out/'config.json',config);versions={n:p._version for n,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(core.parameters(),lr=.001,weight_decay=0.)
    setup=time.perf_counter()-started;steps=32 if args.profile else 600;log=[];diagnostics=[];first_gradients=[];io_seconds=0.
    exercised={n:False for n,_ in core.named_parameters()};zero=None
    for step in range(1,steps+1):
        tick=time.perf_counter();batch=order[(step-1)*8:step*8];sids=[sid for row in batch for sid in row['sids']]
        need(len(batch)==8 and len(sids)==16 and all([cache['scenes'][sid]['n_frames'] for sid in row['sids']]==[8,16] for row in batch),
             'Optimization batch broke intact N8/N16 pairs')
        h,g,layout,valid=base.batch_states(torch,cache,states,index,sids)
        optimizer.zero_grad(set_to_none=True);rate=learning_rate(step)
        for group in optimizer.param_groups:group['lr']=rate
        _,ce,residual,parts,cap=base.losses(torch,core,h,g,layout,valid,norm,head)
        total=ce+coefficient*residual;need(bool(torch.isfinite(total)),'Nonfinite fixed pilot loss')
        if step==1:zero=base.zero_gradients(torch,core,residual)
        if step in POLICY['gradient_steps']:
            raw,geometry=gradient_geometry(torch,core,ce,residual,coefficient,cap)
            path=dataout/f'gradients_{step:04d}.pt';torch.save(raw,path)
            diagnostics.append(dict(step=step,sids=sids,consistency_coefficient=coefficient,raw_file=str(path),raw_sha256=sha(path),**geometry))
        queries=query_diagnostics(torch,parts,cap,valid,layout)
        total.backward();need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in core.parameters()),
            'Missing/nonfinite pilot core gradient')
        for name,p in core.named_parameters():exercised[name]|=bool((p.grad!=0).any())
        if step<=2:first_gradients.append(dict(step=step,pre_clip={n:base.v7.tensor_info(p.grad) for n,p in core.named_parameters()}))
        gradient_norm=torch.nn.utils.clip_grad_norm_(core.parameters(),1.);need(bool(torch.isfinite(gradient_norm)),'Nonfinite clipping norm')
        optimizer.step();need(all(bool(torch.isfinite(p).all()) for p in core.parameters())
            and not states.requires_grad and not h.requires_grad and not g.requires_grad
            and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Core or frozen-input/weight gradient invariant failed')
        torch.cuda.synchronize()
        log.append(dict(step=step,arm=args.arm,condition=mode,lr=rate,loss=float(total),ce_loss=float(ce),consistency_loss=float(residual),
            weighted_consistency_loss=coefficient*float(residual),consistency_coefficient=coefficient,
            gradient_norm=float(gradient_norm),clipped=float(gradient_norm)>1.,seconds=time.perf_counter()-tick,
            target_ids=layout['targets'],sids=sids,pair_ids=[r['pair_id'] for r in batch],cycles=[r['cycle'] for r in batch],
            query_diagnostics=queries,**parts))
        if step%100==0:
            tick=time.perf_counter();save(out/'training.json',log);save(out/'gradient_diagnostics.json',diagnostics)
            io_seconds+=time.perf_counter()-tick
            print(json.dumps(dict(step=step,arm=args.arm,ce=float(ce),consistency=float(residual))),flush=True)
    tick=time.perf_counter();save(out/'training.json',log);save(out/'gradient_diagnostics.json',diagnostics)
    save(out/'first_gradients.json',first_gradients);io_seconds+=time.perf_counter()-tick
    optimizer.zero_grad(set_to_none=True);del optimizer,total,ce,residual,cap
    core.eval().requires_grad_(False);final=base.v7.state_info(core);checkpoint=ckpt/('profile.pt' if args.profile else 'final.pt')
    torch.save(dict(branch=core.state_dict(),step=steps,config=config),checkpoint);checkpoint_sha=sha(checkpoint)
    core.load_state_dict(initial['branch']);need(base.v7.state_info(core)==initialized,'Exact initial reset failed')
    restored=torch.load(checkpoint,map_location=loaded.device,weights_only=True);core.load_state_dict(restored['branch'])
    need(restored['step']==steps and restored['config']==config and base.v7.state_info(core)==final,'Restricted final checkpoint reload differs')
    before=base.v7.state_info(core);core_versions={n:p._version for n,p in core.named_parameters()}
    result=dict(config,passed=True,completed=True,computational_integrity_passed=True,steps=steps,
        checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,selected_parameter_sha256=object_sha(final),
        checkpoint_roundtrip_passed=True,zero_initialization_core_gradients=zero,all_parameter_gradients_exercised=exercised,
        training_file=str(out/'training.json'),training_sha256=sha(out/'training.json'),
        gradient_diagnostics_file=str(out/'gradient_diagnostics.json'),gradient_diagnostics_sha256=sha(out/'gradient_diagnostics.json'),
        first_gradients_file=str(out/'first_gradients.json'),first_gradients_sha256=sha(out/'first_gradients.json'),
        training_seconds=sum(r['seconds'] for r in log),first_four_step_seconds=sum(r['seconds'] for r in log[:4]),
        step_seconds_max_steady=max(r['seconds'] for r in log[4:]),setup_before_training_seconds=setup,
        setup_includes_model_and_feature_load=True,training_log_io_seconds=io_seconds)
    if args.profile:
        timing=base.profile_native(torch,model,loaded.processor,core,plan,cache,states,index,dataout,mode)
        save(out/'native_timing.json',timing)
        result.update(native_timing=timing,native_timing_file=str(out/'native_timing.json'),native_timing_sha256=sha(out/'native_timing.json'),
            no_dev_or_test_evaluation=True)
        if not timing['passed']:
            result.update(passed=False,computational_integrity_passed=False);save(out/'summary.json',result)
            need(False,'Native same-state head replay failed; all observations retained')
    else:
        rows=read(plan['rows_file'])
        training=base.evaluate(torch,model,loaded.processor,core,[(r['cell'],r['sample']) for r in rows['train']],
            out,'train_final',dataout,mode,plan['native_identity_sha256'])
        dev=base.evaluate(torch,model,loaded.processor,core,[(r['cell'],r['sample']) for r in rows['dev']],
            out,'dev_final',dataout,mode,plan['native_identity_sha256'])
        result.update(train_file=str(out/'train_final.json'),train_sha256=sha(out/'train_final.json'),
            dev_file=str(out/'dev_final.json'),dev_sha256=sha(out/'dev_final.json'),trainability=criteria(training['rows'],True),
            id_generalization=criteria(dev['rows'],False),native_train_count=108,native_dev_count=54,native_test_count=0,
            fixed_final_checkpoint_only=True,dev_evaluated_once=True)
    after=base.v7.state_info(core);after_weights=dict(norm=base.v7.tensor_info(norm.weight),head=base.v7.tensor_info(head.weight))
    need(before==after==final and after_weights==weights and versions=={n:p._version for n,p in model.named_parameters()}
         and core_versions=={n:p._version for n,p in core.named_parameters()} and not any(p.grad is not None or p.requires_grad for p in model.parameters())
         and sha(checkpoint)==checkpoint_sha and sha(native_file)==config['native_model_sha256'],'Evaluation changed frozen endpoint/model artifacts')
    endpoint=dict(schema_version=1,step=steps,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
        before_evaluation=before,after_evaluation=after,parameter_sha256=object_sha(final),
        native_weight_identity_before=weights,native_weight_identity_after=after_weights,
        checkpoint_matches_deployed_endpoint=True,core_versions_unchanged=True,native_versions_unchanged=True)
    save(out/'final_endpoint.json',endpoint);validation=time.perf_counter();verify_plan(args.plan)
    if release:need(sha(release['file'])==release['sha256'],'Independent release changed during run')
    result.update(final_endpoint_file=str(out/'final_endpoint.json'),final_endpoint_sha256=sha(out/'final_endpoint.json'),
        final_endpoint_unchanged=True,frozen_backbone_gradient_state_preserved=True,final_validation_seconds=time.perf_counter()-validation,
        elapsed_seconds=time.perf_counter()-started)
    return result


def tensor_self_test(torch):
    """Tiny analytical graph exercises the new diagnostic, never a model/head."""
    import io
    from transformers.utils.quantization_config import QuantizationMethod
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.selection_weight=torch.nn.Parameter(torch.tensor([.2,-.4]))
            self.local=torch.nn.Linear(2,2,bias=False);self.up=torch.nn.Linear(2,1,bias=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);core=Tiny();payload=core.local(torch.tensor([[1.,2.],[-1.,3.]]))
        scores=payload@core.selection_weight;output=core.up(payload).squeeze(-1)+scores
        ce=(output-1).square().mean();residual=(output[0]-output[1]).square()
        raw,geometry=gradient_geometry(torch,core,ce,residual,1.,dict(scores=scores,payload=payload))
        (ce+residual).backward()
        for name,p in core.named_parameters():
            need(torch.allclose(p.grad,raw['ce'][name]+raw['consistency'][name],rtol=1e-5,atol=1e-6),
                 'Separate diagnostic gradients do not reconstruct the same live combined objective')
        combined=sum(float(p.grad.double().square().sum()) for p in core.parameters())**.5
        need(math.isclose(combined,geometry['groups']['all']['combined_norm'],rel_tol=1e-5),'Gradient norm/dot relation differs')
        parts=dict(position_losses=dict(ce=[1.,2.,3.,4.,5.]));layout=dict(offsets=[0,2,5])
        gates=torch.tensor([[0.,.1,.5,.2,1.],[0.,.2,0.,.4,1.]])
        q=query_diagnostics(torch,parts,dict(gates=gates),torch.ones_like(gates,dtype=torch.bool),layout)
        need(q['first_token_nll']==[1.,3.] and q['continuation_token_nll']==[4.] and q['eos_token_nll']==[2.,5.]
             and q['first_all_gates_closed']==[True,False],'First/continuation/EOS gate and loss segmentation differs')
        metadata=json.loads(json.dumps(dict(quantization=QuantizationMethod.BITS_AND_BYTES)))
        stream=io.BytesIO();torch.save(dict(config=metadata,branch=core.state_dict()),stream);stream.seek(0)
        loaded=torch.load(stream,weights_only=True)
        need(loaded['config']==metadata and all(torch.equal(loaded['branch'][n],v) for n,v in core.state_dict().items()),
             'Plain metadata restricted checkpoint roundtrip differs')
    return dict(passed=True,live_gradient_decomposition=True,first_vs_suffix_segmentation=True,
        json_hardware_weights_only_roundtrip=True,no_model_or_head_loaded=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    for name in ('check','profile','run','release','report'):action.add_argument('--'+name,action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--main-release',type=Path);parser.add_argument('--profiles',type=Path,nargs='+')
    parser.add_argument('--runs',type=Path,nargs='+');args=parser.parse_args()
    gpu=args.profile or args.run;native.require_slurm(gpu=gpu)
    if not gpu:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only planning/report required')
    if gpu:need(args.plan is not None and args.arm is not None,'GPU profile/run requires plan and fixed arm')
    if args.release:need(args.plan is not None and args.profiles and len(args.profiles)==4,'Release needs all four fixed profiles')
    if args.report:need(args.plan is not None and args.runs and len(args.runs)==4 and args.main_release is not None,'Report needs all four mains and release')
    phase=next(k for k in ('check','profile','run','release','report') if getattr(args,k));job=os.environ['SLURM_JOB_ID']
    runid=f'{phase}_{args.arm}_s24_{job}' if gpu else f'{phase}_{job}';out=OUT/runid;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();frozen=snapshot(out)
    def plain(value):
        if isinstance(value,Path):return str(value)
        if isinstance(value,list):return [plain(v) for v in value]
        return value
    save(out/'request.json',dict(arguments={k:plain(v) for k,v in vars(args).items()},source_sha256=frozen))
    try:
        if args.check:result=check(out,frozen)
        elif gpu:result=run(args,out,frozen,started)
        else:
            from scripts import report_native_identity_join_optimization as reporter
            result=reporter.release(args,out,frozen) if args.release else reporter.report(args,out,frozen)
        need(sources()==frozen,'Source changed during pilot execution')
        result.update(elapsed_seconds=time.perf_counter()-started)
        save(out/'summary.json',result);print(json.dumps(dict(completed=True,passed=result['passed'],directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

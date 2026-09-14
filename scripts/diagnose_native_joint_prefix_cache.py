"""DRAFT bounded ordinary joint image-prefix cache preparation/software profile.

No memory compressor, training, labels or efficacy gate is included.
"""
from __future__ import annotations
import argparse
from collections import Counter
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_joint_prefix_cache as runtime
from scripts import diagnose_native_identity_join_joint_lora_v2 as p
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,object_sha,model_metadata
OUT=REPO/'outputs/native_aggregation_vlm/native_joint_prefix_cache'
DATA=Path('/mnt/data/gabriele/gnn_transformer/native_joint_prefix_cache')
PROTOCOL='native_joint_prefix_cache_software'
PARENT=p.OUT/'check_443888/plan.json'
PARENT_SHA='f2314c7c689fb489c1efea43c7da625cf53a2822ae1d8467e5208c784010f430'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_JOINT_PREFIX_CACHE_PROPOSAL.md'
PROPOSAL_SHA='bdd89eb327e943e8824630eaa0685f59c3272a63e7c3a3d16b4378008639f8af'
JOB='native_joint_prefix_cache_profile'
OWN=('scripts/native_joint_prefix_cache.py','scripts/diagnose_native_joint_prefix_cache.py',PROPOSAL,
    'slurm/native_joint_prefix_cache_check.sbatch','slurm/native_joint_prefix_cache_profile.sbatch')
POLICY=dict(protocol=PROTOCOL,collections=2,question_count=4,trajectories=24,maximum_new_tokens=4,full_trajectories=8,
    forward_branches=8,reverse_branches=8,prefix_calls=2,model_calls_cap=96,head_calls_cap=96,language_calls_cap=98,
    norm_calls_cap=98,vision_calls=10,cpu_seconds=300,gpu_seconds=240,max_gpu_attempts=1,
    full_split_tv_max=.02,full_split_generated_ids_exact=True,branch_order_logits_exact=True,
    no_lora=True,no_fitting=True,no_compression=True,no_efficacy_claim=True,no_replay_heads=True)


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Held prefix-cache proposal changed');return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result={**p.inherited_sources(),**p.sources()};need(len(result)==166,'Frozen native V2 closure differs');return result


def snapshot(out):
    own=sources();(out/'source').mkdir()
    for name,digest in own.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'New source snapshot differs')
    save(out/'source_hashes.json',own);save(out/'inherited_sources.json',dict(source_sha256=inherited_sources()));return own


def check(out,frozen):
    import torch
    from transformers import AutoProcessor,__version__ as transformers_version
    from transformers.cache_utils import DynamicCache
    torch.set_num_threads(4);bindings={};p.bind(PARENT,bindings,PARENT_SHA);parent=p.verify_plan(PARENT,streaming=True)
    rows=read(parent['rows_file']);by_sid={r['sid']:r for r in rows};sids=parent['profile_cases']['training_sids'];questions=list(dict.fromkeys(r['question'] for r in rows))[:4]
    need(len(questions)==4 and len(sids)==2 and [by_sid[s]['n_frames'] for s in sids]==[8,16],'Exact fixed two collections/four canonical questions required')
    p.bind(parent['rows_file'],bindings,parent['runtime_bindings'][parent['rows_file']]);p.bind(PARENT.parent/'summary.json',bindings)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True);owner,fn,api=p.backend.native_api(processor)
    identity=parent['native_identity']
    need(api==identity['native_api'] and p.fingerprint(processor,str(transformers_version))==identity['processor']
         and p.backend.runtime_identity()==identity['runtime'] and model_metadata()==identity['model'],'Pinned processor/native identity differs')
    rope=lambda **kw:fn(owner,**kw)
    cache_source=Path(inspect.getfile(DynamicCache)).resolve();p.bind(cache_source,bindings)
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);collections=[];cases=[];files={}
    for index,sid in enumerate(sids):
        row=by_sid[sid]
        for image in row['image_files']:p.bind(image['path'],bindings,image['sha256'])
        prefix=runtime.prepare_image_prefix(processor,row['image_files'],rope);file=data/f'prefix_{index}.pt';torch.save(prefix,file);files[str(file)]=sha(file)
        collections.append(dict(index=index,sid=sid,n_frames=row['n_frames'],file=str(file),sha256=sha(file),metadata=prefix['metadata']))
        for j,question in enumerate(questions):
            view=dict(sid=sid,n_frames=row['n_frames'],question=question,image_files=row['image_files'])
            bundle=p.prepare_joint(processor,view);suffix=runtime.question_suffix(torch,prefix,bundle,rope)
            file=data/f'case_{index}_{j}.pt';torch.save(dict(bundle=bundle,suffix=suffix),file);files[str(file)]=sha(file)
            cases.append(dict(collection=index,question_index=j,sid=sid,n_frames=row['n_frames'],question=question,question_sha256=object_sha(question),
                file=str(file),sha256=sha(file),metadata=bundle['metadata'],suffix_input_identity=p.tensor_info(suffix['input_ids']),suffix_position_identity=p.tensor_info(suffix['position_ids'])))
    tests=runtime.self_test(torch);save(out/'selftests.json',tests)
    for name,value in (('collections.json',collections),('cases.json',cases),('questions.json',questions)):
        save(out/name,value);files[str(out/name)]=sha(out/name)
    plan=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),parent_plan=dict(file=str(PARENT),sha256=PARENT_SHA),
        native_identity=identity,native_identity_sha256=parent['native_identity_sha256'],precision=parent['precision'],packages=parent['packages'],
        cache_source=dict(file=str(cache_source),sha256=sha(cache_source)),collections=collections,cases=cases,questions=questions,input_bindings=bindings,files=files,
        tests=tests,question_gold_accessed=False,model_forward_calls=0,head_forward_calls=0)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,collections=2,questions=4,model_forward_calls=0,head_forward_calls=0)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);proof=read(path.parent/'summary.json')
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources() and plan['inherited_source_sha256']==inherited_sources()
         and proof['passed'] is proof['completed'] is True and proof['plan_sha256']==sha(path)==path.with_suffix('.sha256').read_text().strip(),'Passed frozen prefix-cache CPU plan required')
    for name,digest in sources().items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Source copy differs')
    for mapping in (plan['input_bindings'],plan['files']):
        for file,digest in mapping.items():need(sha(file)==digest,'Bound native input changed')
    need(plan['parent_plan']==dict(file=str(PARENT),sha256=PARENT_SHA) and sha(PARENT)==PARENT_SHA,'Parent plan changed')
    p.verify_plan(PARENT,streaming=True);return plan


def accounting(out):
    need(os.environ.get('SLURM_JOB_NAME')==JOB,'Registered cache profile job name required')
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;(out/'launch_sacct.psv').write_text(raw);rows=[]
    for line in raw.splitlines():
        f=line.split('|')
        if len(f)!=9 or f[1]!=JOB:continue
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',f[6]);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',f[6])
        rows.append(dict(job_id=f[0],name=f[1],partition=f[2],state=f[3],seconds=int(f[5]),gpus=int(generic[0]) if generic else sum(map(int,typed))))
    need(len(rows)==1 and rows[0]['job_id']==os.environ['SLURM_JOB_ID'] and rows[0]['partition']=='gpu' and rows[0]['gpus']==1 and rows[0]['seconds']<=240,'Exactly one bounded GPU attempt permitted')
    save(out/'launch_accounting.json',dict(command=command,matching_rows=rows,file=str(out/'launch_sacct.psv'),sha256=sha(out/'launch_sacct.psv'),cap_gpu_seconds=240))


def compare(torch,a,b):
    n=min(len(a['generated_ids']),len(b['generated_ids']));metrics=[]
    for i in range(n):
        x,y=a['raw_logits'][i],b['raw_logits'][i];finite=bool(x.isfinite().all() and y.isfinite().all())
        tv=float((x.double().softmax(-1)-y.double().softmax(-1)).abs().sum()/2) if finite else None
        metrics.append(dict(step=i,tv=tv,finite=finite,max_abs_error=float((x-y).abs().max()) if finite else None,
            same_input_history=a['generated_ids'][:i]==b['generated_ids'][:i],passed=finite and tv<=.02))
    exact_ids=a['generated_ids']==b['generated_ids']
    return dict(passed=exact_ids and all(r['passed'] for r in metrics),generated_ids_exact=exact_ids,
        full_generated_ids=a['generated_ids'],split_generated_ids=b['generated_ids'],raw_logits_exact=torch.equal(a['raw_logits'],b['raw_logits']),metrics=metrics)


def profile(args,out,frozen,started):
    import torch
    from gnnformer.runtime import load_runtime
    torch.set_num_threads(4);accounting(out);plan=verify_plan(args.plan)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','Exactly one B200 required')
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda');model=loaded.model.eval().requires_grad_(False)
    hardware=p.frozen_joint.live_identity(torch,loaded,plan);refs=list(model.named_parameters());base=p.base_metadata(refs);save(out/'base_before.json',base)
    need(not any('.lora_' in name for name,_ in refs),'Original native weights only; no adapter or compressor')
    config=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),plan_file=str(args.plan.resolve()),plan_sha256=sha(args.plan),
        hardware=hardware,native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],data_directory=str(data),software_only=True)
    save(out/'config.json',config);counts=Counter();records=[];comparisons=[];order_checks=[];results={};setup=time.perf_counter()-started
    for collection in plan['collections']:
        ci=collection['index'];cases=[c for c in plan['cases'] if c['collection']==ci]
        for case in cases:
            key=f'full_{ci}_{case["question_index"]}';tick=time.perf_counter();packet=torch.load(case['file'],map_location='cpu',weights_only=True);evidence={}
            try:result=p.generate_joint(model,loaded.processor,packet['bundle'],native_identity_sha256=plan['native_identity_sha256'],capture_head=True,evidence=evidence)
            except BaseException:
                torch.save(p.cpu_tree(torch,dict(case=case,evidence=evidence)),data/(key+'_partial.pt'));raise
            file=data/(key+'.pt');torch.save(dict(case=case,result=result),file);record=dict(key=key,file=str(file),sha256=sha(file),counters=result['counters'],generated_ids=result['generated_ids'])
            records.append(record);results[(ci,case['question_index'],'full')]=result;counts.update(result['counters']);save(out/(key+'.json'),record);p.check_base(refs,base,model)
            torch.cuda.synchronize();save(out/(key+'_timing.json'),dict(seconds=time.perf_counter()-tick))
        tick=time.perf_counter();prefix_packet=torch.load(collection['file'],map_location='cpu',weights_only=True);evidence={}
        try:snapshot_value,proof=runtime.prefill_images(model,prefix_packet,native_identity_sha256=plan['native_identity_sha256'],evidence=evidence)
        except BaseException:
            torch.save(p.cpu_tree(torch,dict(collection=collection,evidence=evidence)),data/f'prefix_{ci}_partial.pt');raise
        file=data/f'prefix_{ci}.pt';torch.save(dict(collection=collection,snapshot=runtime.export_snapshot(torch,snapshot_value),proof=proof),file)
        prefix_record=dict(key=f'prefix_{ci}',file=str(file),sha256=sha(file),counters=proof['counters'],prefix_length=snapshot_value.length,
            persistent_kv_bytes=sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in snapshot_value.layers))
        records.append(prefix_record);counts.update(proof['counters']);save(out/f'prefix_{ci}.json',prefix_record);p.check_base(refs,base,model)
        torch.cuda.synchronize();save(out/f'prefix_{ci}_timing.json',dict(seconds=time.perf_counter()-tick))
        for phase,ordered in (('forward',cases),('reverse',list(reversed(cases)))):
            for case in ordered:
                qi=case['question_index'];key=f'{phase}_{ci}_{qi}';tick=time.perf_counter();packet=torch.load(case['file'],map_location='cpu',weights_only=True);evidence={}
                try:result=runtime.generate_branch(model,loaded.processor,snapshot_value,packet['suffix'],evidence=evidence)
                except BaseException:
                    torch.save(p.cpu_tree(torch,dict(case=case,evidence=evidence)),data/(key+'_partial.pt'));raise
                file=data/(key+'.pt');torch.save(dict(case=case,result=result),file);record=dict(key=key,file=str(file),sha256=sha(file),counters=result['counters'],generated_ids=result['generated_ids'])
                records.append(record);results[(ci,qi,phase)]=result;counts.update(result['counters']);save(out/(key+'.json'),record);p.check_base(refs,base,model)
                torch.cuda.synchronize();save(out/(key+'_timing.json'),dict(seconds=time.perf_counter()-tick))
        runtime.verify_snapshot(snapshot_value)
        for case in cases:
            qi=case['question_index'];a,b,c=(results[(ci,qi,phase)] for phase in ('full','forward','reverse'))
            comparisons.append(dict(collection=ci,question_index=qi,**compare(torch,a,b)))
            order_checks.append(dict(collection=ci,question_index=qi,passed=b['generated_ids']==c['generated_ids'] and torch.equal(b['raw_logits'],c['raw_logits']),
                generated_ids_exact=b['generated_ids']==c['generated_ids'],raw_logits_exact=torch.equal(b['raw_logits'],c['raw_logits'])))
        del snapshot_value
    save(out/'records.json',records);save(out/'comparisons.json',comparisons);save(out/'order_checks.json',order_checks)
    calls=sum(len(r['generated_ids']) for r in records if 'generated_ids' in r)
    need(len(records)==26 and counts['model']==counts['head']==calls<=96 and counts['language']==counts['norm']==calls+2<=98
         and counts['visual']==10 and counts['prefix_decoder']==2 and all(counts.get(k,0)==0 for k in ('broadcast','fusion','conditioning','probe_head')),'Exact24 trajectories/two headless prefills required')
    p.check_base(refs,base,model);files={str(f):sha(f) for f in sorted(data.rglob('*')) if f.is_file()};save(out/'artifacts.json',files)
    passed=all(c['passed'] for c in comparisons+order_checks);save(out/'analysis.json',dict(passed=passed,software_only=True,comparisons=comparisons,order_checks=order_checks,
        counters=dict(counts),setup_seconds=setup,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
        no_compression=True,no_efficacy_claim=True,all24_trajectories_retained=True))
    need(passed,'Full/split TV or exact IDs/order comparison failed; all scheduled trajectories retained')
    return dict(**config,passed=True,completed=True,phase='profile',analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),records_file=str(out/'records.json'),records_sha256=sha(out/'records.json'),
        counters=dict(counts),no_downstream_release=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--check',action='store_true');mode.add_argument('--profile',action='store_true');parser.add_argument('--plan',type=Path);args=parser.parse_args()
    p.native.require_slurm(gpu=args.profile);need(args.plan is None if args.check else args.plan is not None,'Check takes no plan; profile requires one')
    if args.check:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only preparation required')
    phase='check' if args.check else 'profile';out=OUT/f'{phase}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen=snapshot(out)
    save(out/'request.json',dict(phase=phase,plan=None if args.plan is None else str(args.plan.resolve()),source_sha256=frozen))
    try:
        value=check(out,frozen) if args.check else profile(args,out,frozen,started)
        need(sources()==frozen and inherited_sources()==value['inherited_source_sha256'],'Frozen cache profile sources changed');elapsed=time.perf_counter()-started
        need(elapsed<=POLICY['cpu_seconds' if args.check else 'gpu_seconds'],'Fixed cache phase cap exceeded');save(out/'summary.json',dict(value,elapsed_seconds=elapsed));print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

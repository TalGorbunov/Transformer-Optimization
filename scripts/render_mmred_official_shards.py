"""Render four fixed original-MMReD cache shards and merge ordered input rows."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import profile_mmred_official_renderer as p
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
PROTOCOL='mmred_official_render_shards'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_RENDER_SHARDS.md'
PROPOSAL_SHA='d790c5a70dd8eec51764fdb58d733f28a6c938de077f267a2a9b6c2d506a994c'
OWN=('scripts/render_mmred_official_shards.py',PROPOSAL,'slurm/mmred_official_render_shards.sbatch','slurm/mmred_official_render_merge.sbatch')
PARENT=p.OUT/'profile_443954/summary.json'
PARENT_SHA='0589f7720e6e74707d9a4df0e0d30eff5606b805ad572ae9bfad1ccea7369abc'
PARENT_PLAN_SHA='728a0b69a769c2012dd139f55fbb3e8ec70d47d3902fc4225fecc4985b76449d'
NAMESPACE='8318a772d95253321a86406565902da0e808d02edaa66a221f4ba76110adc0d4'
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_render'
DATA=p.recovery.DATA/'rendered'
POLICY=dict(protocol=PROTOCOL,shards=4,keys_per_shard=8677,unique_keys=34708,worlds=5000,frame_occurrences=40800,
    cpu_seconds_per_shard=4500,total_shard_allocation_seconds=18000,merge_cpu_seconds=600,cpu_cores=4,memory_gib=16,
    lengths=[1,2,4,8,16,32],
    maximum_array_attempts=1,render_workers_per_shard=1,cache_namespace=NAMESPACE,rerender_all_keys=True,
    no_N64_N128_render=True,no_model_or_head_calls=True,no_training_or_inference_release=True)


def read(path):return json.loads(Path(path).read_text())


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Held shard proposal changed');return {name:sha(REPO/name) for name in OWN}


def inherited_sources():return {**p.sources(),**p.inherited_sources()}


def check_time(started,phase):
    cap=POLICY['cpu_seconds_per_shard' if phase=='shard' else 'merge_cpu_seconds']
    need(time.perf_counter()-started<cap,'Fixed CPU rendering/merge cap exceeded')


def dependencies(bindings):
    p.bind(PARENT,bindings,PARENT_SHA);profile=p.verify_profile(PARENT)
    need(sha(read(PARENT)['plan_file'])==PARENT_PLAN_SHA and profile['cache_namespace']==NAMESPACE,'Exact passed renderer profile required')
    p.bind(read(PARENT)['plan_file'],bindings,PARENT_PLAN_SHA)
    for file,digest in {**profile['input_bindings'],**profile['artifacts']}.items():p.bind(file,bindings,digest)
    shards=read(profile['shards_file']);frames=read(profile['frame_inventory_file'])
    need(len(frames)==34708 and [s['index'] for s in shards]==[0,1,2,3]
         and all(len(s['keys'])==8677 for s in shards)
         and [key for s in shards for key in s['keys']]==[key for i in range(4) for j,key in enumerate(sorted(frames)) if j%4==i],
         'Exact disjoint four-shard inventory differs')
    projection=read(profile['projection_file'])
    need(all(s['projected_seconds']<=4500 and s['unique_keys']==8677 for s in projection['shards']),'Measured profile does not support fixed shard caps')
    return profile,frames,shards


def common_plan(out,frozen,phase,bindings,artifacts,**extra):
    return dict(protocol=PROTOCOL,phase=phase,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),
        profile=dict(file=str(PARENT),sha256=PARENT_SHA,plan_sha256=PARENT_PLAN_SHA),cache_namespace=NAMESPACE,
        input_bindings=bindings,artifacts=artifacts,no_training_or_inference_release=True,**extra)


def shard(index,out,data,frozen,started):
    bindings={};profile,frames,shards=dependencies(bindings)
    recovery_summary=read(profile['recovery']['file']);recovery_plan=read(recovery_summary['plan_file'])
    visualization,Image,environment=p.renderer_environment(recovery_plan,bindings)
    need(environment==read(profile['environment_file']) and object_sha(environment)==NAMESPACE,'Renderer environment/cache namespace changed')
    save(out/'render_environment.json',environment);keys=shards[index]['keys'];images={};setup=time.perf_counter()-started
    journal=out/'images.jsonl'
    with journal.open('x') as stream:
        for ordinal,key in enumerate(keys):
            check_time(started,'shard');frame=frames[key];tick=time.perf_counter();target=data/key;target.mkdir()
            visualization.frame2png(frame['assignment'],frame['step'],target)
            entry=p.inspect_png(target/f'frame_{frame["step"]:04d}.png',Image)
            entry['path']=entry.pop('file');entry.update(cache_key=key,step=frame['step'],cache_namespace=NAMESPACE)
            images[key]=entry
            stream.write(json.dumps(dict(ordinal=ordinal,key=key,image=entry,seconds=time.perf_counter()-tick),sort_keys=True,allow_nan=False)+'\n')
            if ordinal%128==0:stream.flush()
    need(list(images)==keys and len(images)==8677,'Complete shard image key order differs')
    save(out/'image_index.json',images)
    analysis=dict(passed=True,shard_index=index,images=8677,setup_seconds=setup,render_and_journal_seconds=time.perf_counter()-started-setup,
        png_bytes=sum(r['bytes'] for r in images.values()),image_modes=dict(Counter(r['mode'] for r in images.values())),
        raw_png_preserved=True,no_extra_frame_renders=True)
    save(out/'analysis.json',analysis)
    artifacts={str(path):sha(path) for path in out.glob('*.json')};artifacts[str(journal)]=sha(journal)
    plan=common_plan(out,frozen,'shard',bindings,artifacts,shard_index=index,array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],
        image_index_file=str(out/'image_index.json'),environment_file=str(out/'render_environment.json'),
        image_file_hashes_are_in_bound_index=True,analysis=analysis)
    save(out/'plan.json',plan)
    return dict(shard_index=index,array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],images=8677,
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'))


def verify_plan_summary(path,phase,streaming=False):
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path;summary=read(path)
    need(summary['passed'] is summary['completed'] is True and summary['phase']==phase and summary['protocol']==PROTOCOL
         and summary['policy']==POLICY and summary['source_sha256']==sources() and summary['inherited_source_sha256']==inherited_sources()
         and not (path.parent/'failure.json').exists(),'Passed exact rendering summary required')
    cap=POLICY['cpu_seconds_per_shard' if phase=='shard' else 'merge_cpu_seconds']
    need(0<summary['elapsed_seconds']<=cap and sha(summary['plan_file'])==summary['plan_sha256'],'Rendering plan/time differs')
    plan=read(summary['plan_file'])
    need(plan['protocol']==PROTOCOL and plan['phase']==phase and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and plan['cache_namespace']==NAMESPACE
         and plan['no_training_or_inference_release'] is True,'Rendering source/policy/namespace differs')
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():need(sha(file)==digest,'Bound rendering artifact changed')
    for name,digest in {**sources(),**inherited_sources()}.items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Archived render source changed')
    need(plan['image_index_file'] in plan['artifacts'],'Image index must be directly bound')
    images=read(plan['image_index_file'])
    if not streaming:
        for entry in images.values():need(sha(entry['path'])==entry['sha256'],'Rendered PNG changed')
    return plan,images,dict(file=str(path),sha256=sha(path),plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'])


def merge(paths,out,frozen,started):
    bindings={};profile,frames,shards=dependencies(bindings);need(len(paths)==4,'Exactly four completed shard paths required')
    indices={};descriptors=[];array_ids=set();summary_by_index={}
    for path in paths:
        check_time(started,'merge');plan,images,descriptor=verify_plan_summary(path,'shard')
        index=plan['shard_index'];need(index in range(4) and index not in indices,'Repeated/invalid shard index')
        need(list(images)==shards[index]['keys'] and len(images)==8677,'Exact shard key membership/order differs')
        need(read(plan['environment_file'])==read(profile['environment_file']),'Shard environment differs')
        for key,entry in images.items():
            need(entry['cache_key']==key and entry['cache_namespace']==NAMESPACE and entry['step']==frames[key]['step']
                 and entry['width']==entry['height']==512 and entry['bytes']>0,'PNG/key/Step/geometry ownership differs')
        indices[index]=images;array_ids.add(plan['array_job_id']);descriptors.append(descriptor);summary_by_index[index]=plan['analysis']
        p.bind(descriptor['file'],bindings,descriptor['sha256']);p.bind(descriptor['plan_file'],bindings,descriptor['plan_sha256'])
        for file,digest in plan['artifacts'].items():p.bind(file,bindings,digest)
    need(set(indices)==set(range(4)) and len(array_ids)==1,'Four shards must belong to the same single array')
    images={key:entry for index in range(4) for key,entry in indices[index].items()}
    need(len(images)==len(frames)==34708 and set(images)==set(frames),'Missing/extra/colliding cache keys')
    worlds=read(profile['worlds_file']);rows=[]
    for world in worlds:
        need(world['n'] in POLICY['lengths'],'Held longer world appeared')
        entries=[images[key] for key in world['frame_keys']]
        need(len(entries)==world['n'] and [r['step'] for r in entries]==list(range(1,world['n']+1)),'Ordered native Step image ownership differs')
        rows.append(dict(world,n_frames=world['n'],image_files=[dict(path=r['path'],sha256=r['sha256']) for r in entries]))
    need(len(rows)==len({r['sid'] for r in rows})==5000 and sum(r['n_frames'] for r in rows)==40800
         and Counter(r['pilot_role'] for r in rows)=={'train':4000,'val':400,'test':600},'Original5000-world/40800-frame merged cohort differs')
    save(out/'rows.json',rows);save(out/'image_index.json',images)
    analysis=dict(passed=True,worlds=5000,frame_occurrences=40800,unique_images=34708,
        role_counts={'train':4000,'val':400,'test':600},array_job_id=next(iter(array_ids)),shards=summary_by_index,
        total_png_bytes=sum(r['bytes'] for r in images.values()),all_png_file_hashes_verified=True,
        no_merge_image_decode=True,no_N64_N128_render=True,no_training_or_inference_release=True)
    save(out/'analysis.json',analysis)
    artifacts={str(path):sha(path) for path in out.glob('*.json')}
    plan=common_plan(out,frozen,'merge',bindings,artifacts,rows_file=str(out/'rows.json'),image_index_file=str(out/'image_index.json'),
        shard_reports=sorted(descriptors,key=lambda d:d['file']),image_file_hashes_are_in_bound_index=True,analysis=analysis)
    save(out/'plan.json',plan)
    (out/'REPORT.md').write_text('# Original MMReD input rendering\n\nAll four fixed shards passed. All34,708 PNG hashes and5,000 original worlds/40,800 ordered image occurrences are bound in the merged plan. Original train/validation/primary-test metadata is unchanged. NoN64/N128 images or model/learning/inference work were released.\n')
    return dict(worlds=5000,unique_images=34708,frame_occurrences=40800,array_job_id=next(iter(array_ids)),
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'))


def verify_stage(path,streaming=False):
    plan,images,_=verify_plan_summary(path,'merge',streaming=streaming)
    need(len(images)==34708 and plan['analysis']['worlds']==5000,'Complete merged inputs required')
    rows=read(plan['rows_file']);need(len(rows)==5000 and sum(r['n_frames'] for r in rows)==40800,'Merged row inventory differs')
    return plan


verify_render=verify_stage


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--shard',type=int,choices=range(4));group.add_argument('--merge',nargs=4,type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Only four-core CPU Slurm allowed')
    phase='shard' if args.shard is not None else 'merge';started=time.perf_counter()
    if phase=='shard':
        need(os.environ.get('SLURM_ARRAY_JOB_ID') and int(os.environ.get('SLURM_ARRAY_TASK_COUNT','0'))==4
             and int(os.environ.get('SLURM_ARRAY_TASK_ID','-1'))==args.shard,'Exact four-task CPU array required')
        array=os.environ['SLURM_ARRAY_JOB_ID'];tag=f'shard_{array}_{args.shard}'
        need(all(path.name.split('_')[1]==array for path in OUT.glob('shard_*')),'No second rendering array attempt is released')
    else:tag=f'merge_{os.environ["SLURM_JOB_ID"]}'
    out=OUT/tag;out.mkdir(parents=True,exist_ok=False);frozen=sources();(out/'source').mkdir()
    for name,digest in {**frozen,**inherited_sources()}.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Source archive changed')
    save(out/'request.json',dict(phase=phase,shard=args.shard,merge=None if args.merge is None else [str(p.resolve()) for p in args.merge],source_sha256=frozen))
    try:
        if phase=='shard':
            data=DATA/NAMESPACE/tag;data.mkdir(parents=True,exist_ok=False);value=shard(args.shard,out,data,frozen,started)
        else:value=merge(args.merge,out,frozen,started)
        check_time(started,phase);need(sources()==frozen,'Rendering source changed')
        save(out/'summary.json',dict(passed=True,completed=True,phase=phase,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,
            inherited_source_sha256=inherited_sources(),elapsed_seconds=time.perf_counter()-started,
            no_training_or_inference_release=True,**value))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,phase=phase,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True,no_training_or_inference_release=True));raise


if __name__=='__main__':main()

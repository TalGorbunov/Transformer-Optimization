"""Prospective CPU release of four matched V18 fits after independent checks.

No model forward or fitting. Every failed/zero Slurm allocation stays in the
budget. Timing rules are fixed before profile results, independent of efficacy.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
OUT=REPO/'outputs/native_aggregation_vlm/v18'
FIELDS=('JobIDRaw','JobID','JobName','Partition','State','ExitCode','ElapsedRaw','AllocTRES','Start','End')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
OWN=('scripts/prepare_native_vision_v18_release.py','slurm/native_vision_v18_release.sbatch')


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):digest.update(part)
    return digest.hexdigest()


def save(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')

def is_campaign(row):return row['Partition']=='gpu' and re.match(r'^(?:native_)?v18_',row['JobName']) is not None


def parse_accounting(raw):
    rows=[];seen=set()
    for line in raw.splitlines():
        if not line.strip():continue
        fields=line.split('|');need(len(fields)==len(FIELDS),'Unexpected Slurm accounting columns')
        row=dict(zip(FIELDS,fields))
        if not is_campaign(row):continue
        need(re.fullmatch('[0-9]+',row['JobIDRaw']) and re.fullmatch('[0-9]+(?:_[0-9]+)?',row['JobID'])
             and row['JobIDRaw'] not in seen,'Duplicate or nonallocation campaign row')
        state=row['State'].split()[0].rstrip('+');need(state in TERMINAL,'All prior V18 GPU jobs must be terminal')
        need(row['ElapsedRaw'].isdigit(),'Invalid allocated duration');elapsed=int(row['ElapsedRaw'])
        tres=dict(x.split('=',1) for x in row['AllocTRES'].split(',') if '=' in x)
        generic=tres.get('gres/gpu');typed=[x for k,x in tres.items() if k.startswith('gres/gpu:')]
        need((generic is None or generic.isdigit()) and all(x.isdigit() for x in typed),'Invalid GPU TRES')
        count=int(generic) if generic is not None else sum(map(int,typed))
        need(count in (0,1),'Each registered V18 GPU allocation uses exactly one GPU or zero on failure')
        if count:
            need(row['Start'] not in ('','Unknown','None') and row['End'] not in ('','Unknown','None'),'Missing allocated interval')
            begin,end=datetime.fromisoformat(row['Start']),datetime.fromisoformat(row['End'])
            need(end>=begin,'Reversed allocation interval')
        row.update(state=state,gpu_count=count,allocated_gpu_seconds=count*elapsed);rows.append(row);seen.add(row['JobIDRaw'])
    return sorted(rows,key=lambda x:int(x['JobIDRaw']))


def maximum_concurrency(rows):
    events=[]
    for r in rows:
        if r['gpu_count'] and r['Start']!=r['End']:
            events.extend([(r['Start'],r['gpu_count']),(r['End'],-r['gpu_count'])])
    active=maximum=0
    for _,change in sorted(events):active+=change;maximum=max(maximum,active)
    need(active==0,'Incomplete Slurm allocation intervals')
    return maximum


def projection(setup,first_four,steady,t16,t64):
    inputs=[setup,first_four,steady,t16,t64]
    need(all(math.isfinite(x) and x>=0 for x in inputs) and min(steady,t16,t64)>0,'Invalid timing measurements')
    training=first_four+(4590-4)*steady;evaluation=64*t16+272*t64
    seconds=setup+1.25*(training+evaluation)+120
    return dict(passed=seconds<=2700,projected_seconds=seconds,setup_seconds=setup,first_four_seconds=first_four,
        maximum_steady_step_seconds=steady,training_seconds=training,evaluation_seconds=evaluation,
        T16=t16,T64=t64,dev_examples=64,test_examples=272,steps=4590,margin=1.25,reserve_seconds=120,
        formula='setup +1.25*(first4 +(4590-4)*max_steps5to32 +64*T16 +272*T64)+120')


def self_test():
    def row(j,name,state,elapsed,tres,start='2026-09-11T10:00:00',end='2026-09-11T10:01:00'):
        return '|'.join((str(j),str(j),name,'gpu',state,'0:0' if state=='COMPLETED' else '1:0',str(elapsed),tres,start,end))
    raw='\n'.join((row(1,'v18_semantic_profile','COMPLETED',60,'cpu=4,gres/gpu=1,gres/gpu:b200=1'),
        row(2,'v18_train','FAILED',7,'gres/gpu:b200=1'),row(3,'v18_train','CANCELLED',0,'','Unknown','Unknown'),
        row(4,'unrelated','COMPLETED',100,'gres/gpu=8')))
    result=parse_accounting(raw);need(len(result)==3 and sum(r['allocated_gpu_seconds'] for r in result)==67,'Failed/zero or generic+typed GPU accounting differs')
    need(maximum_concurrency(result)==2,'Parallel interval accounting differs')
    p=projection(10,4,.1,1,2);need(p['projected_seconds']==10+1.25*(4+4586*.1+64+544)+120,'Projection formula differs')
    need(not projection(10,4,1,1,2)['passed'],'Over-budget projection passed')
    for invalid in (raw+'\n'+raw.splitlines()[0],row(7,'v18_train','RUNNING',3,'gres/gpu=1')):
        try:parse_accounting(invalid)
        except ValueError:pass
        else:raise ValueError('Invalid campaign inventory passed')
    return dict(passed=True,tests=['generic_vs_typed_TRES','failed_and_zero_retention','noncampaign_exclusion',
        'concurrency','exact_projection','over_budget','duplicate_rejection','unfinished_rejection'])


def release(args,out):
    import torch
    from transformers import AutoTokenizer
    from scripts import train_native_vision_v18 as train
    from scripts import report_native_vision_v18 as report
    from scripts import stage_native_vision_v18_gates as gates
    from scripts import profile_native_vision_v18_semantic_gate as software
    torch.set_num_threads(4)
    frozen=train.sources();report_sources=report.sources()
    release_sources={**report_sources,**{n:sha(REPO/n) for n in OWN}}
    (out/'source').mkdir(exist_ok=True)
    for name,digest in release_sources.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Release snapshot changed')
    save(out/'source_hashes.json',release_sources)
    plan_path=Path(args.plan).resolve();plan=train.verify_plan(plan_path,ancestors=True)
    cpu=read(plan_path.parent/'summary.json')
    need(cpu['passed'] and cpu['completed'] and cpu['plan_sha256']==sha(plan_path),'Completed exact training CPU plan required')
    data_release=Path(args.data_release).resolve();approved=read(data_release)
    need(approved['passed'] and not approved['manifest_only'] and approved['source_sha256']==report_sources
         and approved.get('cache_audit') is not None and approved.get('gate_cache_audit') is not None,
         'Independent complete data/cache/gate/source check required')
    need(sha(approved['audit_file'])==approved['audit_sha256'],'Independent data audit changed')
    for name,digest in approved['data_bindings'].items():need(sha(name)==digest,'Independent data binding changed')
    data=report.audit_data();tokenizer=AutoTokenizer.from_pretrained(str(train.MODEL),use_fast=False,local_files_only=True)
    cache,cache_audit=report.verify_cache(train.CACHE,data,tokenizer)
    gate_cache=gates.verify_cache(train.GATE_CACHE,ancestors=True,tensors=True)
    need(gate_cache['native_identity_sha256']==plan['native_identity_sha256']==approved['native_identity_sha256'],
         'Training, gate and data release native identities differ')
    software_path=Path(plan['native_profile']['file']);native=software.verify_profile(software_path)
    software_audit=report.verify_software_timing(software_path)
    times=native['timing']['rows']
    t16=max(r['four_token_seconds_bound'] for r in times if r['n_frames']==16)
    t64=max(r['four_token_seconds_bound'] for r in times if r['n_frames']==64)
    profiles={};audits={};projections={};profile_jobs={}
    for supplied in args.profiles:
        directory=Path(supplied).resolve();summary=read(directory/'summary.json');mode=summary['condition']
        need(mode in train.POLICY['conditions'] and mode not in profiles,'Duplicate or unknown training profile')
        audits[mode]=report.verify_training_profile(directory,plan_path,plan,mode,tokenizer,cache,gate_cache)
        logs=read(summary['training_file']);need(len(logs)==32,'Exactly32 training profile updates required')
        first=sum(r['seconds'] for r in logs[:4]);steady=max(r['seconds'] for r in logs[4:])
        need(summary['step_seconds_max_steady']==steady,'Profile steady maximum changed')
        projected=projection(summary['setup_before_training_seconds'],first,steady,t16,t64)
        profiles[mode]=dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'))
        projections[mode]=projected;profile_jobs[mode]=str(summary['slurm_job_id'])
    need(set(profiles)==set(train.POLICY['conditions']),'Both matched training profiles are mandatory')
    cmd=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
         '--format=JobIDRaw,JobID,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(cmd,capture_output=True,text=True,check=True);(out/'all_user_sacct.psv').write_text(result.stdout)
    rows=parse_accounting(result.stdout);lookup={r['JobIDRaw']:r for r in rows}
    for job in [str(native['slurm_job_id']),*profile_jobs.values()]:
        need(job in lookup and lookup[job]['state']=='COMPLETED' and lookup[job]['ExitCode']=='0:0'
             and lookup[job]['gpu_count']==1 and int(lookup[job]['ElapsedRaw'])<=300,'Passed GPU profile allocation is missing or over cap')
    need(all(int(r['ElapsedRaw'])<=300 for r in rows if r['gpu_count']),'Pre-main profile exceeded its registered cap')
    prior=sum(r['allocated_gpu_seconds'] for r in rows);maximum=maximum_concurrency(rows)
    software_cost=sum(r['allocated_gpu_seconds'] for r in rows if r['JobName']=='v18_semantic_profile')
    need(software_cost<=900,'Software-only campaign reserve exceeded')
    budget=dict(passed=prior+10800<=14400 and maximum<=4,allocations=rows,prior_jobs=[r['JobIDRaw'] for r in rows],
        prior_allocated_gpu_seconds=prior,reserved_main_gpu_seconds=10800,projected_campaign_gpu_seconds=prior+10800,
        campaign_gpu_seconds_cap=14400,software_allocated_gpu_seconds=software_cost,software_gpu_seconds_cap=900,
        maximum_concurrent_gpus=maximum,all_failed_and_zero_allocations_retained=True,
        all_user_sacct_file=str(out/'all_user_sacct.psv'),all_user_sacct_sha256=sha(out/'all_user_sacct.psv'),command=cmd)
    # Publish the complete failed decision too; a false gate never releases fits.
    passed=budget['passed'] and all(p['passed'] for p in projections.values())
    fresh=plan['fresh_manifests']['main']
    value=dict(protocol='v18_native_semantic_gate_main_release',passed=passed,plan_file=str(plan_path),plan_sha256=sha(plan_path),
        source_sha256=frozen,report_source_sha256=report_sources,release_source_sha256=release_sources,
        per_main_seconds_cap=2700,campaign_gpu_seconds_cap=14400,campaign_budget=budget,profiles=profiles,
        profile_audits=audits,projections=projections,software_audit=software_audit,
        data_release=dict(file=str(data_release),sha256=sha(data_release)),
        fresh_manifest=dict(file=fresh,sha256=sha(fresh)),cache_audit=cache_audit,
        native_identity_sha256=plan['native_identity_sha256'],no_fit=True,no_efficacy_selection=True,
        main_jobs=4,maximum_concurrent_project_gpus=4,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'release.json',value)
    need(train.sources()==frozen and report.sources()==report_sources and all(sha(REPO/n)==h for n,h in release_sources.items()),
         'Source changed during release audit')
    need(passed,'Timing or campaign cap failed; full projection and accounting retained')
    train.verify_release(out/'release.json',plan_path,plan)
    return dict(passed=True,completed=True,release_file=str(out/'release.json'),release_sha256=sha(out/'release.json'),
        source_sha256=release_sources,projections=projections,prior_allocated_gpu_seconds=prior,no_fit=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--release',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--profiles',type=Path,nargs=2)
    parser.add_argument('--data-release',type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS'),'All release and executable tests require CPU Slurm')
    if args.release:need(args.plan and args.profiles and args.data_release,'Supply exact completed plan, both profiles and independent data release')
    out=OUT/f'{"release_selftest" if args.self_test else "release"}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    # Preserve exact own source even if imports, self-tests or a plan gate fail.
    (out/'source').mkdir();own={name:sha(REPO/name) for name in OWN}
    for name,digest in own.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Initial source snapshot changed')
    save(out/'source_hashes.json',own)
    begin=time.perf_counter();tests=self_test();save(out/'tests.json',tests)
    result=release(args,out) if args.release else dict(passed=True,no_fit=True,tests=tests)
    result.update(seconds=time.perf_counter()-begin,slurm_job_id=os.environ['SLURM_JOB_ID']);save(out/'summary.json',result)
    print(json.dumps(dict(passed=result['passed'],directory=str(out),release_file=result.get('release_file'))),flush=True)


if __name__=='__main__':main()

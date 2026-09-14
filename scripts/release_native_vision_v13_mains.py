"""CPU-only V13 measured release, with complete declared prior GPU accounting.

The campaign inventory is discovered from all user Slurm allocation records,
not inferred from successful outputs. Failed allocations remain in the budget. sacct is read
with -X so batch/extern steps cannot double count allocation time.
"""
from pathlib import Path
import argparse
import json
import getpass
import math
import os
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts import train_native_vision_v13 as train
from scripts.stage_native_vision_v7_features import need, read, sha

OWN = ('scripts/release_native_vision_v13_mains.py', 'slurm/native_vision_v13_main_release.sbatch')
FIELDS = ('JobIDRaw', 'JobID', 'JobName', 'Partition', 'State', 'ElapsedRaw',
          'AllocTRES', 'ExitCode', 'Submit', 'Start', 'End')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def positive(value, name):
    need(isinstance(value, (int, float)) and not isinstance(value, bool)
         and math.isfinite(value) and value > 0, 'Invalid positive timing: '+name)
    return float(value)


def parse_accounting(raw, requested):
    need(requested and len(requested) == len(set(requested))
         and all(re.fullmatch(r'[0-9]+(?:_[0-9]+)?', x) for x in requested),
         'Require distinct explicit allocation/array-task IDs')
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        values = line.split('|')
        need(len(values) == len(FIELDS), 'Unexpected sacct columns')
        row = dict(zip(FIELDS, values))
        need(re.fullmatch(r'[0-9]+(?:_[0-9]+)?', row['JobIDRaw'])
             and re.fullmatch(r'[0-9]+(?:_[0-9]+)?', row['JobID']),
             'Accounting must contain allocation rows only')
        state = row['State'].split()[0].rstrip('+')
        need(state in TERMINAL, 'Unfinished or unknown allocation state: '+row['State'])
        need(row['Partition'] == 'gpu' and row['JobName'].startswith('v13_'),
             'Non-V13 GPU allocation in the declared campaign list')
        need(row['ElapsedRaw'].isdigit(), 'Malformed allocation elapsed seconds')
        tres = dict(item.split('=', 1) for item in row['AllocTRES'].split(',') if '=' in item)
        count = tres.get('gres/gpu')
        if count is None:
            typed = [int(v) for k, v in tres.items() if k.startswith('gres/gpu:')]
            count = sum(typed)
        need(str(count) in ('0','1'), 'Expected at most one allocated GPU per V13 job')
        row.update(state=state, gpu_count=int(count), allocated_gpu_seconds=int(count)*int(row['ElapsedRaw']))
        need(row['gpu_count']==0 or (row['End'] not in ('', 'Unknown', 'None') and row['Start'] not in ('', 'Unknown', 'None')),
             'Missing allocation start/end times')
        rows.append(row)
    need(rows and len({r['JobIDRaw'] for r in rows}) == len(rows),
         'Empty or duplicate allocation accounting')
    aliases = {r[k] for r in rows for k in ('JobIDRaw', 'JobID')}
    need(set(requested) <= aliases, 'A requested job is missing; expand array parents to explicit tasks')
    need(all(any(x in (r['JobIDRaw'], r['JobID']) for x in requested) for r in rows),
         'sacct returned unrequested allocations')
    need(len(rows) == len(requested), 'Duplicate aliases or unexpanded array job request')
    return rows


def measured_projection(load_seconds, steady_seconds, times):
    load = positive(load_seconds, 'load')
    steady = positive(steady_seconds, 'shared maximum steady step')
    t16 = positive(times['16'], 'N16'); t64 = positive(times['64'], 'N64')
    need(t64 >= t16, 'Malformed fixed native timing bound')
    training = 4590*steady; development = 320*t16; test = 272*t64
    projected = load + 1.25*(training+development+test) + 120
    return dict(projected_seconds=projected, passed=projected <= 2700,
                training_seconds=training, development_seconds=development, test_seconds=test,
                generation_seconds=times, N32_uses_N64_bound=True,
                shared_step_seconds_max_steady=steady)


def self_test():
    raw = ('42|42|v13_features_profile|gpu|COMPLETED|30|cpu=4,gres/gpu=1,gres/gpu:b200=1|0:0|s|s|e\n'
           '43|41_1|v13_train_profile|gpu|FAILED|7|gres/gpu=1|1:0|s|s|e\n')
    rows = parse_accounting(raw, ['42', '41_1'])
    need(sum(r['allocated_gpu_seconds'] for r in rows) == 37, 'Failures or typed GPU double counted')
    for text, ids in [(raw, ['42', '43', '41_1']), (raw.replace('FAILED', 'RUNNING'), ['42', '43']),
                      (raw.replace('gres/gpu=1|1:', 'gres/gpu=2|1:'), ['42', '43']),
                      (raw+'43.batch|43.batch|x|gpu|COMPLETED|7|gres/gpu=1|0:0|s|s|e\n', ['42', '43'])]:
        try:
            parse_accounting(text, ids)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed or double-counted accounting accepted')
    p = measured_projection(5, .1, {'16': 1., '64': 2.})
    need(p['training_seconds'] == 459 and p['development_seconds'] == 320
         and p['test_seconds'] == 544 and p['projected_seconds'] == 1778.75,
         'Registered projection arithmetic differs')
    need(5100 + 4*2700 + 300 == 16200 and 5101 + 4*2700 + 300 > 16200,
         'Campaign boundary arithmetic differs')
    return dict(passed=True, tests=7)


def ledger(mapping):
    need(mapping, 'Empty source/artifact ledger')
    for name, digest in mapping.items():
        path = Path(name); path = path if path.is_absolute() else REPO/path
        need(sha(path) == digest, 'Bound source/artifact changed: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path);parser.add_argument('--profiles',nargs=2,type=Path)
    parser.add_argument('--data-release',type=Path);parser.add_argument('--selected-selftest',type=Path)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS'),'CPU Slurm required')
    out=train.OUT/f'main_release_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    own={name:sha(REPO/name) for name in OWN}
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    train.save(out/'source_hashes.json',own);train.save(out/'self_test.json',self_test())
    (out/'INDEX.md').write_text('# V13 measured main release\n\n[Decision](release.json) · [Sources](source_hashes.json) · [GPU accounting](gpu_accounting.json)\n')
    if args.self_test:print(json.dumps(dict(passed=True,output=str(out))),flush=True);return
    need(args.plan and args.profiles and args.data_release and args.selected_selftest,
         'Require training plan, both profiles, independent data release, selected-audit source selftest')
    from scripts import report_native_vision_v13 as reporter
    from scripts.profile_native_vision_v13_null import verify_profile
    import torch
    torch.set_num_threads(4)
    args.plan=args.plan.resolve();args.data_release=args.data_release.resolve()
    plan=train.verify_plan(args.plan);checked=read(args.data_release)
    need(checked['passed'] is True and checked['manifest_only'] is False and checked['source_sha256']==reporter.sources()
         and checked['cache_audit']['features']==53322 and checked['cache_audit']['all_native_tensor_bytes_verified'] is True,
         'Independent data/cache/source release failed')
    ledger(checked['source_sha256']);ledger(checked['data_bindings'])
    need(sha(checked['audit_file'])==checked['audit_sha256'] and sha(checked['pairing_file'])==checked['pairing_sha256'],
         'Independent data/pairing artifact changed')
    data=reporter.audit_data();need(data['summary']['data_bindings']==checked['data_bindings'],'Independent data changed')
    cache=train.cache_binding(plan['cache_binding']['file'])
    software=verify_profile(plan['native_profile']['file'])
    need(sha(plan['native_profile']['file'])==plan['native_profile']['sha256'],'Native profile changed')
    verified_times=reporter.verify_software_timing(software)
    times={n:max(v[n] for v in verified_times.values()) for n in ('16','64')}
    profiles={};summaries={};successful=[]
    for directory in args.profiles:
        directory=directory.resolve();path=directory/'summary.json';s=read(path);condition=s['condition']
        need(condition in train.POLICY['conditions'] and condition not in profiles,'Duplicate/unregistered condition')
        item=dict(directory=str(directory),summary_sha256=sha(path))
        need(s['source_sha256']==train.sources() and s['plan_sha256']==sha(args.plan),'Profile/train source differs')
        s=reporter.verify_training_profile(item,condition,s,data,cache)
        need(s['hardware']['name']==software['hardware']['gpu'],'Native software and profile hardware differs')
        profiles[condition]=item;summaries[condition]=s
        successful.append(dict(role='training_'+condition,job_id=str(s['slurm_job_id']),seconds_cap=300))
    need(set(profiles)==set(train.POLICY['conditions']),'Both profiles required')
    centered=Path(profiles['centered']['directory']);offset=Path(profiles['offset']['directory'])
    need(summaries['centered']['initialized']==summaries['offset']['initialized'],'Matched initialization differs')
    a=read(centered/'first_gradients.json')[0];b=read(offset/'first_gradients.json')[0]
    need(a==b,'Zero-initialized first-step matched gradients differ')
    maximum=max(s['step_seconds_max_steady'] for s in summaries.values())
    projections={c:measured_projection(s['model_and_features_load_seconds'],maximum,times) for c,s in summaries.items()}
    feature=read(Path(cache['profile_directory'])/'summary.json')
    successful.append(dict(role='feature_profile',job_id=str(feature['slurm_job_id']),seconds_cap=180))
    for entry in cache['shards']:
        s=read(Path(entry['directory'])/'summary.json')
        successful.append(dict(role='feature_shard_'+str(s['shard']),job_id=str(s['slurm_job_id']),seconds_cap=600))
    successful.append(dict(role='native_software',job_id=str(software['slurm_job_id']),seconds_cap=300))
    # Discover the complete campaign from Slurm rather than a success-only list.
    command=['sacct','-X','--noheader','--parsable2','--user',getpass.getuser(),'--starttime','2026-09-01',
             '--format',','.join(x+'%100' if x=='JobName' else x for x in FIELDS)]
    train.save(out/'accounting_request.json',dict(command=command,scope='All user v13_ GPU allocations since 2026-09-01, including failed attempts'))
    proc=subprocess.run(command,text=True,capture_output=True,check=False,timeout=60)
    (out/'sacct_all_user.txt').write_text(proc.stdout);(out/'sacct.stderr.txt').write_text(proc.stderr)
    need(proc.returncode==0,'sacct failed')
    chosen=[]
    for line in proc.stdout.splitlines():
        if not line.strip():continue
        values=line.split('|');need(len(values)==len(FIELDS),'Unexpected sacct discovery schema')
        row=dict(zip(FIELDS,values))
        if row['JobName'].startswith('v13_') and row['Partition']=='gpu':chosen.append(line)
    raw='\n'.join(chosen)+'\n';(out/'sacct.txt').write_text(raw)
    ids=[line.split('|')[0] for line in chosen];rows=parse_accounting(raw,ids)
    for item in successful:
        found=[r for r in rows if item['job_id'] in (r['JobIDRaw'],r['JobID'])]
        need(len(found)==1 and found[0]['state']=='COMPLETED' and found[0]['ExitCode']=='0:0'
             and 0<found[0]['allocated_gpu_seconds']<=item['seconds_cap'],'Successful artifact allocation/cap differs: '+item['role'])
        item['accounting_job_id']=found[0]['JobIDRaw']
    need(len({x['accounting_job_id'] for x in successful})==8,'Successful profile/shard job roles incomplete')
    need(not any(r['JobName'] in ('v13_main','v13_selected') for r in rows),'Main outcomes already exist before release')
    feature_cost=sum(r['allocated_gpu_seconds'] for r in rows if r['JobName'] in ('v13_null_profile','v13_null_harvest'))
    software_cost=sum(r['allocated_gpu_seconds'] for r in rows if r['JobName']=='v13_native_profile')
    need(feature_cost<=2700 and software_cost<=600,'Feature/software campaign subcap exceeded')
    prior=sum(r['allocated_gpu_seconds'] for r in rows)
    campaign=dict(passed=prior+10800+300<=16200,prior_jobs=ids,allocations=rows,prior_allocated_gpu_seconds=prior,
        reserved_main_gpu_seconds=10800,reserved_selected_audit_gpu_seconds=300,projected_campaign_gpu_seconds=prior+11100,
        campaign_gpu_seconds_cap=16200,max_concurrent_project_gpus=4,sacct_file=str(out/'sacct.txt'),sacct_sha256=sha(out/'sacct.txt'),
        all_user_sacct_file=str(out/'sacct_all_user.txt'),all_user_sacct_sha256=sha(out/'sacct_all_user.txt'),successful_artifacts=successful,
        feature_allocated_gpu_seconds=feature_cost,native_software_allocated_gpu_seconds=software_cost)
    train.save(out/'gpu_accounting.json',campaign)
    selected_path=args.selected_selftest.resolve();selected=read(selected_path)
    need(selected['passed'] is True and selected['tests_passed'] is True and selected['no_model_loaded'] is True
         and selected['protocol']=='all_selected_v13_native_checkpoint_audit'
         and all(selected['source_sha256'].get(k)==v for k,v in checked['source_sha256'].items()),
         'Selected-audit source/selftest incomplete');ledger(selected['source_sha256'])
    fresh=train.FRESH/'main_manifest.json';need(checked['data_bindings'][str(fresh)]==sha(fresh),'Fresh data not bound')
    result=dict(protocol='v13_learned_null_main_release',passed=campaign['passed'] and all(p['passed'] for p in projections.values()),
        plan_file=str(args.plan),plan_sha256=sha(args.plan),source_sha256=train.sources(),release_source_sha256=own,
        profiles=profiles,projections=projections,shared_step_seconds_max_steady=maximum,
        data_release=dict(file=str(args.data_release),sha256=sha(args.data_release)),report_source_sha256=checked['source_sha256'],
        fresh_manifest=dict(file=str(fresh),sha256=sha(fresh)),per_main_seconds_cap=2700,main_block_gpu_seconds_cap=16200,
        campaign_budget=campaign,selected_audit_freeze=dict(directory=str(selected_path.parent),summary_sha256=sha(selected_path)))
    train.save(out/'release.json',result)
    if result['passed']:train.verify_release(out/'release.json',args.plan,plan)
    ledger(own);ledger(checked['source_sha256'])
    print(json.dumps(dict(passed=result['passed'],release_file=str(out/'release.json'),projections=projections,
        prior_allocated_gpu_seconds=prior)),flush=True)
    need(result['passed'],'Measured time or whole-campaign budget failed; preserve this release')


if __name__=='__main__':main()

"""CPU release of matched V8 mains from frozen checks and measured profiles."""
from pathlib import Path
import argparse
import json
import os
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import train_native_vision_v8 as train
from scripts.stage_native_vision_v7_features import need,read,sha
OWN=('scripts/release_native_vision_v8_mains.py','slurm/native_vision_v8_main_release.sbatch')
TIMING=REPO/'outputs/native_aggregation_vlm/v7/main_runtime_extension_441884/release.json'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--profiles',type=Path,nargs=2,required=True)
    p.add_argument('--data-release',type=Path,required=True)
    a=p.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU Slurm required')
    out=train.OUT/f'main_release_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    own={name:sha(REPO/name) for name in OWN}
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    train.save(out/'source_hashes.json',own)
    a.plan=a.plan.resolve();plan=train.verify_plan(a.plan)
    data=read(a.data_release)
    need(data['passed'] is True and data['source_sha256'],'Completed independent data/report release required')
    for name,digest in data['source_sha256'].items():need(sha(REPO/name)==digest,'Independent report source changed')
    for name,digest in data['data_bindings'].items():need(sha(name)==digest,'Independent data binding changed')
    old=read(TIMING);oldplan=read(old['plan_file'])
    need(old['passed'] is True and sha(old['plan_file'])==old['plan_sha256'],'Timing ancestor changed')
    need(Path(oldplan['native_profile']).resolve()==Path(plan['native_profile']['directory']).resolve()
         and oldplan['native_profile_sha256']==plan['native_profile']['sha256'],'Different native timing profile')
    for name in ('scripts/native_vision_v7_runtime.py','gnnformer/parallel_local_native.py','gnnformer/parallel_local_aggregation.py','gnnformer/parallel_local_prompts.py'):
        need(old['source_sha256'][name]==plan['source_sha256'][name]==sha(REPO/name),'Deployed native computation changed')
    times=old['projections']['parallel']['generation_seconds'];t16=times['16'];t64=times['64']
    need(t16>0 and t64>=t16,'Malformed fixed native timing bound')
    profiles={};projections={}
    for directory in a.profiles:
        directory=directory.resolve();s=read(directory/'summary.json');condition=s['condition']
        need(condition in train.POLICY['conditions'] and condition not in profiles,'Duplicate or incorrect profile')
        need(s['profile'] and s['seed']==10 and s['passed'] and s['steps']==32 and s['no_dev_or_test_evaluation']
             and s['computational_integrity_passed'] and s['active_cache_replay']['passed']
             and s['plan_sha256']==sha(a.plan) and s['source_sha256']==train.sources(),'Failed or unmatched training profile')
        for name,digest in s['source_sha256'].items():need(sha(REPO/name)==digest,'Training source changed')
        old_profile=read(Path(old['profiles']['parallel']['directory'])/'summary.json')
        need(s['hardware']==old_profile['hardware'],'Timing hardware/backend differs')
        training=4860*s['step_seconds_max_steady'];development=360*t16;test=108*t16+344*t64
        projected=s['model_and_features_load_seconds']+1.25*(training+development+test)+120
        projections[condition]=dict(projected_seconds=projected,passed=projected<=2700,
            training_seconds=training,development_seconds=development,test_seconds=test,
            generation_seconds=times,N32_uses_N64_bound=True)
        profiles[condition]=dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'))
    need(set(profiles)==set(train.POLICY['conditions']),'Both conditions required')
    result=dict(passed=all(v['passed'] for v in projections.values()),plan_file=str(a.plan),plan_sha256=sha(a.plan),
        source_sha256=train.sources(),release_source_sha256=own,profiles=profiles,projections=projections,
        data_release=dict(file=str(a.data_release.resolve()),sha256=sha(a.data_release)),report_source_sha256=data['source_sha256'],
        per_main_seconds_cap=2700,main_block_gpu_seconds_cap=11700,
        native_timing_ancestor=dict(file=str(TIMING),sha256=sha(TIMING),plan_file=old['plan_file'],plan_sha256=old['plan_sha256'],
            native_profile_file=plan['native_profile']['file'],native_profile_sha256=plan['native_profile']['sha256'],
            scope='Unchanged native parallel inference; prior measured four-token-plus-CPU-preparation bound, no prior efficacy used'))
    train.save(out/'release.json',result)
    (out/'INDEX.md').write_text('# V8 main release\n\n[Measured decision](release.json) · [Sources](source_hashes.json)\n')
    if result['passed']:train.verify_release(out/'release.json',a.plan,plan)
    for name,digest in own.items():need(sha(REPO/name)==digest,'Release source changed during execution')
    print(json.dumps(dict(passed=result['passed'],release_file=str(out/'release.json'),projections=projections)),flush=True)
    if not result['passed']:raise SystemExit('Main runtime cap failed; preserve this release')

if __name__=='__main__':main()

"""Explicit V7 cache walltime release; original failed profile stays immutable.

Only profile runtime eligibility changes from300 to360 seconds. The original
harvest/merge computation is reused without editing its source or data plan.
This dispatcher adds an execution-source ledger to every resulting artifact.
"""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import cache_native_vision_v7_features as base
from scripts.stage_native_vision_v7_features import need,sha,read,save,verify_plan,OUT,DATA
OWN=('scripts/extend_native_vision_v7_feature_runtime.py','slurm/native_vision_v7_features_extended.sbatch',
     'slurm/native_vision_v7_features_extended_release.sbatch','slurm/native_vision_v7_features_extended_merge.sbatch')
def own():return {p:sha(REPO/p) for p in OWN}
def validate(directory,plan,path):
    directory=Path(directory).resolve();s=read(directory/'summary.json')
    need(directory.is_relative_to(OUT) and s['profile'] and s['completed'] and s['computational_integrity_passed']
         and s['numerical_gate_passed'] and not s['runtime_projection_passed'] and not s['passed'],
         'Expected retained runtime-only failed profile')
    need(s['plan_sha256']==sha(path) and s['source_sha256']==plan['source_sha256']
         and sha(s['raw_file'])==s['raw_sha256'] and sha(s['observations_file'])==s['observations_sha256'],
         'Original profile/source/raw binding differs')
    need(300<max(s['shard_projected_seconds'])<=360,'Extended runtime release is not applicable')
    return s

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--profile-directory',type=Path,required=True);p.add_argument('--release',type=Path)
    p.add_argument('--check',action='store_true');p.add_argument('--shard',type=int);p.add_argument('--merge',nargs=4,type=Path)
    a=p.parse_args();need(os.environ.get('SLURM_JOB_ID'),'Use Slurm')
    plan=verify_plan(a.plan,pixels=True);original=validate(a.profile_directory,plan,a.plan)
    if a.check:
        need(os.environ['SLURM_JOB_PARTITION']=='cpu','CPU release required')
        out=OUT/'runtime_extension'/f'release_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
        ledger=dict(passed=True,scope='runtime_only_300_to_360_seconds',original_profile_passed=False,
            original_runtime_gate_passed=False,numerical_gate_unchanged=True,cap_seconds=360,
            plan_file=str(a.plan.resolve()),plan_sha256=sha(a.plan),
            profile_directory=str(a.profile_directory.resolve()),profile_summary_sha256=sha(a.profile_directory/'summary.json'),
            dispatcher_source_sha256=own(),projected_seconds=original['shard_projected_seconds'])
        save(out/'release.json',ledger);(out/'code').mkdir()
        for f in OWN:(out/'code'/f.replace('/','_')).write_bytes((REPO/f).read_bytes())
        (out/'INDEX.md').write_text('# Explicit cache runtime extension\n\n[Release](release.json). Original five-minute eligibility remains failed.\n')
        print(json.dumps(dict(release=str(out/'release.json'))),flush=True);return
    ledger=read(a.release)
    need(ledger['dispatcher_source_sha256']==own() and ledger['plan_sha256']==sha(a.plan)
         and ledger['profile_summary_sha256']==sha(a.profile_directory/'summary.json') and ledger['passed'],
         'Runtime release binding changed')
    # Explicit process-local eligibility injection; no model/kernel/input code changes.
    base.verify_profile=validate
    if a.merge:
        need(os.environ['SLURM_JOB_PARTITION']=='cpu','CPU merge required')
        for d in a.merge:
            need(read(d/'runtime_release.json')==dict(ledger,release_file=str(a.release.resolve()),release_sha256=sha(a.release)),
                 'A shard lacks its explicit runtime-source release')
        original_save=base.save
        def released_save(path,value):
            if Path(path)==DATA/'feature_cache.json':
                value=dict(value,runtime_extension=dict(ledger,release_file=str(a.release.resolve()),release_sha256=sha(a.release)))
            original_save(path,value)
        base.save=released_save
        base.merge(a)
        out=OUT/f'merge_{os.environ["SLURM_JOB_ID"]}'

    else:
        need(os.environ['SLURM_JOB_PARTITION']=='gpu' and os.environ.get('SLURM_JOB_GPUS') and a.shard in range(4),'GPU shard required')
        a.profile=False;base.gpu(a);out=OUT/f'shard{a.shard}_{os.environ["SLURM_JOB_ID"]}'
    save(out/'runtime_release.json',dict(ledger,release_file=str(a.release.resolve()),release_sha256=sha(a.release)))
    (out/'dispatcher_code').mkdir()
    for f in OWN:(out/'dispatcher_code'/f.replace('/','_')).write_bytes((REPO/f).read_bytes())
if __name__=='__main__':main()

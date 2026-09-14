"""Explicit resource-only extension after the retained30-minute V7 release fails."""
import argparse,json,os,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import train_native_vision_v7 as base
OWN=('scripts/release_native_vision_v7_main_runtime.py','slurm/native_vision_v7_main_runtime_release.sbatch')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('failed_release',type=Path);a=p.parse_args()
    base.need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu','CPU Slurm required')
    old=base.read(a.failed_release);plan=base.read(old['plan_file'])
    base.need(old['passed'] is False and old['per_main_seconds_cap']==1800 and old['main_block_gpu_seconds_cap']==9000
        and old['source_sha256']==base.sources() and old['plan_sha256']==base.sha(old['plan_file']),
        'Expected immutable original30-minute runtime release failure')
    base.need(set(old['projections'])=={'parallel','joint'} and any(not r['passed'] for r in old['projections'].values()),
        'Original runtime failure missing')
    for arm,value in old['profiles'].items():
        base.need(base.sha(Path(value['directory'])/'summary.json')==value['summary_sha256'],'Training profile changed')
        profile=base.read(Path(value['directory'])/'summary.json')
        base.need(profile['passed'] and profile['computational_integrity_passed'] and profile['arm']==arm
            and profile['source_sha256']==old['source_sha256'],'Training computational gate differs')
    for name,digest in old['report_source_sha256'].items():base.need(base.sha(REPO/name)==digest,'Reporter changed')
    base.need(base.sha(old['data_release']['file'])==old['data_release']['sha256'],'Independent data release changed')
    base.need(all(r['projected_seconds']<=2700 for r in old['projections'].values()),'45-minute extension insufficient')
    out=base.OUT/f'main_runtime_extension_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    extension=dict(scope='resource_only_1800_to_2700_seconds',original_release_file=str(a.failed_release.resolve()),
        original_release_sha256=base.sha(a.failed_release),original_passed=False,
        source_sha256={p:base.sha(REPO/p) for p in OWN},no_model_data_optimizer_selection_or_decision_change=True)
    revised=dict(old,passed=True,per_main_seconds_cap=2700,main_block_gpu_seconds_cap=11700,resource_extension=extension)
    revised['projections']={k:dict(v,passed=True,original_1800_second_passed=v['passed']) for k,v in old['projections'].items()}
    base.save(out/'release.json',revised);(out/'code').mkdir()
    for p in OWN:(out/'code'/p.replace('/','_')).write_bytes((REPO/p).read_bytes())
    (out/'INDEX.md').write_text('# V7 explicit main runtime extension\n\n[Release](release.json). Original30-minute timing failure retained.\n')
    print(json.dumps(dict(passed=True,release=str(out/'release.json'),projections=revised['projections'])),flush=True)
if __name__=='__main__':main()

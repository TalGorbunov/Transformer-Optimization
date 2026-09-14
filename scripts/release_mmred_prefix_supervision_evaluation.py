"""CPU-only resource/source gate for a prospectively queued evaluation array."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import evaluate_mmred_prefix_supervision as producer

OWN=('scripts/release_mmred_prefix_supervision_evaluation.py',
     'slurm/mmred_prefix_supervision_evaluation_release.sbatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    producer.need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and os.environ.get('SLURM_JOB_ID')
                  and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4
                  and not os.environ.get('SLURM_JOB_GPUS'),'Four-core CPU-only release verification required')
    started=time.perf_counter();out=producer.OUT/('release_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    own={name:producer.sha(REPO/name) for name in OWN};(out/'source').mkdir()
    for name,digest in own.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        producer.need(producer.sha(target)==digest,'Release-gate source archive differs')
    try:
        plan=producer.verify_plan(args.plan)
        producer.need(plan['resource_eligible'] is True and plan['projection']['passed'] is True,
                      'Published preparation does not release GPU evaluation')
        producer.need({name:producer.sha(REPO/name) for name in OWN}==own and time.perf_counter()-started<300,
                      'Release source/time boundary changed')
        producer.save(out/'summary.json',dict(protocol='mmred_prefix_supervision_evaluation_release',passed=True,completed=True,
            plan_file=str(args.plan.resolve()),plan_sha256=producer.sha(args.plan),source_sha256=own,
            inherited_source_sha256={**producer.inherited_sources(),**producer.sources()},
            projection=plan['projection'],tasks=7,maximum_concurrent_GPUs=3,gpu_seconds_per_task=3600,
            no_accuracy_condition=True,no_model_or_tensor_execution=True,elapsed_seconds=time.perf_counter()-started))
    except BaseException as error:
        producer.save(out/'failure.json',dict(passed=False,type=type(error).__name__,message=str(error),
            elapsed_seconds=time.perf_counter()-started,no_GPU_release=True));raise


if __name__=='__main__':main()

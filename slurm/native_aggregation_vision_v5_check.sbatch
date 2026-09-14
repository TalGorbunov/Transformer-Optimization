#!/bin/bash
#SBATCH -J native_agg_v5_check
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python tests/test_independent_vision_aggregation.py
.venv/bin/python - <<'PY'
import json,os
from pathlib import Path
from scripts.native_aggregation_vlm_v5 import parser,validate_args,load_schedule,source_hashes
root=Path('/mnt/data/gabriele/gnn_transformer/v4_diversity')
checks=[]
for profile in (False,True):
 for condition in ('sum','mean'):
  args=['--dataset-root',str(root),'--condition',condition,'--check-schedule']
  if profile:
   args+=['--profile','--epochs','2','--limit-dev','2','--limit-eval','2','--limit-count','2',
          '--manifest',str(root/'profile_manifest.json'),'--count-manifest',str(root/'profile_count_manifest.json'),
          '--schedule',str(root/'profile_schedule.json')]
  a=parser().parse_args(args);validate_args(a)
  raw,schedule,audit,token_raw,tokens=load_schedule(a)
  checks.append(dict(profile=profile,condition=condition,audit=audit))
out=Path('outputs/native_aggregation_vlm/v5')/f'check_{os.environ["SLURM_JOB_ID"]}'
out.mkdir(parents=True,exist_ok=False)
ledger=source_hashes()
(out/'source_hashes.json').write_text(json.dumps(ledger,indent=2)+'\n')
(out/'core.json').write_text(json.dumps(dict(schema_version=1,passed=True,unit_tests_passed=True,
 schedule_checks_passed=True,source_sha256=ledger,checks=checks,slurm_job_id=os.environ['SLURM_JOB_ID']),indent=2)+'\n')
for name in ledger:
 target=out/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
(out/'INDEX.md').write_text('# V5 CPU validation\n\n- [Core and schedule checks](core.json)\n- [Frozen source ledger](source_hashes.json)\n')
print(json.dumps(dict(output=str(out),passed=True)))
PY

#!/bin/bash
#SBATCH -J mmred_official_pilot_fixtures
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH -t 00:01:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python - <<'PY'
import hashlib,json,os,time
from pathlib import Path
from scripts import report_mmred_official_pilot as stats
assert os.environ['SLURM_JOB_PARTITION']=='cpu'
assert hashlib.sha256(Path(stats.__file__).read_bytes()).hexdigest()=='7f32ab89c8012fab358e91d07fbec2990c87a264a2e3ea69711c9387dbdfe6f5'
assert stats.sha(stats.REPO/stats.PROPOSAL)==stats.PROPOSAL_SHA
started=time.perf_counter()
out=stats.OUT/('fixtures_'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False)
try:
 import numpy as np
 result=stats.self_test(np)
 stats.save(out/'summary.json',dict(result,elapsed_seconds=time.perf_counter()-started,source_sha256=stats.sources(),actual_benchmark_rows=0,bootstrap_actual_data_calls=0,gpu_calls=0,cpu_cap_seconds=60))
 print(json.dumps(result,sort_keys=True))
except BaseException as exc:
 stats.save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),elapsed_seconds=time.perf_counter()-started));raise
PY

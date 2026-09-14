#!/bin/bash
#SBATCH -J v13_null_selftest
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python - <<'PYCODE'
import json,os,hashlib,subprocess,sys,time
from pathlib import Path
root=Path.cwd();job=os.environ['SLURM_JOB_ID'];out=root/'outputs/native_aggregation_vlm/v13/null_software'/('selftest_'+job);out.mkdir(parents=True,exist_ok=False)
files=['gnnformer/conditional_null_mean.py','gnnformer/parallel_local_aggregation.py','tests/test_conditional_null_mean.py','slurm/native_vision_v13_null_selftest.sbatch'];frozen={}
for name in files:
    data=(root/name).read_bytes();frozen[name]=hashlib.sha256(data).hexdigest();(out/name.replace('/','_')).write_bytes(data)
start=time.perf_counter();run=subprocess.run([sys.executable,'tests/test_conditional_null_mean.py'],capture_output=True,text=True)
(out/'stdout.txt').write_text(run.stdout);(out/'stderr.txt').write_text(run.stderr)
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in frozen.items())
summary=dict(passed=run.returncode==0,returncode=run.returncode,source_sha256=frozen,seconds=time.perf_counter()-start,slurm_job_id=job)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'INDEX.md').write_text('# V13 null predictor software tests\n\n[Summary](summary.json) · [Test output](stderr.txt).\n');print(run.stdout,run.stderr);print(json.dumps(summary));sys.exit(run.returncode)
PYCODE

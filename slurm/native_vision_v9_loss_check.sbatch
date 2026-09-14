#!/bin/bash
#SBATCH -J v9_path_loss_test
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,os,subprocess,time
out=Path('outputs/native_aggregation_vlm/v9')/('loss_selftest_'+os.environ['SLURM_JOB_ID'])
out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
names=['gnnformer/aggregation_path_bound.py','tests/test_aggregation_path_bound.py','slurm/native_vision_v9_loss_check.sbatch','gnnformer/parallel_local_aggregation.py','gnnformer/aggregation_consistency.py','tests/test_aggregation_consistency.py']
source={}
for name in names:
 p=Path(name);blob=p.read_bytes();source[name]=hashlib.sha256(blob).hexdigest();(out/'source'/name.replace('/','_')).write_bytes(blob)
(out/'source_hashes.json').write_text(json.dumps(source,indent=2)+'\n')
start=time.perf_counter();runs=[]
for name in ['tests/test_aggregation_path_bound.py','tests/test_aggregation_consistency.py']:
 result=subprocess.run(['.venv/bin/python',name],capture_output=True,text=True)
 runs.append(dict(test=name,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
 print(result.stdout+result.stderr,flush=True)
passed=all(r['returncode']==0 for r in runs)
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==d for n,d in source.items())
(out/'summary.json').write_text(json.dumps(dict(passed=passed,source_sha256=source,runs=runs,seconds=time.perf_counter()-start),indent=2)+'\n')
(out/'INDEX.md').write_text('# Path-bound loss software tests\n\n[Summary](summary.json) · [Sources](source_hashes.json)\n')
if not passed:raise SystemExit('Loss checks failed; outputs preserved')
PY

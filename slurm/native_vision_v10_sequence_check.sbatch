#!/bin/bash
#SBATCH -J v10_sequence_check
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:03:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path.cwd(); out=root/'outputs/native_aggregation_vlm/v10'/('sequence_selftest_'+os.environ['SLURM_JOB_ID'])
out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
names=['gnnformer/paired_sequence_objectives.py','tests/test_paired_sequence_objectives.py','slurm/native_vision_v10_sequence_check.sbatch','gnnformer/aggregation_consistency.py','gnnformer/aggregation_path_bound.py']
ledger={}
for name in names:
 data=(root/name).read_bytes();ledger[name]=hashlib.sha256(data).hexdigest();(out/'source'/name.replace('/','_')).write_bytes(data)
(out/'source_hashes.json').write_text(json.dumps(ledger,indent=2)+'\n')
(out/'INDEX.md').write_text('# General paired sequence objectives\n\n[Summary](summary.json) · [Sources](source_hashes.json)\n')
tick=time.perf_counter();r=subprocess.run([sys.executable,'tests/test_paired_sequence_objectives.py'],text=True,capture_output=True)
summary=dict(passed=r.returncode==0,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_sha256=ledger,seconds=time.perf_counter()-tick)
assert all(hashlib.sha256((root/k).read_bytes()).hexdigest()==v for k,v in ledger.items())
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));raise SystemExit(r.returncode)
PY

#!/bin/bash
#SBATCH -J v11_memory_selftest
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python - <<'PYCODE'
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
start=time.perf_counter()
root=Path.cwd();out=root/'outputs/native_aggregation_vlm/v11/memory_software'/('selftest_'+os.environ['SLURM_JOB_ID'])
out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
files=['gnnformer/parallel_local_memory.py','tests/test_parallel_local_memory.py','scripts/native_vision_v11_last_block.py','tests/test_native_vision_v11_last_block.py','slurm/native_vision_v11_memory_selftest.sbatch']
hashes={}
for name in files:
 data=(root/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();(out/'source'/name.replace('/','_')).write_bytes(data)
(out/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
(out/'INDEX.md').write_text('# V11 CPU memory and last-block self-tests\n\n[Sources](source_hashes.json) · [Test log](unit.stderr.txt) · [Summary](summary.json)\n')
cmd=[sys.executable,'-m','unittest','tests.test_parallel_local_memory','tests.test_native_vision_v11_last_block']
r=subprocess.run(cmd,capture_output=True,text=True)
(out/'unit.stdout.txt').write_text(r.stdout);(out/'unit.stderr.txt').write_text(r.stderr)
summary=dict(passed=r.returncode==0,returncode=r.returncode,source_sha256=hashes,seconds=time.perf_counter()-start,slurm_job_id=os.environ['SLURM_JOB_ID'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(r.stdout);print(r.stderr);print(json.dumps(summary));sys.exit(r.returncode)
PYCODE

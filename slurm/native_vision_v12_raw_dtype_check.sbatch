#!/bin/bash
#SBATCH -J v12_raw_dtype_check
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:02:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python - <<'PYCODE'
import json,os,hashlib
from pathlib import Path
import torch
torch.set_num_threads(4)
assert os.environ['SLURM_JOB_PARTITION']=='cpu' and not os.environ.get('SLURM_JOB_GPUS')
root=Path.cwd();summary_file=root/'outputs/native_aggregation_vlm/v12/reference_software/profile_442314/summary.json'
summary=json.loads(summary_file.read_text());assert summary['passed']
out=root/'outputs/native_aggregation_vlm/v12/reference_software'/('raw_dtype_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
source=root/'slurm/native_vision_v12_raw_dtype_check.sbatch';(out/source.name).write_bytes(source.read_bytes())
rows=[]
for r in summary['timing']['rows']:
    p=Path(r['raw_file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['raw_sha256']
    data=torch.load(p,map_location='cpu',weights_only=True);v=data['raw_logits']
    rows.append(dict(actual_n_frames=r['actual_n_frames'],path=str(p),sha256=r['raw_sha256'],dtype=str(v.dtype),shape=list(v.shape),finite=bool(torch.isfinite(v).all()),exact_fp16_roundtrip=torch.equal(v,v.half().float())))
result=dict(passed=all(r['finite'] and r['exact_fp16_roundtrip'] for r in rows),rows=rows,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),software_summary=str(summary_file),software_sha256=hashlib.sha256(summary_file.read_bytes()).hexdigest(),slurm_job_id=os.environ['SLURM_JOB_ID'])
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'INDEX.md').write_text('# Native generation dtype audit\n\n[Summary](summary.json).\n');print(json.dumps(result),flush=True)
assert result['passed']
PYCODE

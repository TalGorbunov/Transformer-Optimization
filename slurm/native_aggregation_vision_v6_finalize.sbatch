#!/bin/bash
#SBATCH -J v6_finalize
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python - "${1:?Pass the completed post-audit Slurm job ID}" <<'PY'
from pathlib import Path
import hashlib,json,sys
job=sys.argv[1];assert job.isdigit()
root=Path('outputs/native_aggregation_vlm/v6').resolve()
p=root/'checkpoint_audit'/f'post_run_{job}'/'summary.json'
s=json.loads(p.read_text());assert s['computational_integrity_passed'] is True
ledger=dict(schema_version=1,artifacts=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())])
with (root/'post_training_audits.json').open('x') as f:json.dump(ledger,f,indent=2);f.write('\n')
PY
.venv/bin/python scripts/report_native_vision_v6.py

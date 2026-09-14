#!/bin/bash
#SBATCH -J native_agg_v3_report_check
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python scripts/report_native_vision_v3.py --self-test
.venv/bin/python - <<'PY'
from pathlib import Path
import json,torch
from scripts.report_native_vision_v3 import verify_run
base=Path('outputs/native_aggregation_vlm/v2')
paths=list((base/'main/hidden').glob('*/summary.json'))
assert len(paths)==1
rows,info=verify_run(paths[0].parent,'hidden',0,json.loads((base/'main/source_hashes.json').read_text()),torch,historical=True)
assert len(rows)==452 and info['selected_epoch']==7
print('PASS: full historical V2 provenance/checkpoint/manifests/predictions verification')
PY

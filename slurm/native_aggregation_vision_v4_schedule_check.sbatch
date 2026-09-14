#!/bin/bash
#SBATCH -J native_v4_schedule
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
.venv/bin/python - <<'PY'
from pathlib import Path
from types import SimpleNamespace
from scripts.native_aggregation_vlm_v4 import load_schedule
root=Path('/mnt/data/gabriele/gnn_transformer/v4_diversity')
for profile in (False,True):
    a=SimpleNamespace(profile=profile,schedule=root/('profile_schedule.json' if profile else 'schedule.json'),manifest=root/('profile_manifest.json' if profile else 'main_manifest.json'),count_manifest=root/('profile_count_manifest.json' if profile else 'count_manifest.json'))
    raw,schedule,audit,token_raw,token_records=load_schedule(a)
    print('PASS real profile' if profile else 'PASS real main',audit,flush=True)
PY

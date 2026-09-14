#!/bin/bash
#SBATCH -J native_recover_check
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python scripts/fit_native_vision_recoverability.py --self-test
.venv/bin/python - <<'PY'
import os
import subprocess
import sys
env = dict(os.environ)
env.pop('SLURM_JOB_ID', None)
commands = [
    ['scripts/fit_native_vision_recoverability.py', '--self-test'],
    ['scripts/probe_native_vision_recoverability.py', '--manifest', '/missing/main.json',
     '--count-manifest', '/missing/count.json', '--data-output', '/missing/features', '--output', '/missing/report'],
]
for arguments in commands:
    result = subprocess.run([sys.executable, *arguments], env=env, text=True, capture_output=True)
    if result.returncode == 0 or 'requires Slurm' not in result.stderr:
        raise AssertionError((arguments[0], result.returncode, result.stdout, result.stderr))
print('PASS: both operational Slurm guards fire before data/model access', flush=True)
PY

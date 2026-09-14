#!/bin/bash
#SBATCH -J v7_main_runtime_release
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
.venv/bin/python scripts/release_native_vision_v7_main_runtime.py "$@"

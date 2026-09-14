#!/bin/bash
#SBATCH -J mmred_official_recovery
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_ETAG_TIMEOUT=15 HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_DISABLE_TELEMETRY=1 GIT_TERMINAL_PROMPT=0
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
.venv/bin/python scripts/stage_mmred_official_recovery.py "$@"

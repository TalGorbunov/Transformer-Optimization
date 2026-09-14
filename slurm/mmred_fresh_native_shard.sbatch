#!/bin/bash
#SBATCH -J mmred_fresh_native_shard
#SBATCH -p cpu
#SBATCH --array=0-3
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:50:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/stage_mmred_fresh_native.py --shard "$@"

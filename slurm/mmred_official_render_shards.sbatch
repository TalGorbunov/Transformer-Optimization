#!/bin/bash
#SBATCH -J mmred_official_render_shards
#SBATCH -p cpu
#SBATCH --array=0-3
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:15:00
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/render_mmred_official_shards.py --shard "${SLURM_ARRAY_TASK_ID:?}" "$@"

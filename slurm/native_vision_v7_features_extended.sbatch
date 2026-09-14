#!/bin/bash
#SBATCH -J v7_features_ext
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:06:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python scripts/extend_native_vision_v7_feature_runtime.py --shard "${SLURM_ARRAY_TASK_ID:?}" "$@"

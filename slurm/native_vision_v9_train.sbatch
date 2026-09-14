#!/bin/bash
#SBATCH -J v9_main
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH --array=0-3%4
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V9_CONDITIONS=(residual path residual path)
V9_SEEDS=(12 12 13 13)
.venv/bin/python scripts/train_native_vision_v9.py --run --condition "${V9_CONDITIONS[$SLURM_ARRAY_TASK_ID]}" --seed "${V9_SEEDS[$SLURM_ARRAY_TASK_ID]}" "$@"

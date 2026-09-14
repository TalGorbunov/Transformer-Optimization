#!/bin/bash
#SBATCH -J v8_train_profile
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:03:00
#SBATCH --array=0-1%2
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V8_CONDITIONS=(ce consistency)
.venv/bin/python scripts/train_native_vision_v8.py --profile --condition "${V8_CONDITIONS[$SLURM_ARRAY_TASK_ID]}" --seed 10 "$@"

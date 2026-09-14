#!/bin/bash
#SBATCH -J v13_train_profile
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:05:00
#SBATCH --array=0-1%2
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V13_CONDITIONS=(centered offset)
.venv/bin/python scripts/train_native_vision_v13.py --profile --condition "${V13_CONDITIONS[$SLURM_ARRAY_TASK_ID]}" --seed 16 "$@"

#!/bin/bash
#SBATCH -J identity_join_learned_main
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=b200-8x-gpu
#SBATCH --exclude=slurm-b200-8x-gpu-131-255
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH --array=0-5%4
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JOIN_TRAIN_CONDITIONS=(clip sigmoid softmax clip sigmoid softmax)
JOIN_TRAIN_SEEDS=(22 22 22 23 23 23)
.venv/bin/python scripts/train_native_identity_join_learned.py --run --condition "${JOIN_TRAIN_CONDITIONS[$SLURM_ARRAY_TASK_ID]}" --seed "${JOIN_TRAIN_SEEDS[$SLURM_ARRAY_TASK_ID]}" "$@"

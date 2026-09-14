#!/bin/bash
#SBATCH -J v18_main
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=b200-8x-gpu
#SBATCH --exclude=slurm-b200-8x-gpu-131-255
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
V18_CONDITIONS=(native_gate all_open native_gate all_open)
V18_SEEDS=(20 20 21 21)
.venv/bin/python scripts/train_native_vision_v18.py --run --condition "${V18_CONDITIONS[$SLURM_ARRAY_TASK_ID]}" --seed "${V18_SEEDS[$SLURM_ARRAY_TASK_ID]}" "$@"

#!/bin/bash
#SBATCH -J mmred_official_native_evaluation_run
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=b200-8x-gpu
#SBATCH --exclude=slurm-b200-8x-gpu-131-255
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH --array=0-2%3
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/evaluate_mmred_official_native_memory.py --run --arm "${SLURM_ARRAY_TASK_ID:?}" "$@"

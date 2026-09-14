#!/bin/bash
#SBATCH -J identity_join_optimization_profile
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=b200-8x-gpu
#SBATCH --exclude=slurm-b200-8x-gpu-131-255
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:02:30
#SBATCH --array=0-3%4
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PILOT_ARMS=(clip_ce clip_consistency sigmoid_ce sigmoid_consistency)
.venv/bin/python scripts/diagnose_native_identity_join_optimization_v2.py --profile --arm "${PILOT_ARMS[$SLURM_ARRAY_TASK_ID]}" "$@"

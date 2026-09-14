#!/bin/bash
#SBATCH -J learned_memory_software
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=b200-8x-gpu
#SBATCH --exclude=slurm-b200-8x-gpu-131-255
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:10:00
#SBATCH --array=0-1%2
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
models=(qwen cosmos)
.venv/bin/python scripts/profile_native_learned_memory.py --profile --model "${models[${SLURM_ARRAY_TASK_ID:?}]}" "$@"

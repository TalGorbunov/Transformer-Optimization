#!/bin/bash
#SBATCH -J v8_main
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
V8_CONDITIONS=(ce consistency ce consistency)
V8_SEEDS=(10 10 11 11)
.venv/bin/python scripts/train_native_vision_v8.py --run --condition "${V8_CONDITIONS[$SLURM_ARRAY_TASK_ID]}" --seed "${V8_SEEDS[$SLURM_ARRAY_TASK_ID]}" "$@"

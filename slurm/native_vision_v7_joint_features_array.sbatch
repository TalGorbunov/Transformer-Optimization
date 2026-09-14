#!/bin/bash
#SBATCH -J v7_joint_harvest
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:08:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
V7_JOINT_RELEASE="${1:?Pass exact joint harvest release.json}"
shift
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/harvest_native_vision_v7_joint_features.py --release "$V7_JOINT_RELEASE" --shard "${SLURM_ARRAY_TASK_ID:?}" "$@"

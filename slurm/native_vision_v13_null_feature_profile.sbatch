#!/bin/bash
#SBATCH -J v13_null_profile
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:03:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V13_FEATURE_PLAN="${1:?Pass exact CPU worker plan.json}"
shift
.venv/bin/python scripts/cache_native_vision_v13_null_features.py --plan "$V13_FEATURE_PLAN" --profile "$@"

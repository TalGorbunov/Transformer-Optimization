#!/bin/bash
#SBATCH -J mixed_cache_localization
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:03:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LOCALIZE_CACHE_PLAN="${1:?Pass frozen CPU plan.json}"
shift
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/probe_native_vision_mixed_cache_localization.py --run --plan "$LOCALIZE_CACHE_PLAN" "$@"

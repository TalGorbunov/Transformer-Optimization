#!/bin/bash
#SBATCH -J v11_features_merge
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
V11_FEATURE_PLAN="${1:?Pass exact CPU plan.json and four shard directories}"
shift
.venv/bin/python scripts/cache_native_vision_v11_features.py --plan "$V11_FEATURE_PLAN" --merge "$@"

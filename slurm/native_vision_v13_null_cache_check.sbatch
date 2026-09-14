#!/bin/bash
#SBATCH -J v13_null_cache_check
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
V13_REGISTRATION="${1:?Pass exact prospective registration.json}"
shift
.venv/bin/python scripts/cache_native_vision_v13_null_features.py --check --registration "$V13_REGISTRATION" "$@"

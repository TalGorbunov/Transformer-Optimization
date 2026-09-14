#!/bin/bash
#SBATCH -J v7_feature_release
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
V7_RELEASE_PLAN="${1:?Pass frozen parallel plan.json}"
V7_RELEASE_STAGE="${2:?Pass completed parallel CPU stage output directory}"
shift 2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python scripts/release_native_vision_v7_feature_metadata.py --plan "$V7_RELEASE_PLAN" --stage-directory "$V7_RELEASE_STAGE" "$@"

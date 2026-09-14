#!/bin/bash
#SBATCH -J v7_joint_stage
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
V7_PARALLEL_PLAN="${1:?Pass completed parallel feature CPU plan.json}"
shift
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python scripts/stage_native_vision_v7_joint_features.py --parallel-plan "$V7_PARALLEL_PLAN" "$@"

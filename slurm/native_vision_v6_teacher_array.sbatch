#!/bin/bash
#SBATCH -J native_v6_teacher_cache
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:15:00
#SBATCH --array=0-3%4
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
V6_TEACHER_PLAN="${1:?Pass exact CPU plan.json}"
V6_TEACHER_PROFILE="${2:?Pass successful software profile directory}"
shift 2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/cache_native_vision_v6_teacher.py --plan "$V6_TEACHER_PLAN" --shard "${SLURM_ARRAY_TASK_ID:?}" --profile-directory "$V6_TEACHER_PROFILE" "$@"

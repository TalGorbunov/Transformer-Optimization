#!/bin/bash
#SBATCH -J cosmos_long_main
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:55:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LONG_PLAN="${1:?Pass the exact long-budget CPU plan.json}"
LONG_PROFILE="${2:?Pass the completed matching software-profile directory}"
shift 2
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/probe_native_vision_cosmos_long.py --run --plan "$LONG_PLAN" --profile-run "$LONG_PROFILE" "$@"

#!/bin/bash
#SBATCH -J native_v2_prefix
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PREFIX_PLAN="${1:?Pass the exact plan.json emitted by the CPU prefix check}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/probe_native_vision_v2_prefix.py --plan "$PREFIX_PLAN"

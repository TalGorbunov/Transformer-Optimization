#!/bin/bash
#SBATCH -J needles_cot
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
.venv/bin/python outputs/_scratch/dbg/needles_cot.py --model "${MODEL:?}" --arms "${ARMS:?}" --n "${N:-16}" ${EXTRA:-}

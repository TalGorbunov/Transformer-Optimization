#!/bin/bash
#SBATCH -J needles_logz
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:30:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
.venv/bin/python outputs/_scratch/dbg/needles_logz.py --model "${MODEL:?}" --layer "${LAYER:?}" --n "${N:-50}" \
  --regimes "${REGIMES:-H0+HF+HC}" --ks "${KS:-1+2+3+4+6+8+12+16+24+32+48+64}" --output "${OUTPUT:-outputs/needles_logz}"

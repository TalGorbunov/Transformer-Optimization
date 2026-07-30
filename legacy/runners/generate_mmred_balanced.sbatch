#!/bin/bash
#SBATCH --job-name=gen_balanced
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u datasets/mmred/generate_mmred_balanced.py \
  --task "${TASK}" --seq-len 8 --per-count "${PER_COUNT:-120}" --n-chars "${NCHARS:-5}" --rooms "${ROOMS4:-Kitchen,Bathroom,Garden,Office,Bedroom,Park}" \
  --out-root "${OUT_ROOT}"

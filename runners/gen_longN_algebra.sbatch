#!/bin/bash
#SBATCH --job-name=gen_alg
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u datasets/mmred/generate_mmred_balanced.py \
  --task "${TASK:?}" --seq-len "${SEQ_LEN:?}" --per-count "${PER_COUNT:-40}" \
  --n-chars 5 --rooms "Kitchen,Bathroom,Garden,Office,Bedroom,Park" \
  ${COUNTS:+--counts "$COUNTS"} --seed "${SEQ_LEN}" \
  --out-root "data/mmred_longN_${TASK}"

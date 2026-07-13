#!/bin/bash
#SBATCH --job-name=gen_longN
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Q1 long-N MMRED gen (plan 2026-07-08, approved 2026-07-09):
# steps_in_room, crowded 5-char, all 6 park rooms, native 512px render,
# counts = low band 0-8 + spread {12,16,24,32,48,64,96,128} clipped to N.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u datasets/mmred/generate_mmred_balanced.py \
  --task steps_in_room \
  --seq-len "${SEQ_LEN}" \
  --per-count "${PER_COUNT:-30}" \
  --n-chars "${NCHARS:-5}" \
  --rooms "Kitchen,Bathroom,Garden,Office,Bedroom,Park" \
  --counts "${COUNTS}" \
  --seed "${SEED:-${SEQ_LEN}}" \
  --out-root "${OUT_ROOT:-data/mmred_longN_park}"

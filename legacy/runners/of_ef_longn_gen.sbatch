#!/bin/bash
#SBATCH --job-name=ef_gen
#SBATCH --partition=a100-public
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:50:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u datasets/mmred/generate_mmred_balanced.py --task steps_in_room \
  --seq-len "${SEQLEN:?}" --per-count "${PERCOUNT:?}" --counts "${COUNTS:?}" \
  --seed "${GSEED:-7}" --out-root data/mmred_longN_park2

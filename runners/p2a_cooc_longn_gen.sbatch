#!/bin/bash
#SBATCH --job-name=cooc_gen
#SBATCH --partition=a100-public
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:50:00
#SBATCH --output=logs/%x-%j.out
# P2a LOTO prep (p0p2 campaign, 2026-07-23): co-occupancy N=32 eval data — the held-out
# task's length cell (Arm 1). Same generator/defaults as mmred_cooc_balanced (6 park rooms),
# longN count convention (low band 0-8 + spread 12/16/24/32), per-count 23 -> n=299.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u datasets/mmred/generate_mmred_balanced.py --task co_occupancy \
  --seq-len 32 --per-count 23 --counts 0,1,2,3,4,5,6,7,8,12,16,24,32 \
  --seed 7 --out-root data/mmred_cooc_longN

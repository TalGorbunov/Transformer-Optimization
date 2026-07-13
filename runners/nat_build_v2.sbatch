#!/bin/bash
#SBATCH --job-name=nat_v2
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/natural/build_mmred_natural.py --per-cell 50 --seed 1 \
  --out data/mmred_natural_v2 --exclude data/mmred_natural/judge_curation.json

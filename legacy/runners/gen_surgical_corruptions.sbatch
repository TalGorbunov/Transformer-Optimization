#!/bin/bash
#SBATCH --job-name=gen_surgcorrupt
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
TASK="${TASK:?}"; OUT="${OUT:?}"; LIMIT="${LIMIT:-70}"
python -u datasets/mmred/generate_surgical_corruptions_new_tasks.py --task "${TASK}" --out_root "${OUT}" --limit "${LIMIT}"

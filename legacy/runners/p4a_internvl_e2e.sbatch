#!/bin/bash
#SBATCH --job-name=ivl_e2e
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/pipeline/e2e_internvl.py --limit "${LIMIT:-100}" \
  --output outputs/pipeline/e2e_internvl

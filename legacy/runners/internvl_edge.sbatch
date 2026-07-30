#!/bin/bash
#SBATCH --job-name=ivl-edge
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/internvl/edge_cut.py --limit "${LIMIT:-150}" --win-layers "${WIN:-8-19}" --post-layers "${POST:-20-27}" \
  --output "${OUTPUT:-outputs/frame_axis/internvl/edge_cut}"

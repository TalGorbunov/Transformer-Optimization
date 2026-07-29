#!/bin/bash
#SBATCH --job-name=rep_unmix
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/glstm/replica_carrier_probe.py --no-mask \
  --unmix-dir "${UNMIX_DIR:-outputs/ladder/image_longN/unmixer_saved/weights}" \
  --unmix-layer "${UNMIX_LAYER:-16}" \
  --limit "${LIMIT:-300}" \
  --output "${OUTPUT:-outputs/ladder/image_longN/replica_unmix}"

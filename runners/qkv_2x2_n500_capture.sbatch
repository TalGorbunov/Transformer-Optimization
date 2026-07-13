#!/bin/bash
#SBATCH --job-name=qkv500_cap
#SBATCH --partition=l40s-shared,rtx6k-shared,a100-public,l40s-public
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
python -u experiments/glstm/qkv_2x2_capture.py \
  --limit 500 --layers 14,16 --resize 392 --sample-seed 7 \
  --output outputs/ladder/image_longN/qkv_2x2/20260712_n500

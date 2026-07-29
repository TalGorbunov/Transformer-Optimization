#!/bin/bash
#SBATCH --job-name=unmix_save
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/glstm/encoding_unmixer.py \
  --capture "${CAPTURE:-outputs/ladder/image_longN/qkv_2x2/20260712_n500}" \
  --output "${OUTPUT:-outputs/ladder/image_longN/unmixer_saved/$(date +%Y%m%d_%H%M%S)}" \
  --kind mlp --layers "${LAYERS:-16}" \
  --save-dir "${SAVE_DIR:-outputs/ladder/image_longN/unmixer_saved/weights}"

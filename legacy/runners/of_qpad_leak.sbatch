#!/bin/bash
#SBATCH --job-name=qpad_leak
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/glstm/qpad_leak_probe.py \
  --capture "${CAPTURE:-outputs/ladder/image_longN/qkv_2x2/20260712_n500}" \
  --output "${OUTPUT:-outputs/ladder/image_longN/broadcast_gate/qpad_leak/$(date +%Y%m%d_%H%M%S)}"

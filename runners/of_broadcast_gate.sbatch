#!/bin/bash
#SBATCH --job-name=bcast_gate
#SBATCH --partition=a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/glstm/broadcast_gate_probe.py \
  --capture "${CAPTURE:-outputs/ladder/image_longN/qkv_2x2/20260712_n500}" \
  --output "${OUTPUT:-outputs/ladder/image_longN/broadcast_gate/$(date +%Y%m%d_%H%M%S)}" \
  --layer "${LAYER:-16}" \
  --epochs "${EPOCHS:-400}" \
  --arms "${ARMS:-content,qcond}"

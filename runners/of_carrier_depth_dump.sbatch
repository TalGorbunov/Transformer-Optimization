#!/bin/bash
#SBATCH --job-name=carrier_depth
#SBATCH --partition=a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Carrier-state DEPTH profile (2026-07-27): at which layer does the frame's evidence reach the
# carrier in the DEPLOYED L*=12 model? Eval-only on the fmt-caption ckpt, tiny decode budget;
# dumps carrier-row hidden states per layer to carrier_states_cache.pt (gate_tally-compatible).
# LAYERS is hardcoded: `sbatch --export` splits values on commas, so it cannot be passed as env.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LAYERS="2,4,6,8,10,12,14,16,20,24"
python -u experiments/glstm/carrier_layer_lora.py --eval-only \
  --dump-carrier-states "$LAYERS" \
  --ckpt "${CKPT:?set CKPT}" \
  --dirs-file "${DIRS_FILE:?set DIRS_FILE}" \
  --limit "${LIMIT:-150}" \
  --decode-tokens "${DEC:-8}" \
  --dump-decodes 2 \
  --output "${OUTPUT:?set OUTPUT}"

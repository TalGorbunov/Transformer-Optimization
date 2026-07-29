#!/bin/bash
#SBATCH --job-name=fmt_eval
#SBATCH --partition=a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x-%j.out
# Scratchpad FORMAT sweep eval cell (2026-07-22): streaming greedy eval on a dirs-file.
# Env: CKPT, DIRS_FILE, DEC (decode budget), OUTPUT (all required); LIMIT (default 300),
# DUMPS (default 200 — transcripts stream to the log for walltime-kill recovery).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/glstm/carrier_layer_lora.py --eval-only ${EXTRA_FLAGS:-} \
  --ckpt "${CKPT:?set CKPT}" \
  --dirs-file "${DIRS_FILE:?set DIRS_FILE}" \
  --limit "${LIMIT:-300}" \
  --decode-tokens "${DEC:?set DEC}" \
  --dump-decodes "${DUMPS:-200}" \
  --output "${OUTPUT:?set OUTPUT}"

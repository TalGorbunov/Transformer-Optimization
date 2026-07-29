#!/bin/bash
#SBATCH --job-name=trunc_eval
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
# TRUNC campaign eval runner (2026-07-24): carrier_layer_lora --eval-only + truncation
# flags. Env: CKPT, DIRS_FILE, DEC, OUTPUT (required); LIMIT (default 300), DUMPS (200);
# optional flag switches: DROPKV=1, TRUNC_AT=<L>, FAST=1, EXACT=1.
# (No comma-separated --export values — the sbatch comma-split gotcha.)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
FLAGS=()
[[ "${DROPKV:-0}" == "1" ]] && FLAGS+=(--drop-frame-kv)
[[ -n "${TRUNC_AT:-}" ]] && FLAGS+=(--truncate-at "${TRUNC_AT}")
[[ "${FAST:-0}" == "1" ]] && FLAGS+=(--fast-decode)
[[ "${EXACT:-0}" == "1" ]] && FLAGS+=(--exactness-check)
[[ "${CHUNKED:-0}" == "1" ]] && FLAGS+=(--chunked-prefill)
[[ "${VERIFY:-0}" == "1" ]] && FLAGS+=(--verify-chunked)
# DUMP_CS is SPACE-separated layers (sbatch --export comma gotcha); converted here
[[ -n "${DUMP_CS:-}" ]] && FLAGS+=(--dump-carrier-states "${DUMP_CS// /,}")
python -u experiments/glstm/carrier_layer_lora.py --eval-only \
  --ckpt "${CKPT:?set CKPT}" \
  --dirs-file "${DIRS_FILE:?set DIRS_FILE}" \
  --limit "${LIMIT:-300}" \
  --decode-tokens "${DEC:?set DEC}" \
  --dump-decodes "${DUMPS:-200}" \
  --output "${OUTPUT:?set OUTPUT}" \
  "${FLAGS[@]}"

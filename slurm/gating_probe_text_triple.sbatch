#!/bin/bash
#SBATCH --job-name=gate_p0_text
#SBATCH -p a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
# P0 of the gating campaign: G1 ablation on the QwQZh/gated_attention 1B triple.
# Knobs: MODEL_ROOT ROOTS_FILE LIMIT ATTN_N OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/gating/p0_text_triple}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
ROOTS_FILE="${ROOTS_FILE:-slurm/lib/roots_text_tally.txt}"
ROOTS=$(grep -v '^\s*#' "$ROOTS_FILE" | grep -v '^\s*$' | tr '\n' ' ')
run_logged "$OUTPUT" python -u scripts/gating/probe_text_triple.py \
  --model-root "${MODEL_ROOT:?set MODEL_ROOT to the gated_attention snapshot dir}" \
  --roots $ROOTS \
  --limit "${LIMIT:-200}" \
  --attn-samples "${ATTN_N:-6}" \
  --output "$OUTPUT"

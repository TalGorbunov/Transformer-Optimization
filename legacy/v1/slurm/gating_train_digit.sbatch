#!/bin/bash
#SBATCH --job-name=gate_digit
#SBATCH -p l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH -o logs/%x-%j.out
# DIGIT-READOUT gating arm: NO scratchpad anywhere. The model emits the count directly
# from the aggregated state, so there is no transcript and nothing to copy -- this is the
# only readout that actually forces in-model aggregation, and the only one whose accuracy
# can be read as "can the model learn the task".
# --digit-multi trains CE on the answer's DIGIT SEQUENCE (Qwen splits 128 -> '1','2','8'),
# so ANY count works -- no 0-9 cap. Still nothing to copy: the context never contains the
# answer, unlike the caption scratchpad whose running tally made the answer copyable.
# Knobs: GATE GATE_LAYERS GATE_LR GATE_B0 GATE_ONLY ROOTS_FILE LIMIT EPOCHS SEED OUTPUT
# Every partition defaults to a 2h walltime -- always submit with an explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/gating/p7_digit/${GATE:-none}}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
ROOTS_FILE="${ROOTS_FILE:-slurm/lib/roots_inlength.txt}"
ROOTS="$(grep -v '^\s*#' "$ROOTS_FILE" | grep -v '^\s*$' | paste -sd, -)"
run_logged "$OUTPUT" python -u scripts/train_carrier_layer.py \
  --data_root "$ROOTS" \
  --carrier-ckpt "${CARRIER_CKPT:-checkpoints/carrier_token_room_k1_best.pt}" \
  --digit-multi \
  --jitter-gap "${JITTER:-16}" --grad-ckpt \
  --l-open "${L_OPEN:-12}" --limit "${LIMIT:-900}" --epochs "${EPOCHS:-5}" \
  --seed "${SEED:-0}" --split-seed "${SPLIT_SEED:-0}" \
  --gate "${GATE:-none}" --gate-layers "${GATE_LAYERS:-open}" \
  --gate-lr "${GATE_LR:-1e-3}" --gate-b0 "${GATE_B0:-2.0}" \
  ${GATE_ONLY:+--gate-only} \
  --output "$OUTPUT"

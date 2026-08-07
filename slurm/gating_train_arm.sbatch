#!/bin/bash
#SBATCH --job-name=gate_train
#SBATCH -p l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -o logs/%x-%j.out
# P3 of the gating campaign: one ablation arm of the production carrier trainer.
# Same recipe as slurm/train_carrier_layer.sbatch (caption / running-tally / jitter 16 /
# grad-ckpt / L*=12) plus the OFF-by-default gate flags. GATE=none reproduces the anchor.
# Knobs: GATE GATE_LAYERS GATE_LR GATE_B0 GATE_ONLY ROOTS_FILE LIMIT EPOCHS SEED OUTPUT
# NOTE: every partition defaults to a 2h walltime -- always submit with an explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/gating/p3_arms/${GATE:-none}}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
ROOTS_FILE="${ROOTS_FILE:-slurm/lib/roots_inlength.txt}"
ROOTS="$(grep -v '^\s*#' "$ROOTS_FILE" | grep -v '^\s*$' | paste -sd, -)"
run_logged "$OUTPUT" python -u scripts/train_carrier_layer.py \
  --data_root "$ROOTS" \
  --carrier-ckpt "${CARRIER_CKPT:-checkpoints/carrier_token_room_k1_best.pt}" \
  --scratchpad-format "${FMT:-caption}" --running-tally \
  --jitter-gap "${JITTER:-16}" --grad-ckpt \
  --l-open "${L_OPEN:-12}" --limit "${LIMIT:-900}" --epochs "${EPOCHS:-5}" \
  --seed "${SEED:-0}" --split-seed "${SPLIT_SEED:-0}" \
  --gate "${GATE:-none}" --gate-layers "${GATE_LAYERS:-open}" \
  --gate-lr "${GATE_LR:-3e-4}" --gate-b0 "${GATE_B0:-2.0}" \
  ${GATE_ONLY:+--gate-only} \
  --output "$OUTPUT"

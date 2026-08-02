#!/bin/bash
#SBATCH --job-name=train_carrier_layer
#SBATCH -p l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -o logs/%x-%j.out
# THE production trainer (caption recipe defaults; ~14h on a100 at full mixture).
# Knobs: ROOTS (default $ROOTS_INLENGTH from lib/roots_inlength.txt) CARRIER_CKPT FMT
#        LIMIT EPOCHS L_OPEN SEED SPLIT_SEED JITTER OUTPUT
#        ROOTS_FILE (roots list file, comma-joined here — beats ROOTS; use for lists
#        too long / comma-unsafe for --export)  TASK (park|mmred_hf)  EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/carrier/train}"
if [ -n "${ROOTS_FILE:-}" ]; then
    ROOTS="$(grep -v '^\s*#' "$ROOTS_FILE" | grep -v '^\s*$' | paste -sd, -)"
fi
run_logged "$OUTPUT" python -u scripts/train_carrier_layer.py \
  ${TASK:+--task "$TASK"} \
  --data_root "${ROOTS:-$ROOTS_INLENGTH}" \
  --carrier-ckpt "${CARRIER_CKPT:-checkpoints/carrier_token_room_k1_best.pt}" \
  --scratchpad-format "${FMT:-caption}" --running-tally \
  --jitter-gap "${JITTER:-16}" --grad-ckpt \
  --l-open "${L_OPEN:-12}" --limit "${LIMIT:-900}" --epochs "${EPOCHS:-5}" \
  --seed "${SEED:-0}" ${SPLIT_SEED:+--split-seed "$SPLIT_SEED"} ${EXTRA:-} \
  --output "$OUTPUT"

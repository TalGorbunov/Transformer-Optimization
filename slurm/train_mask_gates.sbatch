#!/bin/bash
#SBATCH --job-name=learnmask_gates
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
# LEARNMASK gate trainer (outputs/learnmask/CAMPAIGN_BRIEF.md). The run emits its own
# ep0 handfence reference row + fence-init row + startup engine parity before training.
# Knobs: ARM ESTIMATOR DATA_ROOT EVAL_ROOT LIMIT EVAL_LIMIT EPOCHS LR LAM_OPEN
#        LAM_CLOSE RESIZE CKPT EXTRA OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/learnmask/train}"
# mixed-length roots come from FILES (sbatch --export silently splits comma values)
if [ -n "${ROOTS_FILE:-}" ]; then
    DATA_ROOT="$(grep -v '^\s*#' "$ROOTS_FILE" | grep -v '^\s*$' | paste -sd, -)"
fi
if [ -n "${EVAL_ROOTS_FILE:-}" ]; then
    EVAL_ROOT="$(grep -v '^\s*#' "$EVAL_ROOTS_FILE" | grep -v '^\s*$' | paste -sd, -)"
fi
run_logged "$OUTPUT" python -u scripts/learnmask/train_mask_gates.py \
  --arm "${ARM:-s1}" --estimator "${ESTIMATOR:-st-gumbel}" \
  --gate-init "${GATE_INIT:-fence}" \
  ${S0_SHARE:+--s0-share-layers} \
  --scaffold "${SCAFFOLD:-replica}" --target "${TARGET:-class}" \
  --train "${TRAIN:-gates}" --fixed-regime "${FIXED_REGIME:-s2open}" \
  ${READOUT_CKPT:+--readout-ckpt "$READOUT_CKPT"} \
  --data_root "${DATA_ROOT:-data/mmred_hf/dirs/seq_len_8_train_steps_in_room}" \
  ${EVAL_ROOT:+--eval-root "$EVAL_ROOT"} \
  --limit "${LIMIT:-120}" --eval-limit "${EVAL_LIMIT:-60}" \
  --epochs "${EPOCHS:-8}" --lr-gate "${LR:-3e-2}" --lr-lora "${LR_LORA:-1e-4}" \
  --lam-open "${LAM_OPEN:-1e-2}" --lam-close "${LAM_CLOSE:-0}" \
  --resize "${RESIZE:-512}" \
  ${QTYPES:+--qtype-filter "$QTYPES"} \
  --tail-style "${TAIL:-canonical}" \
  ${EP0:+--ep0-regimes "$EP0"} \
  ${HINT:+--answer-hint "$HINT"} ${PRIME:+--answer-prime "$PRIME"} \
  --carrier-ckpt "${CKPT:-checkpoints/carrier_layer_digit_p7a_lora_best.pt}" \
  --grad-ckpt ${EXTRA:-} \
  --output "$OUTPUT"

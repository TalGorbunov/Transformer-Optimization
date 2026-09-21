#!/bin/bash
#SBATCH --job-name=sp_train
#SBATCH -p rtx6k-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# SPARSE gated fenced-SFT trainer / eval-only runner (4-D mask => EFFICIENT/MATH sdpa).
# Knobs: ROOTS|ROOTS_FILE EPOCHS LIMIT ADAPTER (eval-only) DIRS_FILES EXCLUDES TASKS (space-sep) NFREE=1
#        LONGN_LIMIT GATE (oracle) GATE_FROM SHARPEN (tau) SHARPEN_FROM OUTPUT EXTRA
# Always submit with explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/sparse/s1_gated}"
if [ -n "${ROOTS_FILE:-}" ]; then
  ROOTS="$(grep -v '^\s*#' "$ROOTS_FILE" | grep -v '^\s*$' | paste -sd, -)"
fi
DF_ARGS=""; for f in ${DIRS_FILES:-}; do DF_ARGS="$DF_ARGS --eval-dirs-file $f"; done
EX_ARGS=""; for f in ${EXCLUDES:-}; do EX_ARGS="$EX_ARGS --exclude-dirs-file $f"; done
# TASKS: SPACE-separated task list ("count exists majority"), joined with commas HERE — never put a
# comma list in --export (sbatch splits it; 2026-09-21 it reduced the REDUX multitask trainers to count-only
# and dropped every flag after the comma). NFREE=1 adds --nfree-prompt.
TASKS_ARG=""; if [ -n "${TASKS:-}" ]; then TASKS_ARG="--tasks $(echo ${TASKS} | tr ' ' ',')"; fi
NFREE_ARG=""; if [ "${NFREE:-0}" = "1" ]; then NFREE_ARG="--nfree-prompt"; fi
run_logged "$OUTPUT" python -u scripts/sparse/train_sft_gated.py \
  --data_root "${ROOTS:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-900}" --epochs "${EPOCHS:-5}" \
  ${GATE:+--gate "$GATE"} ${GATE_FROM:+--gate-from-layer "$GATE_FROM"} \
  ${SHARPEN:+--attn-sharpen "$SHARPEN"} \
  ${SHARPEN_FROM:+--sharpen-from-layer "$SHARPEN_FROM"} \
  ${ADAPTER:+--eval-only-adapter "$ADAPTER"} $DF_ARGS $EX_ARGS $TASKS_ARG $NFREE_ARG ${EXTRA:-} \
  --eval-longn-limit "${LONGN_LIMIT:-150}" --output "$OUTPUT"

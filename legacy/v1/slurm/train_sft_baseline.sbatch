#!/bin/bash
#SBATCH --job-name=train_sft
#SBATCH -p h200-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -o logs/%x-%j.out
# Plain-LoRA SFT control. In-length (N>=32) training fits ONLY on h200 (P4.1 lesson).
# Knobs: ROOTS EPOCHS ADAPTER (eval-only) DIRS_FILES (space-separated) LIMIT OUTPUT
#        FRAMES_FIRST=1 (LORAMECH canonical regime) EXCLUDES (space-separated dirs-files)
#        LORA_R (anchor adapters are r=8)
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/carrier/sft}"
# ROOTS_FILE (newline list, # comments) beats ROOTS — comma-lists cannot ride --export.
if [ -n "${ROOTS_FILE:-}" ]; then
  ROOTS="$(grep -v '^\s*#' "$ROOTS_FILE" | grep -v '^\s*$' | paste -sd, -)"
fi
DF_ARGS=""; for f in ${DIRS_FILES:-}; do DF_ARGS="$DF_ARGS --eval-dirs-file $f"; done
EX_ARGS=""; for f in ${EXCLUDES:-}; do EX_ARGS="$EX_ARGS --exclude-dirs-file $f"; done
run_logged "$OUTPUT" python -u scripts/train_sft_baseline.py \
  --data_root "${ROOTS:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-900}" --epochs "${EPOCHS:-5}" \
  ${LORA_R:+--lora-r "$LORA_R"} ${FRAMES_FIRST:+--frames-first} \
  ${ADAPTER:+--eval-only-adapter "$ADAPTER"} $DF_ARGS $EX_ARGS ${EXTRA:-} \
  ${LONGN:+--eval-longn "$LONGN"} --eval-longn-limit "${LONGN_LIMIT:-100}" \
  --output "$OUTPUT"

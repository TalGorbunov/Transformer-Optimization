#!/bin/bash
#SBATCH --job-name=sg_train
#SBATCH -p rtx6k-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# SELFGATE G1 trainer. Knobs: ROOTS_FILE EPOCHS LIMIT TASKS ADAPTER (eval-only)
# DIRS_FILES EXCLUDES LONGN_LIMIT EXTRA OUTPUT. Explicit --time at submit.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:?set OUTPUT}"
if [ -n "${ROOTS_FILE:-}" ]; then
  ROOTS="$(grep -v '^\s*#' "$ROOTS_FILE" | grep -v '^\s*$' | paste -sd, -)"
fi
DF_ARGS=""; for f in ${DIRS_FILES:-}; do DF_ARGS="$DF_ARGS --eval-dirs-file $f"; done
EX_ARGS=""; for f in ${EXCLUDES:-}; do EX_ARGS="$EX_ARGS --exclude-dirs-file $f"; done
run_logged "$OUTPUT" python -u scripts/selfgate/train_sft_selfgate.py \
  --data_root "${ROOTS:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-900}" --epochs "${EPOCHS:-10}" \
  ${TASKS:+--tasks "$TASKS"} ${NFREE:+--nfree-prompt} \
  ${ADAPTER:+--eval-only-adapter "$ADAPTER"} $DF_ARGS $EX_ARGS ${EXTRA:-} \
  --eval-longn-limit "${LONGN_LIMIT:-150}" --output "$OUTPUT"

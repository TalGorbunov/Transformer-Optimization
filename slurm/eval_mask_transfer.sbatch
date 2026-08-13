#!/bin/bash
#SBATCH --job-name=learnmask_transfer
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
# LEARNMASK zero-shot transfer eval. Knobs: DIRS_FILE (one root[=LIMIT] per line —
# NEVER a comma list in --export) or DIRS, GATES, REGIMES, SCAFFOLD, LIMIT, OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/learnmask/transfer}"
if [ -n "${DIRS_FILE:-}" ]; then
    DIRS="$(grep -v '^\s*#' "$DIRS_FILE" | grep -v '^\s*$' | paste -sd, -)"
fi
run_logged "$OUTPUT" python -u scripts/learnmask/eval_mask_transfer.py \
  --dirs "${DIRS:?set DIRS or DIRS_FILE}" \
  ${GATES:+--gates "$GATES"} \
  --regimes "${REGIMES:-hand,init,nofence}" \
  --scaffold "${SCAFFOLD:-replica}" --limit "${LIMIT:-100}" \
  --output "$OUTPUT"

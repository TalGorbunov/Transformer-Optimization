#!/bin/bash
#SBATCH --job-name=redux_evaltasks
#SBATCH -p rtx6k-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# REDUX C1/C3 plain behavioral ladder. Knobs: DIRS_FILES (space-separated) LIMIT
# ADAPTER OUTPUT. Always submit with explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:?set OUTPUT}"
DF_ARGS=""; for f in ${DIRS_FILES:?set DIRS_FILES}; do DF_ARGS="$DF_ARGS --eval-dirs-file $f"; done
run_logged "$OUTPUT" python -u scripts/redux/eval_tasks.py \
  $DF_ARGS ${LIMIT:+--limit "$LIMIT"} ${ADAPTER:+--peft-adapter "$ADAPTER"} \
  --output "$OUTPUT"

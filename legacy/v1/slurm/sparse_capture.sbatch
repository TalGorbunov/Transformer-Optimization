#!/bin/bash
#SBATCH --job-name=sp_capture
#SBATCH -p l40s-shared
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# SPARSE S2 pass-1 capture. Knobs: DIRS_FILES (space-sep) LIMIT LAYER ADAPTER OUTPUT
# Always submit with explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
DF_ARGS=""; for f in ${DIRS_FILES:?set DIRS_FILES}; do DF_ARGS="$DF_ARGS --dirs-file $f"; done
run_logged "${OUTPUT:?set OUTPUT}" python -u scripts/sparse/capture_verdicts.py \
  $DF_ARGS ${LIMIT:+--limit "$LIMIT"} --layer "${LAYER:-20}" \
  ${ADAPTER:+--peft-adapter "$ADAPTER"} ${EXTRA:-} --output "$OUTPUT"

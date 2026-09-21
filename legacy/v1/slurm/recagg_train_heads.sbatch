#!/bin/bash
#SBATCH --job-name=recagg_heads
#SBATCH -p l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH -o logs/%x-%j.out
# RECAGG P0: CPU-only aggregator heads (R1-R4) over an existing leaf-state capture.
# Knobs: NPZ N LABEL OUTPUT EXTRA  (OUTPUT = the run dir itself, timestamped by caller)
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
run_logged "${OUTPUT:?set OUTPUT}" python -u scripts/recagg/train_heads.py \
  --npz "${NPZ:?set NPZ}" --n "${N:-8}" --label "${LABEL:-capture}" \
  ${EXTRA:-} --output "$OUTPUT"

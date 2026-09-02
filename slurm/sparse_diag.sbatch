#!/bin/bash
#SBATCH --job-name=sp_diag
#SBATCH -p l40s-shared
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -o logs/%x-%j.out
# SPARSE per-k gated-loss diagnostic. Knobs: LIMIT ADAPTER EXTRA OUTPUT. Explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
run_logged "${OUTPUT:?}" python -u scripts/sparse/diag_gated_loss.py \
  --limit "${LIMIT:-60}" ${ADAPTER:+--peft-adapter "$ADAPTER"} ${EXTRA:-} --output "$OUTPUT"

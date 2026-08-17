#!/bin/bash
#SBATCH --job-name=recagg_extrap
#SBATCH -p l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH -o logs/%x-%j.out
# RECAGG P2: CPU length-extrapolation (fit @{8,16} -> zero-shot @{32,64}).
# Knobs: FIT EVAL LABEL OUTPUT EXTRA  (FIT/EVAL = space-separated NPZ:N specs)
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
run_logged "${OUTPUT:?set OUTPUT}" python -u scripts/recagg/eval_extrap.py \
  --fit ${FIT:?set FIT} --eval ${EVAL:?set EVAL} \
  --label "${LABEL:-extrap}" ${EXTRA:-} --output "$OUTPUT"

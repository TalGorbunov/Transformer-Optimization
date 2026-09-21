#!/bin/bash
#SBATCH --job-name=armor_sp
#SBATCH -p l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH -o logs/%x-%j.out
# ARMOR Exp B: state-passing control (CPU). Knobs: FITS EVALS LABEL EPOCHS SEEDS OUTPUT EXTRA
# FITS/EVALS are space-separated NPZ:N specs (never comma — sbatch --export splits commas).
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/armor/b_statepass/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/armor/train_heads_sp.py \
  --fit ${FITS:?set FITS} --eval ${EVALS:?set EVALS} \
  --label "${LABEL:-sp}" --epochs "${EPOCHS:-20000}" --seeds "${SEEDS:-5}" \
  ${EXTRA:-} --output "$OUTPUT"

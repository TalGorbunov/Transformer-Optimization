#!/bin/bash
#SBATCH --job-name=ninv_evalreg
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# ninv Phase 1 step-4 eval. Knobs: CKPT OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/ninv/$(date +%Y%m%d_%H%M%S)_evalreg_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/ninv/eval_registers.py \
  --ckpt "$CKPT" ${EXTRA:-} --output "$OUTPUT"

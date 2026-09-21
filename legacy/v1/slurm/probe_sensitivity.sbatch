#!/bin/bash
#SBATCH --job-name=probe_sensitivity
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# Jacobian sensitivity probe (GPU). Knobs: FIT_N GRAD_N LAYER LIMIT_DATA OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/presentation/sensitivity/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/presentation_diagnostics/probe_sensitivity.py \
  --fit-n "${FIT_N:-64}" --grad-n "${GRAD_N:-12}" --layer "${LAYER:-16}" \
  ${EXTRA:-} --output "$OUTPUT"

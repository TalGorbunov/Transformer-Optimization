#!/bin/bash
#SBATCH --job-name=probe_anchor_sweep
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# Anchor-choice sweep on the teacher config (GPU). Knobs: LIMIT LAYER OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/presentation/anchor_sweep/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/presentation_diagnostics/probe_anchor_sweep.py \
  --limit "${LIMIT:-150}" --layer "${LAYER:-16}" ${EXTRA:-} --output "$OUTPUT"

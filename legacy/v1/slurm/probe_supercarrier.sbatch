#!/bin/bash
#SBATCH --job-name=probe_supercarrier
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# Super-carrier probe: one reader over N coarse nodes, N=8..64 in one job. Knobs: READ_LAYERS OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/presentation/supercarrier/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/presentation_diagnostics/probe_supercarrier.py \
  --read-layers "${READ_LAYERS:-16,20}" ${EXTRA:-} --output "$OUTPUT"

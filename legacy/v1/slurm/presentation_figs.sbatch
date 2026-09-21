#!/bin/bash
#SBATCH --job-name=presentation_figs
#SBATCH -p l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -o logs/%x-%j.out
# CPU presentation figures from frozen caches. Knobs: FIG (pca|curves|saturation), OUTPUT, SEEDS
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/presentation/${FIG:?set FIG}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/presentation_diagnostics/presentation_figs.py \
  --fig "${FIG:?set FIG}" --seeds "${SEEDS:-5}" --output "$OUTPUT"

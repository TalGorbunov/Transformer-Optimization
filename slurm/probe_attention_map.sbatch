#!/bin/bash
#SBATCH --job-name=probe_attnmap
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# Attention-map figure (GPU, minutes). Knobs: LIMIT LAYER OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/presentation/attnmap/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/presentation_diagnostics/probe_attention_map.py \
  --limit "${LIMIT:-8}" --layer "${LAYER:-16}" ${EXTRA:-} --output "$OUTPUT"

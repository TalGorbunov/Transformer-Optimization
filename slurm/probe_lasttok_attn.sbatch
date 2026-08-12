#!/bin/bash
#SBATCH --job-name=probe_lasttok_attn
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# Last-token attention rows across layers (plain forward). Knobs: LAYERS OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/presentation/attnmap_lasttok/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/presentation_diagnostics/probe_lasttok_attn.py \
  --layers "${LAYERS:-16 20 24 27}" ${EXTRA:-} --output "$OUTPUT"

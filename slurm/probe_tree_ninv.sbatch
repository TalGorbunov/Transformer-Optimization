#!/bin/bash
#SBATCH --job-name=probe_tree_ninv
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# ninv Phase 0: tree-superquery capture WITH node posreset.
# Knobs: NS READ_LAYERS FIT_LAYERS LIMIT OUTPUT EXTRA  (LIMIT=0 -> the script's per-N table)
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/ninv/$(date +%Y%m%d_%H%M%S)_tree_ninv_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/ninv/probe_tree_ninv.py \
  --ns "${NS:-8,16,32,64}" --read-layers "${READ_LAYERS:-12,16,20,24,27}" \
  --fit-layers "${FIT_LAYERS:-16,20,24,27}" --limit "${LIMIT:-0}" \
  ${EXTRA:-} --output "$OUTPUT"

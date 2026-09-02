#!/bin/bash
#SBATCH --job-name=armor_capture
#SBATCH -p l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# ARMOR B: N=128 leaf capture (probe_tree_ninv via the armor NS-table wrapper).
# Knobs: NS READ_LAYERS FIT_LAYERS LIMIT OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/armor/p1_captures_128/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/armor/capture_leaf.py \
  --ns "${NS:-128}" --read-layers "${READ_LAYERS:-16 20}" \
  --fit-layers "${FIT_LAYERS:-none}" --limit "${LIMIT:-50}" \
  ${EXTRA:-} --output "$OUTPUT"

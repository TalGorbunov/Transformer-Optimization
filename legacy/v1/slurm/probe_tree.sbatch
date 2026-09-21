#!/bin/bash
#SBATCH --job-name=probe_tree
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# Tree-superquery capture+fits (GPU). Knobs: NS READ_LAYERS FIT_LAYERS OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/superquery/$(date +%Y%m%d_%H%M%S)_tree_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/superquery/probe_tree.py \
  --ns "${NS:-8,16,32,64}" --read-layers "${READ_LAYERS:-12,16,20,24,27}" \
  --fit-layers "${FIT_LAYERS:-16,20,24,27}" ${EXTRA:-} --output "$OUTPUT"

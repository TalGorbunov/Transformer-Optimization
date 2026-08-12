#!/bin/bash
#SBATCH --job-name=ninv_quantcap
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# ninv items 2+3: capture with verdict codes at leaf spans. Knobs: NS READ_LAYERS FIT_LAYERS LIMIT OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/ninv/$(date +%Y%m%d_%H%M%S)_quantcap_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/ninv/quant_capture.py \
  --ns "${NS:-8}" --read-layers "${READ_LAYERS:-20}" \
  --fit-layers "${FIT_LAYERS:-none}" --limit "${LIMIT:-0}" \
  ${EXTRA:-} --output "$OUTPUT"

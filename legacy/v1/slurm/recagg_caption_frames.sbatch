#!/bin/bash
#SBATCH --job-name=armB_caption
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# RECAGG Arm B step 1: per-frame VLM captions on the Arm C sample subset.
# Knobs: NS PER_TYPE OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
run_logged "${OUTPUT:?set OUTPUT}" python -u scripts/recagg/caption_frames.py \
  --ns "${NS:-16,32,64,128}" --per-type "${PER_TYPE:-4}" \
  ${EXTRA:-} --output "$OUTPUT"

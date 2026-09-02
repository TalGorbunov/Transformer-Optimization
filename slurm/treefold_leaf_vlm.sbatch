#!/bin/bash
#SBATCH --job-name=tf_leafvlm
#SBATCH -p l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x-%j.out
# TREEFOLD T3 perception: question-conditioned VLM leaf notes (the long pole).
# Knobs: ASKS OUTPUT [NS PER_TYPE LIMIT BATCH EXTRA]. Resumable (partial json).
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
run_logged "${OUTPUT:?set OUTPUT}" python -u scripts/treefold/leaf_vlm.py \
  --asks "${ASKS:?set ASKS}" --ns "${NS:-16-32-64-128}" \
  --per-type "${PER_TYPE:-4}" --batch "${BATCH:-16}" \
  ${LIMIT:+--limit "$LIMIT"} ${EXTRA:-} --output "$OUTPUT"

#!/bin/bash
#SBATCH --job-name=tf_ask
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
# TREEFOLD P0 ASK. Knobs: MODE(zero|few) OUTPUT [NS PER_TYPE LIMIT EXTRA]
# NS uses dashes (16-32-64-128) — never commas in --export.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
run_logged "${OUTPUT:?set OUTPUT}" python -u scripts/treefold/ask.py \
  --mode "${MODE:?set MODE}" --ns "${NS:-16-32-64-128}" \
  --per-type "${PER_TYPE:-8}" ${LIMIT:+--limit "$LIMIT"} ${EXTRA:-} \
  --output "$OUTPUT"

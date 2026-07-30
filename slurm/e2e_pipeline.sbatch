#!/bin/bash
#SBATCH --job-name=e2e_pipeline
#SBATCH -p l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
# E2E tally pipeline. Knobs: MODE SEQ_LENS LIMIT CHUNK_K TRAIN_CACHE OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:?set OUTPUT}"
run_logged "$OUTPUT" python -u scripts/e2e_pipeline.py \
  --mode "${MODE:-retrieve}" --seq-lens "${SEQ_LENS:-8,32,128}" \
  --limit "${LIMIT:-150}" --chunk-k "${CHUNK_K:-8}" \
  --train-cache "${TRAIN_CACHE:-checkpoints/msgcache_joint_N8.pt}" --output "$OUTPUT"

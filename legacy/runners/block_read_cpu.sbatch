#!/bin/bash
#SBATCH --job-name=block_read
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# CPU block-read / locus-completeness on an existing messages_cache.pt.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/glstm/block_read_completeness.py \
  --cache "${CACHE}" \
  --output "${OUTPUT}" \
  ${LAYERS:+--layers "$LAYERS"} ${OFFSETS:+--offsets "$OFFSETS"}

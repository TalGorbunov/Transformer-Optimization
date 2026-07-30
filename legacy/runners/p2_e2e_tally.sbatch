#!/bin/bash
#SBATCH --job-name=e2e_tally
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/pipeline/e2e_tally.py \
  --mode "${MODE:-chunked}" --seq-lens "${SEQ_LENS:-8,16,32,64,128}" \
  --chunk-k "${CHUNK_K:-8}" --limit "${LIMIT:-150}" --calib-n "${CALIB:-30}" \
  --sample-seed "${SEED:-2}" \
  --output "${OUTPUT:-outputs/pipeline/e2e_tally_${MODE:-chunked}}"

#!/bin/bash
#SBATCH --job-name=e2e_alg
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/pipeline/e2e_task_algebra.py --task "${TASK:?}" --limit "${LIMIT:-150}" \
  ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
  --sample-seed 3 --output "${OUTPUT:-outputs/pipeline/e2e_algebra_${TASK}}"

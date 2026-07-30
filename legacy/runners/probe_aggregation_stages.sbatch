#!/bin/bash
#SBATCH --job-name=aggprobe
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u evaluations/scripts/patch_importence/probe_aggregation_stages.py \
  --task "${TASK:-rooms_visited}" \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --limit "${LIMIT:-60}" \
  --layers "${LAYERS:-4,8,12,16,20,24,27}" \
  --output "${OUTPUT:-outputs/probe_aggregation_stages}"

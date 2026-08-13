#!/bin/bash
#SBATCH --job-name=probe_pertask_extract
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u evaluations/scripts/probe_pertask_extraction.py \
  --model-name "${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}" \
  --seq-lens "${SEQ_LENS:-4,6,8}" --max-samples "${MAX_SAMPLES:-90}" ${EXTRA:-} \
  --output "${OUTPUT:-outputs/probe_pertask_extraction}"

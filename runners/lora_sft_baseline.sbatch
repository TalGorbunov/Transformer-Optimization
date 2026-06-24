#!/bin/bash
#SBATCH --job-name=lora_sft
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/glstm/lora_sft_baseline.py \
  --task "${TASK}" --data-root "${DATA_ROOT}" --epochs "${EPOCHS:-20}" \
  ${EXTRA:-} \
  --output "${OUTPUT}"

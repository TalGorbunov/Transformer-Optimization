#!/bin/bash
#SBATCH --job-name=text_pool_sweep
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u evaluations/scripts/probe_text_pooling_sweep.py \
  --task "${TASK}" --data-root "${DATA_ROOT}" --seq-lens "${SEQ_LENS:-8}" \
  --max-samples "${MAX_SAMPLES:-120}" --output "${OUTPUT:-outputs/frame_axis/probes/text_pooling_sweep}"

#!/bin/bash
#SBATCH --job-name=vnb_beh
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/mlvu/eval_ac_behavior.py \
  --data-root data/vnbench_cnt --n-frames "${NFRAMES:-32}" --resize 392 \
  --protocols mcq,open --sample-seed 0 \
  --output "outputs/ladder/vnbench/behavior_N${NFRAMES:-32}"

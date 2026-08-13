#!/bin/bash
#SBATCH --job-name=hb_own
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS=$(date +%Y%m%d_%H%M%S)
python -u experiments/herbench/eval_ac_own_answer.py \
  --data-root "${DATA_ROOT_A:-data/herbench_ac/armA_evidence_only}" \
  --limit "${LIMIT:-0}" \
  --output "${OUTPUT_ROOT:-outputs/herbench/own_answer}/armA_${TS}"
python -u experiments/herbench/eval_ac_own_answer.py \
  --data-root "${DATA_ROOT_B:-data/herbench_ac/armB_ev_fill16}" \
  --limit "${LIMIT:-0}" \
  --output "${OUTPUT_ROOT:-outputs/herbench/own_answer}/armB_${TS}"

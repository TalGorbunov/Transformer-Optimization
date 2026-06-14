#!/bin/bash
#SBATCH --job-name=rooms_visited_base
#SBATCH --partition=h200-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEQ_LENS="${SEQ_LENS:-2,4,6,8}"
MAX_SAMPLES="${MAX_SAMPLES:-120}"
OUTPUT="${OUTPUT:-outputs/eval_mmred_rooms_visited_baseline}"

TASK="${TASK:-rooms_visited}"
python -u evaluations/scripts/eval_mmred_rooms_visited_baseline.py --task "${TASK}" \
  --seq-lens "${SEQ_LENS}" --max-samples "${MAX_SAMPLES}" --output "${OUTPUT}"

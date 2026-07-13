#!/bin/bash
#SBATCH --job-name=b2_gate
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/glstm/gate_calibration_vs_n.py \
  --train-cache "${TRAIN:?}" --eval-caches "$(cat "${EVAL_FILE:?}")" \
  --train-cache-fenced "${TRAINF:-}" \
  --eval-caches-fenced "$( [ -n "${EVALF_FILE:-}" ] && cat "$EVALF_FILE" || echo "" )" \
  --output "${OUTPUT:?}"

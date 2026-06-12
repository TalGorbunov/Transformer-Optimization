#!/usr/bin/env bash
#SBATCH --job-name=mm_carrier_raw_matrix
#SBATCH --partition=l40s-shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tal.gorbunov/projects/Transformer-Optimization}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
cd "${PROJECT_ROOT}"
mkdir -p logs

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

"${PYTHON}" experiments/carrier_probes/run_message_memory_carrier_update_seq8_7b.py \
  --run-baseline-and-layer-local-raw-matrix \
  --run-name "${RUN_NAME}" \
  --message-mode auto \
  --layer-start 14 \
  --layer-end 17 \
  --d-mem 256 \
  ${EXTRA_ARGS}

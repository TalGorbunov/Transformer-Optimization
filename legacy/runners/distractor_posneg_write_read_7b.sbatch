#!/bin/bash
#SBATCH --job-name=posneg_write_read
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -d "${PWD}/evaluations" && -d "${PWD}/models" ]]; then
    REPO_ROOT="${PWD}"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/evaluations" && -d "${SLURM_SUBMIT_DIR}/models" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
  fi
fi

cd "${REPO_ROOT}"
mkdir -p logs

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/distractor_posneg_write_read_adapter_seq8_7b}"
SOURCE_RUN="${SOURCE_RUN:-outputs_oh_man/frame_to_carrier_evidence_sum_probe_seq8_7b_20260521_164621}"
GATE_MODE="${GATE_MODE:-learned}"
STREAMS="${STREAMS:-posneg}"
GATE_SOURCE="${GATE_SOURCE:-write_layer}"
GATE_HARD="${GATE_HARD:-0}"
INIT_STREAMS_FROM="${INIT_STREAMS_FROM:-}"
FREEZE_STREAMS="${FREEZE_STREAMS:-0}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
WRITE_LAYER_START="${WRITE_LAYER_START:-14}"
WRITE_LAYER_END="${WRITE_LAYER_END:-17}"
READ_LAYER_START="${READ_LAYER_START:-20}"
READ_LAYER_END="${READ_LAYER_END:-27}"
ORACLE_MASK_NOISE="${ORACLE_MASK_NOISE:-0.0}"
LAMBDA_CE="${LAMBDA_CE:-1.0}"
LAMBDA_MASK="${LAMBDA_MASK:-1.0}"
LAMBDA_COUNT="${LAMBDA_COUNT:-0.05}"
RUN_NAME="${RUN_NAME:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_ROOT}/slurm"

python -u experiments/distractor/distractor_posneg_write_read_adapter_seq8_7b.py \
  --run-all \
  --source-run "${SOURCE_RUN}" \
  --gate-mode "${GATE_MODE}" \
  --streams "${STREAMS}" \
  --gate-source "${GATE_SOURCE}" \
  $([ "${GATE_HARD}" = "1" ] && echo --gate-hard) \
  $([ -n "${INIT_STREAMS_FROM}" ] && echo --init-streams-from "${INIT_STREAMS_FROM}") \
  $([ "${FREEZE_STREAMS}" = "1" ] && echo --freeze-streams) \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --write-layer-start "${WRITE_LAYER_START}" \
  --write-layer-end "${WRITE_LAYER_END}" \
  --read-layer-start "${READ_LAYER_START}" \
  --read-layer-end "${READ_LAYER_END}" \
  --oracle-mask-noise "${ORACLE_MASK_NOISE}" \
  --lambda-ce "${LAMBDA_CE}" \
  --lambda-mask "${LAMBDA_MASK}" \
  --lambda-count "${LAMBDA_COUNT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  ${EXTRA_ARGS} \
  2>&1 | tee "${OUTPUT_ROOT}/slurm/runner-${SLURM_JOB_ID:-local}.log"

cp "logs/${SLURM_JOB_NAME:-posneg_write_read}-${SLURM_JOB_ID:-local}.out" "${OUTPUT_ROOT}/slurm/" 2>/dev/null || true

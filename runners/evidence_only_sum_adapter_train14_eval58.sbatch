#!/bin/bash
#SBATCH --job-name=sum_adapter_train14_eval58
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/evidence_only_sum_adapter_train14_eval58_7b}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
TRAIN_SEQ_LENS="${TRAIN_SEQ_LENS:-1 2 3 4}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
D_MEM="${D_MEM:-256}"

mkdir -p "${OUTPUT_DIR}"

python -u experiments/evidence_only/evidence_only_sum_evidence_adapter_seq1_8_7b.py \
  --run-all \
  --train-seq-lens ${TRAIN_SEQ_LENS} \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --d-mem "${D_MEM}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/runner-${SLURM_JOB_ID:-local}.log"

cp "logs/${SLURM_JOB_NAME:-sum_adapter_train14_eval58}-${SLURM_JOB_ID:-local}.out" "${OUTPUT_DIR}/" 2>/dev/null || true

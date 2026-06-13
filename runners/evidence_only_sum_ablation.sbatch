#!/bin/bash
#SBATCH --job-name=sum_ablation
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/evidence_only_sum_evidence_adapter_seq1_8_7b}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
TRAIN_SEQ_LENS="${TRAIN_SEQ_LENS:-1 2 3 4}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
D_MEM="${D_MEM:-256}"
LAYER_START="${LAYER_START:-14}"
LAYER_END="${LAYER_END:-17}"
POOL="${POOL:-sum}"
SHARE_WEIGHTS="${SHARE_WEIGHTS:-0}"
INJECT_TOKEN_GROUP="${INJECT_TOKEN_GROUP:-last_token}"
SEED="${SEED:-0}"

mkdir -p "${OUTPUT_DIR}"

python -u experiments/evidence_only/evidence_only_sum_evidence_adapter_seq1_8_7b.py \
  --run-all \
  --train-seq-lens ${TRAIN_SEQ_LENS} \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --d-mem "${D_MEM}" \
  --layer-start "${LAYER_START}" \
  --layer-end "${LAYER_END}" \
  --pool "${POOL}" \
  --seed "${SEED}" \
  --inject-token-group "${INJECT_TOKEN_GROUP}" \
  $([ "${SHARE_WEIGHTS}" = "1" ] && echo --share-weights) \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/runner-${SLURM_JOB_ID:-local}.log"

cp "logs/${SLURM_JOB_NAME:-sum_ablation}-${SLURM_JOB_ID:-local}.out" "${OUTPUT_DIR}/" 2>/dev/null || true

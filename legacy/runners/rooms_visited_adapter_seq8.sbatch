#!/bin/bash
#SBATCH --job-name=rv_adapter
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314

set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -d "${PWD}/evaluations" && -d "${PWD}/models" ]]; then
    REPO_ROOT="${PWD}"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/evaluations" ]]; then
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

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rooms_visited_adapter}"
SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-data/mmred_images_park}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
VARIANTS="${VARIANTS:-carrier_direct_sum}"
LAYERS="${LAYERS:-14,15,16,17}"
SEQ_LEN="${SEQ_LEN:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
PER_LABEL="${PER_LABEL:-60}"
RUN_PREFIX="${RUN_PREFIX:-rv}"
TASK="${TASK:-rooms_visited}"
SMOKE="${SMOKE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_ROOT}/slurm"

SMOKE_FLAG=""
if [[ "${SMOKE}" == "1" ]]; then SMOKE_FLAG="--smoke-test"; fi

python -u experiments/glstm/layerwise_frame_message_glstm.py \
  --task "${TASK}" \
  --model-name "${MODEL_NAME}" \
  --source-dataset-root "${SOURCE_DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --variants "${VARIANTS}" \
  --layers "${LAYERS}" \
  --train-lengths "${SEQ_LEN}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --rooms-visited-per-label "${PER_LABEL}" \
  --candidate-min 0 --candidate-max 8 \
  --run-prefix "${RUN_PREFIX}" \
  --no-probes --no-plots \
  ${SMOKE_FLAG} \
  ${EXTRA_ARGS} \
  2>&1 | tee "${OUTPUT_ROOT}/slurm/runner-${SLURM_JOB_ID:-local}.log"

cp "logs/${SLURM_JOB_NAME:-rv_adapter}-${SLURM_JOB_ID:-local}.out" "${OUTPUT_ROOT}/slurm/" 2>/dev/null || true

#!/bin/bash
#SBATCH --job-name=glstm_train14_ood58
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
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

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/layerwise_glstm_train14_ood58_7b}"
SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-data/mmred_images_park}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
VARIANTS="${VARIANTS:-carrier_glstm_layerwise}"
LAYERS="${LAYERS:-14,15,16,17}"
TRAIN_LENGTHS="${TRAIN_LENGTHS:-1,2,3,4}"
LENGTH_OOD_LENGTHS="${LENGTH_OOD_LENGTHS:-5,6,7,8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
TRAIN_PER_COUNT="${TRAIN_PER_COUNT:-50}"

mkdir -p "${OUTPUT_ROOT}/slurm"

python -u experiments/glstm/layerwise_frame_message_glstm.py \
  --model-name "${MODEL_NAME}" \
  --source-dataset-root "${SOURCE_DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --variants "${VARIANTS}" \
  --layers "${LAYERS}" \
  --train-lengths "${TRAIN_LENGTHS}" \
  --length-ood-lengths "${LENGTH_OOD_LENGTHS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --train-per-count "${TRAIN_PER_COUNT}" \
  2>&1 | tee "${OUTPUT_ROOT}/slurm/runner-${SLURM_JOB_ID:-local}.log"

cp "logs/${SLURM_JOB_NAME:-glstm_train14_ood58}-${SLURM_JOB_ID:-local}.out" "${OUTPUT_ROOT}/slurm/" 2>/dev/null || true

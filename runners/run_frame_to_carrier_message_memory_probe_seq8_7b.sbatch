#!/bin/bash
#SBATCH --job-name=frame-msg-7b
#SBATCH --mem=128G
#SBATCH -p l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/tal.gorbunov/projects/Transformer-Optimization
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

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/mmred_images_park}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/frame_to_carrier_message_memory_probe_seq8_7b_${RUN_NAME}}"

SEQ_LEN="${SEQ_LEN:-8}"
SPLIT="${SPLIT:-all_uniform}"
LAYERS="${LAYERS:-14 15 16 17}"
CARRIER_GROUPS="${CARRIER_GROUPS:-question_tokens last_token target_char_room_tokens}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LAMBDA_COUNT="${LAMBDA_COUNT:-1.0}"
LR="${LR:-1e-3}"
DROPOUT="${DROPOUT:-0.1}"
PROBE_FAMILIES="${PROBE_FAMILIES:-linear mlp}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
FORCE_RECACHE="${FORCE_RECACHE:-0}"
NO_PLOTS="${NO_PLOTS:-0}"
INCLUDE_DIRECT_COUNT_PROBE="${INCLUDE_DIRECT_COUNT_PROBE:-1}"

mkdir -p "${OUTPUT_DIR}"
SLURM_LOG_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME:-frame-msg-7b}-${SLURM_JOB_ID:-local}.out"
copy_slurm_log() {
  if [[ -f "${SLURM_LOG_PATH}" ]]; then
    cp "${SLURM_LOG_PATH}" "${OUTPUT_DIR}/"
  fi
}
trap copy_slurm_log EXIT
exec > >(tee -a "${OUTPUT_DIR}/runner-${SLURM_JOB_ID:-local}.log") 2>&1

read -ra LAYER_ARRAY <<< "${LAYERS//,/ }"
read -ra CARRIER_GROUP_ARRAY <<< "${CARRIER_GROUPS//,/ }"

CMD=(
  python -u scripts/probes/run_frame_to_carrier_message_memory_probe.py
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --seq-len "${SEQ_LEN}"
  --layers "${LAYER_ARRAY[@]}"
  --carrier-groups "${CARRIER_GROUP_ARRAY[@]}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --hidden-dim "${HIDDEN_DIM}"
  --lambda-count "${LAMBDA_COUNT}"
  --lr "${LR}"
  --dropout "${DROPOUT}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --seed "${SEED}"
)

if [[ "${LOAD_IN_4BIT}" == "1" || "${LOAD_IN_4BIT}" == "true" || "${LOAD_IN_4BIT}" == "yes" ]]; then
  CMD+=(--load-in-4bit)
fi

if [[ -n "${MAX_PIXELS}" ]]; then
  CMD+=(--max-pixels "${MAX_PIXELS}")
fi

if [[ -n "${MIN_PIXELS}" ]]; then
  CMD+=(--min-pixels "${MIN_PIXELS}")
fi

if [[ "${FORCE_RECACHE}" == "1" || "${FORCE_RECACHE}" == "true" || "${FORCE_RECACHE}" == "yes" ]]; then
  CMD+=(--force-recache)
fi

if [[ "${NO_PLOTS}" == "1" || "${NO_PLOTS}" == "true" || "${NO_PLOTS}" == "yes" ]]; then
  CMD+=(--no-plots)
fi

if [[ "${INCLUDE_DIRECT_COUNT_PROBE}" == "0" || "${INCLUDE_DIRECT_COUNT_PROBE}" == "false" || "${INCLUDE_DIRECT_COUNT_PROBE}" == "no" ]]; then
  CMD+=(--no-direct-count-probe)
fi

printf 'Output dir: %s\n' "${OUTPUT_DIR}"
printf 'Probe families: %s\n' "${PROBE_FAMILIES}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"
echo "Finished frame-to-carrier message-memory probe run: ${OUTPUT_DIR}"

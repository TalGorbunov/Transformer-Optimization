#!/bin/bash
#SBATCH --job-name=glstm-mem-7b
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
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs_kitkat/glstm_memory_adapter_7b_seq8/${RUN_NAME}}"

SEQ_LEN="${SEQ_LEN:-8}"
LAYERS="${LAYERS:-14 15 16 17}"
TARGET_MODE="${TARGET_MODE:-both}"
D_MEM="${D_MEM:-64}"
TRAIN_SAMPLES_PER_COUNT="${TRAIN_SAMPLES_PER_COUNT:-40}"
VAL_SAMPLES_PER_COUNT="${VAL_SAMPLES_PER_COUNT:-15}"
TEST_SAMPLES_PER_COUNT="${TEST_SAMPLES_PER_COUNT:-15}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
ALPHA_INIT="${ALPHA_INIT:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
NO_PLOTS="${NO_PLOTS:-0}"

mkdir -p "${OUTPUT_DIR}"
SLURM_LOG_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME:-glstm-mem-7b}-${SLURM_JOB_ID:-local}.out"
copy_slurm_log() {
  if [[ -f "${SLURM_LOG_PATH}" ]]; then
    cp "${SLURM_LOG_PATH}" "${OUTPUT_DIR}/"
  fi
}
trap copy_slurm_log EXIT
exec > >(tee -a "${OUTPUT_DIR}/runner-${SLURM_JOB_ID:-local}.log") 2>&1

read -ra LAYER_ARRAY <<< "${LAYERS//,/ }"

CMD=(
  python -u experiments/train_memory_adapter_frozen_qwen.py
  --model_name "${MODEL_NAME}"
  --dataset_root "${DATASET_ROOT}"
  --seq_len "${SEQ_LEN}"
  --layers "${LAYER_ARRAY[@]}"
  --target_mode "${TARGET_MODE}"
  --d_mem "${D_MEM}"
  --output_dir "${OUTPUT_DIR}"
  --train_samples_per_count "${TRAIN_SAMPLES_PER_COUNT}"
  --val_samples_per_count "${VAL_SAMPLES_PER_COUNT}"
  --test_samples_per_count "${TEST_SAMPLES_PER_COUNT}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --alpha_init "${ALPHA_INIT}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip "${GRAD_CLIP}"
)

if [[ "${LOAD_IN_4BIT}" == "1" || "${LOAD_IN_4BIT}" == "true" || "${LOAD_IN_4BIT}" == "yes" ]]; then
  CMD+=(--load_in_4bit)
else
  CMD+=(--no-load_in_4bit)
fi

if [[ -n "${MAX_PIXELS}" ]]; then
  CMD+=(--max_pixels "${MAX_PIXELS}")
fi

if [[ -n "${MIN_PIXELS}" ]]; then
  CMD+=(--min_pixels "${MIN_PIXELS}")
fi

if [[ "${NO_PLOTS}" == "1" || "${NO_PLOTS}" == "true" || "${NO_PLOTS}" == "yes" ]]; then
  CMD+=(--no_plots)
fi

printf 'Output dir: %s\n' "${OUTPUT_DIR}"
printf 'Conditions: base memory_rc memory_q\n'
printf 'Layers: %s\n' "${LAYERS}"
printf 'Train/val/test per count: %s/%s/%s\n' "${TRAIN_SAMPLES_PER_COUNT}" "${VAL_SAMPLES_PER_COUNT}" "${TEST_SAMPLES_PER_COUNT}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"
echo "Finished gLSTM additive memory adapter run: ${OUTPUT_DIR}"

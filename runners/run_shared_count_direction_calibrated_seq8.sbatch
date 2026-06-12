#!/bin/bash
#SBATCH --job-name=shared-cal-count-7b
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
SOURCE_RUN="${SOURCE_RUN:-${REPO_ROOT}/outputs/frame_to_carrier_evidence_sum_probe_seq8_7b_20260521_164621}"
BASE_SOURCE_RUN="${BASE_SOURCE_RUN:-${REPO_ROOT}/outputs/frame_to_carrier_message_memory_probe_seq8_7b_multilayer_20260521_154136}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/shared_count_direction_calibrated_seq8_7b_${RUN_NAME}}"

SEQ_LEN="${SEQ_LEN:-8}"
SPLIT="${SPLIT:-all_uniform}"
EVIDENCE_COUNTS="${EVIDENCE_COUNTS:-0 1 2 3 4 5 6 7 8}"
LAYERS="${LAYERS:-14 15 16 17}"
CARRIERS="${CARRIERS:-target_char target_room}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-100}"
SEED="${SEED:-0}"

BOTTLENECK_DIM="${BOTTLENECK_DIM:-256}"
KEY_DIM="${KEY_DIM:-64}"
VALUE_DIM="${VALUE_DIM:-128}"
DROPOUT="${DROPOUT:-0.1}"
FRAME_GATE_BCE_WEIGHT="${FRAME_GATE_BCE_WEIGHT:-0.1}"
COUNT_DIRECTION_MSE_WEIGHT="${COUNT_DIRECTION_MSE_WEIGHT:-0.5}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.1}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE1_LR="${STAGE1_LR:-1e-3}"
STAGE1_WEIGHT_DECAY="${STAGE1_WEIGHT_DECAY:-1e-2}"
STAGE1_PATIENCE="${STAGE1_PATIENCE:-3}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-64}"

STAGE3_EPOCHS="${STAGE3_EPOCHS:-3}"
STAGE3_LR="${STAGE3_LR:-1e-4}"
STAGE3_WEIGHT_DECAY="${STAGE3_WEIGHT_DECAY:-1e-2}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-1}"
STAGE3_GRAD_ACCUM="${STAGE3_GRAD_ACCUM:-8}"
STAGE3_PATIENCE="${STAGE3_PATIENCE:-3}"
STAGE3_GRAD_CLIP="${STAGE3_GRAD_CLIP:-1.0}"
AUX_MEM_CE_WEIGHT="${AUX_MEM_CE_WEIGHT:-0.2}"
COUNT_DIRECTION_MSE_WEIGHT_STAGE3="${COUNT_DIRECTION_MSE_WEIGHT_STAGE3:-0.1}"
RESIDUAL_L2_WEIGHT="${RESIDUAL_L2_WEIGHT:-1e-5}"
LAMBDA_COUNTS="${LAMBDA_COUNTS:-0.05 0.1 0.2}"
LAMBDA_RES="${LAMBDA_RES:-0.001 0.01}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
INJECT_LAYER="${INJECT_LAYER:-18}"
GAMMA_INIT="${GAMMA_INIT:-1.0}"
TRAIN_GAMMA="${TRAIN_GAMMA:-1}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
NO_PLOTS="${NO_PLOTS:-0}"

mkdir -p "${OUTPUT_DIR}"
SLURM_LOG_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME:-shared-cal-count-7b}-${SLURM_JOB_ID:-local}.out"
copy_slurm_log() {
  if [[ -f "${SLURM_LOG_PATH}" ]]; then
    cp "${SLURM_LOG_PATH}" "${OUTPUT_DIR}/"
  fi
}
trap copy_slurm_log EXIT
exec > >(tee -a "${OUTPUT_DIR}/runner-${SLURM_JOB_ID:-local}.log") 2>&1

read -ra EVIDENCE_COUNT_ARRAY <<< "${EVIDENCE_COUNTS//,/ }"
read -ra LAYER_ARRAY <<< "${LAYERS//,/ }"
read -ra CARRIER_ARRAY <<< "${CARRIERS//,/ }"
read -ra LAMBDA_COUNT_ARRAY <<< "${LAMBDA_COUNTS//,/ }"
read -ra LAMBDA_RES_ARRAY <<< "${LAMBDA_RES//,/ }"

SCRIPT_PATH="${REPO_ROOT}/scripts/probes/run_shared_count_direction_calibrated_seq8.py"

CMD=(
  python -u "${SCRIPT_PATH}"
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --source-run "${SOURCE_RUN}"
  --base-source-run "${BASE_SOURCE_RUN}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --seq-len "${SEQ_LEN}"
  --evidence-counts "${EVIDENCE_COUNT_ARRAY[@]}"
  --layers "${LAYER_ARRAY[@]}"
  --carriers "${CARRIER_ARRAY[@]}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --candidate-min 0
  --candidate-max 8
  --seed "${SEED}"
  --bottleneck-dim "${BOTTLENECK_DIM}"
  --key-dim "${KEY_DIM}"
  --value-dim "${VALUE_DIM}"
  --dropout "${DROPOUT}"
  --frame-gate-bce-weight "${FRAME_GATE_BCE_WEIGHT}"
  --count-direction-mse-weight "${COUNT_DIRECTION_MSE_WEIGHT}"
  --residual-scale "${RESIDUAL_SCALE}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage1-lr "${STAGE1_LR}"
  --stage1-weight-decay "${STAGE1_WEIGHT_DECAY}"
  --stage1-patience "${STAGE1_PATIENCE}"
  --stage1-batch-size "${STAGE1_BATCH_SIZE}"
  --stage3-epochs "${STAGE3_EPOCHS}"
  --stage3-lr "${STAGE3_LR}"
  --stage3-weight-decay "${STAGE3_WEIGHT_DECAY}"
  --stage3-batch-size "${STAGE3_BATCH_SIZE}"
  --stage3-grad-accum "${STAGE3_GRAD_ACCUM}"
  --stage3-patience "${STAGE3_PATIENCE}"
  --stage3-grad-clip "${STAGE3_GRAD_CLIP}"
  --aux-mem-ce-weight "${AUX_MEM_CE_WEIGHT}"
  --count-direction-mse-weight-stage3 "${COUNT_DIRECTION_MSE_WEIGHT_STAGE3}"
  --residual-l2-weight "${RESIDUAL_L2_WEIGHT}"
  --lambda-counts "${LAMBDA_COUNT_ARRAY[@]}"
  --lambda-res "${LAMBDA_RES_ARRAY[@]}"
  --huber-delta "${HUBER_DELTA}"
  --inject-layer "${INJECT_LAYER}"
  --gamma-init "${GAMMA_INIT}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
)

if [[ "${TRAIN_GAMMA}" == "0" || "${TRAIN_GAMMA}" == "false" || "${TRAIN_GAMMA}" == "no" ]]; then
  CMD+=(--no-train-gamma)
else
  CMD+=(--train-gamma)
fi

if [[ "${LOAD_IN_4BIT}" == "1" || "${LOAD_IN_4BIT}" == "true" || "${LOAD_IN_4BIT}" == "yes" ]]; then
  CMD+=(--load-in-4bit)
else
  CMD+=(--no-load-in-4bit)
fi

if [[ "${NO_PLOTS}" == "1" || "${NO_PLOTS}" == "true" || "${NO_PLOTS}" == "yes" ]]; then
  CMD+=(--no-plots)
fi

if [[ -n "${MAX_PIXELS}" ]]; then
  CMD+=(--max-pixels "${MAX_PIXELS}")
fi

if [[ -n "${MIN_PIXELS}" ]]; then
  CMD+=(--min-pixels "${MIN_PIXELS}")
fi

printf 'Script path: %s\n' "${SCRIPT_PATH}"
printf 'Runner path: %s\n' "${REPO_ROOT}/runners/run_shared_count_direction_calibrated_seq8.sbatch"
printf 'Output dir: %s\n' "${OUTPUT_DIR}"
printf 'Source run: %s\n' "${SOURCE_RUN}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"
echo "Finished shared count-direction calibrated run: ${OUTPUT_DIR}"

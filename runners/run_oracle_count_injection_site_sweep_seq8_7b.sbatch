#!/bin/bash
#SBATCH --job-name=oracle-count-inj-7b
#SBATCH --mem=128G
#SBATCH -p l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%2
#SBATCH --chdir=/home/tal.gorbunov/projects/Transformer-Optimization
#SBATCH --output=logs/oracle_count_injection_site_sweep_seq8_7b/%x-%A_%a.out

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
mkdir -p logs/oracle_count_injection_site_sweep_seq8_7b

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/mmred_images_park}"
SOURCE_RUN="${SOURCE_RUN:-${REPO_ROOT}/outputs/frame_to_carrier_evidence_sum_probe_seq8_7b_20260521_164621}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/oracle_count_injection_site_sweep_seq8_7b}"

SEQ_LEN="${SEQ_LEN:-8}"
SPLIT="${SPLIT:-all_uniform}"
EVIDENCE_COUNTS="${EVIDENCE_COUNTS:-0 1 2 3 4 5 6 7 8}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-100}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-360}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-135}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
ALPHA="${ALPHA:-1.0}"
MARGIN="${MARGIN:-1.0}"
LAMBDA_MARGIN="${LAMBDA_MARGIN:-0.2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
SEED="${SEED:-0}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
NO_PLOTS="${NO_PLOTS:-0}"
INCLUDE_RANDOM_CONTROL="${INCLUDE_RANDOM_CONTROL:-1}"

CONFIG_NAMES=(
  "baseline_no_injection"
  "layer_specific_room_char_l14_17"
  "layer_specific_room_char_l12_18"
  "layer_specific_all_question_tokens_l18_24"
  "layer_specific_last_token_l20_28"
  "layer_specific_all_question_tokens_last_token_l18_24"
  "layer_specific_all_question_tokens_last_token_l20_28"
  "layer_specific_room_char_all_question_tokens_last_token_l14_24"
)
TOKEN_GROUPS=(
  "none"
  "room_char"
  "room_char"
  "all_question_tokens"
  "last_token"
  "all_question_tokens+last_token"
  "all_question_tokens+last_token"
  "room_char+all_question_tokens+last_token"
)
LAYER_STARTS=(0 14 12 18 20 18 20 14)
LAYER_ENDS=(0 17 18 24 28 24 28 24)
CODEBOOK_TYPES=(
  "none"
  "layer_specific"
  "layer_specific"
  "layer_specific"
  "layer_specific"
  "layer_specific"
  "layer_specific"
  "layer_specific"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG_NAME="${CONFIG_NAMES[${TASK_ID}]}"
TOKEN_GROUP="${TOKEN_GROUPS[${TASK_ID}]}"
LAYER_START="${LAYER_STARTS[${TASK_ID}]}"
LAYER_END="${LAYER_ENDS[${TASK_ID}]}"
CODEBOOK_TYPE="${CODEBOOK_TYPES[${TASK_ID}]}"

read -ra EVIDENCE_COUNT_ARRAY <<< "${EVIDENCE_COUNTS//,/ }"

SCRIPT_PATH="${REPO_ROOT}/scripts/experiments/oracle_count_injection_site_sweep_seq8_7b.py"

CMD=(
  python -u "${SCRIPT_PATH}"
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --source-run "${SOURCE_RUN}"
  --seq-len "${SEQ_LEN}"
  --split "${SPLIT}"
  --output-root "${OUTPUT_ROOT}"
  --config-name "${CONFIG_NAME}"
  --token-group "${TOKEN_GROUP}"
  --layer-start "${LAYER_START}"
  --layer-end "${LAYER_END}"
  --codebook-type "${CODEBOOK_TYPE}"
  --alpha "${ALPHA}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --max-train-samples "${MAX_TRAIN_SAMPLES}"
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --evidence-counts "${EVIDENCE_COUNT_ARRAY[@]}"
  --candidate-min 0
  --candidate-max 8
  --margin "${MARGIN}"
  --lambda-margin "${LAMBDA_MARGIN}"
  --weight-decay "${WEIGHT_DECAY}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
)

if [[ "${LOAD_IN_4BIT}" == "1" || "${LOAD_IN_4BIT}" == "true" || "${LOAD_IN_4BIT}" == "yes" ]]; then
  CMD+=(--load-in-4bit)
else
  CMD+=(--no-load-in-4bit)
fi

if [[ "${INCLUDE_RANDOM_CONTROL}" == "0" || "${INCLUDE_RANDOM_CONTROL}" == "false" || "${INCLUDE_RANDOM_CONTROL}" == "no" ]]; then
  CMD+=(--no-include-random-control)
else
  CMD+=(--include-random-control)
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
printf 'Runner path: %s\n' "${REPO_ROOT}/runners/run_oracle_count_injection_site_sweep_seq8_7b.sbatch"
printf 'Output root: %s\n' "${OUTPUT_ROOT}"
printf 'Array task: %s\n' "${TASK_ID}"
printf 'Config: %s token_group=%s layers=%s-%s codebook=%s\n' "${CONFIG_NAME}" "${TOKEN_GROUP}" "${LAYER_START}" "${LAYER_END}" "${CODEBOOK_TYPE}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"
echo "Finished oracle count injection config: ${CONFIG_NAME}"

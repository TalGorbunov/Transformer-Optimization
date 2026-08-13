#!/bin/bash
#SBATCH --job-name=oracle-count-inject-7b
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
PREVIOUS_SHARED_RUN="${PREVIOUS_SHARED_RUN:-${REPO_ROOT}/outputs/shared_count_direction_memory_seq8_7b_20260527_203756}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/oracle_count_multilayer_injection}"
RUN_NAME="${RUN_NAME:-oracle_count_multilayer_injection_seq8_7b}"

SEQ_LEN="${SEQ_LEN:-8}"
SPLIT="${SPLIT:-all_uniform}"
EVIDENCE_COUNTS="${EVIDENCE_COUNTS:-0 1 2 3 4 5 6 7 8}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-100}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
SEED="${SEED:-0}"

EPOCHS="${EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"

ALPHAS="${ALPHAS:-0.5 1.0 2.0}"
TRAIN_ALPHA="${TRAIN_ALPHA:-1.0}"
TOKEN_GROUPS="${TOKEN_GROUPS:-room_char all_question_tokens last_token question_plus_last room_char_question_last}"
LAYER_WINDOWS="${LAYER_WINDOWS:-14-17 18-21 18-24 20-28 14-24}"
PARAM_TYPES="${PARAM_TYPES:-layer_specific_count_codebook static_count_codebook}"
MARGIN_LAMBDA="${MARGIN_LAMBDA:-0.0}"
MARGIN="${MARGIN:-1.0}"
RESIDUAL_L2="${RESIDUAL_L2:-1e-4}"
REFT_RANK="${REFT_RANK:-8}"
FULL_GRID="${FULL_GRID:-0}"
NORMALIZE_INJECTION_ENERGY="${NORMALIZE_INJECTION_ENERGY:-0}"
INCLUDE_OLD_ORACLE_BASELINE="${INCLUDE_OLD_ORACLE_BASELINE:-1}"
QWEN_CODEBOOK_INIT_NORM="${QWEN_CODEBOOK_INIT_NORM:-}"
SMOKE="${SMOKE:-0}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
NO_PLOTS="${NO_PLOTS:-0}"

if [[ "${SMOKE}" == "1" || "${SMOKE}" == "true" || "${SMOKE}" == "yes" ]]; then
  RUN_NAME="${RUN_NAME:-oracle_count_multilayer_injection_seq8_7b}"
  SAMPLE_LIMIT="${SAMPLE_LIMIT:-2}"
  EPOCHS="${EPOCHS:-1}"
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2}"
  MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2}"
  TOKEN_GROUPS="${TOKEN_GROUPS:-room_char}"
  LAYER_WINDOWS="${LAYER_WINDOWS:-14-17}"
  ALPHAS="${ALPHAS:-1.0}"
  PARAM_TYPES="${PARAM_TYPES:-layer_specific_count_codebook}"
  MARGIN_LAMBDA="${MARGIN_LAMBDA:-0.0}"
fi

mkdir -p "${OUTPUT_ROOT}"

read -ra EVIDENCE_COUNT_ARRAY <<< "${EVIDENCE_COUNTS//,/ }"
read -ra ALPHA_ARRAY <<< "${ALPHAS//,/ }"
read -ra TOKEN_GROUP_ARRAY <<< "${TOKEN_GROUPS//,/ }"
read -ra LAYER_WINDOW_ARRAY <<< "${LAYER_WINDOWS//,/ }"
read -ra PARAM_TYPE_ARRAY <<< "${PARAM_TYPES//,/ }"
read -ra MARGIN_LAMBDA_ARRAY <<< "${MARGIN_LAMBDA//,/ }"

SCRIPT_PATH="${REPO_ROOT}/experiments/oracle_bounds/run_oracle_count_multilayer_injection_seq8.py"

CMD=(
  python -u "${SCRIPT_PATH}"
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --source-run "${SOURCE_RUN}"
  --previous-shared-run "${PREVIOUS_SHARED_RUN}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --split "${SPLIT}"
  --seq-len "${SEQ_LEN}"
  --evidence-counts "${EVIDENCE_COUNT_ARRAY[@]}"
  --sample-limit "${SAMPLE_LIMIT}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --candidate-min 0
  --candidate-max 8
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip "${GRAD_CLIP}"
  --max-train-steps "${MAX_TRAIN_STEPS}"
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
  --alphas "${ALPHA_ARRAY[@]}"
  --train-alpha "${TRAIN_ALPHA}"
  --token-groups "${TOKEN_GROUP_ARRAY[@]}"
  --layer-windows "${LAYER_WINDOW_ARRAY[@]}"
  --param-types "${PARAM_TYPE_ARRAY[@]}"
  --margin-lambda "${MARGIN_LAMBDA_ARRAY[@]}"
  --margin "${MARGIN}"
  --residual-l2 "${RESIDUAL_L2}"
  --reft-rank "${REFT_RANK}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
)

if [[ "${SMOKE}" == "1" || "${SMOKE}" == "true" || "${SMOKE}" == "yes" ]]; then
  CMD+=(--smoke)
fi

if [[ "${FULL_GRID}" == "1" || "${FULL_GRID}" == "true" || "${FULL_GRID}" == "yes" ]]; then
  CMD+=(--full-grid)
fi

if [[ "${NORMALIZE_INJECTION_ENERGY}" == "1" || "${NORMALIZE_INJECTION_ENERGY}" == "true" || "${NORMALIZE_INJECTION_ENERGY}" == "yes" ]]; then
  CMD+=(--normalize-injection-energy)
else
  CMD+=(--no-normalize-injection-energy)
fi

if [[ "${INCLUDE_OLD_ORACLE_BASELINE}" == "0" || "${INCLUDE_OLD_ORACLE_BASELINE}" == "false" || "${INCLUDE_OLD_ORACLE_BASELINE}" == "no" ]]; then
  CMD+=(--no-include-old-oracle-baseline)
else
  CMD+=(--include-old-oracle-baseline)
fi

if [[ "${LOAD_IN_4BIT}" == "1" || "${LOAD_IN_4BIT}" == "true" || "${LOAD_IN_4BIT}" == "yes" ]]; then
  CMD+=(--load-in-4bit)
else
  CMD+=(--no-load-in-4bit)
fi

if [[ "${NO_PLOTS}" == "1" || "${NO_PLOTS}" == "true" || "${NO_PLOTS}" == "yes" ]]; then
  CMD+=(--no-plots)
fi

if [[ -n "${QWEN_CODEBOOK_INIT_NORM}" ]]; then
  CMD+=(--qwen-codebook-init-norm "${QWEN_CODEBOOK_INIT_NORM}")
fi

if [[ -n "${MAX_PIXELS}" ]]; then
  CMD+=(--max-pixels "${MAX_PIXELS}")
fi

if [[ -n "${MIN_PIXELS}" ]]; then
  CMD+=(--min-pixels "${MIN_PIXELS}")
fi

printf 'Script path: %s\n' "${SCRIPT_PATH}"
printf 'Runner path: %s\n' "${REPO_ROOT}/runners/oracle_count_multilayer_injection_seq8_7b.sbatch"
printf 'Output root: %s\n' "${OUTPUT_ROOT}"
printf 'Smoke: %s\n' "${SMOKE}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"

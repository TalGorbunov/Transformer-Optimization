#!/bin/bash
#SBATCH --job-name=memory-inject-sweep-7b
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
BASELINE_RUN="${BASELINE_RUN:-${REPO_ROOT}/outputs/shared_count_direction_memory_seq8_7b_20260527_203756}"
STAGE3_CHECKPOINT="${STAGE3_CHECKPOINT:-${BASELINE_RUN}/checkpoints/stage3_shared_count_direction_plus_small_residual_best.pt}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/memory_injection_site_sweep_seq8_7b_${RUN_NAME}}"

SEQ_LEN="${SEQ_LEN:-8}"
SPLIT="${SPLIT:-all_uniform}"
EVIDENCE_COUNTS="${EVIDENCE_COUNTS:-0 1 2 3 4 5 6 7 8}"
LAYERS="${LAYERS:-14 15 16 17}"
TARGETS="${TARGETS:-room_char all_question_tokens last_token room_char_plus_last question_plus_last}"
SINGLE_LAYERS="${SINGLE_LAYERS:-14 16 18 20 22 24}"
LAYER_WINDOWS="${LAYER_WINDOWS:-14-17 16-19 18-21 20-23}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-100}"
SEED="${SEED:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
NO_PLOTS="${NO_PLOTS:-0}"
NO_AUTO_CALIBRATED="${NO_AUTO_CALIBRATED:-0}"

mkdir -p "${OUTPUT_DIR}"
SLURM_LOG_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME:-memory-inject-sweep-7b}-${SLURM_JOB_ID:-local}.out"
copy_slurm_log() {
  if [[ -f "${SLURM_LOG_PATH}" ]]; then
    cp "${SLURM_LOG_PATH}" "${OUTPUT_DIR}/"
  fi
}
trap copy_slurm_log EXIT
exec > >(tee -a "${OUTPUT_DIR}/runner-${SLURM_JOB_ID:-local}.log") 2>&1

read -ra EVIDENCE_COUNT_ARRAY <<< "${EVIDENCE_COUNTS//,/ }"
read -ra LAYER_ARRAY <<< "${LAYERS//,/ }"
read -ra TARGET_ARRAY <<< "${TARGETS//,/ }"
read -ra SINGLE_LAYER_ARRAY <<< "${SINGLE_LAYERS//,/ }"
read -ra LAYER_WINDOW_ARRAY <<< "${LAYER_WINDOWS//,/ }"

SCRIPT_PATH="${REPO_ROOT}/experiments/carrier_probes/run_memory_injection_site_sweep_seq8.py"

CMD=(
  python -u "${SCRIPT_PATH}"
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --source-run "${SOURCE_RUN}"
  --base-source-run "${BASE_SOURCE_RUN}"
  --baseline-run "${BASELINE_RUN}"
  --stage3-checkpoint "${STAGE3_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --seq-len "${SEQ_LEN}"
  --evidence-counts "${EVIDENCE_COUNT_ARRAY[@]}"
  --layers "${LAYER_ARRAY[@]}"
  --targets "${TARGET_ARRAY[@]}"
  --single-layers "${SINGLE_LAYER_ARRAY[@]}"
  --layer-windows "${LAYER_WINDOW_ARRAY[@]}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --candidate-min 0
  --candidate-max 8
  --seed "${SEED}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
)

if [[ "${LOAD_IN_4BIT}" == "1" || "${LOAD_IN_4BIT}" == "true" || "${LOAD_IN_4BIT}" == "yes" ]]; then
  CMD+=(--load-in-4bit)
else
  CMD+=(--no-load-in-4bit)
fi

if [[ "${NO_AUTO_CALIBRATED}" == "1" || "${NO_AUTO_CALIBRATED}" == "true" || "${NO_AUTO_CALIBRATED}" == "yes" ]]; then
  CMD+=(--no-auto-calibrated)
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
printf 'Runner path: %s\n' "${REPO_ROOT}/runners/run_memory_injection_site_sweep_seq8.sbatch"
printf 'Output dir: %s\n' "${OUTPUT_DIR}"
printf 'Stage3 checkpoint: %s\n' "${STAGE3_CHECKPOINT}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"
echo "Finished memory injection-site sweep: ${OUTPUT_DIR}"

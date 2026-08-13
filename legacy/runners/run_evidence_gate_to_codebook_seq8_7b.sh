#!/bin/bash
#SBATCH --job-name=evidence-gate-codebook
#SBATCH --mem=96G
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
PREVIOUS_SHARED_RUN="${PREVIOUS_SHARED_RUN:-${REPO_ROOT}/outputs/shared_count_direction_memory_seq8_7b_20260527_203756}"

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/evidence_gate_to_codebook_seq8_7b_${RUN_NAME}}"

SEQ_LEN="${SEQ_LEN:-8}"
SPLIT="${SPLIT:-all_uniform}"
EVIDENCE_COUNTS="${EVIDENCE_COUNTS:-0 1 2 3 4 5 6 7 8}"
LAYERS="${LAYERS:-14 15 16 17}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-100}"
VARIANTS="${VARIANTS:-baseline oracle_codebook evidence_gate_codebook_question_18_21 evidence_gate_codebook_room_char_14_17 two_stage_gate_plus_codebook softmax_memory_baseline}"
ALPHA_LIST="${ALPHA_LIST:-0.25 0.5 1.0 2.0}"
BETA_LIST="${BETA_LIST:-1.0 2.0 5.0 10.0}"
GAMMA_LIST="${GAMMA_LIST:-0.25 0.5 1.0}"
TAU_STRATEGY="${TAU_STRATEGY:-quantile_grid}"
TAU_GRID="${TAU_GRID:-}"
MAX_CALIBRATED_CONFIGS="${MAX_CALIBRATED_CONFIGS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DEVICE="${DEVICE:-cuda}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
NO_PLOTS="${NO_PLOTS:-0}"
SKIP_QWEN="${SKIP_QWEN:-0}"
QWEN_CODEBOOK_INIT_NORM="${QWEN_CODEBOOK_INIT_NORM:-}"

mkdir -p "${OUTPUT_DIR}"
SLURM_LOG_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME:-evidence-gate-codebook}-${SLURM_JOB_ID:-local}.out"
copy_slurm_log() {
  if [[ -f "${SLURM_LOG_PATH}" ]]; then
    cp "${SLURM_LOG_PATH}" "${OUTPUT_DIR}/"
  fi
}
trap copy_slurm_log EXIT
exec > >(tee -a "${OUTPUT_DIR}/runner-${SLURM_JOB_ID:-local}.log") 2>&1

read -ra EVIDENCE_COUNT_ARRAY <<< "${EVIDENCE_COUNTS//,/ }"
read -ra LAYER_ARRAY <<< "${LAYERS//,/ }"
read -ra VARIANT_ARRAY <<< "${VARIANTS//,/ }"
read -ra ALPHA_ARRAY <<< "${ALPHA_LIST//,/ }"
read -ra BETA_ARRAY <<< "${BETA_LIST//,/ }"
read -ra GAMMA_ARRAY <<< "${GAMMA_LIST//,/ }"

CMD=(
  python -u scripts/probes/run_evidence_gate_to_codebook_seq8.py
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --source-run "${SOURCE_RUN}"
  --base-source-run "${BASE_SOURCE_RUN}"
  --previous-shared-run "${PREVIOUS_SHARED_RUN}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --seq-len "${SEQ_LEN}"
  --evidence-counts "${EVIDENCE_COUNT_ARRAY[@]}"
  --layers "${LAYER_ARRAY[@]}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --variants "${VARIANT_ARRAY[@]}"
  --alpha-list "${ALPHA_ARRAY[@]}"
  --beta-list "${BETA_ARRAY[@]}"
  --gamma-list "${GAMMA_ARRAY[@]}"
  --tau-strategy "${TAU_STRATEGY}"
  --max-calibrated-configs "${MAX_CALIBRATED_CONFIGS}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --dtype "${DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --device "${DEVICE}"
)

if [[ -n "${TAU_GRID}" ]]; then
  read -ra TAU_GRID_ARRAY <<< "${TAU_GRID//,/ }"
  CMD+=(--tau-grid "${TAU_GRID_ARRAY[@]}")
fi

if [[ "${LOAD_IN_4BIT}" == "1" || "${LOAD_IN_4BIT}" == "true" || "${LOAD_IN_4BIT}" == "yes" ]]; then
  CMD+=(--load-in-4bit)
else
  CMD+=(--no-load-in-4bit)
fi

if [[ -n "${MAX_PIXELS}" ]]; then
  CMD+=(--max-pixels "${MAX_PIXELS}")
fi

if [[ -n "${MIN_PIXELS}" ]]; then
  CMD+=(--min-pixels "${MIN_PIXELS}")
fi

if [[ -n "${QWEN_CODEBOOK_INIT_NORM}" ]]; then
  CMD+=(--qwen-codebook-init-norm "${QWEN_CODEBOOK_INIT_NORM}")
fi

if [[ "${NO_PLOTS}" == "1" || "${NO_PLOTS}" == "true" || "${NO_PLOTS}" == "yes" ]]; then
  CMD+=(--no-plots)
fi

if [[ "${SKIP_QWEN}" == "1" || "${SKIP_QWEN}" == "true" || "${SKIP_QWEN}" == "yes" ]]; then
  CMD+=(--skip-qwen)
fi

printf 'Output dir: %s\n' "${OUTPUT_DIR}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"
echo "Finished evidence gate to codebook run: ${OUTPUT_DIR}"

#!/bin/bash
#SBATCH --job-name=entity-room-soft-bias
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -p l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00
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

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-32B-Instruct}"
DATA_ROOT="${DATA_ROOT:-data/mmred_images}"
METADATA_PATH="${METADATA_PATH:-data/mmred}"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-outputs/entity_room_soft_bias_T30}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/${RUN_NAME}}"
CACHE_DIR="${CACHE_DIR:-${EXPERIMENT_DIR}/cache}"

SEQ_LENS="${SEQ_LENS:-2 4 6 8}"
SPLIT="${SPLIT:-all_uniform}"
EVIDENCE_COUNTS="${EVIDENCE_COUNTS:-all}"
MAX_SAMPLES_PER_SEQ_LEN="${MAX_SAMPLES_PER_SEQ_LEN:-}"

TRANSITION_LAYER="${TRANSITION_LAYER:-30}"
ROOM_BBOX_PADDING_FRAC="${ROOM_BBOX_PADDING_FRAC:-0.02}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-true}"
FORCE_EXTRACT="${FORCE_EXTRACT:-false}"
CLEAN_TOP1_ONLY="${CLEAN_TOP1_ONLY:-false}"
SEED="${SEED:-0}"

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/run-${SLURM_JOB_ID:-local}.log") 2>&1

read -ra SEQ_LEN_ARRAY <<< "${SEQ_LENS//,/ }"
read -ra EVIDENCE_COUNT_ARRAY <<< "${EVIDENCE_COUNTS//,/ }"

CMD=(
  python -u
  evaluations/scripts/eval_mmred_entity_room_soft_bias.py
  --model-id "${MODEL_ID}"
  --data-root "${DATA_ROOT}"
  --seq-lens "${SEQ_LEN_ARRAY[@]}"
  --split "${SPLIT}"
  --evidence-counts "${EVIDENCE_COUNT_ARRAY[@]}"
  --transition-layer "${TRANSITION_LAYER}"
  --room-bbox-padding-frac "${ROOM_BBOX_PADDING_FRAC}"
  --output-dir "${OUTPUT_DIR}"
  --cache-dir "${CACHE_DIR}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --clean-top1-only "${CLEAN_TOP1_ONLY}"
  --seed "${SEED}"
)

if [[ -d "${METADATA_PATH}" ]]; then
  CMD+=(--metadata-path "${METADATA_PATH}")
fi

if [[ -n "${MAX_SAMPLES_PER_SEQ_LEN}" ]]; then
  CMD+=(--max-samples-per-seq-len "${MAX_SAMPLES_PER_SEQ_LEN}")
fi

if [[ "${LOAD_IN_4BIT}" == "true" ]]; then
  CMD+=(--load-in-4bit)
else
  CMD+=(--no-load-in-4bit)
fi

if [[ "${FORCE_EXTRACT}" == "true" ]]; then
  CMD+=(--force-extract)
fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"

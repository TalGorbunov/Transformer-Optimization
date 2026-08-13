#!/bin/bash
#SBATCH --job-name=assoc-mem-7b
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
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/assoc_memory_probe_seq8_7b_${RUN_NAME}}"

SEQ_LEN="${SEQ_LEN:-8}"
SPLIT="${SPLIT:-all_uniform}"
LAYERS="${LAYERS:-14 16 18}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
MEMORY_DIM="${MEMORY_DIM:-64}"
LR="${LR:-1e-3}"
DROPOUT="${DROPOUT:-0.1}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
OVERWRITE_CACHE="${OVERWRITE_CACHE:-0}"
NO_PLOTS="${NO_PLOTS:-0}"

mkdir -p "${OUTPUT_DIR}"
SLURM_LOG_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME:-assoc-mem-7b}-${SLURM_JOB_ID:-local}.out"
copy_slurm_log() {
  if [[ -f "${SLURM_LOG_PATH}" ]]; then
    cp "${SLURM_LOG_PATH}" "${OUTPUT_DIR}/"
  fi
}
trap copy_slurm_log EXIT
exec > >(tee -a "${OUTPUT_DIR}/runner-${SLURM_JOB_ID:-local}.log") 2>&1

read -ra LAYER_ARRAY <<< "${LAYERS//,/ }"

CMD=(
  python -u scripts/probes/run_associative_memory_probe.py
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --seq-len "${SEQ_LEN}"
  --layers "${LAYER_ARRAY[@]}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --hidden-dim "${HIDDEN_DIM}"
  --memory-dim "${MEMORY_DIM}"
  --lr "${LR}"
  --dropout "${DROPOUT}"
  --dtype "${DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
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

if [[ "${OVERWRITE_CACHE}" == "1" || "${OVERWRITE_CACHE}" == "true" || "${OVERWRITE_CACHE}" == "yes" ]]; then
  CMD+=(--overwrite-cache)
fi

if [[ "${NO_PLOTS}" == "1" || "${NO_PLOTS}" == "true" || "${NO_PLOTS}" == "yes" ]]; then
  CMD+=(--no-plots)
fi

printf 'Output dir: %s\n' "${OUTPUT_DIR}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"
echo "Finished associative memory probe run: ${OUTPUT_DIR}"

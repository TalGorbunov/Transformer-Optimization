#!/bin/bash
#SBATCH --job-name=qwen7b-acc
#SBATCH --mem=64G
#SBATCH -p l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=04:00:00
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

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-/home/tal.gorbunov/projects/Transformer-Optimization/data/mmred_images_park_no_step_marker}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization/outputs/qwen7b_accuracy_heatmap}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/${RUN_NAME}}"

SEQ_LENS="${SEQ_LENS:-2 4 8}"
EVIDENCE_COUNTS="${EVIDENCE_COUNTS:-0 1 2 3 4 5 6 7 8}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-100}"
SPLIT="${SPLIT:-all_uniform}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

mkdir -p "${OUTPUT_DIR}"
SLURM_LOG_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME:-qwen7b-acc}-${SLURM_JOB_ID:-local}.out"
copy_slurm_log() {
  if [[ -f "${SLURM_LOG_PATH}" ]]; then
    cp "${SLURM_LOG_PATH}" "${OUTPUT_DIR}/"
  fi
}
trap copy_slurm_log EXIT
exec > >(tee -a "${OUTPUT_DIR}/run-${SLURM_JOB_ID:-local}.log") 2>&1

read -ra SEQ_LEN_ARRAY <<< "${SEQ_LENS//,/ }"
read -ra EVIDENCE_COUNT_ARRAY <<< "${EVIDENCE_COUNTS//,/ }"

CMD=(
  python -u experiments/qwen7b_accuracy_heatmap.py
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --seq-lens "${SEQ_LEN_ARRAY[@]}"
  --evidence-counts "${EVIDENCE_COUNT_ARRAY[@]}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --split "${SPLIT}"
  --dtype "${DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
)

printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"

#!/bin/bash
#SBATCH --job-name=msg-probe-frame-7b
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
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/message_probe_sum_vs_joint_mlp_seq8_${RUN_NAME}}"

SEQ_LEN="${SEQ_LEN:-8}"
SPLIT="${SPLIT:-all_uniform}"
EVIDENCE_COUNTS="${EVIDENCE_COUNTS:-0 1 2 3 4 5 6 7 8}"
LAYERS="${LAYERS:-14 15 16 17}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-100}"
SEED="${SEED:-0}"

LINEAR_SOLVER="${LINEAR_SOLVER:-lbfgs}"
LINEAR_CLASS_WEIGHT="${LINEAR_CLASS_WEIGHT:-balanced}"
FRAME_LINEAR_C="${FRAME_LINEAR_C:-1.0}"
FRAME_LINEAR_MAX_ITER="${FRAME_LINEAR_MAX_ITER:-1000}"
SKLEARN_N_JOBS="${SKLEARN_N_JOBS:-}"

BASE_CANDIDATE_MIN="${BASE_CANDIDATE_MIN:-0}"
BASE_CANDIDATE_MAX="${BASE_CANDIDATE_MAX:-8}"
COMPUTE_MISSING_BASE="${COMPUTE_MISSING_BASE:-1}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
FORCE_RECACHE="${FORCE_RECACHE:-0}"
RECOMPUTE_MISSING_CARRIERS="${RECOMPUTE_MISSING_CARRIERS:-1}"
NO_PLOTS="${NO_PLOTS:-0}"

mkdir -p "${OUTPUT_DIR}"
SLURM_LOG_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME:-msg-probe-frame-7b}-${SLURM_JOB_ID:-local}.out"
copy_slurm_log() {
  if [[ -f "${SLURM_LOG_PATH}" ]]; then
    cp "${SLURM_LOG_PATH}" "${OUTPUT_DIR}/"
  fi
}
trap copy_slurm_log EXIT
exec > >(tee -a "${OUTPUT_DIR}/runner-${SLURM_JOB_ID:-local}.log") 2>&1

read -ra EVIDENCE_COUNT_ARRAY <<< "${EVIDENCE_COUNTS//,/ }"
read -ra LAYER_ARRAY <<< "${LAYERS//,/ }"

CMD=(
  python -u experiments/message_probe_sum_vs_joint_mlp.py
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --source-run "${SOURCE_RUN}"
  --base-source-run "${BASE_SOURCE_RUN}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --seq-len "${SEQ_LEN}"
  --evidence-counts "${EVIDENCE_COUNT_ARRAY[@]}"
  --layers "${LAYER_ARRAY[@]}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --seed "${SEED}"
  --linear-solver "${LINEAR_SOLVER}"
  --linear-class-weight "${LINEAR_CLASS_WEIGHT}"
  --frame-linear-c "${FRAME_LINEAR_C}"
  --frame-linear-max-iter "${FRAME_LINEAR_MAX_ITER}"
  --base-candidate-min "${BASE_CANDIDATE_MIN}"
  --base-candidate-max "${BASE_CANDIDATE_MAX}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
)

if [[ -n "${SKLEARN_N_JOBS}" ]]; then
  CMD+=(--sklearn-n-jobs "${SKLEARN_N_JOBS}")
fi

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

if [[ "${RECOMPUTE_MISSING_CARRIERS}" == "0" || "${RECOMPUTE_MISSING_CARRIERS}" == "false" || "${RECOMPUTE_MISSING_CARRIERS}" == "no" ]]; then
  CMD+=(--no-recompute-missing-carriers)
fi

if [[ "${COMPUTE_MISSING_BASE}" == "0" || "${COMPUTE_MISSING_BASE}" == "false" || "${COMPUTE_MISSING_BASE}" == "no" ]]; then
  CMD+=(--no-compute-missing-base)
fi

if [[ "${NO_PLOTS}" == "1" || "${NO_PLOTS}" == "true" || "${NO_PLOTS}" == "yes" ]]; then
  CMD+=(--no-plots)
fi

printf 'Output dir: %s\n' "${OUTPUT_DIR}"
printf 'Message source run: %s\n' "${SOURCE_RUN}"
printf 'Base source run: %s\n' "${BASE_SOURCE_RUN}"
printf 'Evidence counts: %s\n' "${EVIDENCE_COUNTS}"
printf 'Layers: %s\n' "${LAYERS}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"
echo "Finished base-vs-per-frame message probe run: ${OUTPUT_DIR}"

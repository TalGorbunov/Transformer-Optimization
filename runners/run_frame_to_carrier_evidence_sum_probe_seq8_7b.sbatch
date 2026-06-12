#!/bin/bash
#SBATCH --job-name=frame-evidence-sum-7b
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
SOURCE_RUN="${SOURCE_RUN:-${REPO_ROOT}/outputs/frame_to_carrier_message_memory_probe_seq8_7b_multilayer_20260521_154136}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/frame_to_carrier_evidence_sum_probe_seq8_7b_${RUN_NAME}}"

SEQ_LEN="${SEQ_LEN:-8}"
SPLIT="${SPLIT:-all_uniform}"
LAYERS="${LAYERS:-14 15 16 17}"
CONDITIONS="${CONDITIONS:-target_char target_room target_char_room wrong_non_target raw_frame_memory content_only_memory}"
PROBE_FAMILIES="${PROBE_FAMILIES:-linear mlp}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
DROPOUT="${DROPOUT:-0.1}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
FORCE_RECACHE="${FORCE_RECACHE:-0}"
RECOMPUTE_MISSING_CARRIERS="${RECOMPUTE_MISSING_CARRIERS:-1}"
NO_PLOTS="${NO_PLOTS:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"

mkdir -p "${OUTPUT_DIR}"
SLURM_LOG_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME:-frame-evidence-sum-7b}-${SLURM_JOB_ID:-local}.out"
copy_slurm_log() {
  if [[ -f "${SLURM_LOG_PATH}" ]]; then
    cp "${SLURM_LOG_PATH}" "${OUTPUT_DIR}/"
  fi
}
trap copy_slurm_log EXIT
exec > >(tee -a "${OUTPUT_DIR}/runner-${SLURM_JOB_ID:-local}.log") 2>&1

read -ra LAYER_ARRAY <<< "${LAYERS//,/ }"
read -ra CONDITION_ARRAY <<< "${CONDITIONS//,/ }"
read -ra PROBE_FAMILY_ARRAY <<< "${PROBE_FAMILIES//,/ }"

CMD=(
  python -u scripts/probes/run_frame_to_carrier_evidence_sum_probe.py
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --source-run "${SOURCE_RUN}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --seq-len "${SEQ_LEN}"
  --layers "${LAYER_ARRAY[@]}"
  --conditions "${CONDITION_ARRAY[@]}"
  --probe-families "${PROBE_FAMILY_ARRAY[@]}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --hidden-dim "${HIDDEN_DIM}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --dropout "${DROPOUT}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --bootstrap-iters "${BOOTSTRAP_ITERS}"
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

if [[ "${FORCE_RECACHE}" == "1" || "${FORCE_RECACHE}" == "true" || "${FORCE_RECACHE}" == "yes" ]]; then
  CMD+=(--force-recache)
fi

if [[ "${RECOMPUTE_MISSING_CARRIERS}" == "0" || "${RECOMPUTE_MISSING_CARRIERS}" == "false" || "${RECOMPUTE_MISSING_CARRIERS}" == "no" ]]; then
  CMD+=(--no-recompute-missing-carriers)
fi

if [[ "${NO_PLOTS}" == "1" || "${NO_PLOTS}" == "true" || "${NO_PLOTS}" == "yes" ]]; then
  CMD+=(--no-plots)
fi

printf 'Output dir: %s\n' "${OUTPUT_DIR}"
printf 'Source run: %s\n' "${SOURCE_RUN}"
printf 'Conditions: %s\n' "${CONDITIONS}"
printf 'Probe families: %s\n' "${PROBE_FAMILIES}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"
echo "Finished frame-to-carrier evidence-sum probe run: ${OUTPUT_DIR}"

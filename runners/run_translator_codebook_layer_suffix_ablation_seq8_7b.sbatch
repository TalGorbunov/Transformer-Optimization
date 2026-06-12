#!/bin/bash
#SBATCH --job-name=translator-suffix-7b
#SBATCH --mem=128G
#SBATCH -p l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-16%2
#SBATCH --chdir=/home/tal.gorbunov/projects/Transformer-Optimization
#SBATCH --output=logs/translator_codebook_layer_suffix_ablation_seq8_7b/%x-%A_%a.out

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
mkdir -p logs/translator_codebook_layer_suffix_ablation_seq8_7b

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/mmred_images_park}"
SOURCE_RUN="${SOURCE_RUN:-${REPO_ROOT}/outputs/frame_to_carrier_evidence_sum_probe_seq8_7b_20260521_164621}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/translator_codebook_layer_suffix_ablation_seq8_7b}"

SEQ_LEN="${SEQ_LEN:-8}"
SPLIT="${SPLIT:-all_uniform}"
EVIDENCE_COUNTS="${EVIDENCE_COUNTS:-0 1 2 3 4 5 6 7 8}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-100}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-360}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-135}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MARGIN_TARGET="${MARGIN_TARGET:-1.0}"
LAMBDA_MARGIN="${LAMBDA_MARGIN:-0.2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
SEED="${SEED:-0}"
RANK="${RANK:-16}"
STATE_BOTTLENECK="${STATE_BOTTLENECK:-64}"
COUNT_CONTROL="${COUNT_CONTROL:-none}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
NO_PLOTS="${NO_PLOTS:-0}"

CONFIG_NAMES=(
  "A_room_char_l14_17_static_fixed_alpha"
  "A_room_char_l15_17_static_fixed_alpha"
  "A_room_char_l16_17_static_fixed_alpha"
  "A_room_char_l17_static_fixed_alpha"
  "A_room_char_l14_17_layer_fixed_alpha"
  "A_room_char_l15_17_layer_fixed_alpha"
  "A_room_char_l16_17_layer_fixed_alpha"
  "A_room_char_l17_layer_fixed_alpha"
  "A_room_char_l14_17_static_energy_norm"
  "A_room_char_l15_17_static_energy_norm"
  "A_room_char_l16_17_static_energy_norm"
  "A_room_char_l17_static_energy_norm"
  "A_room_char_l14_17_layer_energy_norm"
  "A_room_char_l15_17_layer_energy_norm"
  "A_room_char_l16_17_layer_energy_norm"
  "A_room_char_l17_layer_energy_norm"
  "baseline_no_injection_reference"
)

METHODS=(
  "static"
  "static"
  "static"
  "static"
  "layer"
  "layer"
  "layer"
  "layer"
  "static"
  "static"
  "static"
  "static"
  "layer"
  "layer"
  "layer"
  "layer"
  "baseline"
)

ALPHA_REGIMES=(
  "fixed_alpha"
  "fixed_alpha"
  "fixed_alpha"
  "fixed_alpha"
  "fixed_alpha"
  "fixed_alpha"
  "fixed_alpha"
  "fixed_alpha"
  "energy_norm"
  "energy_norm"
  "energy_norm"
  "energy_norm"
  "energy_norm"
  "energy_norm"
  "energy_norm"
  "energy_norm"
  "baseline"
)

LAYER_STARTS=(14 15 16 17 14 15 16 17 14 15 16 17 14 15 16 17 14)
LAYER_ENDS=(17 17 17 17 17 17 17 17 17 17 17 17 17 17 17 17 17)
ALPHAS=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.1547 1.4142 2.0 1.0 1.1547 1.4142 2.0 0.0)
DEFAULT_LRS=(5e-3 5e-3 5e-3 5e-3 5e-3 5e-3 5e-3 5e-3 5e-3 5e-3 5e-3 5e-3 5e-3 5e-3 5e-3 5e-3 0)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG_NAME="${CONFIG_NAMES[${TASK_ID}]}"
TOKEN_GROUP="room_char"
LAYER_START="${LAYER_STARTS[${TASK_ID}]}"
LAYER_END="${LAYER_ENDS[${TASK_ID}]}"
METHOD="${METHODS[${TASK_ID}]}"
ALPHA_REGIME="${ALPHA_REGIMES[${TASK_ID}]}"
TASK_ALPHA="${ALPHAS[${TASK_ID}]}"
TASK_LR="${DEFAULT_LRS[${TASK_ID}]}"
ALPHA="${ALPHA:-${TASK_ALPHA}}"
LR="${LR:-${TASK_LR}}"

read -ra EVIDENCE_COUNT_ARRAY <<< "${EVIDENCE_COUNTS//,/ }"

SCRIPT_PATH="${REPO_ROOT}/experiments/oracle_bounds/translator_ablation_gold_count_seq8_7b.py"

CMD=(
  python -u "${SCRIPT_PATH}"
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --source-run "${SOURCE_RUN}"
  --seq-len "${SEQ_LEN}"
  --split "${SPLIT}"
  --output-root "${OUTPUT_ROOT}"
  --config-name "${CONFIG_NAME}"
  --method "${METHOD}"
  --token-group "${TOKEN_GROUP}"
  --layer-start "${LAYER_START}"
  --layer-end "${LAYER_END}"
  --alpha "${ALPHA}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --rank "${RANK}"
  --state-bottleneck "${STATE_BOTTLENECK}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --max-train-samples "${MAX_TRAIN_SAMPLES}"
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --evidence-counts "${EVIDENCE_COUNT_ARRAY[@]}"
  --candidate-min 0
  --candidate-max 8
  --lambda-margin "${LAMBDA_MARGIN}"
  --margin-target "${MARGIN_TARGET}"
  --weight-decay "${WEIGHT_DECAY}"
  --count-control "${COUNT_CONTROL}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --submit-mode "sbatch"
)

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
printf 'Runner path: %s\n' "${REPO_ROOT}/runners/run_translator_codebook_layer_suffix_ablation_seq8_7b.sbatch"
printf 'Output root: %s\n' "${OUTPUT_ROOT}"
printf 'Array task: %s\n' "${TASK_ID}"
printf 'Config: %s method=%s token_group=%s layers=%s-%s alpha=%s alpha_regime=%s lr=%s\n' \
  "${CONFIG_NAME}" "${METHOD}" "${TOKEN_GROUP}" "${LAYER_START}" "${LAYER_END}" "${ALPHA}" "${ALPHA_REGIME}" "${LR}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" || "${DRY_RUN:-0}" == "yes" ]]; then
  echo "DRY_RUN requested; command was assembled but not executed."
  exit 0
fi

"${CMD[@]}" "$@"
echo "Finished translator suffix ablation config: ${CONFIG_NAME} method=${METHOD}"

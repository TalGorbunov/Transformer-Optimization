#!/bin/bash
#SBATCH --job-name=msg-memory-carrier-7b
#SBATCH --mem=128G
#SBATCH -p l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/tal.gorbunov/projects/Transformer-Optimization
#SBATCH --output=logs/message_memory_carrier_update_seq8_7b/%x-%j.out

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
mkdir -p logs/message_memory_carrier_update_seq8_7b

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/mmred_images_park}"
SOURCE_RUN="${SOURCE_RUN:-${REPO_ROOT}/outputs/frame_to_carrier_evidence_sum_probe_seq8_7b_20260521_164621}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/message_memory_carrier_update_seq8_7b}"

SEQ_LEN="${SEQ_LEN:-8}"
SPLIT="${SPLIT:-all_uniform}"
EVIDENCE_COUNTS="${EVIDENCE_COUNTS:-0 1 2 3 4 5 6 7 8}"
MAX_SAMPLES_PER_COUNT="${MAX_SAMPLES_PER_COUNT:-100}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-360}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-135}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
D_MEM="${D_MEM:-256}"
LAYER_START="${LAYER_START:-14}"
LAYER_END="${LAYER_END:-17}"
GAMMA_INIT="${GAMMA_INIT:-0.05}"
MESSAGE_MODE="${MESSAGE_MODE:-auto}"
MARGIN_TARGET="${MARGIN_TARGET:-1.0}"
LAMBDA_MARGIN="${LAMBDA_MARGIN:-0.2}"
LAMBDA_UPDATE_ENERGY="${LAMBDA_UPDATE_ENERGY:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
SEED="${SEED:-0}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
NO_PLOTS="${NO_PLOTS:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-1}"

STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${OUTPUT_ROOT}/${STAMP}_message_memory_carrier_update_seq8_7b}"
SCRIPT_PATH="${REPO_ROOT}/scripts/probes/run_message_memory_carrier_update_seq8_7b.py"
SUMMARY_SCRIPT="${REPO_ROOT}/scripts/probes/summarize_message_memory_carrier_update_seq8_7b.py"

read -ra EVIDENCE_COUNT_ARRAY <<< "${EVIDENCE_COUNTS//,/ }"

common_args=(
  --model-name "${MODEL_NAME}"
  --dataset-root "${DATASET_ROOT}"
  --source-run "${SOURCE_RUN}"
  --seq-len "${SEQ_LEN}"
  --split "${SPLIT}"
  --output-root "${OUTPUT_ROOT}"
  --layer-start "${LAYER_START}"
  --layer-end "${LAYER_END}"
  --d-mem "${D_MEM}"
  --gamma-init "${GAMMA_INIT}"
  --message-mode "${MESSAGE_MODE}"
  --lr "${LR}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --max-samples-per-count "${MAX_SAMPLES_PER_COUNT}"
  --evidence-counts "${EVIDENCE_COUNT_ARRAY[@]}"
  --candidate-min 0
  --candidate-max 8
  --lambda-margin "${LAMBDA_MARGIN}"
  --margin-target "${MARGIN_TARGET}"
  --lambda-update-energy "${LAMBDA_UPDATE_ENERGY}"
  --weight-decay "${WEIGHT_DECAY}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --submit-mode "sbatch"
)

if [[ "${LOAD_IN_4BIT}" == "1" || "${LOAD_IN_4BIT}" == "true" || "${LOAD_IN_4BIT}" == "yes" ]]; then
  common_args+=(--load-in-4bit)
else
  common_args+=(--no-load-in-4bit)
fi

if [[ "${NO_PLOTS}" == "1" || "${NO_PLOTS}" == "true" || "${NO_PLOTS}" == "yes" ]]; then
  common_args+=(--no-plots)
fi

if [[ -n "${MAX_PIXELS}" ]]; then
  common_args+=(--max-pixels "${MAX_PIXELS}")
fi

if [[ -n "${MIN_PIXELS}" ]]; then
  common_args+=(--min-pixels "${MIN_PIXELS}")
fi

run_variant() {
  local variant="$1"
  local out_dir="$2"
  local epochs="$3"
  local max_train="$4"
  local max_eval="$5"
  shift 5
  local cmd=(
    python -u "${SCRIPT_PATH}"
    "${common_args[@]}"
    --memory-variant "${variant}"
    --output-dir "${out_dir}"
    --run-name "${variant}"
    --epochs "${epochs}"
    --max-train-samples "${max_train}"
    --max-eval-samples "${max_eval}"
  )
  printf 'Running variant=%s output=%s\n' "${variant}" "${out_dir}"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}" "$@"
}

check_smoke() {
  local variant="$1"
  local out_dir="$2"
  python - "$variant" "$out_dir" <<'PY'
import json
import sys
from pathlib import Path

variant = sys.argv[1]
out_dir = Path(sys.argv[2])
diag = json.loads((out_dir / "diagnostics.json").read_text())
if variant != "baseline":
    required = {
        "hooks_ok": diag.get("hooks_ok"),
        "frame_rows_ok": diag.get("frame_rows_ok"),
        "room_char_positions_found": diag.get("room_char_positions_found"),
        "nonzero_gates": diag.get("nonzero_gates"),
        "nonzero_updates": diag.get("nonzero_updates"),
    }
    bad = {key: value for key, value in required.items() if int(value or 0) != 1}
    backward = diag.get("backward_diagnostics", {})
    if int(backward.get("only_adapter_params_updated", 0)) != 1:
        bad["only_adapter_params_updated"] = backward
    if bad:
        raise SystemExit(f"Smoke diagnostics failed for {variant}: {bad}")
else:
    if not (out_dir / "metrics.csv").is_file() or not (out_dir / "summary.csv").is_file():
        raise SystemExit(f"Baseline smoke did not write metrics/summary in {out_dir}")
print(f"Smoke diagnostics passed for {variant}: {out_dir}")
PY
}

mkdir -p "${EXPERIMENT_DIR}"

printf 'Script path: %s\n' "${SCRIPT_PATH}"
printf 'Runner path: %s\n' "${REPO_ROOT}/runners/run_message_memory_carrier_update_seq8_7b.sbatch"
printf 'Experiment dir: %s\n' "${EXPERIMENT_DIR}"
printf 'Output root: %s\n' "${OUTPUT_ROOT}"

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" || "${DRY_RUN:-0}" == "yes" ]]; then
  echo "DRY_RUN requested; commands were assembled but not executed."
  exit 0
fi

if [[ "${RUN_SMOKE}" == "1" || "${RUN_SMOKE}" == "true" || "${RUN_SMOKE}" == "yes" ]]; then
  SMOKE_DIR="${EXPERIMENT_DIR}/smoke"
  run_variant "baseline" "${SMOKE_DIR}/baseline" 1 16 16 "$@"
  check_smoke "baseline" "${SMOKE_DIR}/baseline"
  run_variant "layer_local" "${SMOKE_DIR}/layer_local" 1 16 16 "$@"
  check_smoke "layer_local" "${SMOKE_DIR}/layer_local"
  run_variant "cumulative_sum" "${SMOKE_DIR}/cumulative_sum" 1 16 16 "$@"
  check_smoke "cumulative_sum" "${SMOKE_DIR}/cumulative_sum"
  echo "Smoke tests passed."
fi

if [[ "${RUN_FULL}" == "1" || "${RUN_FULL}" == "true" || "${RUN_FULL}" == "yes" ]]; then
  FULL_DIR="${EXPERIMENT_DIR}/full"
  run_variant "baseline" "${FULL_DIR}/baseline" 1 "${MAX_TRAIN_SAMPLES}" "${MAX_EVAL_SAMPLES}" "$@"
  run_variant "layer_local" "${FULL_DIR}/layer_local" "${EPOCHS}" "${MAX_TRAIN_SAMPLES}" "${MAX_EVAL_SAMPLES}" "$@"
  run_variant "cumulative_sum" "${FULL_DIR}/cumulative_sum" "${EPOCHS}" "${MAX_TRAIN_SAMPLES}" "${MAX_EVAL_SAMPLES}" "$@"
  python -u "${SUMMARY_SCRIPT}" --output-root "${EXPERIMENT_DIR}"
  python -u "${SUMMARY_SCRIPT}" --output-root "${OUTPUT_ROOT}"
fi

echo "Finished message memory carrier update experiment: ${EXPERIMENT_DIR}"

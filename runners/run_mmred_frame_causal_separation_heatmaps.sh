#!/bin/bash
#SBATCH --job-name=transofrmer-optimization
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -p h200-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/tal.gorbunov/projects/Transformer-Optimization
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-32B-Instruct}"
CLEAN_ROOT="${CLEAN_ROOT:-data/mmred_images_emptyframe}"
CORRUPT_ROOT="${CORRUPT_ROOT:-data/mmred_emptyframe_corrupted}"
SEQ_LENS="${SEQ_LENS:-2,4,8}"
SAMPLE_SOURCE="${SAMPLE_SOURCE:-auto}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
LIMIT_PER_SEQ="${LIMIT_PER_SEQ:-}"
MAX_SAMPLES_PER_CELL="${MAX_SAMPLES_PER_CELL:-}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmred_frame_causal_separation_heatmaps/${RUN_TAG}_seq_lens_${SEQ_LENS//,/_}}"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/run-${SLURM_JOB_ID:-local}.log") 2>&1

CMD=(
  python -u evaluations/scripts/mmred_frame_causal_separation_heatmaps.py
  --model-id "${MODEL_ID}"
  --clean-root "${CLEAN_ROOT}"
  --corrupt-root "${CORRUPT_ROOT}"
  --seq-lens "${SEQ_LENS}"
  --sample-source "${SAMPLE_SOURCE}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --score-batch-size "${SCORE_BATCH_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --seed "${SEED}"
)

if [[ -n "${LIMIT_PER_SEQ}" ]]; then
  CMD+=(--limit-per-seq "${LIMIT_PER_SEQ}")
fi

if [[ -n "${MAX_SAMPLES_PER_CELL}" ]]; then
  CMD+=(--max-samples-per-cell "${MAX_SAMPLES_PER_CELL}")
fi

"${CMD[@]}" "$@"

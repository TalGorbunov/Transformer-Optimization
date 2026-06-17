#!/bin/bash
#SBATCH --job-name=rv_corrupt
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASK="${TASK:-rooms_visited}"
DATA_ROOT="${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}"
LIMIT="${LIMIT:-25}"
LAYERS="${LAYERS:-0,2,4,6,8,10,12,14,16,18,20,22,24,26}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"   # these scripts DEFAULT to 32B; force 7B
OUTPUT="${OUTPUT:-outputs/token_group_corruption_new_tasks/${TASK}}"
CORRUPTION_MODE="${CORRUPTION_MODE:-evidence}"
CLEAN_TOP1="${CLEAN_TOP1:-false}"

python -u evaluations/scripts/patch_importence/token_group_corruption_new_tasks.py \
  --task "${TASK}" --data_root "${DATA_ROOT}" --limit "${LIMIT}" --layers "${LAYERS}" \
  --model "${MODEL_NAME}" \
  --output "${OUTPUT}" --corruption_mode "${CORRUPTION_MODE}" \
  --clean_top1_must_match_gold "${CLEAN_TOP1}" --disable_plots

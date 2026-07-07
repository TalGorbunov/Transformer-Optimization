#!/bin/bash
#SBATCH --job-name=flow_restore
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314

# Stage 2 flow diagnostic: token-group RESTORATION on a fully (evidence-)corrupted input, for the
# FRONT-QUESTION format ("Question: ...\nThe following are the N frames..."). Patches clean activations
# for {evidence frames, last token, question span} at each selected layer and reports normalized rescue
# = (patched-corrupted)/(clean-corrupted). Answers: at which (latter-half) layers does restoring the
# frames vs the last token recover the answer -> where frame info reaches the answer position.
# NOTE: these scripts DEFAULT to 32B; we force 7B.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_ROOT="${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}"
CORRUPTED_ROOT="${CORRUPTED_ROOT:-data/mmred_corrupted_park/seq_len_8/all_uniform}"
LIMIT="${LIMIT:-40}"
LAYERS="${LAYERS:-13,15,17,19,21,23,25,27}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
FRAME_PATCH_MODE="${FRAME_PATCH_MODE:-evidence_only}"
CLEAN_TOP1="${CLEAN_TOP1:-true}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${OUTPUT:-outputs/frame_axis/probes/flow_diagnostic/${RUN_NAME}}"
mkdir -p "${OUTPUT}"

echo "DATA_ROOT=${DATA_ROOT} CORRUPTED_ROOT=${CORRUPTED_ROOT} LAYERS=${LAYERS} LIMIT=${LIMIT} MODEL=${MODEL_NAME}"
python -u evaluations/scripts/patch_importence/group_restoration_importance.py \
  --data_root "${DATA_ROOT}" --corrupted_root "${CORRUPTED_ROOT}" \
  --limit "${LIMIT}" --layers "${LAYERS}" --model "${MODEL_NAME}" \
  --frame_patch_mode "${FRAME_PATCH_MODE}" \
  --clean_top1_must_match_gold "${CLEAN_TOP1}" \
  --output "${OUTPUT}"

#!/bin/bash
#SBATCH --job-name=rv_surgcorr
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TASK="${TASK:?}"; CORRUPTED_ROOT="${CORRUPTED_ROOT:?}"; OUTPUT="${OUTPUT:?}"
LIMIT="${LIMIT:-5}"; LAYERS="${LAYERS:-0,2,4,6,8,10,12,14,16,18,20,22,24,26}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"   # explicit 7B — script defaults to 32B otherwise
python -u evaluations/scripts/patch_importence/token_group_corruption_new_tasks.py \
  --task "${TASK}" --model_name "${MODEL}" --data_root data/mmred_images_park/seq_len_8/all_uniform \
  --corruption_mode surgical --corrupted_root "${CORRUPTED_ROOT}" \
  --limit "${LIMIT}" --layers "${LAYERS}" --output "${OUTPUT}" --disable_plots

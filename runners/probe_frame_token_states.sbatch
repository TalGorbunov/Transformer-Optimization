#!/bin/bash
#SBATCH --job-name=rv_ftprobe
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
LIMIT="${LIMIT:-90}"
LAYERS="${LAYERS:-4,6,8,12,16,20}"
OUTPUT="${OUTPUT:-outputs/probe_frame_token_states/${TASK}}"

python -u evaluations/scripts/patch_importence/probe_frame_token_states.py \
  --task "${TASK}" --data_root "${DATA_ROOT}" --limit "${LIMIT}" --layers "${LAYERS}" --output "${OUTPUT}"

#!/bin/bash
#SBATCH --job-name=msgprobe
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
  --task "${TASK:-rooms_visited}" --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-150}" --layers "${LAYERS:-8,12,14,16,18,19,20,22,24}" \
  --output "${OUTPUT:-outputs/probe_frame_to_carrier_message}"

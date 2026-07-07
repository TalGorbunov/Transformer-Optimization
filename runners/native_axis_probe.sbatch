#!/bin/bash
#SBATCH --job-name=nataxis
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
python -u evaluations/scripts/patch_importence/native_axis_probe.py \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --layers "${LAYERS:-16,20}" --offset "${OFFSET:-9}" --limit "${LIMIT:-100}" \
  --sample-seed "${SAMPLE_SEED:-0}" --output "${OUTPUT:-outputs/frame_axis/probes/native_axis}"

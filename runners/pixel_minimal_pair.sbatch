#!/bin/bash
#SBATCH --job-name=pixpair
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u evaluations/scripts/patch_importence/pixel_minimal_pair.py \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-150}" --layer "${LAYER:-16}" --offset "${OFFSET:-9}" \
  --output "${OUTPUT:-outputs/frame_axis/probes/pixel_minimal_pair}"

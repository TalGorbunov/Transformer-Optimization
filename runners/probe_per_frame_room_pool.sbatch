#!/bin/bash
#SBATCH --job-name=rv_roomprobe2
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u evaluations/scripts/patch_importence/probe_per_frame_room_identity.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct --limit 40 --layers "${LAYERS:-8,12,16,20,24}" \
  --pool "${POOL:-meanmax}" --output "${OUTPUT:-outputs/probe_per_frame_room_identity_meanmax}"

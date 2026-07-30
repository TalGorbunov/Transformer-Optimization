#!/bin/bash
#SBATCH --job-name=a1fu_smoke
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%j.out
# Smoke for A1-fu1 (text multipass cache) + A1-fu2 (compact text style) — n=6 each.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== smoke 1: text multipass cache (n=6) ==="
python -u experiments/glstm/text_multipass_cache.py \
  --data_root data/mmred_images_park/seq_len_8/all_uniform \
  --limit 6 --layers 14,16 --decode-offsets 0,9,13 \
  --output outputs/_scratch/a1fu1_mp_smoke

echo "=== smoke 2: compact text style probe (n=6) ==="
python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
  --task count --model Qwen/Qwen2.5-VL-7B-Instruct --text-frames --text-style compact \
  --data_root data/mmred_images_park/seq_len_8/all_uniform \
  --limit 6 --layers 14,16 --carrier per_token --n-frames 8 \
  --max-offset 13 --decode-offsets 9,13 --sample-seed 0 \
  --output outputs/_scratch/a1fu2_easy_smoke
echo "SMOKE DONE"

#!/bin/bash
#SBATCH --job-name=a1_fence
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# A1 follow-up: FENCED text-MMRED cache (cross-frame text attention blocked in L0..13) —
# does frame isolation recover carrier d' on text as it does on images (3.1->5.2)?
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
  --task count --model Qwen/Qwen2.5-VL-7B-Instruct --text-frames --fence-cross-frame \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-400}" --layers "${LAYERS:-14,16,18,20}" \
  --carrier per_token --n-frames 8 \
  --max-offset 13 --decode-offsets 9,13 \
  --sample-seed 0 --save-messages \
  --output "outputs/ladder/text_mmred/locmap_cache_fenced/${TS}"

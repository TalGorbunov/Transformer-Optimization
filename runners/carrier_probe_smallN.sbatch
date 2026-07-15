#!/bin/bash
#SBATCH --job-name=carr_smallN
#SBATCH --partition=l40s-shared,rtx6k-shared,a100-public,l40s-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for s in 1 2 3 4 5 6 7 8; do
  echo "########## seq_len $s ##########"
  python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
    --task count --model Qwen/Qwen2.5-VL-7B-Instruct \
    --data_root data/mmred_images_park/seq_len_$s/all_uniform \
    --limit 150 --layers 8,12,14,16,18,20 --carrier per_token --n-frames $s \
    --max-offset 14 --decode-offsets 9,13 --sample-seed 1 \
    --output outputs/ladder/image_smallN/carrier_probe/N$s
done
echo "carrier probe sweep done $(date)"

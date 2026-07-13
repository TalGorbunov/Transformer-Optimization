#!/bin/bash
#SBATCH --job-name=p1a_pos
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
for PAD in 16 32; do
  echo "=== P1a pad-to-frames $PAD ==="
  python -u experiments/glstm/text_multipass_cache.py \
    --modality image --resize 392 --pad-to-frames "$PAD" \
    --data_root data/mmred_images_park/seq_len_8/all_uniform \
    --limit "${LIMIT:-200}" --layers 14,16 --decode-offsets 9,13 --n-frames 8 --sample-seed 0 \
    --output "outputs/ladder/image_longN/position_ctrl/pad${PAD}/${TS}"
done
echo P1A_DONE

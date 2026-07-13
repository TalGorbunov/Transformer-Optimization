#!/bin/bash
#SBATCH --job-name=p1ab_smk
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "=== P1a smoke: pad-to-frames 32 ==="
python -u experiments/glstm/text_multipass_cache.py \
  --modality image --resize 392 --pad-to-frames 32 \
  --data_root data/mmred_images_park/seq_len_8/all_uniform \
  --limit 20 --layers 14,16 --decode-offsets 9,13 --n-frames 8 --sample-seed 0 \
  --output outputs/_scratch/p1a_pad32_smoke
echo "=== P1b smoke: companion gray ==="
python -u experiments/glstm/text_multipass_cache.py \
  --modality image --resize 392 --companion gray \
  --data_root data/mmred_images_park/seq_len_8/all_uniform \
  --limit 20 --layers 14,16 --decode-offsets 9,13 --n-frames 8 --sample-seed 0 \
  --output outputs/_scratch/p1b_gray_smoke
echo "=== P1b smoke: companion samescene ==="
python -u experiments/glstm/text_multipass_cache.py \
  --modality image --resize 392 --companion samescene \
  --data_root data/mmred_images_park/seq_len_8/all_uniform \
  --limit 20 --layers 14,16 --decode-offsets 9,13 --n-frames 8 --sample-seed 0 \
  --output outputs/_scratch/p1b_same_smoke
echo P1AB_SMOKE_DONE

#!/bin/bash
#SBATCH --job-name=a1_mp
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# A1-fu1: text-MMRED MULTIPASS message cache (isolated per-frame forwards) — write-cap test.
# Offsets match the joint blockcache {0,2,5,8,9,11,13} so block_read_completeness compares 1:1.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
python -u experiments/glstm/text_multipass_cache.py \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-400}" --layers "${LAYERS:-14,16,18}" \
  --decode-offsets "${DECODE_OFFSETS:-0,2,5,8,9,11,13}" \
  --text-style "${TEXT_STYLE:-rooms}" --sample-seed 0 \
  --output "outputs/ladder/text_mmred/multipass_cache/${TS}"

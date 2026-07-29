#!/bin/bash
#SBATCH --job-name=p42_digit
#SBATCH --partition=a100-public
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x-%j.out
# P4.2 (p0p2, PREREG plans/p4_PREREG.md): carrier + DIGIT readout, in-length — the
# caption-winner data/recipe VERBATIM minus the scratchpad (no --running-tally, poslist
# fmt flag unused in digit mode). Digit path skips gold>9 at train (documented).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROOTS="data/mmred_images_park/seq_len_2/all_uniform,data/mmred_images_park/seq_len_3/all_uniform,data/mmred_images_park/seq_len_4/all_uniform,data/mmred_images_park/seq_len_5/all_uniform,data/mmred_images_park/seq_len_6/all_uniform,data/mmred_images_park/seq_len_7/all_uniform,data/mmred_images_park/seq_len_8/all_uniform,data/mmred_cooc_balanced/seq_len_8/all_uniform,data/mmred_rooms_balanced/seq_len_8/all_uniform,data/mmred_niah_which/seq_len_8/all_uniform,data/mmred_union_or/seq_len_8/all_uniform,data/mmred_longN_park/seq_len_16/all_uniform,data/mmred_longN_park/seq_len_32/all_uniform,data/mmred_longN_park/seq_len_64/all_uniform,data/mmred_longN_park2/seq_len_32/all_uniform,data/mmred_longN_park2/seq_len_48/all_uniform"
python -u experiments/glstm/carrier_layer_cached.py \
  --jitter-gap 16 --grad-ckpt \
  --carrier-ckpt outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/carrier_best.pt \
  --data_root "$ROOTS" \
  --limit 900 \
  --epochs 5 \
  --l-open 12 \
  --seed 0 --split-seed 0 \
  --output outputs/ladder/image_longN/carrier_digit_inlength

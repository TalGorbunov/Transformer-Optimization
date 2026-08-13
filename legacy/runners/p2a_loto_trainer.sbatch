#!/bin/bash
#SBATCH --job-name=loto_train
#SBATCH --partition=a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x-%j.out
# P2a LOTO T1 writer (p0p2 campaign, PREREG: plans/scratchpad_loto_PREREG.md): the l12v2
# recipe VERBATIM minus the held-out task — 15 data roots (mmred_cooc_balanced EXCLUDED).
# FMT = the P0.1 format-sweep winner. --split-seed 0 pins arm A's train/eval permutation
# so every shared eval dirs-file stays held-out. DATA_ROOT hardcoded (--export comma gotcha).
# Env: FMT (required), OUTPUT (required), EPOCHS (default 5), LIMIT (default 900), SEED (0).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROOTS="data/mmred_images_park/seq_len_2/all_uniform,data/mmred_images_park/seq_len_3/all_uniform,data/mmred_images_park/seq_len_4/all_uniform,data/mmred_images_park/seq_len_5/all_uniform,data/mmred_images_park/seq_len_6/all_uniform,data/mmred_images_park/seq_len_7/all_uniform,data/mmred_images_park/seq_len_8/all_uniform,data/mmred_rooms_balanced/seq_len_8/all_uniform,data/mmred_niah_which/seq_len_8/all_uniform,data/mmred_union_or/seq_len_8/all_uniform,data/mmred_longN_park/seq_len_16/all_uniform,data/mmred_longN_park/seq_len_32/all_uniform,data/mmred_longN_park/seq_len_64/all_uniform,data/mmred_longN_park2/seq_len_32/all_uniform,data/mmred_longN_park2/seq_len_48/all_uniform"
python -u experiments/glstm/carrier_layer_cached.py \
  --scratchpad-format "${FMT:?set FMT}" \
  --running-tally --jitter-gap 16 --grad-ckpt \
  --carrier-ckpt outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/carrier_best.pt \
  --data_root "$ROOTS" \
  --limit "${LIMIT:-900}" \
  --epochs "${EPOCHS:-5}" \
  --l-open 12 \
  --seed "${SEED:-0}" \
  --split-seed "${SPLIT_SEED:-0}" \
  --output "${OUTPUT:?set OUTPUT}"

#!/bin/bash
#SBATCH --job-name=nat_train
#SBATCH --partition=a100-public
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x-%j.out
# P3a L3 (p0p2, PREREG plans/p3a_natural_PREREG.md): WINNING recipe verbatim (caption,
# L12, r8, park-distilled e_c, fixed criterion) on the 4 natural TRAIN roots (image-half
# A only; n=890, N in {8,16}). Exams run separately on the image-held-out EVAL roots.
# DATA_ROOT hardcoded (--export comma gotcha).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROOTS="data/mmred_natural_mm/seq_len_8/dist_far_train,data/mmred_natural_mm/seq_len_8/dist_near_train,data/mmred_natural_mm/seq_len_16/dist_far_train,data/mmred_natural_mm/seq_len_16/dist_near_train"
python -u experiments/glstm/carrier_layer_cached.py \
  --scratchpad-format caption \
  --running-tally --jitter-gap 16 --grad-ckpt \
  --carrier-ckpt outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/carrier_best.pt \
  --data_root "$ROOTS" \
  --limit "${LIMIT:-900}" \
  --epochs "${EPOCHS:-5}" \
  --l-open 12 \
  --seed 0 --split-seed 0 \
  --output "${OUTPUT:-outputs/ladder/natural_mm/carrier_caption_nat}"

#!/bin/bash
#SBATCH --job-name=st3_smokeC
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# Track-C smoke: (A) --no-qfirst, (B) --no-posreset tiny arms (occ/pos plumbing);
# (C) NIAH which-frame root through the cached digit trainer (parse + prep).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CK=outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/carrier_best.pt

echo "=== smokeC-A: no-qfirst tiny (steps8, 1 ep) ==="
python -u experiments/glstm/carrier_layer_cached.py --no-qfirst \
  --data_root data/mmred_images_park/seq_len_8/all_uniform \
  --shuffle-dirs 0 --limit 18 --train-frac 0.5 --epochs 1 \
  --carrier-ckpt "$CK" --output outputs/_scratch/st3_smokeC/noqfirst

echo "=== smokeC-B: no-posreset tiny (steps8, 1 ep) ==="
python -u experiments/glstm/carrier_layer_cached.py --no-posreset \
  --data_root data/mmred_images_park/seq_len_8/all_uniform \
  --shuffle-dirs 0 --limit 18 --train-frac 0.5 --epochs 1 \
  --carrier-ckpt "$CK" --output outputs/_scratch/st3_smokeC/noposreset

echo "=== smokeC-C: NIAH root through cached digit trainer (1 ep) ==="
python -u experiments/glstm/carrier_layer_cached.py \
  --data_root "data/mmred_images_park/seq_len_8/all_uniform,data/mmred_niah_which/seq_len_8/all_uniform" \
  --shuffle-dirs 0 --limit 9 --train-frac 0.5 --epochs 1 \
  --carrier-ckpt "$CK" --output outputs/_scratch/st3_smokeC/niah

echo "=== smokeC done ==="

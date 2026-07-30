#!/bin/bash
#SBATCH --job-name=st3_smokeA
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# Track-A smoke: scratchpad targets + jitter in carrier_layer_cached.py, then the
# eval-only scratchpad greedy decode (carrier_layer_lora.py) on the tiny ckpt via
# --dirs-file. Verifies: target render/roundtrip, ext-mask/pos plumbing, loss falls,
# TF metrics, ckpt flag, decode+parse (parse-fail rate reported).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== smokeA-1: scratchpad+jitter tiny train (steps5+steps8+cooc+rooms, 3 ep) ==="
python -u experiments/glstm/carrier_layer_cached.py \
  --data_root "data/mmred_images_park/seq_len_5/all_uniform,data/mmred_images_park/seq_len_8/all_uniform,data/mmred_cooc_balanced/seq_len_8/all_uniform,data/mmred_rooms_balanced/seq_len_8/all_uniform" \
  --shuffle-dirs 0 --limit 9 --train-frac 0.67 --epochs 3 \
  --scratchpad --jitter-gap 12 \
  --carrier-ckpt outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/carrier_best.pt \
  --output outputs/_scratch/st3_smokeA/train

RUN=$(ls -d outputs/_scratch/st3_smokeA/train/*/ | tail -1)
echo "=== smokeA-2: eval-only scratchpad decode on ${RUN} ==="
python -u experiments/glstm/carrier_layer_lora.py --eval-only \
  --ckpt "${RUN}carrier_layer_best.pt" \
  --dirs-file "${RUN}eval_dirs.txt" \
  --data_root ignored --limit 999 --decode-tokens 48 \
  --output outputs/_scratch/st3_smokeA/evalonly

echo "=== smokeA done ==="

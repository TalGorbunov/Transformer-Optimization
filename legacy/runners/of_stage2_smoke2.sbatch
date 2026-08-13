#!/bin/bash
#SBATCH --job-name=st2_smoke2
#SBATCH --partition=a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# Smoke round 2: the restructured carrier_layer_lora.py (multi-root mixture loader,
# streaming --eval-only + greedy multi-digit decode). outputs/_scratch/ only.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CKPT=outputs/ladder/image_longN/carrier_layer/20260718_122503_L17_r8/carrier_layer_best.pt

echo "=== smoke2 A: train path, single task (must match round-1 behavior) ==="
python -u experiments/glstm/carrier_layer_lora.py \
  --shuffle-dirs 0 --limit 27 --train-n 18 --epochs 1 \
  --output outputs/_scratch/st2_smoke2/train_single

echo "=== smoke2 B: train path, steps+cooc MIXTURE (task parse, stratified split, per-task eval) ==="
python -u experiments/glstm/carrier_layer_lora.py \
  --data_root "data/mmred_images_park/seq_len_8/all_uniform,data/mmred_cooc_balanced/seq_len_8/all_uniform" \
  --shuffle-dirs 0 --limit 14 --train-n 14 --epochs 1 \
  --output outputs/_scratch/st2_smoke2/train_mixture

echo "=== smoke2 C: eval-only streaming, steps N=8 ==="
python -u experiments/glstm/carrier_layer_lora.py \
  --eval-only --ckpt "$CKPT" \
  --shuffle-dirs 0 --limit 9 \
  --output outputs/_scratch/st2_smoke2/evalonly_n8

echo "=== smoke2 D: eval-only streaming, longN N=32 (multi-digit decode; gold up to 32) ==="
python -u experiments/glstm/carrier_layer_lora.py \
  --eval-only --ckpt "$CKPT" \
  --data_root data/mmred_longN_park/seq_len_32/all_uniform \
  --shuffle-dirs 0 --limit 13 \
  --output outputs/_scratch/st2_smoke2/evalonly_n32

echo "=== smoke2 done ==="

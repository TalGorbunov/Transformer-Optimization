#!/bin/bash
#SBATCH --job-name=st2_smoke4
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# Smoke round 4 (P1 prerequisite): pooled multi-N multi-task loader + lazy-mask cache.
# A) variable-N mixture (steps N2/N5/N8 + cooc + rooms) tiny train — verify mask/pos-debug,
#    per-group gold hists, group-stratified split, lazy-mask training path.
# B) eval-only regression on the truncated-prior ckpt (cached-mask path unchanged).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== smoke4 A: pooled variable-N mixture, tiny train ==="
python -u experiments/glstm/carrier_layer_lora.py \
  --data_root "data/mmred_images_park/seq_len_2/all_uniform,data/mmred_images_park/seq_len_5/all_uniform,data/mmred_images_park/seq_len_8/all_uniform,data/mmred_cooc_balanced/seq_len_8/all_uniform,data/mmred_rooms_balanced/seq_len_8/all_uniform" \
  --shuffle-dirs 0 --limit 9 --train-n 33 --epochs 1 \
  --carrier-init outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/carrier_best.pt \
  --output outputs/_scratch/st2_smoke4/train_pooled

echo "=== smoke4 B: eval-only regression (cached-mask path) ==="
python -u experiments/glstm/carrier_layer_lora.py --eval-only \
  --ckpt outputs/ladder/image_longN/carrier_layer/20260718_122503_L17_r8/carrier_layer_best.pt \
  --data_root data/mmred_images_park/seq_len_8/all_uniform \
  --shuffle-dirs 0 --limit 18 \
  --output outputs/_scratch/st2_smoke4/evalonly_regression

echo "=== smoke4 done ==="

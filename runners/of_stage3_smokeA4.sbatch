#!/bin/bash
#SBATCH --job-name=st3_smokeA4
#SBATCH --partition=a100-public
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%j.out
# A4 smoke: per-root LIMIT syntax + gold>9 scratchpad prep + N=32 hi-phase backward memory.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/glstm/carrier_layer_cached.py \
  --data_root "data/mmred_images_park/seq_len_8/all_uniform=4,data/mmred_longN_park/seq_len_32/all_uniform=14" \
  --shuffle-dirs 0 --limit 900 --train-frac 0.6 --epochs 1 \
  --scratchpad --jitter-gap 12 --grad-ckpt \
  --carrier-ckpt outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/carrier_best.pt \
  --output outputs/_scratch/st3_smokeA4/train
echo "=== smokeA4 done ==="

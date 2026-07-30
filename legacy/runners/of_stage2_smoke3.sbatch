#!/bin/bash
#SBATCH --job-name=st2_smoke3
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%j.out
# Smoke round 3: rooms-visited task parse in carrier_layer_lora (E5 prerequisite).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== smoke3: 3-task mixture steps+cooc+rooms, tiny ==="
python -u experiments/glstm/carrier_layer_lora.py \
  --data_root "data/mmred_images_park/seq_len_8/all_uniform,data/mmred_cooc_balanced/seq_len_8/all_uniform,data/mmred_rooms_balanced/seq_len_8/all_uniform" \
  --shuffle-dirs 0 --limit 8 --train-n 12 --epochs 1 \
  --output outputs/_scratch/st2_smoke3/train_mixture3

echo "=== smoke3 done ==="

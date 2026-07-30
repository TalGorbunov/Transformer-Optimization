#!/bin/bash
#SBATCH --job-name=carrier_cached
#SBATCH --partition=a100-public
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/glstm/carrier_layer_cached.py \
  ${EXTRA_FLAGS:-} \
  --carrier-ckpt "${CKPT:?}" \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-900}" \
  --epochs "${EPOCHS:-15}" \
  --l-open "${L_OPEN:-17}" \
  --lr-lora "${LR_LORA:-1e-4}" \
  --output "${OUTPUT:-outputs/ladder/image_longN/carrier_layer_cached}"

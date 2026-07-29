#!/bin/bash
#SBATCH --job-name=carrier_tok
#SBATCH --partition=a100-public
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/glstm/carrier_token_distill.py \
  ${EXTRA_FLAGS:-} \
  --objective "${OBJECTIVE:-proxy}" \
  --init "${INIT:-room}" \
  --k "${K:-1}" \
  --epochs "${EPOCHS:-15}" \
  --lr "${LR:-2e-3}" \
  --limit "${LIMIT:-300}" \
  --train-n "${TRAIN_N:-150}" \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --output "${OUTPUT:-outputs/ladder/image_longN/carrier_token}"

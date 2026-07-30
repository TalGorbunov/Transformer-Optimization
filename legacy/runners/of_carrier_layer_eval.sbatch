#!/bin/bash
#SBATCH --job-name=cl_eval
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# Stage-2 carrier-layer STREAMING eval (E3 length / E4a task transfer).
# Env: CKPT (required), DATA_ROOT, LIMIT, OUTPUT, EXTRA_FLAGS.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/glstm/carrier_layer_lora.py --eval-only \
  --ckpt "${CKPT:?set CKPT}" \
  ${EXTRA_FLAGS:-} \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-300}" \
  --output "${OUTPUT:-outputs/ladder/image_longN/carrier_layer_eval}"

#!/bin/bash
#SBATCH --job-name=c1_tokif
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# C1 [preapproved-smoke]: oracle tally as digits in the prompt, zero training. 5 arms, n<=150.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/readout/c1_token_interface.py \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-150}" --sample-seed 0 ${TEXT_FRAMES:+--text-frames} \
  --arms "${ARMS:-base_std,oracle_std,oracle_open,cf_in_open,cf_ood_open}" \
  --output "${OUTPUT:-outputs/_scratch/c1_token_interface}"

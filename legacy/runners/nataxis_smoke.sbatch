#!/bin/bash
#SBATCH --job-name=nataxis_smk
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u evaluations/scripts/patch_importence/native_axis_probe.py \
  --data_root data/mmred_longN_park/seq_len_32/all_uniform \
  --n-frames 32 --resize 392 --layers 14,16 --offset 9 --limit 4 --sample-seed 0 \
  --output outputs/_scratch/nataxis_memfix_smoke

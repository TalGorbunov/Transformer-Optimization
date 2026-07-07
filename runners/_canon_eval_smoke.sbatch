#!/bin/bash
#SBATCH --job-name=canon_eval
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=0:40:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u evaluations/scripts/eval_mmred_qwen25_vl_accuracy.py \
  --data-root data/mmred_images_park --seq-lens 1,2,8 --limit-per-seq 20 --load-in-4bit

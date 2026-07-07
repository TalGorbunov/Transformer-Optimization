#!/bin/bash
#SBATCH --job-name=readout_align
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u evaluations/scripts/patch_importence/probe_count_readout_alignment.py \
  --data_root "${DATA_ROOT}" --limit "${LIMIT:-150}" --tag "${TAG}" \
  --output "${OUTPUT:-outputs/frame_axis/probes/count_readout_alignment}"

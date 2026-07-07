#!/bin/bash
#SBATCH --job-name=extr_pool
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# Exp-1 Stage A: mean-pool vs max-pool vs attn-pool per-frame is-evidence AUC, by crowding.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u evaluations/scripts/patch_importence/extraction_pooling_probe.py \
  --max-samples "${MAX_SAMPLES:-40}" --output "${OUTPUT:-outputs/frame_axis/probes/extraction_pooling}"
echo "DONE"

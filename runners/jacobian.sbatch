#!/bin/bash
#SBATCH --job-name=jacobian
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/glstm/frame_jacobian_sensitivity.py --limit "${LIMIT:-40}"

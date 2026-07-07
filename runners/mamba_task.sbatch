#!/bin/bash
#SBATCH --job-name=mamba_task
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/glstm/frame_axis_aggregator_adapter.py --aggregator mamba \
  --tasks "${TASK}" --data-root "${DATA_ROOT}" --train-seq-lens 8 --ood-seq-lens "" \
  --epochs 6 --output "${OUTPUT}"

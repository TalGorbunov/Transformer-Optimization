#!/bin/bash
#SBATCH --job-name=lora_fix
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/glstm/lora_sft_baseline.py \
  --task "${TASK:-steps_in_room}" --data-root "${DATA_ROOT}" --train-seq-lens 8 \
  --lora-layers 8-18 --target "${TARGET}" --frame-sup-weight "${FSW:-0}" \
  --lora-r 16 --lr 2e-4 --epochs "${EPOCHS:-6}" --patience 4 \
  --holdout-counts "${HOLDOUT:-}" ${ISO:+--frame-isolation} --decompose --decomp-layer 19 --output "${OUTPUT}"

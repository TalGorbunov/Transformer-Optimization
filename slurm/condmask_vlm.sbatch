#!/bin/bash
# condmask VISION trainer wrapper (frozen 4-bit Qwen2.5-VL-7B on B200).
# Knobs: ARM (required), SEED, EXTRA, OUTPUT.
#   sbatch --export=ALL,ARM=C,SEED=0 slurm/condmask_vlm.sbatch
#SBATCH -J condmask-vlm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python scripts/condmask/train_condmask_vlm.py \
  --arm "${ARM:?set ARM=A..F}" \
  --seed "${SEED:-0}" \
  --output "${OUTPUT:-outputs/condmask_vlm}" \
  ${EXTRA:-}

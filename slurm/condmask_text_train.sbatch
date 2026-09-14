#!/bin/bash
# condmask text-pilot trainer wrapper — Decart CoreWeave cluster (B200 nodes).
# NOTE: this cluster is NOT the CLAUDE.md cluster; partitions here are
# gpu (default) / gpu-preemptable / gpu-prod, no QOS required (verified 2026-08-17).
# Knobs via --export: ARM (required), SEED, MODEL, EXTRA (free-form flags).
#   sbatch --export=ALL,ARM=C,SEED=0 slurm/condmask_text_train.sbatch
#SBATCH -J condmask-train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python scripts/condmask/train_condmask_text.py \
  --arm "${ARM:?set ARM=A..F}" \
  --seed "${SEED:-0}" \
  --model "${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
  --output "${OUTPUT:-outputs/condmask}" \
  ${EXTRA:-}

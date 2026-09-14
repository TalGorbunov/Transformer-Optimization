#!/bin/bash
# Many-needle dose-response (PREREG_AGG.md): one job = one MODEL. Lists use '+' separators.
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-3B-Instruct,LAYER=27 slurm/needles.sbatch
#   sbatch -p gpu-short -t 00:30:00 --export=ALL,MODEL=...,LAYER=27,N=5,KS=1+8+32,OUTPUT=outputs/_scratch/needles slurm/needles.sbatch
#SBATCH -J needles
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/condmask/needles.py \
  --model "${MODEL:?}" --layer "${LAYER:?}" --n "${N:-50}" --ks "${KS:-1+2+3+4+6+8+12+16+24+32+48+64}" \
  --regimes "${REGIMES:-H0+HF+HC+H32+DUP}" --output "${OUTPUT:-outputs/needles}" ${EXTRA:-}

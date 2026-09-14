#!/bin/bash
#SBATCH -J bindcount
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python outputs/_scratch/dbg/train_bindcount.py --model "${MODEL:-Qwen/Qwen2.5-7B-Instruct}" --arm "${ARM:?}" ${EXTRA:-}

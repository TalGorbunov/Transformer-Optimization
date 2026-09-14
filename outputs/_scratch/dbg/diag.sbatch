#!/bin/bash
#SBATCH -J diag
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
.venv/bin/python "outputs/_scratch/dbg/${SCRIPT:?}.py" --model "${MODEL:?}" --layer "${LAYER:?}" --n "${N:-24}" --output "${OUTPUT:-outputs/needles_${SCRIPT#needles_}}" ${EXTRA:-}

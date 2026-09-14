#!/bin/bash
#SBATCH -J run
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
# generic: SCRIPT=<name in outputs/_scratch/dbg> ARGS="<args, no commas>"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python "outputs/_scratch/dbg/${SCRIPT:?}.py" ${ARGS:-}

#!/bin/bash
#SBATCH -J v5_kernel_check
#SBATCH -t 00:03:00
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/diagnose_native_vision_v5_kernel_route.py --check

#!/bin/bash
#SBATCH -J v10_finalize
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
.venv/bin/python scripts/finalize_native_vision_v10.py "$@"

#!/bin/bash
#SBATCH -J identity_join_learned_plot
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python scripts/plot_native_identity_join_learned.py "$@"

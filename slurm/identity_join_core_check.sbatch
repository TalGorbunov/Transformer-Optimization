#!/bin/bash
#SBATCH -J identity_join_core_check
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:02:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python tests/test_parallel_local_binary_aggregation.py

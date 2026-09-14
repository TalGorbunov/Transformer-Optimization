#!/bin/bash
#SBATCH -J native_agg_v1_report
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python scripts/report_native_vision_v1.py

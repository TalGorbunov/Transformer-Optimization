#!/bin/bash
#SBATCH -J native_agg_report
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/native_aggregation_mpl
.venv/bin/python scripts/report_native_aggregation_p1.py

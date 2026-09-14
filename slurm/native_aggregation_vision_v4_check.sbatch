#!/bin/bash
#SBATCH -J native_agg_v4_check
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python tests/test_native_aggregation_vlm_v4.py
.venv/bin/python -m py_compile scripts/native_aggregation_vlm_v4.py tests/test_native_aggregation_vlm_v4.py
.venv/bin/python scripts/native_aggregation_vlm_v4.py --help >/dev/null

#!/bin/bash
#SBATCH -J native_recover_fit
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
RECOVERABILITY_INPUT="${RECOVERABILITY_DATA_INPUT:-/mnt/data/gabriele/gnn_transformer/v2_recoverability/harvest_${HARVEST_JOB:?Set HARVEST_JOB or RECOVERABILITY_DATA_INPUT}}"
.venv/bin/python scripts/fit_native_vision_recoverability.py \
    --data-input "$RECOVERABILITY_INPUT" \
    --output outputs/native_aggregation_vlm/v2/recoverability

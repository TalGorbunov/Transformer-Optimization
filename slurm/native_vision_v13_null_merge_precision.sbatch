#!/bin/bash
#SBATCH -J v13_null_merge_precision
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [[ "${1:-}" == "--self-test" ]]; then
    .venv/bin/python scripts/merge_native_vision_v13_precision.py --self-test
else
    V13_PRECISION_PLAN="${1:?Pass exact original worker plan and four completed shard directories}"
    shift
    .venv/bin/python scripts/merge_native_vision_v13_precision.py --plan "$V13_PRECISION_PLAN" --merge "$@"
fi

#!/bin/bash
#SBATCH -J native_marginal_audit
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
.venv/bin/python scripts/audit_native_vision_marginal_shortcuts.py "$@"

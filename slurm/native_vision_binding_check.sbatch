#!/bin/bash
#SBATCH -J native_binding_check
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
.venv/bin/python scripts/probe_native_vision_binding.py --self-test
.venv/bin/python scripts/probe_native_vision_binding.py --check-plan

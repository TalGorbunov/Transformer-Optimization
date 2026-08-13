#!/bin/bash
#SBATCH --job-name=b1_dprime
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# CACHES_FILE: file whose content is the comma cache map (env vars can't carry commas
# through sbatch --export — it splits on them).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/glstm/dprime_vs_n.py --caches "$(cat "${CACHES_FILE:?}")" --offset "${OFFSET:-9}" \
  --output "${OUTPUT:?}"

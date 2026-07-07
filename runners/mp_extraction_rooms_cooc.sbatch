#!/bin/bash
#SBATCH --job-name=mp_extract_probe
#SBATCH --partition=l40s-public
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/glstm/mp_extraction_rooms_cooc.py

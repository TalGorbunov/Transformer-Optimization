#!/bin/bash
#SBATCH --job-name=distinct_eval
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/glstm/benchmark_distinct.py

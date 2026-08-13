#!/bin/bash
#SBATCH --job-name=verify_sum
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/glstm/benchmark_verify.py --holdout-from 5

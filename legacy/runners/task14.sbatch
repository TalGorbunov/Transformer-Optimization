#!/bin/bash
#SBATCH --job-name=task14
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/glstm/task14_most_visited.py

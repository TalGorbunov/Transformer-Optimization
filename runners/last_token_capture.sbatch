#!/bin/bash
#SBATCH --job-name=lasttok
#SBATCH --partition=l40s-shared,rtx6k-shared,a100-public,l40s-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/glstm/last_token_capture.py --seq-lens 1,2,3,4,5,6,7,8 --limit 150 --resize 392 --layers 20,24,27
echo "last-token capture done $(date)"

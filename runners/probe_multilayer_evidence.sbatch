#!/bin/bash
#SBATCH --job-name=probe_multilayer
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u evaluations/scripts/probe_multilayer_evidence.py --seq-lens "${SEQ_LENS:-4,6,8}" --max-samples "${MAX_SAMPLES:-100}"

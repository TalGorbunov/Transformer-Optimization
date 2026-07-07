#!/bin/bash
#SBATCH --job-name=cache_ff
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/glstm/cache_frames_first_reps.py --task "${TASK:-steps_in_room}" \
  --data-root "${DATA_ROOT}" --seq-len "${SEQ_LEN:-8}" --read-layer 19 --limit "${LIMIT:-800}" \
  --output "${OUTPUT}"

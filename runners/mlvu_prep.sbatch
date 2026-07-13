#!/bin/bash
#SBATCH --job-name=mlvu_prep
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# MLVU-AC prep: decode downloaded count videos -> 128 uniform 448px frames + dup-detect GT
# -> data/mlvu_ac/. Idempotent (skips prepped qids); run once per download batch.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:/home/tal.gorbunov/.local/pyav-py39"
python -u experiments/mlvu/prep_ac_frames.py \
  --videos-root /home/tal.gorbunov/mlvu_transient/MLVU/video/count \
  --out data/mlvu_ac ${LIMIT:+--limit "$LIMIT"}

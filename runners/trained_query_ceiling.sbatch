#!/bin/bash
#SBATCH --job-name=trainq_ceil
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
CAP=${CAP:-outputs/ladder/image_longN/qkv_2x2/20260712_n500}
python -u experiments/glstm/trained_query_ceiling.py \
  --capture "$CAP" \
  --output  "$CAP/trained_query" \
  --layer   "${LAYER:-16}" \
  --epochs  "${EPOCHS:-400}" \
  --batch   "${BATCH:-512}" \
  --lr      "${LR:-1e-3}" \
  --inits   "${INITS:-joint,mp,random}"

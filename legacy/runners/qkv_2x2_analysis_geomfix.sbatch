#!/bin/bash
#SBATCH --job-name=qkv2x2_gf
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/glstm/qkv_2x2_analysis.py \
  --capture outputs/ladder/image_longN/qkv_2x2/20260712_geomfix \
  --output  outputs/ladder/image_longN/qkv_2x2/20260712_geomfix/analysis

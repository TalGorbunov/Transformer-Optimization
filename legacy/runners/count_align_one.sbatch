#!/bin/bash
#SBATCH --job-name=count_align1
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Single-condition count readout-alignment probe. Env: DATA, TAG, OUTDIR, TEXT(0/1), LIMIT.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LIMIT="${LIMIT:-160}"
TEXT_FLAG=""; [[ "${TEXT:-0}" == "1" ]] && TEXT_FLAG="--text"
mkdir -p "$OUTDIR"
echo "DATA=$DATA TAG=$TAG OUTDIR=$OUTDIR TEXT=$TEXT_FLAG LIMIT=$LIMIT"
python -u evaluations/scripts/patch_importence/probe_count_readout_alignment.py \
  --data_root "$DATA" $TEXT_FLAG --tag "$TAG" --limit "$LIMIT" --output "$OUTDIR" 2>&1 | tee "$OUTDIR/${TAG}.log"
echo "DONE -> $OUTDIR/$TAG"

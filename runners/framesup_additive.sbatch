#!/bin/bash
#SBATCH --job-name=framesup_add
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
# Correct per-frame-supervision test: ADDITIVE readout (Σσ of the supervised per-frame scores = answer).
# Reads the 'add' accuracy. FSW=0 vs FSW=1 isolates the calibration lever. seq8 IID.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/glstm/frame_axis_aggregator_adapter.py \
  --aggregator sum --tasks "${TASK:?}" \
  --frame-pool mean --count-readout additive --frame-sup-weight "${FSW:-0}" \
  --data-root "${DATA_ROOT:?}" \
  --train-seq-lens "8" --ood-seq-lens "" \
  --epochs 8 --patience 3 --read-layer 19 --d-mem 256 --lr 1e-3 \
  --output "${OUTPUT:?}"

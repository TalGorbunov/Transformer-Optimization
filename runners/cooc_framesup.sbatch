#!/bin/bash
#SBATCH --job-name=cooc_framesup
#SBATCH --partition=h200-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
# Tests whether per-frame supervision is the lever for co_occupancy (clean per-frame reps ~0.98,
# but count-only CE caps acc). sum aggregator, joint 1-forward, seq8 IID. FSW=0 vs FSW=1.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FSW="${FSW:-0}"; OUT="${OUTPUT:?}"
python -u experiments/glstm/frame_axis_aggregator_adapter.py \
  --aggregator sum --tasks co_occupancy \
  --frame-pool mean --count-readout ce --frame-sup-weight "${FSW}" \
  --data-root data/mmred_cooc_balanced \
  --train-seq-lens "8" --ood-seq-lens "" \
  --epochs 8 --patience 3 --read-layer 19 --d-mem 256 --lr 1e-3 \
  --output "${OUT}"

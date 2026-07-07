#!/bin/bash
#SBATCH --job-name=temporal_adapter
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Temporal MMRED tasks (first_in_room / last_in_room / span_in_room): order-dependent,
# so a permutation-invariant sum CANNOT compute them -> sum is the control, mamba (last-state,
# order-aware) is the test. Joint 1-forward read, LM-CE readout. Seq-lens baked in here to
# avoid the sbatch --export comma-split bug. Env knobs (all comma-free): TASK, AGG, MAMBA_READOUT, OUTPUT.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK="${TASK:?set TASK=first_in_room|last_in_room|span_in_room}"
AGG="${AGG:-mamba}"
MAMBA_READOUT="${MAMBA_READOUT:-last}"
OUT="${OUTPUT:?set OUTPUT}"

python -u experiments/glstm/frame_axis_aggregator_adapter.py \
  --aggregator "${AGG}" --mamba-readout "${MAMBA_READOUT}" \
  --tasks "${TASK}" \
  --count-readout ce --frame-pool mean \
  --train-seq-lens "5 6 7 8" --ood-seq-lens "4" \
  --max-train-per-seq 400 --epochs 8 --patience 3 \
  --read-layer 19 --d-mem 256 --lr 1e-3 \
  --output "${OUT}"

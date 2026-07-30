#!/bin/bash
#SBATCH --job-name=frame_agg_adapter
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

AGG="${AGGREGATOR:-seqmodel}"
EXTRA="${EXTRA:-}"
EPOCHS="${EPOCHS:-6}"
TRAIN_SEQ_LENS="${TRAIN_SEQ_LENS:-1,2,3,4}"
OOD_SEQ_LENS="${OOD_SEQ_LENS:-5,6,7,8}"

python -u experiments/glstm/frame_axis_aggregator_adapter.py \
  --aggregator "${AGG}" --epochs "${EPOCHS}" \
  --train-seq-lens "${TRAIN_SEQ_LENS}" --ood-seq-lens "${OOD_SEQ_LENS}" ${EXTRA} \
  --output "${OUTPUT:-outputs/frame_axis_aggregator_adapter}"

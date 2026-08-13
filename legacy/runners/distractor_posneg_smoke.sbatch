#!/bin/bash
#SBATCH --job-name=posneg_smoke
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SR="outputs_oh_man/frame_to_carrier_evidence_sum_probe_seq8_7b_20260521_164621"

python -u experiments/distractor/distractor_posneg_write_read_adapter_seq8_7b.py \
  --run-all --smoke --source-run "${SR}" \
  --gate-mode learned --streams posneg --run-name smoke_learned_posneg

python -u experiments/distractor/distractor_posneg_write_read_adapter_seq8_7b.py \
  --run-all --smoke --source-run "${SR}" \
  --gate-mode oracle --streams pos_only --run-name smoke_oracle_pos_only

echo SMOKE_ALL_OK

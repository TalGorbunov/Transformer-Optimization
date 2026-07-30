#!/bin/bash
#SBATCH --job-name=frozenro_smoke
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u experiments/distractor/distractor_posneg_write_read_adapter_seq8_7b.py \
  --run-all --smoke \
  --source-run outputs_oh_man/frame_to_carrier_evidence_sum_probe_seq8_7b_20260521_164621 \
  --gate-mode learned --streams posneg \
  --init-streams-from outputs_no_train/distractor_oracle_posneg_write_read_adapter_seq8_7b/oracle_posneg_write_read_5epochs/checkpoints/oracle_posneg_write_read_adapter_best.pt \
  --freeze-streams \
  --run-name smoke_frozenreadout

echo SMOKE_FROZENRO_OK

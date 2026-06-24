#!/bin/bash
#SBATCH --job-name=cd_extrap
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/glstm/probe_count_direction_extrapolation.py \
  --task "${TASK}" --data-root "${DATA_ROOT}" --read-layer "${READ_LAYER:-19}" \
  --fit-counts "${FIT_COUNTS:-0,1,2,3,4}" --test-counts "${TEST_COUNTS:-5,6,7,8}" \
  ${EXTRA:-} \
  --output "${OUTPUT:-outputs/frame_axis/probes/count_direction_extrap}"

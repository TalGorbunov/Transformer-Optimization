#!/bin/bash
#SBATCH --job-name=mamba_steps_ho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

AGG="${AGGREGATOR:-mamba}"
EPOCHS="${EPOCHS:-6}"
DATA_ROOT="${DATA_ROOT:-data/mmred_steps_1char}"
OUT="${OUTPUT:-outputs/frame_axis/steps1char_holdout58/${AGG}}"

# steps_in_room on minimal-crowding (1 char) data; count-extrapolation: train counts 0-4, OOD 5-8.
# --count-readout additive => extensive Σsigmoid count head that can extrapolate past trained labels.
python -u experiments/glstm/frame_axis_aggregator_adapter.py \
  --aggregator "${AGG}" \
  --tasks steps_in_room \
  --data-root "${DATA_ROOT}" \
  --train-seq-lens 8 --ood-seq-lens 8 \
  --holdout-counts 5,6,7,8 \
  --count-readout additive \
  --epochs "${EPOCHS}" \
  --output "${OUT}"

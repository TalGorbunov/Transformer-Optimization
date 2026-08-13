#!/bin/bash
#SBATCH --job-name=hb_locmap
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
  --task herbench_ac \
  --data_root "${DATA_ROOT:-data/herbench_ac/armB_ev_fill16}" \
  --n-frames "${N_FRAMES:-16}" \
  --limit "${LIMIT:-134}" \
  --layers "${LAYERS:-12,14,16,18,20}" \
  --carrier per_token --max-offset "${MAXOFF:-24}" \
  ${DECODE_OFFSETS:+--decode-offsets "$DECODE_OFFSETS"} \
  ${SAVE_MESSAGES:+--save-messages} \
  ${FENCE:+--fence-cross-frame} \
  --sample-seed "${SAMPLE_SEED:-0}" \
  --output "${OUTPUT:-outputs/herbench/probes/ac_locmap/$(date +%Y%m%d_%H%M%S)}"

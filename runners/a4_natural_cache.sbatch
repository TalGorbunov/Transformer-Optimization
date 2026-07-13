#!/bin/bash
#SBATCH --job-name=a4_cache
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# A4 instruments: carrier locmap + message cache per mmred_natural_v2 cell (herbench_ac
# meta format; model digit-argmax acc = the behavioral number for the cell).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
CELL="${CELL:?}"
python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
  --task herbench_ac --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data_root "data/mmred_natural_v2/${CELL}" \
  --limit "${LIMIT:-300}" --layers "${LAYERS:-14,16}" \
  --carrier per_token --n-frames 8 \
  --max-offset "${MAXOFF:-16}" --decode-offsets "${DECODE_OFFSETS:-8,9,10,11,12}" \
  --sample-seed 0 --save-messages \
  --output "outputs/ladder/natural/${CELL}/${TS}"

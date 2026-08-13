#!/bin/bash
#SBATCH --job-name=mlvu_cache
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# P0c: MLVU-AC carrier message cache on judge-labeled N=32 frames.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
  --task herbench_ac --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data_root data/mlvu_ac_n32judge \
  --limit 206 --layers 14,16 --carrier per_token --n-frames 32 --resize 392 \
  --max-offset 24 --decode-offsets 0,14,15,16,17,18 \
  --sample-seed 0 --save-messages \
  --output "outputs/ladder/mlvu_ac/msgcache_n32judge/${TS}"

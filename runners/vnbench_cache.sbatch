#!/bin/bash
#SBATCH --job-name=vnb_cache
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
  --task herbench_ac --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data_root data/vnbench_cnt_n32exact \
  --limit 300 --layers 14,16 --carrier per_token --n-frames 32 --resize 392 \
  --max-offset 20 --decode-offsets 0,12,13,14,15,16 \
  --sample-seed 0 --save-messages \
  --output "outputs/ladder/vnbench/msgcache_n32exact/${TS}"

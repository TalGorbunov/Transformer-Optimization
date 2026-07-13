#!/bin/bash
#SBATCH --job-name=b0a_res
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# B0a [preapproved-smoke]: deployed carrier d' (room-token messages, frames-first) at
# {512,392,336}px, crowded 5-char steps N=8, n=150. One job, three sequential probe runs.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ROOT="outputs/_scratch/b0a_res_smoke/${TS}"
mkdir -p "$ROOT"
for PX in ${RES_LIST:-512 392 336}; do
  echo "=== B0a resolution ${PX}px ==="
  python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
    --task count --model Qwen/Qwen2.5-VL-7B-Instruct \
    --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
    --limit "${LIMIT:-150}" --layers "${LAYERS:-14,16,20}" \
    --carrier per_token --n-frames 8 --resize "$PX" \
    --max-offset "${MAXOFF:-11}" --decode-offsets "${DECODE_OFFSETS:-9}" \
    --sample-seed 0 --save-messages \
    --output "${ROOT}/res${PX}"
done
echo "B0a smoke done -> ${ROOT}"

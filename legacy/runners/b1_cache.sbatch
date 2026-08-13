#!/bin/bash
#SBATCH --job-name=b1_cache
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=11:55:00
#SBATCH --output=logs/%x-%j.out
# B1: deployed carrier message caches vs N — JOINT then FENCED arms, images @392px.
# Env: SEQ_LEN (16/32/64/128 -> data/mmred_longN_park; 8 -> mmred_images_park), LIMIT,
#      ARMS="joint fenced" (subset ok), LAYERS, DECODE_OFFSETS, RESIZE.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
N="${SEQ_LEN:?}"
if [[ "$N" == "8" ]]; then
  DR="${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}"
else
  DR="${DATA_ROOT:-data/mmred_longN_park/seq_len_${N}/all_uniform}"
fi
for ARM in ${ARMS:-joint fenced}; do
  FLAG=""
  [[ "$ARM" == "fenced" ]] && FLAG="--fence-cross-frame"
  echo "=== B1 N=${N} arm=${ARM} ==="
  python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
    --task count --model Qwen/Qwen2.5-VL-7B-Instruct \
    --data_root "$DR" \
    --limit "${LIMIT:-300}" --layers "${LAYERS:-14,16}" \
    --carrier per_token --n-frames "$N" --resize "${RESIZE:-392}" \
    --max-offset "${MAXOFF:-13}" --decode-offsets "${DECODE_OFFSETS:-9,13}" \
    --sample-seed 0 --save-messages $FLAG \
    --output "outputs/ladder/image_longN/${ARM}/N${N}/${TS}"
done

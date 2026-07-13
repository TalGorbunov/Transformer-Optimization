#!/bin/bash
#SBATCH --job-name=a1_longN
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# Text-MMRED d' vs N preview (C-range design + B1's dilution math on text, no image data needed):
# joint carrier caches at N in {16,24,40} on states-only text data (data/mmred_text_longN).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
for NF in ${N_LIST:-16 24 40}; do
  echo "=== text-MMRED N=${NF} ==="
  python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
    --task count --model Qwen/Qwen2.5-VL-7B-Instruct --text-frames \
    --data_root "data/mmred_text_longN/seq_len_${NF}/all_uniform" \
    --limit "${LIMIT:-200}" --layers "${LAYERS:-14,16,18,20}" \
    --carrier per_token --n-frames "$NF" \
    --max-offset 13 --decode-offsets 9,13 \
    --sample-seed 0 --save-messages \
    --output "outputs/ladder/text_mmred/longN/${TS}/N${NF}"
done
echo "text longN done (ts=${TS})"

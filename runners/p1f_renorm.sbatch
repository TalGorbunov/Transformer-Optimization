#!/bin/bash
#SBATCH --job-name=renorm
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
N="${SEQ_LEN:?}"
if [[ "$N" == "8" ]]; then DR="data/mmred_images_park/seq_len_8/all_uniform";
else DR="data/mmred_longN_park/seq_len_${N}/all_uniform"; fi
python -u experiments/glstm/attn_renorm_patch.py \
  --data_root "$DR" --n-frames "$N" --limit "${LIMIT:-300}" --resize 392 \
  --layers 14,16 --patch-layers "${PATCH:-14,15,16,17}" --decode-offsets 9,13 \
  --sample-seed 0 \
  --output "outputs/ladder/image_longN/renorm/N${N}/${TS}"

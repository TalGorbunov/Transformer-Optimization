#!/bin/bash
#SBATCH --job-name=b1_mp
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# B1 multipass arm: each frame fed ALONE (image modality), carrier messages cached @392px.
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
python -u experiments/glstm/text_multipass_cache.py \
  --modality image --resize "${RESIZE:-392}" \
  --data_root "$DR" \
  --limit "${LIMIT:-300}" --layers "${LAYERS:-14,16}" \
  --decode-offsets "${DECODE_OFFSETS:-9,13}" \
  --n-frames "$N" --sample-seed 0 \
  --output "outputs/ladder/image_longN/multipass/N${N}/${TS}"

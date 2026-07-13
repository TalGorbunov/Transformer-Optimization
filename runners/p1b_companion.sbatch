#!/bin/bash
#SBATCH --job-name=p1b_comp
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
for COMP in ${COMPS:?}; do
  echo "=== P1b companion $COMP ==="
  python -u experiments/glstm/text_multipass_cache.py \
    --modality image --resize 392 --companion "$COMP" \
    --data_root data/mmred_images_park/seq_len_8/all_uniform \
    --limit "${LIMIT:-200}" --layers 14,16 --decode-offsets 9,13 --n-frames 8 --sample-seed 0 \
    --output "outputs/ladder/image_longN/companion/${COMP}/${TS}"
done
echo P1B_DONE

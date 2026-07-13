#!/bin/bash
#SBATCH --job-name=chunk_sweep
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# P1e: chunk-size sweep at N=32 @392px — carrier d' vs frames-per-forward k.
# k=1 (multipass N32 cache) and k=32 (joint N32 cache) already exist.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
for K in ${KS:-2 4 8 16}; do
  echo "=== chunk k=$K ==="
  python -u experiments/glstm/text_multipass_cache.py \
    --modality image --resize 392 --chunk-k "$K" \
    --data_root data/mmred_longN_park/seq_len_32/all_uniform \
    --limit "${LIMIT:-200}" --layers 14,16 --decode-offsets 9,13 \
    --n-frames 32 --sample-seed 0 \
    --output "outputs/ladder/image_longN/chunk_sweep/k${K}/${TS}"
done
echo CHUNK_SWEEP_DONE

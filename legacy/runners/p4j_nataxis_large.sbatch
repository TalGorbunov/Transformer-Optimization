#!/bin/bash
#SBATCH --job-name=nataxis_lg
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out
# P4j (N=32/64 legs): gradient backward needs an 80GB GPU.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
for N in 32 64; do
  python -u evaluations/scripts/patch_importence/native_axis_probe.py \
    --data_root "data/mmred_longN_park/seq_len_${N}/all_uniform" \
    --n-frames "$N" --resize 392 \
    --layers 14,16 --offset 9 --limit "${LIMIT:-60}" --sample-seed 0 \
    --output "outputs/ladder/image_longN/native_axis/N${N}/${TS}" || echo "N=$N FAILED"
done
echo NATAXIS_LG_DONE

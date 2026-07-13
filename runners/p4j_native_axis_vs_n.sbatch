#!/bin/bash
#SBATCH --job-name=nataxis_N
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out
# P4j: the model's native reading axis vs N (images @392px). Axes saved -> cos(axis_N, axis_8)
# and d' along each axis computed downstream on CPU.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
for N in 8 16 32 64; do
  if [[ "$N" == "8" ]]; then DR="data/mmred_images_park/seq_len_8/all_uniform";
  else DR="data/mmred_longN_park/seq_len_${N}/all_uniform"; fi
  python -u evaluations/scripts/patch_importence/native_axis_probe.py \
    --data_root "$DR" --n-frames "$N" --resize 392 \
    --layers 14,16 --offset 9 --limit "${LIMIT:-100}" --sample-seed 0 \
    --output "outputs/ladder/image_longN/native_axis/N${N}/${TS}"
done
echo NATAXIS_DONE

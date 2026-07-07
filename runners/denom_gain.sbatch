#!/bin/bash
#SBATCH --job-name=denom_gain
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u evaluations/scripts/patch_importence/denom_gain_vs_temp.py \
  --only "${ONLY:-all}" --per-seqlen "${PERSEQ:-15}" --seq-lens "${SEQLENS:-2,4,6,8}" \
  --mults "${MULTS:-2,4,8}" --band "${BAND:-late}" \
  --output "${OUTPUT:-outputs/frame_axis/probes/denom_gain/$(date +%Y%m%d_%H%M%S)}"

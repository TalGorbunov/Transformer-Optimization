#!/bin/bash
#SBATCH --job-name=probe_attnmap_dep
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# Deployed-stack attention map (GPU, minutes). Knobs: CKPT LIMIT LAYERS OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/presentation/attnmap_deployed/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/probe_attnmap_deployed.py \
  --ckpt "${CKPT:-checkpoints/carrier_layer_fmt_caption_best.pt}" \
  --limit "${LIMIT:-8}" --layers "${LAYERS:-8,20}" ${EXTRA:-} --output "$OUTPUT"

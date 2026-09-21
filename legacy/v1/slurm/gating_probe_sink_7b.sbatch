#!/bin/bash
#SBATCH --job-name=gate_p1_sink
#SBATCH -p l40s-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
# P1 of the gating campaign: sink diagnostic on the frozen 7B, plain vs deployed layout.
# Knobs: CKPT DATA_ROOT LIMIT EXTRA OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/gating/p1_sink_7b}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
run_logged "$OUTPUT" python -u scripts/gating/probe_sink_7b.py \
  --ckpt "${CKPT:-checkpoints/carrier_layer_fmt_caption_best.pt}" \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-8}" ${EXTRA:-} \
  --output "$OUTPUT"

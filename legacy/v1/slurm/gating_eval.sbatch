#!/bin/bash
#SBATCH --job-name=gate_eval
#SBATCH -p l40s-public
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%j.out
# P3.5 of the gating campaign: evaluate one trained arm across a root sweep.
# Knobs: CKPT ROOTS_FILE LIMIT PER_GOLD MODE DEC EXTRA OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/gating/p35_discriminator}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
run_logged "$OUTPUT" python -u scripts/gating/eval_gated.py \
  --ckpt "${CKPT:?set CKPT}" \
  --roots-file "${ROOTS_FILE:-slurm/lib/roots_gating_grid.txt}" \
  --limit "${LIMIT:-900}" --per-gold "${PER_GOLD:-40}" \
  --mode "${MODE:-tf}" --decode-tokens "${DEC:-320}" ${EXTRA:-} \
  --output "$OUTPUT"

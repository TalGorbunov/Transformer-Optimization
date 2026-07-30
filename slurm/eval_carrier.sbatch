#!/bin/bash
#SBATCH --job-name=eval_carrier
#SBATCH -p l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
# The exam. Knobs: CKPT DIRS_FILE (or DATA_ROOT) LIMIT DEC EXTRA OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/carrier/exam}"
run_logged "$OUTPUT" python -u scripts/eval_carrier.py \
  --ckpt "${CKPT:-checkpoints/carrier_layer_fmt_caption_best.pt}" \
  ${DIRS_FILE:+--dirs-file "$DIRS_FILE"} \
  ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
  --limit "${LIMIT:-150}" --decode-tokens "${DEC:-320}" ${EXTRA:-} \
  --output "$OUTPUT"

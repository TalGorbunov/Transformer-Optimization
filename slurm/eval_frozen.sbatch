#!/bin/bash
#SBATCH --job-name=eval_frozen
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# Frozen baseline / LoRA-drift arm. Knobs: DATA_ROOT DIRS_FILE LIMIT LORA_CKPT EXTRA OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/carrier/frozen}"
run_logged "$OUTPUT" python -u scripts/eval_frozen.py \
  ${DATA_ROOT:+--data_root "$DATA_ROOT"} ${DIRS_FILE:+--dirs-file "$DIRS_FILE"} \
  --limit "${LIMIT:-300}" --shuffle-dirs "${SHUFFLE_SEED:-0}" \
  ${LORA_CKPT:+--lora-ckpt "$LORA_CKPT"} ${EXTRA:-} --output "$OUTPUT"

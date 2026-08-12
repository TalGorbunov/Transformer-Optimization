#!/bin/bash
#SBATCH --job-name=ninv_registers
#SBATCH -p l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
# ninv Phase 1 register trainer. Knobs: ARM (raw|quantized) EPOCHS LIMIT OUTPUT EXTRA
# NOTE: pass --time explicitly at sbatch (DefaultTime is 2h on ALL partitions).
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/ninv/p1_${ARM:-raw}_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/ninv/train_registers.py \
  --leaf-input "${ARM:-raw}" --epochs "${EPOCHS:-10}" --limit "${LIMIT:-0}" \
  ${EXTRA:-} --output "$OUTPUT"

#!/bin/bash
#SBATCH --job-name=sp_probe
#SBATCH -p rtx6k-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# SPARSE gated Hahn probe — single invocation.
# Knobs: DATA_ROOT LIMIT CONTROLS ADAPTER ARMS GATE GOLD_SET OUTPUT EXTRA
# Always submit with explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/sparse/s3/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/sparse/probe_hahn_gated.py \
  --data_root "${DATA_ROOT:?set DATA_ROOT}" --limit "${LIMIT:-50}" \
  --controls "${CONTROLS:-12}" --seed "${SEED:-0}" \
  ${ADAPTER:+--peft-adapter "$ADAPTER"} ${ARMS:+--arms "$ARMS"} \
  ${GATE:+--gate "$GATE"} ${GOLD_SET:+--gold-set "$GOLD_SET"} ${EXTRA:-} \
  --output "$OUTPUT"

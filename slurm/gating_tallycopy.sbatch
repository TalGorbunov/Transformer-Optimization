#!/bin/bash
#SBATCH --job-name=gate_tallycopy
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
# Is teacher-forced COUNT accuracy measuring counting or copying the running tally?
# Knobs: CKPT ROOTS_FILE LIMIT DELTA OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/gating/p35_tallycopy}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
ROOTS_FILE="${ROOTS_FILE:-slurm/lib/roots_tallycopy.txt}"
ROOTS=$(grep -v '^\s*#' "$ROOTS_FILE" | grep -v '^\s*$' | tr '\n' ' ')
run_logged "$OUTPUT" python -u scripts/gating/probe_tally_copy.py \
  --ckpt "${CKPT:?set CKPT}" --roots $ROOTS \
  --limit "${LIMIT:-40}" --delta "${DELTA:-3}" \
  --output "$OUTPUT"

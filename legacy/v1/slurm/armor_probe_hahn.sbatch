#!/bin/bash
#SBATCH --job-name=armor_hahn
#SBATCH -p a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# ARMOR Exp A: Hahn sensitivity/margin probe. Knobs: DATA_ROOT LIMIT CONTROLS OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/armor/hahn/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/armor/probe_hahn.py \
  --data_root "${DATA_ROOT:?set DATA_ROOT}" --limit "${LIMIT:-50}" \
  --controls "${CONTROLS:-12}" ${EXTRA:-} --output "$OUTPUT"

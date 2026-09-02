#!/bin/bash
#SBATCH --job-name=armor_mlvu_cap
#SBATCH -p a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# ARMOR C step 1: per-frame VLM records on MLVU-AC. Knobs: LIMIT OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
run_logged "${OUTPUT:?set OUTPUT}" python -u scripts/armor/mlvu_caption.py \
  --limit "${LIMIT:-0}" ${EXTRA:-} --output "$OUTPUT"

#!/bin/bash
#SBATCH --job-name=ninv_possym
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/ninv/$(date +%Y%m%d_%H%M%S)_possym_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/ninv/pos_symmetry_check.py --output "$OUTPUT"

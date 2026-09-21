#!/bin/bash
#SBATCH --job-name=sp_s10b_smoke
#SBATCH -p rtx6k-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -o logs/%x-%j.out
# S10b smoke: tau=1.0 must reproduce the S10 smoke anchor (outputs/_scratch/sparse_w3/s10_smoke); tau=2.0 must differ.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
run_logged outputs/_scratch/sparse_w3/s10b_smoke_tau1 python -u scripts/sparse/probe_attn_photo.py \
  --arm p1b --n 8 --k-list 2,4 --limit 3 --attn-sharpen 1.0 --sharpen-from-layer 12 --output outputs/_scratch/sparse_w3/s10b_smoke_tau1
run_logged outputs/_scratch/sparse_w3/s10b_smoke_tau2 python -u scripts/sparse/probe_attn_photo.py \
  --arm p1b --n 8 --k-list 2,4 --limit 3 --attn-sharpen 2.0 --sharpen-from-layer 12 --output outputs/_scratch/sparse_w3/s10b_smoke_tau2
echo "S10b SMOKE COMPLETE"

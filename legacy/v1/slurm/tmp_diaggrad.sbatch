#!/bin/bash
#SBATCH --job-name=sp_diaggrad
#SBATCH -p l40s-shared
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH -o logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
run_logged outputs/_scratch/sparse_smoke/diag_grad python -u scripts/sparse/diag_grad_mask.py \
  --output outputs/_scratch/sparse_smoke/diag_grad

#!/bin/bash
#SBATCH --job-name=sp_s10
#SBATCH -p l40s-shared
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH -o logs/%x-%j.out
# S10 attention photograph chain. MODE=smoke|p1b|gated|frozen. Explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
MODE="${MODE:?}"
case "$MODE" in
smoke)
  run_logged outputs/_scratch/sparse_w3/s10_smoke python -u scripts/sparse/probe_attn_photo.py \
    --arm p1b --n 8 --k-list 2,4 --limit 3 --output outputs/_scratch/sparse_w3/s10_smoke
  run_logged outputs/_scratch/sparse_w3/s10_smoke_g python -u scripts/sparse/probe_attn_photo.py \
    --arm gated --n 8 --k-list 2,4 --limit 3 --output outputs/_scratch/sparse_w3/s10_smoke_g
  ;;
p1b)
  for N in 8 16 32 64; do
    run_logged "outputs/sparse/s10/p1b_N${N}" python -u scripts/sparse/probe_attn_photo.py \
      --arm p1b --n $N --k-list 2,4,8 --limit 30 --output "outputs/sparse/s10/p1b_N${N}"
  done
  ;;
gated)
  for N in 8 32 128; do
    run_logged "outputs/sparse/s10/gated_N${N}" python -u scripts/sparse/probe_attn_photo.py \
      --arm gated --n $N --k-list 2,4,8 --limit 30 --output "outputs/sparse/s10/gated_N${N}"
  done
  ;;
frozen)
  run_logged outputs/sparse/s10/frozen_N32 python -u scripts/sparse/probe_attn_photo.py \
    --arm frozen --n 32 --k-list 2,4,8 --limit 30 --output outputs/sparse/s10/frozen_N32
  ;;
esac
echo "S10 ($MODE) COMPLETE"

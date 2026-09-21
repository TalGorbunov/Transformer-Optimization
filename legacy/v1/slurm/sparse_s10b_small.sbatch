#!/bin/bash
#SBATCH --job-name=sp_s10b_small
#SBATCH -p rtx6k-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -o logs/%x-%j.out
# S10b small-N photograph: N=2 (k 1,2) and N=4 (k 1,2,4) for frozen / p1b / gated. Explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
for ARM in frozen p1b gated; do
  run_logged "outputs/sparse/s10/${ARM}_N2" python -u scripts/sparse/probe_attn_photo.py --arm $ARM --n 2 --k-list 1,2 --limit 30 --output "outputs/sparse/s10/${ARM}_N2"
  run_logged "outputs/sparse/s10/${ARM}_N4" python -u scripts/sparse/probe_attn_photo.py --arm $ARM --n 4 --k-list 1,2,4 --limit 30 --output "outputs/sparse/s10/${ARM}_N4"
done
echo "S10b SMALL COMPLETE"

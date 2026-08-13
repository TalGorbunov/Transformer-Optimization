#!/bin/bash
#SBATCH --job-name=nat_frozen
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# P3a L0 (p0p2): frozen baseline on the 4 image-held-out natural eval roots (plain prompt).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for ROOT in \
  data/mmred_natural_mm/seq_len_8/dist_far_eval \
  data/mmred_natural_mm/seq_len_8/dist_near_eval \
  data/mmred_natural_mm/seq_len_16/dist_far_eval \
  data/mmred_natural_mm/seq_len_16/dist_near_eval; do
  python -u experiments/glstm/frozen_baseline_eval.py \
    --data_root "$ROOT" --limit 300 \
    --output outputs/ladder/natural_mm/frozen_baseline
done

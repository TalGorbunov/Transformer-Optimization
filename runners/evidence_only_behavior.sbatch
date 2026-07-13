#!/bin/bash
#SBATCH --job-name=evonly_beh
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# P3: frozen evidence-only behavioral number (approved 2026-07-09 "misc" item) —
# pure-aggregation regime: every shown frame is evidence, gold = seq_len; n~19/seq len.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
for N in 1 2 3 4 5 6 7 8; do
  python -u experiments/readout/behavior_vs_n.py \
    --data_root "data/mmred_images_park_evidence_only_seq1_8/seq_len_${N}/all_uniform" \
    --limit "${LIMIT:-19}" --resize 0 --sample-seed 0 \
    --output "outputs/ladder/evidence_only_behavior/${TS}/N${N}"
done
echo EVONLY_DONE

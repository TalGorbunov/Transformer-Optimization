#!/bin/bash
#SBATCH --job-name=text_frames_prec
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEQ_LENS="${SEQ_LENS:-1,2,3,4,5,6,7,8}"
TASKS="${TASKS:-steps_in_room rooms_visited co_occupancy}"
SCRIPT=evaluations/scripts/eval_mmred_text_frames_acc.py

# Precision lever (Barbero: low precision worsens representational collapse / saturation).
# Same code path, plain single-token, full-scene text; compare 4-bit nf4 vs full bf16 weights
# (compute dtype bf16 in both, so this isolates WEIGHT precision). Compare the bias / saturation.
for TASK in ${TASKS}; do
  echo "=== NF4 (4-bit) task=${TASK} ==="
  python -u "${SCRIPT}" --task "${TASK}" --seq-lens "${SEQ_LENS}" --dtype bfloat16 --load-in-4bit \
    --max-samples "${MAXS:-80}" --max-new-tokens 8 --log-predictions \
    --output "${OUT_NF4:-outputs/eval_mmred_text_frames_acc_prec_nf4}"
done
for TASK in ${TASKS}; do
  echo "=== BF16 (full) task=${TASK} ==="
  python -u "${SCRIPT}" --task "${TASK}" --seq-lens "${SEQ_LENS}" --dtype bfloat16 --no-load-in-4bit \
    --max-samples "${MAXS:-80}" --max-new-tokens 8 --log-predictions \
    --output "${OUT_BF16:-outputs/eval_mmred_text_frames_acc_prec_bf16}"
done

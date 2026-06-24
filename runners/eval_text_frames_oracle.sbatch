#!/bin/bash
#SBATCH --job-name=text_frames_oracle
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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

# Oracle-list = only the queried entities' per-frame rooms (no scene to scan). Isolates FUSION
# from RETRIEVAL (HERBench's oracle-frame study, in text). Plain single-token + CoT.
for TASK in ${TASKS}; do
  echo "=== ORACLE PLAIN task=${TASK} ==="
  python -u "${SCRIPT}" --task "${TASK}" --seq-lens "${SEQ_LENS}" --oracle-list \
    --max-samples "${PLAIN_MAX:-80}" --max-new-tokens 8 --log-predictions \
    --output "${OUTPUT:-outputs/eval_mmred_text_frames_acc_oracle}"
done
for TASK in ${TASKS}; do
  echo "=== ORACLE COT task=${TASK} ==="
  python -u "${SCRIPT}" --task "${TASK}" --seq-lens "${SEQ_LENS}" --oracle-list --cot \
    --max-samples "${COT_MAX:-50}" --max-new-tokens 512 --log-predictions \
    --output "${COT_OUTPUT:-outputs/eval_mmred_text_frames_acc_oracle_cot}"
done

#!/bin/bash
#SBATCH --job-name=frame_agg_logic
#SBATCH --partition=l40s-shared
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCRIPT=experiments/glstm/frame_axis_aggregator_cached.py
COMMON="--train-seq-lens 1,2,3,4,5,6,7,8 --ood-seq-lens 99 --cap-per-seq 400 --epochs ${EPOCHS:-120}"
for AGG in logic summax deepsets; do
  echo "================ LOGIC-TEST: ${AGG} ================"
  python -u "${SCRIPT}" ${COMMON} --aggregator "${AGG}" --output outputs/frame_axis_logic_test/${AGG}
done
echo "==== LOGIC-TEST DONE ===="

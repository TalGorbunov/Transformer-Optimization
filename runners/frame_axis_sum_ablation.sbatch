#!/bin/bash
#SBATCH --job-name=frame_agg_sumabl
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

SCRIPT=experiments/glstm/frame_axis_aggregator_cached.py
COMMON="--train-seq-lens 1,2,3,4,5,6 --ood-seq-lens 7,8 --epochs ${EPOCHS:-120}"
OUT=outputs/frame_axis_sum_ablation

for AGG in sum summax deepsets; do
  echo "================ SUM-ABLATION: ${AGG} ================"
  python -u "${SCRIPT}" ${COMMON} --aggregator "${AGG}" --output "${OUT}/${AGG}"
done
echo "==== SUM-ABLATION DONE ===="

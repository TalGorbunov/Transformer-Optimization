#!/bin/bash
#SBATCH --job-name=c2_codebook
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# C2: digit-compositional soft-token injection vs per-count C-control (approved Q5,
# incl. fact-slot site amendment). Text-MMRED N=40 context, frozen 7B.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/readout/c2_digit_codebook.py \
  --data_root "${DATA_ROOT:-data/mmred_text_longN/seq_len_40/all_uniform}" \
  --routes "${ROUTES:-digit_fact,digit_ans,count_fact,token_fact}" \
  --reps-per-count "${REPS:-40}" --epochs "${EPOCHS:-2}" --lr "${LR:-5e-2}" \
  --eval-per-count "${EVAL_PER:-4}" --seed 0 \
  --inject-level "${INJECT_LEVEL:-embed}" \
  --output "${OUTPUT:-outputs/readout/c2_digit_codebook}"

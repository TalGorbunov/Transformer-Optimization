#!/bin/bash
#SBATCH --job-name=mamba_lm
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Frozen Mamba-LM baseline on text-MMRED (pure-PyTorch path, no mamba-ssm).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1   # use the cached download; no network on compute node
python -u evaluations/scripts/mamba_lm_text_eval.py \
  --model "${MODEL:-tiiuae/falcon-mamba-7b-instruct}" \
  --max-samples "${MAX_SAMPLES:-40}" --seq-lens "${SEQ_LENS:-4,6,8}" \
  --output "${OUTPUT:-outputs/frame_axis/mamba_lm_text}"
echo "DONE"

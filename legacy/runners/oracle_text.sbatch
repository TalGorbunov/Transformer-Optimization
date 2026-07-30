#!/bin/bash
#SBATCH --job-name=rv_textoracle
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u evaluations/scripts/oracle_text_distinct_count.py --task "${TASK:?}" --model-name Qwen/Qwen2.5-VL-7B-Instruct --max-samples "${N:-120}"

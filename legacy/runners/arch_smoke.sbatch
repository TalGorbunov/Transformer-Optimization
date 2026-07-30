#!/bin/bash
#SBATCH --job-name=arch_smoke
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
# NEW venv (py3.11 on /rg) -- NEVER the shared .venv
source /rg/shocher_prj/tal.gorbunov/venv_arch/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/rg/shocher_prj/tal.gorbunov/hf_arch
mkdir -p "$HF_HOME" outputs/_scratch/arch_smoke
echo "node: $(hostname)"; df -h /rg | tail -1; nvidia-smi -L
python -u experiments/arch_battery/smoke_models.py \
  --only "${ONLY:-}" \
  --out outputs/_scratch/arch_smoke/smoke_results_${SLURM_JOB_ID}.json

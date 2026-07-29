#!/bin/bash
#SBATCH --job-name=p43_noharm
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# P4.3 (p0p2, PREREG plans/p4_PREREG.md): no-harm bench on the plain SFT adapter —
# same 500+500 MME/POPE protocol as job 124508 (carrier ref: -0.2/-1.4 pts). Band |d|<=2.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/glstm/noharm_bench.py \
  --peft-adapter outputs/ladder/image_longN/sft_control_le8_v2/20260720_191541_lora/adapter \
  --output outputs/ladder/image_longN/noharm_bench_sft

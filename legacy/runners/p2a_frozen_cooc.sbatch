#!/bin/bash
#SBATCH --job-name=loto_frozen
#SBATCH --partition=a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# P2a Arm 4 (PREREG plans/scratchpad_loto_PREREG.md): frozen floor on the SAME cooc items
# as Arms 1/3 (dirs-files; digit protocol -> gold>9 skipped at N=32, documented).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=outputs/ladder/image_longN/carrier_tally_l12v2/20260721_071710_L12_r8
L=outputs/ladder/image_longN/carrier_loto_nococ/20260724_064756_L12_r8
python -u experiments/glstm/frozen_baseline_eval.py \
  --dirs-file $A/eval_dirs_cooc_all.txt --limit 300 \
  --output outputs/ladder/image_longN/loto_frozen_coocN8
python -u experiments/glstm/frozen_baseline_eval.py \
  --dirs-file $L/eval_dirs_coocN32.txt --limit 300 \
  --output outputs/ladder/image_longN/loto_frozen_coocN32

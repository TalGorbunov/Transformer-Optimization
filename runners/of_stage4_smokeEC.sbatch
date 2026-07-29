#!/bin/bash
#SBATCH --job-name=st4_smokeEC
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# E-C smoke: carriers-at-end layout in carrier_token_distill.py — verify the [mask-debug E-C]
# line (carrier allowed-keys identical = prefix+question+one_block+self), frames-first teacher
# span parse (NF+1 occurrences), distill loss falls on n=8.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/glstm/carrier_token_distill.py \
  --carriers-at-end ${SMOKE_EXTRA:-} --objective distill \
  --data_root data/mmred_images_park/seq_len_8/all_uniform \
  --shuffle-dirs 0 --limit 8 --train-n 5 --epochs 2 \
  --output outputs/_scratch/st4_smokeEC
echo "=== smokeEC done ==="

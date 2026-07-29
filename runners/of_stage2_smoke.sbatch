#!/bin/bash
#SBATCH --job-name=st2_smoke
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# Smoke for the E1 code changes (--shuffle-dirs, [gold-hist], carrier-layer [mask-debug]/
# [pos-debug]). Tiny limits, everything to outputs/_scratch/.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== smoke 1: carrier_layer_lora (shuffle, hist, mask/pos-debug) ==="
python -u experiments/glstm/carrier_layer_lora.py \
  --shuffle-dirs 0 --limit 27 --train-n 18 --epochs 1 \
  --output outputs/_scratch/st2_smoke/carrier_layer

echo "=== smoke 2: carrier_token_distill (shuffle, hist) ==="
python -u experiments/glstm/carrier_token_distill.py \
  --objective distill --shuffle-dirs 0 --limit 18 --train-n 9 --epochs 1 \
  --output outputs/_scratch/st2_smoke/carrier_token

echo "=== smoke 3: replica_carrier_probe qfirst blockfence (shuffle, hist) ==="
python -u experiments/glstm/replica_carrier_probe.py \
  --question-first --fence-frames --fence-blocks --reset-positions \
  --shuffle-dirs 0 --limit 18 --layers 16 \
  --output outputs/_scratch/st2_smoke/replica_probe

echo "=== smoke done ==="

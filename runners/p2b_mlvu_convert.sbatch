#!/bin/bash
#SBATCH --job-name=mlvu_conv
#SBATCH --partition=a100-public
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# P2b prep (p0p2): MLVU-AC -> MMRED-format sample dirs at N=32 and N=128 (392px), so the
# carrier eval (carrier_layer_lora.py --eval-only --alien-task) consumes them unchanged.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/mlvu/convert_ac_to_mmred.py --n-frames 32 --resize 392 \
  --out-root data/mlvu_ac_mmred
python -u experiments/mlvu/convert_ac_to_mmred.py --n-frames 128 --resize 392 \
  --out-root data/mlvu_ac_mmred

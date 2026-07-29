#!/bin/bash
#SBATCH --job-name=mlvu_carrier
#SBATCH --partition=a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x-%j.out
# P2b (p0p2, PREREG in plans/p0p2_STATE.md): winning-format carrier ckpt on MLVU-AC,
# zero-shot, 32-frame arm (dense N=128 ruled prohibitive: ~20 min/sample x 206 > 60h).
# --alien-task: MLVU questions match no MMRED template — decode-only scoring; the MCQ
# nearest-option mapping is applied at collection from the dumped transcripts + mcq.json.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/glstm/carrier_layer_lora.py --eval-only --alien-task \
  --ckpt "${CKPT:?set CKPT}" \
  --dirs-file "${DIRS_FILE:?set DIRS_FILE}" \
  --limit "${LIMIT:-300}" \
  --decode-tokens "${DEC:-320}" \
  --dump-decodes "${DUMPS:-206}" \
  --output "${OUTPUT:?set OUTPUT}"

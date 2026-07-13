#!/bin/bash
#SBATCH --job-name=qkv500_anl
#SBATCH --partition=l40s-shared,rtx6k-shared,a100-public,l40s-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# Firm-up battery on the n=500 capture (13.4GB → needs high RAM; 4h_0g's 16G cap OOM'd).
# GPU QOS used only to get a high-mem node; analyses are CPU-only.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
CAP=outputs/ladder/image_longN/qkv_2x2/20260712_n500

echo "=== [1/3] 2x2 grid + own-geometry anchor (n=500) $(date) ==="
python -u experiments/glstm/qkv_2x2_analysis.py --capture "$CAP" --output "$CAP/analysis" || echo "2x2 FAILED"

echo "=== [2/3] encoding un-mixer (n=500) $(date) ==="
python -u experiments/glstm/encoding_unmixer.py --capture "$CAP" --output "$CAP/unmixer" --kind both || echo "encoding-unmix FAILED"

echo "=== [3/3] query un-mixer (~4000 examples) $(date) ==="
python -u experiments/glstm/query_unmixer.py --capture "$CAP" --output "$CAP/query_unmixer" || echo "query-unmix FAILED"

echo "=== BATTERY DONE $(date) ==="

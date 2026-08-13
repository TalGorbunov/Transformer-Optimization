#!/bin/bash
#SBATCH --job-name=consol_ladder
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# Consolidation ladder: decode-then-count (joint frame tokens) vs last-token count -> where is precision lost?
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LIMIT="${LIMIT:-160}"
RUN="${OUT_ROOT:-$PWD/outputs/frame_axis/probes/consolidation_ladder}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"; echo "OUT=$RUN LIMIT=$LIMIT"
DATA="${DATA:-$PWD/data/mmred_steps_balanced/seq_len_8/all_uniform}"
python -u evaluations/scripts/patch_importence/consolidation_ladder.py \
  --data_root "$DATA" --tag "${TAG:-steps_5char}" --limit "$LIMIT" --layer "${LAYER:-19}" --output "$RUN" 2>&1 | tee "$RUN/ladder.log"
cp "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$RUN/" 2>/dev/null || true
echo "DONE -> $RUN"

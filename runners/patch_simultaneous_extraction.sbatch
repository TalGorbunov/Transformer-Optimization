#!/bin/bash
#SBATCH --job-name=sim_extract
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:45:00
#SBATCH --output=logs/%x-%j.out

# Simultaneous-extraction causal patch: patch evidence-frames-only (uncompeted) reps into the
# full N-frame run; does the gold count recover? (competed-extraction vs pure-aggregation)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LIMIT="${LIMIT:-40}"
LAYERS="${LAYERS:-14,19}"
OUT_ROOT="${OUT_ROOT:-$PWD/outputs/frame_axis/probes/simultaneous_extraction_patch}"
RUN="$OUT_ROOT/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
echo "OUT=$RUN LIMIT=$LIMIT LAYERS=$LAYERS"
P=evaluations/scripts/patch_importence/patch_simultaneous_extraction.py

python -u "$P" --data_root "$PWD/data/mmred_steps_balanced/seq_len_8/all_uniform" \
  --tag steps_5char --layers "$LAYERS" --limit "$LIMIT" --output "$RUN" 2>&1 | tee "$RUN/steps_5char.log"

cp "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$RUN/" 2>/dev/null || true
echo "DONE -> $RUN"

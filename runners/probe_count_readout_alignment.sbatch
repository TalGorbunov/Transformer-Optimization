#!/bin/bash
#SBATCH --job-name=count_align
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:45:00
#SBATCH --output=logs/%x-%j.out

# Absent-vs-misaligned probe across 3 conditions of the steps_in_room count task:
#   image_uncrowded (1char), text_uncrowded (1char as text), image_crowded (5char).
# For each: last-token Ridge R^2 (count present?) + classification acc/majority + cosine of the
# count direction to the count-token unembedding rows (present-but-misaligned?).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LIMIT="${LIMIT:-120}"
OUT_ROOT="${OUT_ROOT:-$PWD/outputs/frame_axis/probes/count_readout_alignment}"
RUN="$OUT_ROOT/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
echo "OUT=$RUN LIMIT=$LIMIT"

UNCROWD="$PWD/data/mmred_steps_1char/seq_len_8/all_uniform"
CROWD="$PWD/data/mmred_steps_balanced/seq_len_8/all_uniform"
P=evaluations/scripts/patch_importence/probe_count_readout_alignment.py

echo "########## image_uncrowded (1char) ##########"
python -u "$P" --data_root "$UNCROWD" --tag image_uncrowded --limit "$LIMIT" --output "$RUN" 2>&1 | tee "$RUN/image_uncrowded.log"

echo "########## text_uncrowded (1char as text) ##########"
python -u "$P" --data_root "$UNCROWD" --text --tag text_uncrowded --limit "$LIMIT" --output "$RUN" 2>&1 | tee "$RUN/text_uncrowded.log"

echo "########## image_crowded (5char) ##########"
python -u "$P" --data_root "$CROWD" --tag image_crowded --limit "$LIMIT" --output "$RUN" 2>&1 | tee "$RUN/image_crowded.log"

cp "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$RUN/" 2>/dev/null || true
echo "DONE -> $RUN"

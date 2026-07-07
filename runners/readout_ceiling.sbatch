#!/bin/bash
#SBATCH --job-name=readceil
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:45:00
#SBATCH --output=logs/%x-%j.out

# Readout-ceiling: model-acc vs linear(digit-repair) vs MLP readout, across the 2x2 (modality x crowding).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LIMIT="${LIMIT:-160}"
RUN="${OUT_ROOT:-$PWD/outputs/frame_axis/probes/readout_ceiling}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"; echo "OUT=$RUN LIMIT=$LIMIT"
UNCROWD="$PWD/data/mmred_steps_1char/seq_len_8/all_uniform"
CROWD="$PWD/data/mmred_steps_balanced/seq_len_8/all_uniform"
P=evaluations/scripts/patch_importence/readout_ceiling.py

python -u "$P" --data_root "$UNCROWD"        --tag image_uncrowded --limit "$LIMIT" --output "$RUN" 2>&1 | tee "$RUN/image_uncrowded.log"
python -u "$P" --data_root "$UNCROWD" --text  --tag text_uncrowded  --limit "$LIMIT" --output "$RUN" 2>&1 | tee "$RUN/text_uncrowded.log"
python -u "$P" --data_root "$CROWD"           --tag image_crowded   --limit "$LIMIT" --output "$RUN" 2>&1 | tee "$RUN/image_crowded.log"
python -u "$P" --data_root "$CROWD"    --text  --tag text_crowded    --limit "$LIMIT" --output "$RUN" 2>&1 | tee "$RUN/text_crowded.log"
cp "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$RUN/" 2>/dev/null || true
echo "DONE -> $RUN"

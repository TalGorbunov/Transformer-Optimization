#!/bin/bash
#SBATCH --job-name=extr_crowd
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# Phase-1 extraction experiment: single-frame P(yes) detection ("Is C in R in THIS frame?")
# across entity-crowding levels (1char / 2char / 5char-balanced). The 1char number is the
# clean UNCROWDED BASELINE (the model is good at single-frame detection even though it fails
# at counting); the 1->2->5 drop is the crowding degradation curve. No training; forward only.
set -euo pipefail

# locate repo root via SLURM_SUBMIT_DIR (SLURM copies the script to spool, so BASH_SOURCE is unreliable)
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
REPO_ROOT="$(pwd)"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK="${TASK:-steps_in_room}"
SEQ_LENS="${SEQ_LENS:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-40}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs/frame_axis/probes/crowding_extraction}"
RUN_NAME="$(date +%Y%m%d_%H%M%S)"
OUT="$OUT_ROOT/$RUN_NAME"
mkdir -p "$OUT"
echo "REPO_ROOT=$REPO_ROOT  OUT=$OUT  TASK=$TASK  SEQ_LENS=$SEQ_LENS  MAX_SAMPLES=$MAX_SAMPLES"

# crowding level -> dataset root (1 char = uncrowded baseline ... 5 char = crowded)
declare -A DATASETS=(
  [1char]="$REPO_ROOT/data/mmred_steps_1char"
  [2char]="$REPO_ROOT/data/mmred_steps_2char"
  [5char]="$REPO_ROOT/data/mmred_steps_balanced"
)

for level in 1char 2char 5char; do
  droot="${DATASETS[$level]}"
  echo "=================== crowding=$level  data_root=$droot ==================="
  python -u evaluations/scripts/eval_per_frame_verification.py \
    --model-name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --data-root "$droot" \
    --split all_uniform \
    --seq-lens "$SEQ_LENS" \
    --task "$TASK" \
    --max-samples "$MAX_SAMPLES" \
    --output "$OUT/$level" 2>&1 | tee "$OUT/${level}.log"
done

# copy the slurm log into the run dir on exit
cp "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$OUT/" 2>/dev/null || true
echo "DONE -> $OUT"

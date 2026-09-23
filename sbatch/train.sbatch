#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# experiments/train.py: fenced LoRA SFT on official rows (P2 recipe, faithful prompt, native 512).
# Anchor: none (this produces the NEW baseline row); P2's 0.900/0.660/0.300 came from the old prompt at 392 px.
# Knobs (env; no commas; lists via files): CONFIGS_FILE (sbatch/lib/splits/configs_train_8_16.txt) QTYPES_FILE
#   GATE (none|oracle) LAYOUT (question-first) EPOCHS (5) LIMIT VAL_LIMIT (100) EXTRA OUTPUT
# Prefer l40s/h200 over the 40 GB a100 (N=16 at 512 px is ~5.3 K tokens). Submit:
#   SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/train.sbatch
#   sbatch --qos=2h_2g --export=ALL,QTYPES_FILE=sbatch/lib/splits/qtypes_steps.txt,LIMIT=40,EPOCHS=1,VAL_LIMIT=20 sbatch/train.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
CONFIGS_FILE="${CONFIGS_FILE:-sbatch/lib/splits/configs_train_8_16.txt}"
mapfile -t cfgs < <(grep -vE '^\s*#|^\s*$' "$CONFIGS_FILE")
for c in "${cfgs[@]}"; do stage_split "${c}_train"; stage_split "${c}_val"; done
OUTPUT="${OUTPUT:-outputs/port/train/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
QT_ARGS=(); if [ -n "${QTYPES_FILE:-}" ]; then mapfile -t qt < <(grep -vE '^\s*#|^\s*$' "$QTYPES_FILE"); QT_ARGS=(--qtypes "${qt[@]}"); fi
run_logged "$OUTPUT" python -u experiments/train.py \
    --data-root "$DATA_ROOT" --configs "${cfgs[@]}" "${QT_ARGS[@]}" --gate "${GATE:-none}" --layout "${LAYOUT:-question-first}" ${NOFENCE:+--no-fence} ${LOGN:+--attn-logn-sref "$LOGN"} \
    --epochs "${EPOCHS:-5}" ${LIMIT:+--limit "$LIMIT"} --val-limit "${VAL_LIMIT:-100}" ${EXTRA:-} --output "$OUTPUT"

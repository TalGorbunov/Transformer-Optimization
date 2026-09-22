#!/usr/bin/env bash
#SBATCH --job-name=gate_capture
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# experiments/gate_capture.py: <|vision_end|> slot states under one (layout x fence) arm, all (or listed) qtypes. DIAG D4.
# Knobs (env; lists space-separated): SPLIT (seq_len_8_train) ARM (fenced_qfirst) QTYPES ("" = all 24) LIMIT (50)
#   HS_LAYERS ("12 20 24") ADAPTER MAX_SEQ (60000) EXTRA OUTPUT
# Submit:  sbatch -p a100-public --qos=12h_4g --time=02:00:00 --export=ALL,SPLIT=seq_len_8_train,ARM=fenced_qfirst sbatch/gate_capture.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
SPLIT="${SPLIT:-seq_len_8_train}"; CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
stage_split "$SPLIT"
ARM="${ARM:-fenced_qfirst}"
read -r -a HS_ARR <<< "${HS_LAYERS:-12 20 24}"
QT_ARGS=(); if [ -n "${QTYPES_FILE:-}" ]; then mapfile -t QT_ARR < <(grep -vE '^\s*#|^\s*$' "$QTYPES_FILE"); QT_ARGS=(--qtypes "${QT_ARR[@]}"); elif [ -n "${QTYPES:-}" ]; then read -r -a QT_ARR <<< "$QTYPES"; QT_ARGS=(--qtypes "${QT_ARR[@]}"); fi
OUTPUT="${OUTPUT:-outputs/diag/gate/N${CONFIG#seq_len_}_${SPLITNAME}/${ARM}}"
run_logged "$OUTPUT" python -u experiments/gate_capture.py --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" \
    --arm "$ARM" "${QT_ARGS[@]}" --limit "${LIMIT:-50}" --hs-layers "${HS_ARR[@]}" --max-seq-tokens "${MAX_SEQ:-60000}" \
    ${ADAPTER:+--adapter "$ADAPTER"} ${EXTRA:-} --output "$OUTPUT"

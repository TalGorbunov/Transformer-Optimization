#!/usr/bin/env bash
#SBATCH --job-name=evaluate
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# experiments/evaluate.py on one official split (frozen or adapter; layouts; fence; oracle gate).
# Anchors (port checks only): PORT_CHECK=arm-a on seq_len_8_test -> 0.533 (all 24) / 0.765-0.559-0.353
#   (QTYPES_FILE=qtypes_anchor3 LIMIT_PER_QTYPE=34); PORT_CHECK=p2 ADAPTER=checkpoints/sft_fenced_hf_adapter
#   QIDS_FILE=sbatch/lib/splits/qids_p2_seq8_test.txt -> 45/50 (seq16 33/50, seq32 15/50).
# Knobs (env; no commas; lists via files): SPLIT (seq_len_8_test) QTYPES_FILE QIDS_FILE ADAPTER (empty = frozen)
#   LAYOUT FENCE (1) GATE (none|oracle) LIMIT LIMIT_PER_QTYPE PORT_CHECK MAX_NEW EXTRA OUTPUT
# Submit:
#   SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/evaluate.sbatch
#   sbatch -p a100-public --qos=2h_2g --export=ALL,SPLIT=seq_len_8_test,LIMIT=20 sbatch/evaluate.sbatch   # smoke
#   sbatch --time=03:00:00 --export=ALL,SPLIT=seq_len_8_test,PORT_CHECK=arm-a sbatch/evaluate.sbatch    # C1 port check
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
SPLIT="${SPLIT:-seq_len_8_test}"
CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
stage_split "$SPLIT"
OUTPUT="${OUTPUT:-outputs/port/evaluate/${SPLIT}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
QT_ARGS=(); if [ -n "${QTYPES_FILE:-}" ]; then mapfile -t qt < <(grep -vE '^\s*#|^\s*$' "$QTYPES_FILE"); QT_ARGS=(--qtypes "${qt[@]}"); fi
MODEL_ARGS=(--frozen); [ -n "${ADAPTER:-}" ] && MODEL_ARGS=(--adapter "$ADAPTER")
FENCE_ARGS=(); [ "${FENCE:-0}" = "1" ] && FENCE_ARGS=(--fence)
run_logged "$OUTPUT" python -u experiments/evaluate.py \
    --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" "${QT_ARGS[@]}" "${MODEL_ARGS[@]}" "${FENCE_ARGS[@]}" \
    ${QIDS_FILE:+--qids-file "$QIDS_FILE"} ${LAYOUT:+--layout "$LAYOUT"} --gate "${GATE:-none}" \
    ${LIMIT:+--limit "$LIMIT"} ${LIMIT_PER_QTYPE:+--limit-per-qtype "$LIMIT_PER_QTYPE"} \
    ${PORT_CHECK:+--port-check "$PORT_CHECK"} ${MAX_NEW:+--max-new-tokens "$MAX_NEW"} ${EXTRA:-} --output "$OUTPUT"

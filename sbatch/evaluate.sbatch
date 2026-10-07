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
# Knobs (env; no commas; lists via files): DATASET (mmred|herbench|minerva) BACKBONE (qwen2.5-vl-7b)
#   MMReD: SPLIT (seq_len_8_test)   video: PROTOCOL (planted|uniform) N (16) SPLIT (test)
#   QTYPES_FILE QIDS_FILE ADAPTER (empty = frozen) ARM (replaces LAYOUT/FENCE/GATE) LAYOUT FENCE (1) GATE (none|oracle)
#   LIMIT LIMIT_PER_QTYPE PER_QTYPE (stratified rows per task) PORT_CHECK MAX_NEW EXTRA OUTPUT
# Submit:
#   SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/evaluate.sbatch
#   sbatch -p a100-public --qos=2h_2g --export=ALL,SPLIT=seq_len_8_test,LIMIT=20 sbatch/evaluate.sbatch   # smoke
#   sbatch --time=03:00:00 --export=ALL,SPLIT=seq_len_8_test,PORT_CHECK=arm-a sbatch/evaluate.sbatch    # C1 port check
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
DATASET="${DATASET:-mmred}"; BACKBONE="${BACKBONE:-qwen2.5-vl-7b}"
if [ "$DATASET" = "mmred" ]; then
  SPLIT="${SPLIT:-seq_len_8_test}"; CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
  stage_split "$SPLIT"
  DATA_ARGS=(--config "$CONFIG" --split "$SPLITNAME")
  OUTPUT="${OUTPUT:-outputs/port/evaluate/${SPLIT}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
else
  PROTOCOL="${PROTOCOL:-planted}"; N="${N:-16}"; SPLITNAME="${SPLIT:-test}"
  case "$DATASET" in herbench) ROOT=data/herbench_v2;; minerva) ROOT=data/minerva;; *) echo "unknown DATASET $DATASET"; exit 2;; esac
  stage_video "$ROOT"
  DATA_ARGS=(--protocol "$PROTOCOL" --n-frames "$N" --split "$SPLITNAME")
  OUTPUT="${OUTPUT:-outputs/scaleup/${DATASET}/${BACKBONE}/s1_eval/${PROTOCOL}_N${N}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
fi
QT_ARGS=(); if [ -n "${QTYPES_FILE:-}" ]; then mapfile -t qt < <(grep -vE '^\s*#|^\s*$' "$QTYPES_FILE"); QT_ARGS=(--qtypes "${qt[@]}"); fi
MODEL_ARGS=(--frozen); [ -n "${ADAPTER:-}" ] && MODEL_ARGS=(--adapter "$ADAPTER")
FENCE_ARGS=(); [ "${FENCE:-0}" = "1" ] && FENCE_ARGS=(--fence)
ARM_ARGS=(); if [ -n "${ARM:-}" ]; then ARM_ARGS=(--arm "$ARM"); else ARM_ARGS=(${LAYOUT:+--layout "$LAYOUT"} --gate "${GATE:-none}"); fi
run_logged "$OUTPUT" python -u experiments/evaluate.py --dataset "$DATASET" --backbone "$BACKBONE" \
    --data-root "$DATA_ROOT" "${DATA_ARGS[@]}" "${QT_ARGS[@]}" "${MODEL_ARGS[@]}" "${FENCE_ARGS[@]}" \
    ${QIDS_FILE:+--qids-file "$QIDS_FILE"} "${ARM_ARGS[@]}" \
    ${LIMIT:+--limit "$LIMIT"} ${LIMIT_PER_QTYPE:+--limit-per-qtype "$LIMIT_PER_QTYPE"} ${PER_QTYPE:+--per-qtype "$PER_QTYPE"} \
    ${PORT_CHECK:+--port-check "$PORT_CHECK"} ${FRAME_SET:+--frame-set "$FRAME_SET"} ${MAX_PIXELS:+--max-pixels "$MAX_PIXELS"} ${FAST:+--fast} ${MAX_NEW:+--max-new-tokens "$MAX_NEW"} ${EXTRA:-} --output "$OUTPUT"

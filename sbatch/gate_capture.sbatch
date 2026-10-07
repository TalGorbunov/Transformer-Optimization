#!/usr/bin/env bash
#SBATCH --job-name=gate_capture
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# experiments/gate_capture.py: per-frame slot states under one (layout x fence) arm. DIAG D4 / SCALEUP S2.
# Knobs (env; lists space-separated): DATASET (mmred) BACKBONE (qwen2.5-vl-7b) ARM (fenced_qfirst)
#   MMReD:  SPLIT (seq_len_8_train)            video: PROTOCOL (planted) N (16) SPLIT (test)
#   QTYPES ("" = all) / QTYPES_FILE  LIMIT (50)  HS_LAYERS ("" = the backbone's probe_layers)  ADAPTER  MAX_SEQ (60000)  EXTRA  OUTPUT
# Submit:  sbatch -p a100-public --qos=12h_4g --time=02:00:00 --export=ALL,SPLIT=seq_len_8_train,ARM=fenced_qfirst sbatch/gate_capture.sbatch
#          sbatch ... --export=ALL,DATASET=herbench,PROTOCOL=planted,N=16,ARM=fenced_qfirst sbatch/gate_capture.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
DATASET="${DATASET:-mmred}"; BACKBONE="${BACKBONE:-qwen2.5-vl-7b}"; ARM="${ARM:-fenced_qfirst}"
if [ "$DATASET" = "mmred" ]; then
  SPLIT="${SPLIT:-seq_len_8_train}"; CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
  stage_split "$SPLIT"
  DATA_ARGS=(--config "$CONFIG" --split "$SPLITNAME")
  OUTPUT="${OUTPUT:-outputs/diag/gate/N${CONFIG#seq_len_}_${SPLITNAME}/${ARM}}"
else
  PROTOCOL="${PROTOCOL:-planted}"; N="${N:-16}"; SPLITNAME="${SPLIT:-test}"
  case "$DATASET" in herbench) ROOT=data/herbench_v2;; minerva) ROOT=data/minerva;; *) echo "unknown DATASET $DATASET"; exit 2;; esac
  stage_video "$ROOT"
  DATA_ARGS=(--protocol "$PROTOCOL" --n-frames "$N" --split "$SPLITNAME")
  OUTPUT="${OUTPUT:-outputs/scaleup/${DATASET}/${BACKBONE}/s2_gate/${PROTOCOL}_N${N}/${ARM}}"
fi
HS_ARGS=(); if [ -n "${HS_LAYERS:-}" ]; then read -r -a HS_ARR <<< "$HS_LAYERS"; HS_ARGS=(--hs-layers "${HS_ARR[@]}"); fi
QT_ARGS=(); if [ -n "${QTYPES_FILE:-}" ]; then mapfile -t QT_ARR < <(grep -vE '^\s*#|^\s*$' "$QTYPES_FILE"); QT_ARGS=(--qtypes "${QT_ARR[@]}"); elif [ -n "${QTYPES:-}" ]; then read -r -a QT_ARR <<< "$QTYPES"; QT_ARGS=(--qtypes "${QT_ARR[@]}"); fi
run_logged "$OUTPUT" python -u experiments/gate_capture.py --dataset "$DATASET" --backbone "$BACKBONE" --data-root "$DATA_ROOT" \
    "${DATA_ARGS[@]}" --arm "$ARM" "${QT_ARGS[@]}" --limit "${LIMIT:-50}" "${HS_ARGS[@]}" --max-seq-tokens "${MAX_SEQ:-60000}" \
    ${FRAME_SET:+--frame-set "$FRAME_SET"} ${MAX_PIXELS:+--max-pixels "$MAX_PIXELS"} ${FAST:+--fast} ${ADAPTER:+--adapter "$ADAPTER"} ${EXTRA:-} --output "$OUTPUT"

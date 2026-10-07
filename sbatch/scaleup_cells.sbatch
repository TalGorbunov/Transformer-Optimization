#!/usr/bin/env bash
#SBATCH --job-name=scaleup
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# SCALEUP S1/S2/S3 cells on a video dataset, several N chained in ONE job (the diag_d2 pattern: stage once, loop N).
#   STAGE=s1  evaluate.py  --frozen --arm $ARM            (plain | qfirst; official comparability = uniform N=16 plain)
#   STAGE=s2  gate_capture.py --arm $ARM                  (fenced_qfirst | qfirst | fenced_qlast)
#   STAGE=s3  evaluate.py  --frozen --arm gated           (fenced + oracle gate = the upper bound)
# Knobs (env; lists space-separated; no commas): DATASET (herbench|minerva) BACKBONE (qwen2.5-vl-7b) STAGE (s1|s2|s3)
#   FRAME_SET (native|512)  MAX_PIXELS (processor cap; empty = default)  FAST (1 = batched fenced path)  CHUNK_TOKENS
#   POOL (1 = also the mean image-token state)  JUDGE (per-block yes/no instruction; s2 + FAST=1)
#   STAGE_DATA (1 = copy tarballs to the node NVMe)  ARM  PROTOCOL (planted|isolated|uniform)  QTYPES ("" = all tasks)  NLIST ("8 16 32 64")  PER_QTYPE (50; 0 = all rows)  MAX_NEW (8)  MAX_SEQ (60000)  ADAPTER  EXTRA
# Masked forwards (fenced arms, gated) at N=128 go to h200-shared (~40k tokens); mask-free cells fit the 40 GB A100s.
# Submit (see sbatch/scaleup_submit.sh for the wave):
#   sbatch -p a100-public --qos=12h_4g --time=06:00:00 --export=ALL,DATASET=herbench,STAGE=s1,ARM=plain,NLIST="8 16 32 64" sbatch/scaleup_cells.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
DATASET="${DATASET:-herbench}"; BACKBONE="${BACKBONE:-qwen2.5-vl-7b}"; STAGE="${STAGE:-s1}"; ARM="${ARM:-plain}"
PROTOCOL="${PROTOCOL:-planted}"; PER_QTYPE="${PER_QTYPE:-50}"
read -r -a NARR <<< "${NLIST:-8 16 32 64}"
case "$DATASET" in herbench) ROOT=data/herbench_v2;; minerva) ROOT=data/minerva;; *) echo "unknown DATASET $DATASET"; exit 2;; esac
_CELL_STAGE="$STAGE"; STAGE="${STAGE_DATA:-1}"; stage_video "$ROOT"; STAGE="$_CELL_STAGE"   # STAGE names the cell here; common.sh reads STAGE=1 as "copy to NVMe"
QT_ARGS=(); if [ -n "${QTYPES:-}" ]; then read -r -a QT <<< "$QTYPES"; QT_ARGS=(--qtypes "${QT[@]}"); fi
PQ_ARGS=(); [ "$PER_QTYPE" != "0" ] && PQ_ARGS=(--per-qtype "$PER_QTYPE")
RES_ARGS=(${FRAME_SET:+--frame-set "$FRAME_SET"} ${MAX_PIXELS:+--max-pixels "$MAX_PIXELS"})
FAST_ARGS=(); [ "${FAST:-0}" = "1" ] && FAST_ARGS=(--fast ${CHUNK_TOKENS:+--chunk-tokens "$CHUNK_TOKENS"})
RTAG="${FRAME_SET:-native}${MAX_PIXELS:+_mp$MAX_PIXELS}"
LIM_ARGS=(--limit "${PER_QTYPE:-50}"); [ "$PER_QTYPE" = "0" ] && LIM_ARGS=(--limit 100000)
for N in "${NARR[@]}"; do
  case "$STAGE" in
    s1) OUT="outputs/scaleup/${DATASET}/${BACKBONE}/s1_eval/${RTAG}/${PROTOCOL}_N${N}/${ARM}"
        run_logged "$OUT" python -u experiments/evaluate.py --dataset "$DATASET" --backbone "$BACKBONE" --data-root "$DATA_ROOT" \
            --protocol "$PROTOCOL" --n-frames "$N" --split test --frozen --arm "$ARM" "${QT_ARGS[@]}" "${PQ_ARGS[@]}" "${RES_ARGS[@]}" \
            --max-new-tokens "${MAX_NEW:-8}" --max-seq-tokens "${MAX_SEQ:-60000}" ${EXTRA:-} --output "$OUT" ;;
    s2) OUT="outputs/scaleup/${DATASET}/${BACKBONE}/s2_gate/${RTAG}/${PROTOCOL}_N${N}/${ARM}${POOL:+_pool}${JUDGE:+_judge}"
        run_logged "$OUT" python -u experiments/gate_capture.py --dataset "$DATASET" --backbone "$BACKBONE" --data-root "$DATA_ROOT" \
            --protocol "$PROTOCOL" --n-frames "$N" --split test --arm "$ARM" "${QT_ARGS[@]}" "${LIM_ARGS[@]}" "${RES_ARGS[@]}" "${FAST_ARGS[@]}" \
            --max-seq-tokens "${MAX_SEQ:-60000}" ${ADAPTER:+--adapter "$ADAPTER"} ${POOL:+--pool} ${JUDGE:+--judge "$JUDGE"} ${EXTRA:-} --output "$OUT" ;;
    s3) OUT="outputs/scaleup/${DATASET}/${BACKBONE}/s3_oracle/${RTAG}/${PROTOCOL}_N${N}/gated"
        run_logged "$OUT" python -u experiments/evaluate.py --dataset "$DATASET" --backbone "$BACKBONE" --data-root "$DATA_ROOT" \
            --protocol "$PROTOCOL" --n-frames "$N" --split test ${ADAPTER:+--adapter "$ADAPTER"} ${ADAPTER:---frozen} --arm gated "${QT_ARGS[@]}" "${PQ_ARGS[@]}" "${RES_ARGS[@]}" "${FAST_ARGS[@]}" \
            --max-new-tokens "${MAX_NEW:-8}" --max-seq-tokens "${MAX_SEQ:-60000}" ${EXTRA:-} --output "$OUT" ;;
    *) echo "unknown STAGE $STAGE"; exit 2;;
  esac
done

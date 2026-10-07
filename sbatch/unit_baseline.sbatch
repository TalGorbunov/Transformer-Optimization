#!/usr/bin/env bash
#SBATCH --job-name=unit
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# UNIT baseline (plan 2026-10-06): how well does the frozen model perceive one evidence unit by itself?
# Three unfenced regimes, all through the model's own output, several units chained in ONE job:
#   REGIME=T  evaluate.py --no-frames                      the question alone (floor; text-answerable tasks)
#   REGIME=E  evaluate.py --unit $u --res $RES             ONLY the evidence units, official layout (perception ceiling)
#   REGIME=D  unit_judge.py --unit $u --res $RES           per-unit yes/no vs hard + easy negatives (detection)
# Knobs (env; lists space-separated; NO commas): DATASET (herbench|minerva|mmred) BACKBONE (qwen2.5-vl-7b) REGIME ("T", "E", "D" or "T E D" in order)
#   UNITS ("frame clip3_d1 clip5_d1 clip5_d2 clip9_d2")  RES (lo|hi; empty = the processor's defaults)
#   QIDS_FILE (sbatch/lib/splits/qids_herbench_native582.txt; empty = PER_QTYPE rows)  PER_QTYPE (50)  QTYPES ("" = all)
#   MMReD: SPLIT (seq_len_8_test)   video T: PROTOCOL (uniform) N (8)   NEG (both|hard|easy)  MAX_NEW (8)  EXTRA  STAGE_DATA (0)
#   OUT_BASE (default outputs/scaleup/<dataset>/<backbone>/s1u_unit; smokes go to outputs/_scratch/unit_smoke/<dataset>)
# Units are read from /rg (units_native/ is not tarred); MMReD frames are staged as usual.
# Submit (see sbatch/unit_submit.sh for the wave):
#   sbatch -p l40s-shared --qos=12h_4g --time=04:00:00 --export=ALL,DATASET=herbench,REGIME=E,UNITS="frame clip5_d1",RES=hi,QIDS_FILE=sbatch/lib/splits/qids_herbench_native582.txt sbatch/unit_baseline.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
DATASET="${DATASET:-herbench}"; BACKBONE="${BACKBONE:-qwen2.5-vl-7b}"; RES="${RES:-}"
# generation budget: MMReD answers are JSON ('{ "answer": "Kitchen" }' ~ 10 tokens; evaluate.py's default 24); the MCQ letter needs 8.
# (2026-10-06 smoke 170567: 8 tokens truncated every MMReD text-only answer -> parse_fail 24/24.)
if [ -z "${MAX_NEW:-}" ]; then if [ "$DATASET" = "mmred" ]; then MAX_NEW=24; else MAX_NEW=8; fi; fi
read -r -a UARR <<< "${UNITS:-frame}"
read -r -a RARR <<< "${REGIME:-E}"
if [ "$DATASET" = "mmred" ]; then
  SPLIT="${SPLIT:-seq_len_8_test}"; CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
  stage_split "$SPLIT"
  DATA_ARGS=(--config "$CONFIG" --split "$SPLITNAME"); TAG="$CONFIG"
else
  case "$DATASET" in herbench) ROOT=data/herbench_v2;; minerva) ROOT=data/minerva;; *) echo "unknown DATASET $DATASET"; exit 2;; esac
  STAGE="${STAGE_DATA:-0}" stage_video "$ROOT"
  DATA_ARGS=(--protocol "${PROTOCOL:-uniform}" --n-frames "${N:-8}" --split test); TAG="${PROTOCOL:-uniform}_N${N:-8}"
fi
QT_ARGS=(); if [ -n "${QTYPES:-}" ]; then read -r -a QT <<< "$QTYPES"; QT_ARGS=(--qtypes "${QT[@]}"); fi
ROW_ARGS=(); if [ -n "${QIDS_FILE:-}" ]; then ROW_ARGS=(--qids-file "$QIDS_FILE"); else ROW_ARGS=(--per-qtype "${PER_QTYPE:-50}"); fi
RES_ARGS=(${RES:+--res "$RES"}); RTAG="${RES:-default}"
BASE="${OUT_BASE:-outputs/scaleup/${DATASET}/${BACKBONE}/s1u_unit}"     # smokes: OUT_BASE=outputs/_scratch/...
for REGIME in "${RARR[@]}"; do
case "$REGIME" in
  T) OUT="$BASE/T/${TAG}"
     run_logged "$OUT" python -u experiments/evaluate.py --dataset "$DATASET" --backbone "$BACKBONE" --data-root "$DATA_ROOT" \
         "${DATA_ARGS[@]}" "${QT_ARGS[@]}" "${ROW_ARGS[@]}" --frozen --arm plain --no-frames \
         --max-new-tokens "$MAX_NEW" ${EXTRA:-} --output "$OUT" ;;
  E) for u in "${UARR[@]}"; do
       OUT="$BASE/E_${u}_${RTAG}"
       if [ "$DATASET" = "mmred" ]; then EDATA=(--config "$CONFIG" --split "$SPLITNAME" --unit "$u"); else EDATA=(--unit "$u" --split test); fi
       run_logged "$OUT" python -u experiments/evaluate.py --dataset "$DATASET" --backbone "$BACKBONE" --data-root "$DATA_ROOT" \
           "${EDATA[@]}" "${QT_ARGS[@]}" "${ROW_ARGS[@]}" --frozen --arm plain "${RES_ARGS[@]}" \
           --max-new-tokens "$MAX_NEW" ${EXTRA:-} --output "$OUT"
     done ;;
  D) for u in "${UARR[@]}"; do
       OUT="$BASE/D_${u}_${RTAG}"
       run_logged "$OUT" python -u experiments/unit_judge.py --dataset "$DATASET" --backbone "$BACKBONE" --data-root "$DATA_ROOT" \
           "${DATA_ARGS[@]}" "${QT_ARGS[@]}" "${ROW_ARGS[@]}" --frozen --unit "$u" --neg "${NEG:-both}" "${RES_ARGS[@]}" \
           ${EXTRA:-} --output "$OUT"
     done ;;
  *) echo "unknown REGIME $REGIME"; exit 2;;
esac
done

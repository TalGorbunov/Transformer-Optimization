#!/usr/bin/env bash
#SBATCH --job-name=probe_attn
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# experiments/probe_attention.py: the attention photograph on official rows, one (arm, N, cond) per job on MMReD;
# on a video dataset (DATASET=vnbench ...) the N of NLIST are chained in one job (the scaleup_cells pattern).
# DIAG D2/D3/D8 (docs/DIAGNOSTICS_2026-09-22.md). Bands: outputs/diag/CAMPAIGN_BRIEF.md.
# Knobs (env; space-separated lists, never commas): SPLIT (seq_len_32_test) ARM (qfirst) QTYPES ("steps_in_room")
#   KLIST ("" = class-interleaved first LIMIT rows; e.g. "2 4 8") LIMIT (30) LAYERS ("12 16 20 24 27")
#   TAU (0 = off) FROM_LAYER (12) LOGN (0 = off) ADAPTER MAX_SEQ (60000) NOBLOCK (1 skips block_mass.csv) EXTRA OUTPUT
#   Video: DATASET (mmred | vnbench ...) BACKBONE (qwen2.5-vl-7b) PROTOCOL (planted) NLIST ("8 16 32 64 128")
#   MAX_PIXELS (the ladder's 151200) STAGE_DATA (1 = copy tarballs to the node NVMe) OUT_BASE
#   (default outputs/scaleup/<dataset>/<backbone>/s5_photo/<rtag>/<protocol>_N<N>/<arm><cond>)
# Submit:
#   SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/probe_attention.sbatch
#   sbatch -p a100-public --qos=12h_4g --time=03:00:00 --export=ALL,SPLIT=seq_len_32_test,ARM=qfirst,QTYPES=char_at_frame,TAU=2 sbatch/probe_attention.sbatch
#   sbatch -p a100-public --qos=12h_4g --time=03:00:00 --export=ALL,DATASET=vnbench,ARM=plain,NLIST="8 16 32 64 128",MAX_PIXELS=151200,LIMIT=20 sbatch/probe_attention.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$$"
DATASET="${DATASET:-mmred}"
ARM="${ARM:-qfirst}"
read -r -a LAYERS_ARR <<< "${LAYERS:-12 16 20 24 27}"
COND=""
[ -n "${TAU:-}" ] && [ "${TAU:-0}" != "0" ] && COND="${COND}_tau${TAU}"
[ -n "${LOGN:-}" ] && [ "${LOGN:-0}" != "0" ] && COND="${COND}_logn${LOGN}"
KL_ARGS=(); [ -n "${KLIST:-}" ] && { read -r -a KL_ARR <<< "$KLIST"; KL_ARGS=(--k-list "${KL_ARR[@]}"); }
QT_ARGS=(); [ -n "${QTYPES:-}" ] && { read -r -a QT_ARR <<< "$QTYPES"; QT_ARGS=(--qtypes "${QT_ARR[@]}"); }
COMMON=("${KL_ARGS[@]}" --limit "${LIMIT:-30}" --layers "${LAYERS_ARR[@]}"
        --attn-sharpen "${TAU:-0}" --sharpen-from-layer "${FROM_LAYER:-12}" --attn-logn-sref "${LOGN:-0}"
        --max-seq-tokens "${MAX_SEQ:-60000}" ${ADAPTER:+--adapter "$ADAPTER"} ${NOBLOCK:+--no-block-csv} ${EXTRA:-})
if [ "$DATASET" = "mmred" ]; then
  SPLIT="${SPLIT:-seq_len_32_test}"
  CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
  stage_split "$SPLIT"
  read -r -a QT_ARR <<< "${QTYPES:-steps_in_room}"
  QT_TAG="${QT_ARR[0]}"; [ "${#QT_ARR[@]}" -gt 1 ] && QT_TAG="multi"
  NTAG="N${CONFIG#seq_len_}"
  OUTPUT="${OUTPUT:-outputs/diag/photo/${QT_TAG}/${ARM}/${NTAG}${COND}}"
  run_logged "$OUTPUT" python -u experiments/probe_attention.py \
      --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" --qtypes "${QT_ARR[@]}" --arm "$ARM" \
      "${COMMON[@]}" --output "$OUTPUT"
else
  BACKBONE="${BACKBONE:-qwen2.5-vl-7b}"; PROTOCOL="${PROTOCOL:-planted}"
  case "$DATASET" in herbench) ROOT=data/herbench_v2;; minerva) ROOT=data/minerva;; vnbench) ROOT=data/vnbench;; *) echo "unknown DATASET $DATASET"; exit 2;; esac
  STAGE="${STAGE_DATA:-1}" stage_video "$ROOT"
  RTAG="native${MAX_PIXELS:+_mp$MAX_PIXELS}"
  read -r -a NARR <<< "${NLIST:-8 16 32 64 128}"
  for N in "${NARR[@]}"; do
    OUTPUT="${OUT_BASE:-outputs/scaleup/${DATASET}/${BACKBONE}}/s5_photo/${RTAG}/${PROTOCOL}_N${N}/${ARM}${COND}"
    run_logged "$OUTPUT" python -u experiments/probe_attention.py --dataset "$DATASET" --backbone "$BACKBONE" --data-root "$DATA_ROOT" \
        --protocol "$PROTOCOL" --n-frames "$N" --split test "${QT_ARGS[@]}" --arm "$ARM" ${MAX_PIXELS:+--max-pixels "$MAX_PIXELS"} \
        "${COMMON[@]}" --output "$OUTPUT"
  done
fi

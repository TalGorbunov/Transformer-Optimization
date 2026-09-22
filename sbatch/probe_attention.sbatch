#!/usr/bin/env bash
#SBATCH --job-name=probe_attn
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# experiments/probe_attention.py: the attention photograph on official rows, one (arm, N, cond) per job.
# DIAG D2/D3/D8 (docs/DIAGNOSTICS_2026-09-22.md). Bands: outputs/diag/CAMPAIGN_BRIEF.md.
# Knobs (env; space-separated lists, never commas): SPLIT (seq_len_32_test) ARM (qfirst) QTYPES ("steps_in_room")
#   KLIST ("" = class-interleaved first LIMIT rows; e.g. "2 4 8") LIMIT (30) LAYERS ("12 16 20 24 27")
#   TAU (0 = off) FROM_LAYER (12) LOGN (0 = off) ADAPTER MAX_SEQ (60000) NOBLOCK (1 skips block_mass.csv) EXTRA OUTPUT
# Submit:
#   SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/probe_attention.sbatch
#   sbatch -p a100-public --qos=12h_4g --time=03:00:00 --export=ALL,SPLIT=seq_len_32_test,ARM=qfirst,QTYPES=char_at_frame,TAU=2 sbatch/probe_attention.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$$"
SPLIT="${SPLIT:-seq_len_32_test}"
CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
stage_split "$SPLIT"
ARM="${ARM:-qfirst}"
read -r -a QT_ARR <<< "${QTYPES:-steps_in_room}"
read -r -a LAYERS_ARR <<< "${LAYERS:-12 16 20 24 27}"
QT_TAG="${QT_ARR[0]}"; [ "${#QT_ARR[@]}" -gt 1 ] && QT_TAG="multi"
COND=""
[ -n "${TAU:-}" ] && [ "${TAU:-0}" != "0" ] && COND="${COND}_tau${TAU}"
[ -n "${LOGN:-}" ] && [ "${LOGN:-0}" != "0" ] && COND="${COND}_logn${LOGN}"
NTAG="N${CONFIG#seq_len_}"
OUTPUT="${OUTPUT:-outputs/diag/photo/${QT_TAG}/${ARM}/${NTAG}${COND}}"
KL_ARGS=(); [ -n "${KLIST:-}" ] && { read -r -a KL_ARR <<< "$KLIST"; KL_ARGS=(--k-list "${KL_ARR[@]}"); }
run_logged "$OUTPUT" python -u experiments/probe_attention.py \
    --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" --qtypes "${QT_ARR[@]}" --arm "$ARM" \
    "${KL_ARGS[@]}" --limit "${LIMIT:-30}" --layers "${LAYERS_ARR[@]}" \
    --attn-sharpen "${TAU:-0}" --sharpen-from-layer "${FROM_LAYER:-12}" --attn-logn-sref "${LOGN:-0}" \
    --max-seq-tokens "${MAX_SEQ:-60000}" ${ADAPTER:+--adapter "$ADAPTER"} ${NOBLOCK:+--no-block-csv} ${EXTRA:-} \
    --output "$OUTPUT"

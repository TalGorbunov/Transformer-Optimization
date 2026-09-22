#!/usr/bin/env bash
#SBATCH --job-name=probe_hahn
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# experiments/probe_hahn.py: single-frame influence at the answer on official rows, one (N, cond) per job, several arms.
# DIAG D1/D3/D6 (docs/DIAGNOSTICS_2026-09-22.md). Bands (outputs/diag/CAMPAIGN_BRIEF.md): plain/qfirst alpha in [0.5,1.0],
#   fenced slot |alpha| <= 0.05, replay floor 0. No legacy cell ran on official rows.
# Knobs (env; space-separated lists, never commas): SPLIT (seq_len_8_test) QTYPES ("steps_in_room") LIMIT (50) CONTROLS (12)
#   MAX_GOLD GOLD_SET ARMS ("plain qfirst fenced") HS_LAYERS ("16 20 28") TAU (0) FROM_LAYER (12) LOGN (0) MAX_SEQ (60000)
#   ADAPTER PORT_CHECK (p2) EXTRA OUTPUT
# Submit:
#   SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/probe_hahn.sbatch
#   sbatch -p a100-public --qos=12h_4g --time=04:00:00 --export=ALL,SPLIT=seq_len_16_test,QTYPES=char_at_frame sbatch/probe_hahn.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$$"
SPLIT="${SPLIT:-seq_len_8_test}"
CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
stage_split "$SPLIT"
read -r -a QT_ARR <<< "${QTYPES:-steps_in_room}"
read -r -a ARMS_ARR <<< "${ARMS:-plain qfirst fenced}"
read -r -a HS_ARR <<< "${HS_LAYERS:-16 20 28}"
QT_TAG="${QT_ARR[0]}"; [ "${#QT_ARR[@]}" -gt 1 ] && QT_TAG="multi"
COND=""
[ -n "${TAU:-}" ] && [ "${TAU:-0}" != "0" ] && COND="${COND}_tau${TAU}"
[ -n "${LOGN:-}" ] && [ "${LOGN:-0}" != "0" ] && COND="${COND}_logn${LOGN}"
NTAG="N${CONFIG#seq_len_}"
OUTPUT="${OUTPUT:-outputs/diag/hahn/${QT_TAG}/${NTAG}${COND}}"
run_logged "$OUTPUT" python -u experiments/probe_hahn.py \
    --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" --qtypes "${QT_ARR[@]}" \
    --limit "${LIMIT:-50}" --controls "${CONTROLS:-12}" ${MAX_GOLD:+--max-gold "$MAX_GOLD"} ${GOLD_SET:+--gold-set $GOLD_SET} \
    --arms "${ARMS_ARR[@]}" --hs-layers "${HS_ARR[@]}" \
    --attn-sharpen "${TAU:-0}" --sharpen-from-layer "${FROM_LAYER:-12}" --attn-logn-sref "${LOGN:-0}" \
    --max-seq-tokens "${MAX_SEQ:-60000}" ${ADAPTER:+--adapter "$ADAPTER"} ${PORT_CHECK:+--port-check "$PORT_CHECK"} \
    ${EXTRA:-} --output "$OUTPUT"

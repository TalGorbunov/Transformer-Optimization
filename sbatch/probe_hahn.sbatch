#!/usr/bin/env bash
#SBATCH --job-name=probe_hahn
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# experiments/probe_hahn.py: one-frame-flip sensitivity on official steps_in_room rows, one N per job.
# Anchor: the pre-registered bands (outputs/armor/CAMPAIGN_BRIEF.md): plain alpha in [0.7,1.3], fenced |alpha|<0.2,
#   replay floor 0. No legacy cell ran on official rows.
# Knobs (env; space-separated lists, never commas): SPLIT (seq_len_8_test) LIMIT (30) CONTROLS (12) GOLD_SET
#   ARMS ("plain fenced gated") HS_LAYERS ("16 20 28") ADAPTER EXTRA OUTPUT
# Submit:
#   SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/probe_hahn.sbatch
#   sbatch --export=ALL,SPLIT=seq_len_16_test sbatch/probe_hahn.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$$"
SPLIT="${SPLIT:-seq_len_8_test}"
CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
stage_split "$SPLIT"
OUTPUT="${OUTPUT:-outputs/port/hahn/${SPLIT}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
read -r -a ARMS_ARR <<< "${ARMS:-plain fenced gated}"
read -r -a HS_ARR <<< "${HS_LAYERS:-16 20 28}"
run_logged "$OUTPUT" python -u experiments/probe_hahn.py \
    --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" --limit "${LIMIT:-30}" --controls "${CONTROLS:-12}" \
    ${GOLD_SET:+--gold-set $GOLD_SET} --arms "${ARMS_ARR[@]}" --hs-layers "${HS_ARR[@]}" \
    ${ADAPTER:+--adapter "$ADAPTER"} ${EXTRA:-} --output "$OUTPUT"

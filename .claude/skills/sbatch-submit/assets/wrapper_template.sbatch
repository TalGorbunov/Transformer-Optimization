#!/usr/bin/env bash
#SBATCH --job-name=CHANGEME
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# CHANGEME: one line on what this runs (experiments/CHANGEME.py) and the anchor numbers it must reproduce.
# Knobs (env vars via --export=ALL,K=V; never a comma inside a value; lists go through *_FILE):
#   SPLIT        benchmark split to stage on the node NVMe        (default seq_len_8_train)
#   QTYPES_FILE  file with one question type per line             (default: all 24)
#   LIMIT EPOCHS EXTRA OUTPUT
# Submit:
#   SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/CHANGEME.sbatch                       # dry-run first
#   sbatch -p a100-public --qos=24h_1g --time=12:00:00 --export=ALL,SPLIT=seq_len_16_train,LIMIT=0 sbatch/CHANGEME.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"

OUTPUT="${OUTPUT:-outputs/GROUP/NAME/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
stage_split "${SPLIT:-seq_len_8_train}"                   # exports DATA_ROOT (STAGE=0 to read from /rg)

QT_ARGS=()
if [ -n "${QTYPES_FILE:-}" ]; then
    mapfile -t qt < <(grep -vE '^\s*#|^\s*$' "$QTYPES_FILE")
    QT_ARGS=(--qtypes "${qt[@]}")
fi

run_logged "$OUTPUT" python -u experiments/CHANGEME.py \
    --data-root "$DATA_ROOT" "${QT_ARGS[@]}" \
    --limit "${LIMIT:-50}" --epochs "${EPOCHS:-5}" \
    ${EXTRA:-} --output "$OUTPUT"

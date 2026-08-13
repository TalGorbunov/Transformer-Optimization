#!/bin/bash
#SBATCH --job-name=mmred_hf_frozen
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# Arm A (frozen) on the original MMReD renders. Knobs: JSON IMAGES [QTYPES(space-sep)]
# [LIMIT_PER_QTYPE] [DEC] [OUTPUT] [MODEL]  — uses the SHARED .venv (GPU).
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"

JSON="${JSON:?set JSON (e.g. data/mmred_hf/json/seq_len_8_test.json)}"
IMAGES="${IMAGES:?set IMAGES (e.g. data/mmred_hf/images/seq_len_8_test)}"
OUTPUT="${OUTPUT:-outputs/mmred_hf/frozen/$(basename "$IMAGES")_${SLURM_JOB_ID:-local}}"

run_logged "$OUTPUT" python -u scripts/mmred_hf/eval_frozen.py \
    --json "$JSON" --images "$IMAGES" \
    ${QTYPES:+--qtypes $QTYPES} \
    --limit-per-qtype "${LIMIT_PER_QTYPE:-50}" --decode "${DEC:-32}" \
    ${MODEL:+--model "$MODEL"} \
    --output "$OUTPUT"

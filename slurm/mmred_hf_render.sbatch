#!/bin/bash
#SBATCH --job-name=mmred_hf_render
#SBATCH -p l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -o logs/%x-%j.out
# CPU render of the HF MMReD dataset with the upstream renderer.
# Knobs: CONFIG (seq_len_8) SPLIT (train|val|test) [LIMIT] [OUTPUT]
# NOTE: runs in .venv_mmred (python3.11 + upstream mmred pkg), NOT the shared .venv —
# common.sh is sourced only for repo-root cd + run_logged; python is called by path.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"

CONFIG="${CONFIG:?set CONFIG (e.g. seq_len_8)}"
SPLIT="${SPLIT:?set SPLIT (train|val|test)}"
PY="$REPO_ROOT/.venv_mmred/bin/python"
JSON="data/mmred_hf/json/${CONFIG}_${SPLIT}.json"
IMAGES="data/mmred_hf/images/${CONFIG}_${SPLIT}"
OUTPUT="${OUTPUT:-outputs/mmred_hf/render/${CONFIG}_${SPLIT}_${SLURM_JOB_ID:-local}}"

if [ ! -f "$JSON" ]; then
    run_logged "$OUTPUT" "$PY" scripts/mmred_hf/prep.py \
        --config "$CONFIG" --split "$SPLIT" ${LIMIT:+--limit "$LIMIT"}
fi
# our capped-worker driver, NOT upstream render_images.py (which spawns
# node-core-count workers and melts NFS from inside an 8-CPU cgroup)
run_logged "$OUTPUT" "$PY" scripts/mmred_hf/render.py \
    --input_path "$JSON" --output_dir "$IMAGES" ${LIMIT:+--limit "$LIMIT"}

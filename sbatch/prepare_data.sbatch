#!/usr/bin/env bash
#SBATCH --job-name=prepare_data
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
# experiments/prepare_data.py: HF rows -> json -> official renders -> per-split tar (CPU only).
# Anchor: VERIFY_N>0 must report "frames md5-identical N/N" against data/mmred_hf on seq_len_8_val.
# Knobs (env via --export=ALL,K=V; no commas): CONFIG (seq_len_8) SPLIT (val) STEP (prep|render|tar|all) LIMIT WORKERS
#   VERIFY_N (0 = real run into the canonical tree; >0 = verify mode into OUTPUT) OUTPUT EXTRA
# Submit:
#   SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/prepare_data.sbatch
#   sbatch --export=ALL,CONFIG=seq_len_8,SPLIT=val,VERIFY_N=50 sbatch/prepare_data.sbatch      # verify
#   sbatch --export=ALL,CONFIG=seq_len_1,SPLIT=train sbatch/prepare_data.sbatch                # render a split
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$$"
CONFIG="${CONFIG:-seq_len_8}"; SPLIT="${SPLIT:-val}"
if [ "${VERIFY_N:-0}" != "0" ]; then
    OUTPUT="${OUTPUT:-outputs/port/prepare_verify/$(date +%Y%m%d_%H%M%S)_${CONFIG}_${SPLIT}}"
    run_logged "$OUTPUT" python -u experiments/prepare_data.py --config "$CONFIG" --split "$SPLIT" \
        --verify-against data/mmred_hf --verify-n "$VERIFY_N" --out "$OUTPUT/tree" \
        --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" ${EXTRA:-}
else
    OUTPUT="${OUTPUT:-outputs/prepare/${CONFIG}_${SPLIT}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
    run_logged "$OUTPUT" python -u experiments/prepare_data.py --config "$CONFIG" --split "$SPLIT" \
        --stage "${STEP:-all}" ${LIMIT:+--limit "$LIMIT"} \
        --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" ${EXTRA:-}
fi

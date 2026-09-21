#!/bin/bash
#SBATCH --job-name=mmred_hf_materialize
#SBATCH -p l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o logs/%x-%j.out
# Materialize MMReD-HF samples as per-qtype qa.txt dirs (hardlinked frames) for the
# Phase 3 training mixture — one dir tree per config x qtype so per-root =LIMIT caps
# give exact qtype balance (mixed-qtype roots skew under the K-tag stratified shuffle).
# Knobs: CONFIGS (space list, default "seq_len_2 seq_len_4 seq_len_8 seq_len_16") SPLIT (train)
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"

SPLIT="${SPLIT:-train}"
QTYPES=$(python -c "from gnnformer.mmred_hf import NIAH_QTYPES, DC_QTYPES; print(' '.join(NIAH_QTYPES + DC_QTYPES))")
OUTPUT="${OUTPUT:-outputs/mmred_hf/materialize/${SPLIT}_${SLURM_JOB_ID:-local}}"

for cfg in ${CONFIGS:-seq_len_2 seq_len_4 seq_len_8 seq_len_16}; do
    for qt in $QTYPES; do
        run_logged "$OUTPUT" python scripts/mmred_hf/materialize_dirs.py \
            --config "$cfg" --split "$SPLIT" --qtypes "$qt"
    done
done

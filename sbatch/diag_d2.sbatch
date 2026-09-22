#!/usr/bin/env bash
#SBATCH --job-name=diag_d2
#SBATCH --partition=a100-public
#SBATCH --qos=12h_4g
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# DIAG D2: the attention photograph at ONE N for every arm in ARMS (space-separated: plain qfirst fenced gated), all QTYPES,
#   base condition. One job per N. Output outputs/diag/photo/<qtype-tag>/<arm>/N<N>/.
# Knobs: SPLIT (seq_len_32_test) ARMS ("plain qfirst fenced") QTYPES ("char_at_frame steps_in_room n_char_at_frame where_spend")
#   LIMIT (50) LAYERS ("12 16 20 24 27") NOBLOCK MAX_SEQ (60000) ADAPTER
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$$"
SPLIT="${SPLIT:-seq_len_32_test}"; CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
stage_split "$SPLIT"
NTAG="N${CONFIG#seq_len_}"
read -r -a QT_ARR <<< "${QTYPES:-char_at_frame steps_in_room n_char_at_frame where_spend}"
read -r -a LAYERS_ARR <<< "${LAYERS:-12 16 20 24 27}"
for ARM in ${ARMS:-plain qfirst fenced}; do
  run_logged "outputs/diag/photo/multi/${ARM}/${NTAG}" python -u experiments/probe_attention.py \
      --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" --qtypes "${QT_ARR[@]}" --arm "$ARM" \
      --limit "${LIMIT:-50}" --layers "${LAYERS_ARR[@]}" --max-seq-tokens "${MAX_SEQ:-60000}" \
      ${ADAPTER:+--adapter "$ADAPTER"} ${NOBLOCK:+--no-block-csv} --output "outputs/diag/photo/multi/${ARM}/${NTAG}"
done
echo "[diag_d2] done"

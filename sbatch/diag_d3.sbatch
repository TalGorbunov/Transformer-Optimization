#!/usr/bin/env bash
#SBATCH --job-name=diag_d3
#SBATCH --partition=a100-public
#SBATCH --qos=12h_4g
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# DIAG D3: the temperature / log-N dissociation at ONE N — for each condition in CONDS (space-separated tokens
#   base | tau<t> | logn<sref>) run the attention photograph (probe_attention, arm ARM) and, unless NOEVAL=1, the
#   exact-match evaluation (evaluate.py, frozen, layout question-first) over QTYPES. One job per N.
# Knobs: SPLIT (seq_len_32_test) ARM (qfirst) QTYPES ("char_at_frame steps_in_room where_spend") CONDS ("base tau1.5 tau2 tau3 tau4 logn3000")
#   LIMIT (50 = every official row per qtype) FROM_LAYER (12) LAYERS ("12 16 20 24 27") NOBLOCK (1) NOEVAL MAX_SEQ (60000) ADAPTER
# Submit: sbatch -p a100-public --qos=12h_4g --time=06:00:00 --export=ALL,SPLIT=seq_len_32_test sbatch/diag_d3.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$$"
SPLIT="${SPLIT:-seq_len_32_test}"; CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
stage_split "$SPLIT"
ARM="${ARM:-qfirst}"; NTAG="N${CONFIG#seq_len_}"
read -r -a QT_ARR <<< "${QTYPES:-char_at_frame steps_in_room where_spend}"
read -r -a LAYERS_ARR <<< "${LAYERS:-12 16 20 24 27}"
for COND in ${CONDS:-base tau1.5 tau2 tau3 tau4 logn3000}; do
  TAU=0; LOGN=0; SUF=""
  case "$COND" in
    base) ;;
    tau*) TAU="${COND#tau}"; SUF="_tau${TAU}" ;;
    logn*) LOGN="${COND#logn}"; SUF="_logn${LOGN}" ;;
    *) echo "unknown cond $COND"; exit 2 ;;
  esac
  run_logged "outputs/diag/photo/d3/${ARM}/${NTAG}${SUF}" python -u experiments/probe_attention.py \
      --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" --qtypes "${QT_ARR[@]}" --arm "$ARM" \
      --limit "${LIMIT:-50}" --layers "${LAYERS_ARR[@]}" --attn-sharpen "$TAU" --sharpen-from-layer "${FROM_LAYER:-12}" \
      --attn-logn-sref "$LOGN" --max-seq-tokens "${MAX_SEQ:-60000}" ${ADAPTER:+--adapter "$ADAPTER"} \
      ${NOBLOCK:+--no-block-csv} --output "outputs/diag/photo/d3/${ARM}/${NTAG}${SUF}"
  if [ "${NOEVAL:-0}" != "1" ]; then
    LAY="question-first"; [ "$ARM" = "plain" ] && LAY="paper"
    run_logged "outputs/diag/eval/d3/${ARM}/${NTAG}${SUF}" python -u experiments/evaluate.py \
        --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" --qtypes "${QT_ARR[@]}" \
        $([ -n "${ADAPTER:-}" ] && echo "--adapter $ADAPTER" || echo "--frozen") --layout "$LAY" \
        --attn-sharpen "$TAU" --sharpen-from-layer "${FROM_LAYER:-12}" --attn-logn-sref "$LOGN" \
        --output "outputs/diag/eval/d3/${ARM}/${NTAG}${SUF}"
  fi
done
echo "[diag_d3] done"

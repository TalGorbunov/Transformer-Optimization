#!/usr/bin/env bash
#SBATCH --job-name=t1_eval
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
# Tier-1 / comparison-table evaluation chain: ONE adapter (or the frozen model), every N in NLIST, the Tier-1 qtypes.
# Knobs: ADAPTER (path; empty = frozen) ARM (name for the output dir) NLIST ("1 2 4 8 16 32 64 128") GATE (none|oracle)
#   LAYOUT (adapter default) FENCE (adapter default; 1 forces) TAU (0) FROM_LAYER (12) LOGN (0; adapter contract wins if set)
#   QTYPES_FILE (sbatch/lib/splits/qtypes_tier1.txt) MAX_SEQ (60000) EXTRA
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
QTYPES_FILE="${QTYPES_FILE:-sbatch/lib/splits/qtypes_tier1.txt}"
mapfile -t qt < <(grep -vE '^\s*#|^\s*$' "$QTYPES_FILE")
ARM="${ARM:-arm}"
COND=""; [ "${TAU:-0}" != "0" ] && COND="${COND}_tau${TAU}"; [ "${LOGN:-0}" != "0" ] && COND="${COND}_logn${LOGN}"
for N in ${NLIST:-1 2 4 8 16 32 64 128}; do
  SPLIT="seq_len_${N}_test"; stage_split "$SPLIT"
  MODEL_ARGS=(--frozen); [ -n "${ADAPTER:-}" ] && MODEL_ARGS=(--adapter "$ADAPTER")
  FENCE_ARGS=(); [ "${FENCE:-0}" = "1" ] && FENCE_ARGS=(--fence)
  run_logged "outputs/diag/eval/tier1/${ARM}${COND}/N${N}" python -u experiments/evaluate.py \
      --data-root "$DATA_ROOT" --config "seq_len_${N}" --split test --qtypes "${qt[@]}" "${MODEL_ARGS[@]}" "${FENCE_ARGS[@]}" \
      ${LAYOUT:+--layout "$LAYOUT"} --gate "${GATE:-none}" --max-seq-tokens "${MAX_SEQ:-60000}" \
      --attn-sharpen "${TAU:-0}" --sharpen-from-layer "${FROM_LAYER:-12}" --attn-logn-sref "${LOGN:-0}" ${EXTRA:-} \
      --output "outputs/diag/eval/tier1/${ARM}${COND}/N${N}"
done
echo "[t1_eval] done"

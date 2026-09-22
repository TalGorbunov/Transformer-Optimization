#!/usr/bin/env bash
#SBATCH --job-name=diag_smoke
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# One-GPU smoke of the three DIAG instruments on official seq_len_8 rows (tiny limits), sequentially:
#   probe_hahn (3 qtypes x 4 arms + tau), probe_attention (4 arms + tau + logn), evaluate (qfirst, tau).
# Knobs: SPLIT (seq_len_8_test) LIMIT (3) OUTPUT (outputs/_scratch/diag_smoke)
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$$"
SPLIT="${SPLIT:-seq_len_8_test}"; CONFIG="${SPLIT%_*}"; SPLITNAME="${SPLIT##*_}"
stage_split "$SPLIT"
LIMIT="${LIMIT:-3}"; OUT="${OUTPUT:-outputs/_scratch/diag_smoke}"
run_logged "$OUT/hahn" python -u experiments/probe_hahn.py --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" \
    --qtypes steps_in_room char_at_frame n_char_at_frame --limit "$LIMIT" --controls 2 --arms plain qfirst fenced gated --output "$OUT/hahn"
run_logged "$OUT/hahn_tau" python -u experiments/probe_hahn.py --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" \
    --qtypes char_at_frame --limit 2 --controls 1 --arms qfirst --attn-sharpen 2 --output "$OUT/hahn_tau"
for ARM in plain qfirst fenced gated; do
  run_logged "$OUT/photo_$ARM" python -u experiments/probe_attention.py --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" \
      --qtypes steps_in_room char_at_frame --arm "$ARM" --limit "$LIMIT" --output "$OUT/photo_$ARM"
done
run_logged "$OUT/photo_tau" python -u experiments/probe_attention.py --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" \
    --qtypes char_at_frame --arm qfirst --limit 2 --attn-sharpen 2 --output "$OUT/photo_tau"
run_logged "$OUT/photo_logn" python -u experiments/probe_attention.py --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" \
    --qtypes char_at_frame --arm qfirst --limit 2 --attn-logn-sref 3000 --output "$OUT/photo_logn"
run_logged "$OUT/eval_tau" python -u experiments/evaluate.py --data-root "$DATA_ROOT" --config "$CONFIG" --split "$SPLITNAME" \
    --qtypes char_at_frame steps_in_room --frozen --layout question-first --limit 6 --attn-sharpen 2 --output "$OUT/eval_tau"
echo "[diag_smoke] done"

#!/bin/bash
#SBATCH --job-name=herb_plain_acc
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:55:00
#SBATCH -o logs/%x-%j.out
# Plain-model yes/no per-unit accuracy sweep (eval-only). Groups by resolution.
#   GROUP=448  -> δ{0,0.5,1,2} @448   |  GROUP=672 -> δ{0,0.5,1,2} @672
#   GROUP=hi   -> δ{3,4,5}@{448,672} (Stage-2, needs those extractions)
#   RESSWEEP=1 GROUP=res -> resolution sweep at δ=1 (needs the res clips)
# CLIPROOT default data/herbench_retrieve_opt_clips ; LIMIT caps #questions.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
GROUP="${GROUP:?set GROUP=448|672|hi|res}"
LIMIT="${LIMIT:-0}"
CLIPROOT="${CLIPROOT:-data/herbench_retrieve_opt_clips}"
BASE="outputs/herbench_retrieve_opt/plain_acc"

cell() {  # cell <arm-tag> <input-mode> <data-root>
  local TAG=$1 MODE=$2 ROOT=$3
  run_logged "$BASE/$TAG" python -u scripts/herbench_retrieve_opt/eval_plain_acc.py \
    --input-mode "$MODE" --data-root "$ROOT" --arm-tag "$TAG" --limit "$LIMIT" \
    --output "$BASE/$TAG"
}

case "$GROUP" in
  448)
    cell d0_r448   armB  data/herbench_ac/armB_ev_fill16
    cell d0.5_r448 clips "$CLIPROOT/d0.5_r448"
    cell d1_r448   clips "$CLIPROOT/d1_r448"
    cell d2_r448   clips "$CLIPROOT/d2_r448" ;;
  672)
    cell d0_r672   armB  data/herbench_ac/armB_ev_fill16_r672
    cell d0.5_r672 clips "$CLIPROOT/d0.5_r672"
    cell d1_r672   clips "$CLIPROOT/d1_r672"
    cell d2_r672   clips "$CLIPROOT/d2_r672" ;;
  hi)
    cell d3_r448 clips "$CLIPROOT/d3_r448"; cell d4_r448 clips "$CLIPROOT/d4_r448"
    cell d5_r448 clips "$CLIPROOT/d5_r448"
    cell d3_r672 clips "$CLIPROOT/d3_r672"; cell d4_r672 clips "$CLIPROOT/d4_r672"
    cell d5_r672 clips "$CLIPROOT/d5_r672" ;;
  res)  # resolution sweep at δ=1
    for R in 336 448 672 896; do cell "d1_r${R}" clips "$CLIPROOT/d1_r${R}"; done ;;
  *) echo "unknown GROUP=$GROUP" >&2; exit 1 ;;
esac

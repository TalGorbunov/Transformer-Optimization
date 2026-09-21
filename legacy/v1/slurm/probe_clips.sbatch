#!/bin/bash
#SBATCH --job-name=herb_probe_clips
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:55:00
#SBATCH -o logs/%x-%j.out
# HERBench retrieve-opt clip-unit supply probe (Phase 1). Batches arms in one 2h_2g job.
# Knobs:
#   STEP=1  -> A0 anchor (armB single-frame units @448) + B-delta0.5@448  [the gate]
#   STEP=2  -> full remaining sweep: A1@672, B-{1,2}@448, C-{0.5,1,2}@672  (extract first)
#   LAYERS  -> read-layer sweep (default 12,14,16,18,20)
#   LIMIT   -> cap #questions (0=all);  MAXUNITS -> units per forward (default 10)
#   CLIPROOT-> base dir of extracted clip arms (default data/herbench_retrieve_opt_clips)
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
STEP="${STEP:?set STEP=1|2}"
LAYERS="${LAYERS:-12,14,16,18,20}"
LIMIT="${LIMIT:-0}"
MAXUNITS="${MAXUNITS:-10}"
CLIPROOT="${CLIPROOT:-data/herbench_retrieve_opt_clips}"
PROBE_BASE="outputs/herbench_retrieve_opt/probe"

cell() {  # cell <name> <input-mode> <data-root>
  local NAME=$1 MODE=$2 ROOT=$3
  local OUT="$PROBE_BASE/$NAME"
  run_logged "$OUT" python -u scripts/herbench_retrieve_opt/probe_clips.py \
    --input-mode "$MODE" --data-root "$ROOT" \
    --layers "$LAYERS" --max-units "$MAXUNITS" --limit "$LIMIT" \
    --output "$OUT"
}

case "$STEP" in
  smoke)  # plumbing check only (no extraction dependency)
    LIMIT="${LIMIT:-4}"
    cell _scratch/smoke_A0 armB data/herbench_ac/armB_ev_fill16 ;;
  1)      # gate + first temporal arm
    cell A0_r448      armB  data/herbench_ac/armB_ev_fill16
    cell B_d0.5_r448  clips "$CLIPROOT/d0.5_r448" ;;
  2-B448) # temporal axis @448
    cell B_d1_r448    clips "$CLIPROOT/d1_r448"
    cell B_d2_r448    clips "$CLIPROOT/d2_r448" ;;
  2-C672) # temporal + resolution @672
    cell C_d0.5_r672  clips "$CLIPROOT/d0.5_r672"
    cell C_d1_r672    clips "$CLIPROOT/d1_r672"
    cell C_d2_r672    clips "$CLIPROOT/d2_r672" ;;
  2-A1)   # pure resolution single-frame
    cell A1_r672      armB  data/herbench_ac/armB_ev_fill16_r672 ;;
  *) echo "unknown STEP=$STEP" >&2; exit 1 ;;
esac

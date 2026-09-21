#!/bin/bash
#SBATCH --job-name=tf_cell
#SBATCH -p l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out
# TREEFOLD tree cell (T1/T2/T4/T5/T3-tree) via run_cell.py. Resumable.
# Knobs: ASKS LEAF(oracle|captions|notes) OUTPUT TITLE
#        [FAN NS PER_TYPE LIMIT CAPTIONS NOTES REUSE_LEAVES BATCH EXTRA]
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
run_logged "${OUTPUT:?set OUTPUT}" python -u scripts/treefold/run_cell.py \
  --asks "${ASKS:?set ASKS}" --leaf "${LEAF:?set LEAF}" \
  --fan "${FAN:-2}" --ns "${NS:-16-32-64-128}" \
  --per-type "${PER_TYPE:-8}" --batch "${BATCH:-48}" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${CAPTIONS:+--captions "$CAPTIONS"} ${NOTES:+--notes "$NOTES"} \
  ${REUSE_LEAVES:+--reuse-leaves "$REUSE_LEAVES"} \
  --prompt-style "${PROMPT_STYLE:-canon}" \
  --title "${TITLE:-TREEFOLD cell}" ${EXTRA:-} --output "$OUTPUT"

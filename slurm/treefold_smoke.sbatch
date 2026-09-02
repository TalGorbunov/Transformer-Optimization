#!/bin/bash
#SBATCH --job-name=tf_smoke
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:45:00
#SBATCH -o logs/%x-%j.out
# TREEFOLD smoke: every stage at LIMIT<=3, N=16, into outputs/_scratch/.
# Chained with set -e: any stage failure aborts the job with its message.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
OUT="${OUT:-outputs/_scratch/treefold_smoke}"
CAPS="outputs/recagg/armB_captions/20260817_203538_full/captions.json"

run_logged "$OUT/ask_zs" python -u scripts/treefold/ask.py \
  --mode zero --ns 16 --limit 3 --output "$OUT/ask_zs"

run_logged "$OUT/ask_fs" python -u scripts/treefold/ask.py \
  --mode few --ns 16 --limit 3 --output "$OUT/ask_fs"

run_logged "$OUT/t1" python -u scripts/treefold/run_cell.py \
  --asks "$OUT/ask_zs/asks.json" --leaf oracle --fan 2 --ns 16 --limit 3 \
  --title "SMOKE T1 fan-2 oracle" --output "$OUT/t1"

run_logged "$OUT/t2_k4" python -u scripts/treefold/run_cell.py \
  --asks "$OUT/ask_zs/asks.json" --leaf oracle --fan 4 --ns 16 --limit 3 \
  --reuse-leaves "$OUT/t1/leaf_notes.json" \
  --title "SMOKE T2 fan-4 (reused leaves)" --output "$OUT/t2_k4"

run_logged "$OUT/t2_kN" python -u scripts/treefold/run_cell.py \
  --asks "$OUT/ask_zs/asks.json" --leaf oracle --fan 0 --ns 16 --limit 3 \
  --reuse-leaves "$OUT/t1/leaf_notes.json" \
  --title "SMOKE T2 fan-N (reused leaves)" --output "$OUT/t2_kN"

run_logged "$OUT/t4" python -u scripts/treefold/run_cell.py \
  --asks "$OUT/ask_zs/asks.json" --leaf captions --per-type 4 --fan 2 \
  --ns 16 --limit 3 --captions "$CAPS" \
  --title "SMOKE T4 fan-2 captions" --output "$OUT/t4"

run_logged "$OUT/leaf_vlm" python -u scripts/treefold/leaf_vlm.py \
  --asks "$OUT/ask_zs/asks.json" --ns 16 --per-type 4 --limit 3 \
  --output "$OUT/leaf_vlm"

run_logged "$OUT/t3" python -u scripts/treefold/run_cell.py \
  --asks "$OUT/ask_zs/asks.json" --leaf notes --per-type 4 --fan 2 \
  --ns 16 --limit 3 --notes "$OUT/leaf_vlm/leaf_notes.json" \
  --title "SMOKE T3 fan-2 VLM leaves" --output "$OUT/t3"

echo "SMOKE COMPLETE"

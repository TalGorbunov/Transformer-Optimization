#!/bin/bash
#SBATCH --job-name=redux_datagen
#SBATCH -p l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
# REDUX C0 datagen: balanced steps_in_room pools for exists/majority exams + probe
# bases. Knobs: N (16|32|64|128). Counts hardcoded per N (comma-lists never ride
# --export). Exam split seed 1 per-count 50; probe_base split seed 3 per-count 80.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
N="${N:?set N}"
OUT=data/mmred_redux
case "$N" in
  16)  COUNTS="0 1 2 4 6 7 8 9 10 12 16" ;;
  32)  COUNTS="0 1 2 4 8 12 14 15 16 17 18 20 24 32" ;;
  64)  COUNTS="0 1 2 4 8 16 24 28 30 31 32 33 34 36 40 48 64" ;;
  128) COUNTS="0 1 2 4 8 16 32 48 56 60 62 63 64 65 66 68 72 80 96 128" ;;
  *) echo "bad N=$N"; exit 1 ;;
esac
run_logged "outputs/redux/datagen_N${N}" python -u datasets/mmred/generate_mmred_balanced.py \
  --task steps_in_room --seq-len "$N" --per-count 50 --counts "$COUNTS" \
  --out-root "$OUT" --split all_uniform --seed 1
run_logged "outputs/redux/datagen_N${N}" python -u datasets/mmred/generate_mmred_balanced.py \
  --task steps_in_room --seq-len "$N" --per-count 80 --counts "$((N / 2))" \
  --out-root "$OUT" --split probe_base --seed 3
echo "DATAGEN N=$N COMPLETE"

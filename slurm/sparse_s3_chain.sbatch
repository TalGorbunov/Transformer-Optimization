#!/bin/bash
#SBATCH --job-name=sp_s3
#SBATCH -p rtx6k-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# SPARSE S3 chains (gated Hahn probe, p1fence arm + adapter).
# CHAIN=n : vs-N chain N in {8,16,32,64,128} (N2 pools/limits: 50/50/50/50/40, ctrl 12)
# CHAIN=k : vs-k chain at N=64, base k in {1,2,4,8,16,32}, 30 pairs/k (redux pool)
# Knobs: CHAIN ADAPTER GATE (oracle|none) OUTPUT. Explicit --time at submit.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
ADAPTER="${ADAPTER:?set ADAPTER}"
GATE="${GATE:-oracle}"
OUTPUT="${OUTPUT:-outputs/sparse/s3}"
CHAIN="${CHAIN:?set CHAIN=n|k}"
if [ "$CHAIN" = n ]; then
  for N in 8 16 32 64 128; do
    if [ "$N" = 8 ]; then POOL=data/mmred_images_park/seq_len_8/all_uniform
    else POOL=data/mmred_longN_park/seq_len_${N}/all_uniform; fi
    LIM=50; [ "$N" = 128 ] && LIM=40
    run_logged "$OUTPUT/gate_${GATE}_N${N}" python -u scripts/sparse/probe_hahn_gated.py \
      --data_root "$POOL" --limit "$LIM" --controls 12 --seed 0 \
      --arms p1fence --peft-adapter "$ADAPTER" --gate "$GATE" ${EXTRA:-} \
      --output "$OUTPUT/gate_${GATE}_N${N}"
  done
elif [ "$CHAIN" = k ]; then
  POOL="${POOL:-data/mmred_redux/seq_len_64/all_uniform}"
  for K in 1 2 4 8 16 32; do
    run_logged "$OUTPUT/gate_${GATE}_N64_k${K}" python -u scripts/sparse/probe_hahn_gated.py \
      --data_root "$POOL" --limit "${LIMIT:-30}" --controls 6 --seed 0 \
      --arms p1fence --peft-adapter "$ADAPTER" --gate "$GATE" --gold-set "$K" ${EXTRA:-} \
      --output "$OUTPUT/gate_${GATE}_N64_k${K}"
  done
else
  echo "unknown CHAIN=$CHAIN" >&2; exit 2
fi
echo "S3 CHAIN $CHAIN COMPLETE"

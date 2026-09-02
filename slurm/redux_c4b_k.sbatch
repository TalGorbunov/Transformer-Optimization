#!/bin/bash
#SBATCH --job-name=rx_c4b_k
#SBATCH -p a100-public
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# REDUX C4b chain (ii): decay-in-k at fixed N — flips at base k -> k+1 for
# k in {1,2,4,8,16,32}, logN ON and OFF, p1fence + P1b adapter.
# Knobs: NLEN (default 64) POOL LIMIT OUTPUT. Explicit --time at submit.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
NLEN="${NLEN:-64}"
POOL="${POOL:-data/mmred_redux/seq_len_${NLEN}/all_uniform}"
ADAPTER="${ADAPTER:-checkpoints/sft_fenced_le16_ep10_adapter}"
OUTPUT="${OUTPUT:-outputs/redux/c4b_k}"
for K in 1 2 4 8 16 32; do
  for MODE in off on; do
    if [ "$MODE" = on ]; then LOGN="--attn-logn-sref 3398"; else LOGN=""; fi
    run_logged "$OUTPUT/N${NLEN}_k${K}_logn${MODE}" python -u scripts/armor/probe_hahn.py \
      --data_root "$POOL" --limit "${LIMIT:-30}" --controls 6 --seed 0 \
      --arms p1fence --peft-adapter "$ADAPTER" --gold-set "$K" $LOGN \
      --output "$OUTPUT/N${NLEN}_k${K}_logn${MODE}"
  done
done
echo "C4B-K CHAIN COMPLETE (N=$NLEN)"

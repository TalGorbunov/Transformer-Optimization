#!/bin/bash
#SBATCH --job-name=lm_l1_hahn
#SBATCH -p a100-public
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# LORAMECH L1: Hahn sensitivity+margin on a LoRA adapter, plain arm, all 5 Ns chained.
# Same pools/limits/seed as ARMOR-A jobs 136128-32 => paired vs the frozen curve.
# Knobs: ADAPTER (required), TAG (run-dir suffix), ARMS (default plain), OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
OUTPUT="${OUTPUT:-outputs/loramech/l1_hahn}"
TAG="${TAG:?set TAG}"
for CELL in "8 data/mmred_images_park/seq_len_8/all_uniform 50" \
            "16 data/mmred_longN_park/seq_len_16/all_uniform 50" \
            "32 data/mmred_longN_park/seq_len_32/all_uniform 50" \
            "64 data/mmred_longN_park/seq_len_64/all_uniform 50" \
            "128 data/mmred_longN_park/seq_len_128/all_uniform 40"; do
  set -- $CELL
  run_logged "$OUTPUT/${TAG}_N$1" python -u scripts/armor/probe_hahn.py \
    --data_root "$2" --limit "$3" --controls 12 --seed 0 \
    --arms "${ARMS:-plain}" ${ADAPTER:+--peft-adapter "$ADAPTER"} ${EXTRA:-} \
    --output "$OUTPUT/${TAG}_N$1"
done
echo "L1 CHAIN COMPLETE ($TAG)"

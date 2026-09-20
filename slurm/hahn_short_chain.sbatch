#!/bin/bash
#SBATCH --job-name=hahn_short
#SBATCH -p a100-public
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -o logs/%x-%j.out
# Margin/sensitivity chain at SHORT N in {1,2,4} for both arms (2026-09-14, for the
# 1..128 margin figure): untouched model (scripts/armor/probe_hahn.py, ARMOR-A protocol)
# and the gated N-free read (scripts/sparse/probe_hahn_gated.py, S11 protocol).
# Knobs: NS (space-sep, default "1 2 4") ADAPTER. Explicit --time at submit.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
ADAPTER="${ADAPTER:-checkpoints/sft_fenced_gated_nfree_adapter}"
for N in ${NS:-1 2 4}; do
  POOL=data/mmred_images_park/seq_len_${N}/all_uniform
  run_logged "outputs/armor/hahn/short_N${N}" python -u scripts/armor/probe_hahn.py \
    --data_root "$POOL" --limit 50 --controls 12 --output "outputs/armor/hahn/short_N${N}"
  run_logged "outputs/sparse/s11/gate_oracle_N${N}" python -u scripts/sparse/probe_hahn_gated.py \
    --data_root "$POOL" --limit 50 --controls 12 --seed 0 --arms p1fence \
    --peft-adapter "$ADAPTER" --gate oracle --nfree-prompt \
    --output "outputs/sparse/s11/gate_oracle_N${N}"
done
echo "HAHN SHORT CHAIN COMPLETE"

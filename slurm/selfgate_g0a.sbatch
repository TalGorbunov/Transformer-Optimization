#!/bin/bash
#SBATCH --job-name=sg_g0a
#SBATCH -p a100-public
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# SELFGATE G0a: per-head evidence-separation scan, ungated forwards.
# Chains: canonical S9b adapter (nfree prompt) + P1b reference, N in {8,32,128}.
# Explicit --time at submit.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
OUT="${OUTPUT:-outputs/selfgate/g0a}"
KLIST="${KLIST:-2-4-8}"; KLIST="${KLIST//-/,}"   # dashes in --export, commas for the CLI
S9B=checkpoints/sft_fenced_gated_vn_nfree_adapter
P1B=checkpoints/sft_fenced_le16_ep10_adapter
for N in 8 32 128; do
  run_logged "$OUT/s9b_N${N}" python -u scripts/selfgate/probe_headscan.py \
    --adapter "$S9B" --nfree-prompt --n "$N" --k-list "$KLIST" \
    --limit "${LIMIT:-30}" --output "$OUT/s9b_N${N}"
  run_logged "$OUT/p1b_N${N}" python -u scripts/selfgate/probe_headscan.py \
    --adapter "$P1B" --n "$N" --k-list "$KLIST" \
    --limit "${LIMIT:-30}" --output "$OUT/p1b_N${N}"
done
echo "G0A COMPLETE"

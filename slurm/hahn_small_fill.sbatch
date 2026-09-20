#!/bin/bash
#SBATCH --job-name=hahn_small
#SBATCH -p rtx6k-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -o logs/%x-%j.out
# Small-N flip fill (2026-09-19): N=2 frozen(plain,repjoint,fenced) + trained read (p1fence+P1b, N2 protocol)
# + gated (S11 protocol); N=4 trained read only (others exist). Explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
P1B=checkpoints/sft_fenced_le16_ep10_adapter
GATED=checkpoints/sft_fenced_gated_nfree_adapter
run_logged outputs/armor/hahn/short_N2 python -u scripts/armor/probe_hahn.py \
  --data_root data/mmred_images_park/seq_len_2/all_uniform --limit 50 --controls 12 --seed 0 --output outputs/armor/hahn/short_N2
for N in 2 4; do
  run_logged "outputs/loramech/n2_hahn/p1fence_ep10_N${N}" python -u scripts/armor/probe_hahn.py \
    --data_root "data/mmred_images_park/seq_len_${N}/all_uniform" --limit 50 --controls 12 --seed 0 --max-gold 8 \
    --arms p1fence --peft-adapter "$P1B" --output "outputs/loramech/n2_hahn/p1fence_ep10_N${N}"
done
run_logged outputs/sparse/s11/gate_oracle_N2 python -u scripts/sparse/probe_hahn_gated.py \
  --data_root data/mmred_images_park/seq_len_2/all_uniform --limit 50 --controls 12 --seed 0 --arms p1fence \
  --peft-adapter "$GATED" --gate oracle --nfree-prompt --output outputs/sparse/s11/gate_oracle_N2
echo "HAHN SMALL FILL COMPLETE"

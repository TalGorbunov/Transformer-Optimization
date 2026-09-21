#!/bin/bash
#SBATCH --job-name=sg_alpha
#SBATCH -p a100-public
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH -o logs/%x-%j.out
# SOFTGATE A1: alpha(B) chain over N (S11 protocol: S9b nfree adapter, oracle gate, nfree prompt, gold<=8).
# Knobs: BONUS (float), NLIST dash-separated. Explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
BONUS="${BONUS:?}"; NLIST="${NLIST:-8-16-32-64-128}"
GATED=checkpoints/sft_fenced_gated_vn_nfree_adapter
pool() { if [ "$1" -le 8 ]; then echo "data/mmred_images_park/seq_len_$1/all_uniform"; else echo "data/mmred_longN_park/seq_len_$1/all_uniform"; fi; }
for N in ${NLIST//-/ }; do
  LIM=50; [ "$N" -ge 128 ] && LIM=40
  run_logged "outputs/softgate/a1_B${BONUS}_N${N}" python -u scripts/sparse/probe_hahn_gated.py \
    --data_root "$(pool $N)" --limit $LIM --controls 12 --seed 0 --arms p1fence --peft-adapter "$GATED" \
    --gate oracle --nfree-prompt --gate-bonus "$BONUS" --output "outputs/softgate/a1_B${BONUS}_N${N}"
done
echo "SOFTGATE A1 (B=$BONUS) COMPLETE"

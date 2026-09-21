#!/bin/bash
#SBATCH --job-name=fixedk
#SBATCH -p a100-public
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%j.out
# FIXEDK: flip response at FIXED base count. COND=frozen|trained|gated; NLIST/GLIST dash-separated.
# frozen: probe_hahn arms plain,fenced (ARMOR protocol). trained: probe_hahn p1fence + P1b (N2 protocol).
# gated: probe_hahn_gated p1fence + S9b nfree adapter, oracle gate, nfree prompt (S11 protocol).
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
COND="${COND:?}"; NLIST="${NLIST:-2-4-8-16-32-64-128}"; GLIST="${GLIST:-0-1-2}"; LIMIT="${LIMIT:-30}"
P1B=checkpoints/sft_fenced_le16_ep10_adapter; GATED="${GATED:-checkpoints/sft_fenced_gated_nfree_adapter}"  # S9 = the S11 endpoint adapter (review item 4)
# pools: park (100/class) for N<=8; mmred_redux (fresh, 50/class at k=0,1,2, NEVER a training root) for N>=16 (review item 6)
pool() { if [ "$1" -le 8 ]; then echo "data/mmred_images_park/seq_len_$1/all_uniform"; else echo "data/mmred_redux/seq_len_$1/all_uniform"; fi; }
for N in ${NLIST//-/ }; do for G in ${GLIST//-/ }; do
  if [ "$N" -eq 2 ] && [ "$G" -ge 2 ]; then continue; fi
  OUT="outputs/fixedk/${COND}_N${N}_g${G}"
  case "$COND" in
    frozen)  run_logged "$OUT" python -u scripts/armor/probe_hahn.py --data_root "$(pool $N)" --arms plain,fenced --gold-set "$G" --limit "$LIMIT" --controls 12 --seed 0 --output "$OUT" ;;
    trained) run_logged "$OUT" python -u scripts/armor/probe_hahn.py --data_root "$(pool $N)" --arms p1fence --peft-adapter "$P1B" --max-gold 8 --gold-set "$G" --limit "$LIMIT" --controls 12 --seed 0 --output "$OUT" ;;
    gated)   run_logged "$OUT" python -u scripts/sparse/probe_hahn_gated.py --data_root "$(pool $N)" --arms p1fence --peft-adapter "$GATED" --gate oracle --nfree-prompt --gold-set "$G" --limit "$LIMIT" --controls 12 --seed 0 --output "$OUT" ;;
  esac
done; done
echo "FIXEDK ($COND) COMPLETE"

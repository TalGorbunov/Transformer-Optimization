#!/bin/bash
#SBATCH --job-name=lay_smoke
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH -o logs/%x-%j.out
# Query-side layout ablation smoke: anchor (no-flag eval-only P1b N8 must read 1.0000)
# + 1-epoch smokes of qfirst-once and qlast-once. Explicit --time at submit.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
SC=outputs/_scratch/layout
run_logged "$SC/anchor_p1b_N8" python -u scripts/sparse/train_sft_gated.py \
  --limit 24 --eval-only-adapter checkpoints/sft_fenced_le16_ep10_adapter \
  --eval-dirs-file outputs/loramech/examdirs/exam_ff_N8.txt --eval-longn-limit 150 \
  --output "$SC/anchor_p1b_N8"
for LAY in qfirst-once qlast-once; do
  run_logged "$SC/smoke_$LAY" python -u scripts/sparse/train_sft_gated.py \
    --data_root data/mmred_images_park/seq_len_8/all_uniform,data/mmred_longN_park/seq_len_16/all_uniform \
    --limit 12 --epochs 1 --val-cap 4 --layout "$LAY" \
    --eval-dirs-file outputs/loramech/examdirs/exam_ff_N8.txt --eval-longn-limit 4 \
    --output "$SC/smoke_$LAY"
done
echo "LAYOUT SMOKE COMPLETE"

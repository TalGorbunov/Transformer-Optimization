#!/bin/bash
#SBATCH --job-name=rx_smoke
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH -o logs/%x-%j.out
# REDUX v3 smoke: trainer anchor after the guard/nfree edits + 1-ep multitask smokes (ungated, gated), nfree.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
S=outputs/_scratch/redux_v3; P1B=checkpoints/sft_fenced_le16_ep10_adapter; EXN8=outputs/loramech/examdirs/exam_ff_N8.txt
run_logged $S/anchor_p1b_N8 python -u scripts/sparse/train_sft_gated.py --limit 24 --eval-only-adapter $P1B --eval-dirs-file $EXN8 --eval-longn-limit 150 --output $S/anchor_p1b_N8
grep -q "1.0000" $S/anchor_p1b_N8/*/longn_eval.csv && echo "[anchor] TRAINER P1b exam_ff_N8 = 1.0000" || { echo "[anchor] TRAINER ANCHOR MISS"; cat $S/anchor_p1b_N8/*/longn_eval.csv; exit 1; }
EV="--eval-dirs-file outputs/redux/examdirs/exam_count_N8.txt --eval-dirs-file outputs/redux/examdirs/exam_exists_N8.txt --eval-dirs-file outputs/redux/examdirs/exam_majority_N8.txt"
run_logged $S/smoke_mt_ungated python -u scripts/sparse/train_sft_gated.py --limit 18 --epochs 1 --val-cap 4 --tasks count,exists,majority --nfree-prompt $EV --eval-longn-limit 4 --output $S/smoke_mt_ungated
run_logged $S/smoke_mt_gated python -u scripts/sparse/train_sft_gated.py --limit 18 --epochs 1 --val-cap 4 --tasks count,exists,majority --gate oracle --nfree-prompt $EV --eval-longn-limit 4 --output $S/smoke_mt_gated
echo "REDUX V3 SMOKE COMPLETE"

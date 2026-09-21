#!/bin/bash
#SBATCH --job-name=sp_w3
#SBATCH -p l40s-shared
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH -o logs/%x-%j.out
# SPARSE wave 3 runner. Explicit --time at submit. MODEs:
#   smoke : ANCHOR (eval-only P1b exam_ff_N8@150 must read 1.0000 with the wave-3
#           deltas in place) + declare-n eval smoke + virtual-n 1-ep train smoke
#   equiv : S8 step-0 equivalence smoke (HARD GATE) — diag_equiv.py, P1g adapter
#   s7    : prompt-lie chain for one true-N dirs file: DECLARES (space-sep) evals
#           with --declare-n each (0 = ref, no flag)
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
P1G=checkpoints/sft_fenced_gated_adapter
P1B=checkpoints/sft_fenced_le16_ep10_adapter
SCRATCH=outputs/_scratch/sparse_w3
MODE="${MODE:?set MODE=smoke|equiv|s7}"
case "$MODE" in
smoke)
  run_logged "$SCRATCH/anchor_p1b_N8" python -u scripts/sparse/train_sft_gated.py \
    --limit 24 --eval-only-adapter "$P1B" \
    --eval-dirs-file outputs/loramech/examdirs/exam_ff_N8.txt \
    --eval-longn-limit 150 --output "$SCRATCH/anchor_p1b_N8"
  run_logged "$SCRATCH/smoke_declaren" python -u scripts/sparse/train_sft_gated.py \
    --limit 24 --eval-only-adapter "$P1G" --gate oracle --declare-n 16 \
    --eval-dirs-file outputs/sparse/s7/dirs_N32.txt \
    --eval-longn-limit 8 --output "$SCRATCH/smoke_declaren"
  run_logged "$SCRATCH/smoke_vn_train" python -u scripts/sparse/train_sft_gated.py \
    --data_root data/mmred_images_park/seq_len_8/all_uniform,data/mmred_longN_park/seq_len_16/all_uniform \
    --limit 12 --epochs 1 --val-cap 4 --gate oracle --virtual-n \
    --eval-dirs-file outputs/loramech/examdirs/exam_ff_N8.txt \
    --eval-longn-limit 4 --output "$SCRATCH/smoke_vn_train"
  ;;
equiv)
  run_logged "$SCRATCH/equiv" python -u scripts/sparse/diag_equiv.py \
    --peft-adapter "$P1G" --limit "${LIMIT:-10}" --output "$SCRATCH/equiv"
  ;;
smoke9b)
  run_logged "$SCRATCH/anchor9b_p1b_N8" python -u scripts/sparse/train_sft_gated.py \
    --limit 24 --eval-only-adapter "$P1B" \
    --eval-dirs-file outputs/loramech/examdirs/exam_ff_N8.txt \
    --eval-longn-limit 150 --output "$SCRATCH/anchor9b_p1b_N8"
  run_logged "$SCRATCH/smoke_9b_train" python -u scripts/sparse/train_sft_gated.py \
    --data_root data/mmred_images_park/seq_len_8/all_uniform,data/mmred_longN_park/seq_len_16/all_uniform \
    --limit 12 --epochs 1 --val-cap 4 --gate oracle --nfree-prompt --virtual-n \
    --eval-dirs-file outputs/loramech/examdirs/exam_ff_N8.txt \
    --eval-longn-limit 4 --output "$SCRATCH/smoke_9b_train"
  ;;
smoke9)
  run_logged "$SCRATCH/anchor9_p1b_N8" python -u scripts/sparse/train_sft_gated.py \
    --limit 24 --eval-only-adapter "$P1B" \
    --eval-dirs-file outputs/loramech/examdirs/exam_ff_N8.txt \
    --eval-longn-limit 150 --output "$SCRATCH/anchor9_p1b_N8"
  run_logged "$SCRATCH/smoke_nfree_train" python -u scripts/sparse/train_sft_gated.py \
    --data_root data/mmred_images_park/seq_len_8/all_uniform,data/mmred_longN_park/seq_len_16/all_uniform \
    --limit 12 --epochs 1 --val-cap 4 --gate oracle --nfree-prompt \
    --eval-dirs-file outputs/loramech/examdirs/exam_ff_N8.txt \
    --eval-longn-limit 4 --output "$SCRATCH/smoke_nfree_train"
  ;;
s7)
  DF="${DIRS_FILE:?set DIRS_FILE}"
  TAG="${TAG:?set TAG (e.g. N32)}"
  for DN in ${DECLARES:?set DECLARES}; do
    if [ "$DN" = 0 ]; then DFLAG=""; NAME=ref; else DFLAG="--declare-n $DN"; NAME="d$DN"; fi
    run_logged "outputs/sparse/s7/${TAG}_${NAME}" python -u scripts/sparse/train_sft_gated.py \
      --limit 24 --eval-only-adapter "$P1G" --gate oracle $DFLAG \
      --eval-dirs-file "$DF" --eval-longn-limit 80 \
      --output "outputs/sparse/s7/${TAG}_${NAME}"
  done
  ;;
*) echo "unknown MODE=$MODE" >&2; exit 2;;
esac
echo "SPARSE-W3 ($MODE) COMPLETE"

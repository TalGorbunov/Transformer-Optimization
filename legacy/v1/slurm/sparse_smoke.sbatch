#!/bin/bash
#SBATCH --job-name=sp_smoke
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -o logs/%x-%j.out
# SPARSE pre-flight: anchors + smokes (brief §5). MODE=trainer|probe.
#   trainer: (1) ANCHOR eval-only P1b on exam_ff_N8 @150 -> must read 1.0000
#            (2) gated-eval smoke (P1b + oracle gate, 6 dirs)
#            (3) S0 smoke (sharpen tau=2 from L12, 5 dirs @N32)
#            (4) gated-train smoke (1 ep, 16 samples, oracle gate)
#   probe:   (1) ANCHOR 3 pairs no flags vs n2_hahn dnorms  (2) 3 pairs gate=oracle
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
SCRATCH=outputs/_scratch/sparse_smoke
P1B=checkpoints/sft_fenced_le16_ep10_adapter
EXN8=outputs/loramech/examdirs/exam_ff_N8.txt
EXN32=outputs/loramech/examdirs/exam_ff_N32.txt
MODE="${MODE:?set MODE=trainer|probe}"
if [ "$MODE" = trainer ]; then
  run_logged "$SCRATCH/anchor_p1b_N8" python -u scripts/sparse/train_sft_gated.py \
    --limit 24 --eval-only-adapter "$P1B" --eval-dirs-file "$EXN8" \
    --eval-longn-limit 150 --output "$SCRATCH/anchor_p1b_N8"
  run_logged "$SCRATCH/smoke_gated_eval" python -u scripts/sparse/train_sft_gated.py \
    --limit 24 --eval-only-adapter "$P1B" --gate oracle --eval-dirs-file "$EXN8" \
    --eval-longn-limit 6 --output "$SCRATCH/smoke_gated_eval"
  run_logged "$SCRATCH/smoke_sharpen" python -u scripts/sparse/train_sft_gated.py \
    --limit 24 --eval-only-adapter "$P1B" --attn-sharpen 2 --sharpen-from-layer 12 \
    --eval-dirs-file "$EXN32" --eval-longn-limit 5 --output "$SCRATCH/smoke_sharpen"
  run_logged "$SCRATCH/smoke_gated_train" python -u scripts/sparse/train_sft_gated.py \
    --limit 16 --epochs 1 --val-cap 4 --gate oracle --eval-dirs-file "$EXN8" \
    --eval-longn-limit 4 --output "$SCRATCH/smoke_gated_train"
elif [ "$MODE" = probe ]; then
  run_logged "$SCRATCH/probe_anchor" python -u scripts/sparse/probe_hahn_gated.py \
    --data_root data/mmred_images_park/seq_len_8/all_uniform --limit 3 --controls 1 \
    --seed 0 --arms p1fence --peft-adapter "$P1B" --output "$SCRATCH/probe_anchor"
  run_logged "$SCRATCH/probe_gated" python -u scripts/sparse/probe_hahn_gated.py \
    --data_root data/mmred_images_park/seq_len_8/all_uniform --limit 3 --controls 1 \
    --seed 0 --arms p1fence --peft-adapter "$P1B" --gate oracle \
    --output "$SCRATCH/probe_gated"
else
  echo "unknown MODE=$MODE" >&2; exit 2
fi
echo "SPARSE SMOKE ($MODE) COMPLETE"

#!/bin/bash
#SBATCH --job-name=sg_a2
#SBATCH -p a100-public
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH -o logs/%x-%j.out
# SOFTGATE A2: the hard-gate-trained canonical adapter evaluated under a SOFT gate (no retrain), N=32/128.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
GATED=checkpoints/sft_fenced_gated_vn_nfree_adapter
# NLIST (default "32 128"): one --eval-dirs-file per N; SUFFIX tags the output dir (used for the post-timeout refill).
EVALS=""; for N in ${NLIST:-32 128}; do EVALS="$EVALS --eval-dirs-file outputs/loramech/examdirs/exam_ff_N${N}.txt"; done
for B in ${BLIST:-2 8 32 128}; do
  run_logged "outputs/softgate/a2_B${B}${SUFFIX:-}" python -u scripts/sparse/train_sft_gated.py --limit 24 \
    --eval-only-adapter $GATED --gate oracle --gate-bonus "$B" --nfree-prompt $EVALS \
    --eval-longn-limit 150 --output "outputs/softgate/a2_B${B}${SUFFIX:-}"
done
echo "SOFTGATE A2 COMPLETE"

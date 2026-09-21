#!/bin/bash
#SBATCH --job-name=ninv_twopass_hf
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
# ninv: frozen two-pass emitted headline on MMReD-HF N=8. Knobs: RESIZE OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/ninv/$(date +%Y%m%d_%H%M%S)_twopass_hf_${SLURM_JOB_ID:-local}}"
run_logged "$OUTPUT" python -u scripts/ninv/frozen_twopass_hf.py \
  --resize "${RESIZE:-512}" ${EXTRA:-} --output "$OUTPUT"

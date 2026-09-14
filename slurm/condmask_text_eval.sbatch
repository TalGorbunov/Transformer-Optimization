#!/bin/bash
# condmask length-transfer eval wrapper (see condmask_text_train.sbatch header).
# Knobs: CKPT (or REGIME=full|blockwise), ROOTS, OUTPUT, EXTRA.
#   sbatch --export=ALL,CKPT=outputs/condmask/<run>/condmask_best.pt slurm/condmask_text_eval.sbatch
#SBATCH -J condmask-eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARGS=()
[ -n "${CKPT:-}" ] && ARGS+=(--ckpt "$CKPT")
[ -n "${REGIME:-}" ] && ARGS+=(--regime "$REGIME")
[ -n "${ROOTS:-}" ] && ARGS+=(--roots "$ROOTS")
exec .venv/bin/python scripts/condmask/eval_condmask_text.py \
  "${ARGS[@]}" --output "${OUTPUT:-outputs/condmask/eval}" ${EXTRA:-}

#!/bin/bash
#SBATCH --job-name=armA_reader
#SBATCH -p a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# RECAGG Arm A: one frozen pretrained reader over oracle captions (venv_arch).
# Knobs: MODEL LABEL OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
export HF_HOME=/rg/shocher_prj/tal.gorbunov/hf_arch
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # compute nodes: cached models only
ARCH_PY=/rg/shocher_prj/tal.gorbunov/venv_arch/bin/python
run_logged "${OUTPUT:?set OUTPUT}" "$ARCH_PY" -u scripts/recagg/frozen_readers.py \
  --model "${MODEL:?set MODEL}" --label "${LABEL:?set LABEL}" \
  ${EXTRA:-} --output "$OUTPUT"

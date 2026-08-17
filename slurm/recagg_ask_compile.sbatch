#!/bin/bash
#SBATCH --job-name=armC_compile
#SBATCH -p a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH -o logs/%x-%j.out
# RECAGG Arm C: Ask-Compile-Execute over oracle records (venv_arch).
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
export HF_HOME=/rg/shocher_prj/tal.gorbunov/hf_arch
export HF_HUB_OFFLINE=1
ARCH_PY=/rg/shocher_prj/tal.gorbunov/venv_arch/bin/python
run_logged "${OUTPUT:?set OUTPUT}" "$ARCH_PY" -u scripts/recagg/ask_compile_execute.py \
  ${EXTRA:-} --output "$OUTPUT"

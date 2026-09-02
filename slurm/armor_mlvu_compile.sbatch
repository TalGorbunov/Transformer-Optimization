#!/bin/bash
#SBATCH --job-name=armor_mlvu_ace
#SBATCH -p a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# ARMOR C step 2: compile+execute on MLVU records (venv_arch). Knobs: RECORDS LIMIT OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
export HF_HOME=/rg/shocher_prj/tal.gorbunov/hf_arch
export HF_HUB_OFFLINE=1
ARCH_PY=/rg/shocher_prj/tal.gorbunov/venv_arch/bin/python
run_logged "${OUTPUT:?set OUTPUT}" "$ARCH_PY" -u scripts/armor/mlvu_compile_execute.py \
  --records "${RECORDS:?set RECORDS}" --limit "${LIMIT:-0}" ${EXTRA:-} --output "$OUTPUT"

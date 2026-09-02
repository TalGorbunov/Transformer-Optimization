#!/bin/bash
#SBATCH --job-name=tf_14b
#SBATCH -p l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH -o logs/%x-%j.out
# TREEFOLD 14B stage-swap control (venv_arch, armC pattern).
# Knobs: MODE(leaves|answers) ASKS OUTPUT [MERGES NS PER_TYPE LIMIT BATCH TITLE EXTRA]
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
export HF_HOME=/rg/shocher_prj/tal.gorbunov/hf_arch
export HF_HUB_OFFLINE=1
ARCH_PY=/rg/shocher_prj/tal.gorbunov/venv_arch/bin/python
run_logged "${OUTPUT:?set OUTPUT}" "$ARCH_PY" -u scripts/treefold/stages_14b.py \
  --mode "${MODE:?set MODE}" --asks "${ASKS:?set ASKS}" \
  --ns "${NS:-16-32-64-128}" --per-type "${PER_TYPE:-8}" \
  --batch "${BATCH:-16}" ${LIMIT:+--limit "$LIMIT"} \
  ${MERGES:+--merges "$MERGES"} --title "${TITLE:-TREEFOLD_14B}" \
  ${EXTRA:-} --output "$OUTPUT"

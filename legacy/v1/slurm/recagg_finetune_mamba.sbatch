#!/bin/bash
#SBATCH --job-name=mamba_count_ft
#SBATCH -p a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH -o logs/%x-%j.out
# RECAGG adapter route: LoRA count fine-tune (train N<=16) then chained
# frozen_readers eval at N=8..128 with the adapter. Knobs: MODEL LABEL OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
export HF_HOME=/rg/shocher_prj/tal.gorbunov/hf_arch
export HF_HUB_OFFLINE=1
ARCH_PY=/rg/shocher_prj/tal.gorbunov/venv_arch/bin/python
OUT="${OUTPUT:?set OUTPUT}"
run_logged "$OUT" "$ARCH_PY" -u scripts/recagg/finetune_mamba.py \
  --model "${MODEL:-state-spaces/mamba-2.8b-hf}" ${EXTRA:-} --output "$OUT" \
  || { echo "[chain] training failed — skipping eval"; exit 1; }
run_logged "$OUT/eval" "$ARCH_PY" -u scripts/recagg/frozen_readers.py \
  --model "${MODEL:-state-spaces/mamba-2.8b-hf}" --adapter "$OUT/adapter" \
  --shots 0 --label "${LABEL:-mamba_ft}_eval" --limit 50 --output "$OUT/eval"

#!/bin/bash
#SBATCH --job-name=bench_noharm
#SBATCH -p l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# MME+POPE no-harm. Knobs: LORA_CKPT or PEFT_ADAPTER; MME_N POPE_N OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/carrier/noharm}"
run_logged "$OUTPUT" python -u scripts/bench_noharm.py \
  ${LORA_CKPT:+--lora-ckpt "$LORA_CKPT"} ${PEFT_ADAPTER:+--peft-adapter "$PEFT_ADAPTER"} \
  --mme-n "${MME_N:-500}" --pope-n "${POPE_N:-500}" --output "$OUTPUT"

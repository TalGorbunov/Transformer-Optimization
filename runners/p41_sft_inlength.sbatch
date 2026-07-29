#!/bin/bash
#SBATCH --job-name=p41_sft
#SBATCH --partition=a100-public
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x-%j.out
# P4.1 (p0p2, PREREG plans/p4_PREREG.md): plain-LoRA SFT retrained IN-LENGTH — config
# identical to sft_control_le8_v2 except the data root spans N=8/16/32/64 (symlink farm
# data/mmred_sft_p41; longN_park2 + non-steps tasks omitted, documented). FLASH wrap on
# the training forward; adapter saved; exams follow separately on arm-A dirs-files.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/glstm/lora_sft_baseline.py \
  --task steps_in_room --data-root data/mmred_sft_p41 \
  --train-seq-lens 8,16,32 \
  --epochs 4 --patience 4 --lora-r 8 \
  --output outputs/ladder/image_longN/sft_inlength_p41

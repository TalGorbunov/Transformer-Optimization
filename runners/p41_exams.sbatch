#!/bin/bash
#SBATCH --job-name=p41_exam
#SBATCH --partition=a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# P4.1 exams (PREREG plans/p4_PREREG.md): in-length SFT adapter, generate-and-parse.
# N=32 = in-length primary band cell (held-out file: own test split + unseen park2);
# N=64 = extrapolation (trained <=32), arm-A 52-dir file — clean for this ckpt.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=outputs/ladder/image_longN/sft_inlength_p41/20260725_031153_lora
A=outputs/ladder/image_longN/carrier_tally_l12v2/20260721_071710_L12_r8
python -u experiments/glstm/lora_sft_baseline.py \
  --task steps_in_room --data-root data/mmred_sft_p41 --train-seq-lens 8,16,32 \
  --epochs 0 \
  --eval-only-adapter $D/adapter \
  --eval-dirs-file $D/eval_dirs_p41_N32.txt \
  --eval-dirs-file $A/eval_dirs_N64.txt \
  --eval-longn-limit 150 \
  --output outputs/ladder/image_longN/sft_inlength_p41_exams

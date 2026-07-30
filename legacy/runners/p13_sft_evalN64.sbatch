#!/bin/bash
#SBATCH --job-name=sft_evalN64
#SBATCH --partition=a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# P1.3 (p0p2 campaign): E-B SFT baseline N=64 cell — rerun of the OOM'd leg of job 124696
# with the EFF_SDPA eval path (efficient-attention backend; the fix that unlocked N=128
# evals on 40GB A100s). Eval-only: loads the saved adapter, reruns test_iid as a sanity
# anchor (expect ~1.000), then the N=64 generate-and-parse cell (limit 100, same
# stratified-shuffled-prefix sampling as the landed N=16/N=32 cells).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u experiments/glstm/lora_sft_baseline.py \
  --task steps_in_room --data-root data/mmred_images_park \
  --epochs 0 \
  --eval-only-adapter outputs/ladder/image_longN/sft_control_le8_v2/20260720_191541_lora/adapter \
  --eval-longn data/mmred_longN_park/seq_len_64/all_uniform \
  --eval-longn-limit 100 \
  --output outputs/ladder/image_longN/sft_control_le8_v2_evalN64

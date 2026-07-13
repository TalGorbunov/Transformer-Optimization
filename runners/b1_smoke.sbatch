#!/bin/bash
#SBATCH --job-name=b1_smoke
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
# B1 smoke: joint probe (with new mass field) + image multipass, N=16 @392px, n=4.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "=== smoke joint N=16 ==="
python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
  --task count --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data_root data/mmred_longN_park/seq_len_16/all_uniform \
  --limit 4 --layers 14,16 --carrier per_token --n-frames 16 --resize 392 \
  --max-offset 13 --decode-offsets 9,13 --sample-seed 0 --save-messages \
  --output outputs/_scratch/b1_joint_smoke
echo "=== smoke multipass image N=16 ==="
python -u experiments/glstm/text_multipass_cache.py \
  --modality image --resize 392 \
  --data_root data/mmred_longN_park/seq_len_16/all_uniform \
  --limit 4 --layers 14,16 --decode-offsets 9,13 --n-frames 16 --sample-seed 0 \
  --output outputs/_scratch/b1_mp_smoke
echo "B1 SMOKE DONE"

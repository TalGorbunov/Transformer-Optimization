#!/bin/bash
#SBATCH --job-name=unified_7b
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n314
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u evaluations/scripts/patch_importence/unified_bottleneck_analysis.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --clean_data_dir data/mmred_images_park/seq_len_8/all_uniform \
  --corrupted_data_dir data/mmred_corrupted_park/seq_len_8/all_uniform \
  --output_dir outputs/unified_count_7b --sample_limit 10 \
  --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26 --clean_top1_must_match_gold true --disable_plots

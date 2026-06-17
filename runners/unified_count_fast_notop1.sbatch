#!/bin/bash
#SBATCH --job-name=unified_gt5
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u evaluations/scripts/patch_importence/unified_bottleneck_analysis.py \
  --clean_data_dir data/mmred_images_park/seq_len_8/all_uniform \
  --corrupted_data_dir data/mmred_corrupted_park/seq_len_8/all_uniform \
  --output_dir outputs/unified_count_fast_notop1 --sample_limit 5 \
  --layers 0,4,8,12,16,20,24 --clean_top1_must_match_gold false --disable_plots

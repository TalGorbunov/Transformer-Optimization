#!/bin/bash
#SBATCH --job-name=unified_gt
#SBATCH --partition=h200-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u evaluations/scripts/patch_importence/unified_bottleneck_analysis.py \
  --clean_data_dir data/mmred_images_park/seq_len_8/all_uniform \
  --corrupted_data_dir data/mmred_corrupted_park/seq_len_8/all_uniform \
  --output_dir outputs/unified_count_groundtruth --sample_limit 20 \
  --layers 0,4,8,12,16,20,24 --disable_plots

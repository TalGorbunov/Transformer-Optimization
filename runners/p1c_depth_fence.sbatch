#!/bin/bash
#SBATCH --job-name=p1c_fence
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# P1c: hybrid-fence depth sweep — frames isolated (no cross-frame visual attention) in layers
# [0,L), shared above; standard N=8 joint config. Known confound: full-depth fencing is
# off-distribution and HURT (2.47->1.93); the sweep tests whether PARTIAL-depth isolation
# recovers toward the gray-companion level at some depth.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
for L in 4 8 12; do
  echo "=== fence-upto $L ==="
  python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
    --task count --model Qwen/Qwen2.5-VL-7B-Instruct --fence-cross-frame --fence-upto "$L" \
    --data_root data/mmred_images_park/seq_len_8/all_uniform \
    --limit 150 --layers 14,16 --carrier per_token --n-frames 8 --resize 392 \
    --max-offset 13 --decode-offsets 9,13 --sample-seed 0 --save-messages \
    --output "outputs/ladder/image_longN/depth_fence/upto${L}/${TS}"
done
echo P1C_DONE

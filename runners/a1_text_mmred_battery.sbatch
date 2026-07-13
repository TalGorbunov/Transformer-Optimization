#!/bin/bash
#SBATCH --job-name=a1_text
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# A1: text-MMRED instrument battery (frozen Qwen-7B, frames as text).
#  1) carrier locmap + message cache (per_token, MAXOFF=20, offsets 9,13, n=400, save-messages)
#  2) behavioral baseline at N=8, n=250 (generation + parse)
#  3) native reading axis at the room token (offset 9), n=100
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PARTS="${PARTS:-1,2,3}"

[[ ",$PARTS," == *,1,* ]] && {
echo "=== A1.1 text locmap + message cache ==="
python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
  --task count --model Qwen/Qwen2.5-VL-7B-Instruct --text-frames \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-400}" --layers "${LAYERS:-4,8,12,14,16,18,20,24}" \
  --carrier per_token --n-frames 8 \
  --max-offset "${MAXOFF:-20}" --decode-offsets "${DECODE_OFFSETS:-9,13}" \
  --sample-seed 0 --save-messages \
  --output "outputs/ladder/text_mmred/locmap_cache/${TS}"
}

[[ ",$PARTS," == *,2,* ]] && {
echo "=== A1.2 behavior N=8 n=250 ==="
python -u evaluations/scripts/eval_mmred_text_frames_acc.py \
  --task steps_in_room --seq-lens 8 --max-samples "${BEH_N:-250}" \
  --data-root "${DATA_ROOT_BEH:-data/mmred_images_park}" --split all_uniform \
  --output "outputs/ladder/text_mmred/behavior/${TS}"
}

[[ ",$PARTS," == *,3,* ]] && {
echo "=== A1.3 native axis (room token, offset 9) ==="
python -u evaluations/scripts/patch_importence/native_axis_probe.py \
  --text-frames --layers "${NAT_LAYERS:-14,16,20}" --offset 9 --limit "${NAT_N:-100}" \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --sample-seed 0 \
  --output "outputs/ladder/text_mmred/native_axis/${TS}"
}
echo "A1 GPU battery done (ts=${TS})"

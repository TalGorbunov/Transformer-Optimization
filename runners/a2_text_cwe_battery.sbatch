#!/bin/bash
#SBATCH --job-name=a2_cwe
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# A2: text-CWE rung (word-in-frame counting, extraction perfect by construction).
# Per N in {8,16,32,64}: behavior (own-answer) + carrier message cache (offsets 9=appear, 11=word).
# N=8 run doubles as the locmap (MAXOFF=16). Then native axis on the CWE question at N=8.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

for NF in ${N_LIST:-8 16 32 64}; do
  MAXOFF=11; [[ "$NF" == "8" ]] && MAXOFF=16
  echo "=== A2 text-CWE N=${NF} ==="
  python -u evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py \
    --task text_cwe --model Qwen/Qwen2.5-VL-7B-Instruct --text-frames \
    --limit "${LIMIT:-150}" --layers "${LAYERS:-8,12,14,16,18,20,24}" \
    --carrier per_token --n-frames "$NF" \
    --max-offset "$MAXOFF" --decode-offsets "${DECODE_OFFSETS:-9,11}" \
    --sample-seed 0 --save-messages \
    --output "outputs/ladder/text_cwe/locmap_cache/${TS}/N${NF}"
done

echo "=== A2 native axis (CWE, N=8, offset 11 = quoted word) ==="
python -u evaluations/scripts/patch_importence/native_axis_probe.py \
  --task text_cwe --text-frames --n-frames 8 \
  --layers "${NAT_LAYERS:-14,16,20}" --offset "${NAT_OFF:-11}" --limit "${NAT_N:-100}" \
  --sample-seed 0 \
  --output "outputs/ladder/text_cwe/native_axis/${TS}"
echo "A2 battery done (ts=${TS})"

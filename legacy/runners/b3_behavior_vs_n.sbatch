#!/bin/bash
#SBATCH --job-name=b3_beh
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# B3: frozen-model behavior vs N (generation reader, @392px). Env: SEQ_LENS="8 16 32", LIMIT.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
for N in ${SEQ_LENS:?}; do
  if [[ "$N" == "8" ]]; then DR="data/mmred_images_park/seq_len_8/all_uniform";
  else DR="data/mmred_longN_park/seq_len_${N}/all_uniform"; fi
  python -u experiments/readout/behavior_vs_n.py \
    --data_root "$DR" --limit "${LIMIT:-150}" --resize 392 --sample-seed 1 \
    --output "outputs/ladder/image_longN/behavior/N${N}"
done
echo B3_DONE

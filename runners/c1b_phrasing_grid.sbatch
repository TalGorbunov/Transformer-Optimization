#!/bin/bash
#SBATCH --job-name=c1b_grid
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# C1b: fact-phrasing robustness grid (approved 2026-07-09) — paraphrases, position,
# digits-vs-words, distractor fact, source-attributed; counterfactual targets so the
# score isolates fact-reading. De-risks C2's injection interface.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u experiments/readout/c1_token_interface.py \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-120}" --sample-seed 0 \
  --arms "${ARMS:-fact_cf_in,fact_cf_ood,para1_cf_in,para1_cf_ood,para2_cf_ood,words_cf_in,words_cf_ood,src_cf_ood,top_cf_ood,factq_cf_ood,dis_cf_in,dis_cf_ood}" \
  --output "${OUTPUT:-outputs/readout/c1b_phrasing}"

#!/bin/bash
#SBATCH -J native_recover_harvest
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/probe_native_vision_recoverability.py \
    --manifest /mnt/data/gabriele/gnn_transformer/v2_clean/main_manifest.json \
    --count-manifest /mnt/data/gabriele/gnn_transformer/v2_clean/count_manifest.json \
    --data-output "/mnt/data/gabriele/gnn_transformer/v2_recoverability/harvest_${SLURM_JOB_ID:?}" \
    --output outputs/native_aggregation_vlm/v2/recoverability

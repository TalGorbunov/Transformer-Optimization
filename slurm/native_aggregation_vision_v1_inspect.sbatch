#!/bin/bash
#SBATCH -J native_agg_v1_inspect
#SBATCH -t 00:15:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/inspect_native_vision_v1.py \
    --root outputs/native_aggregation_vlm/v1 \
    --manifest /mnt/data/gabriele/gnn_transformer/native_aggregation_vision_pilot/profile_manifest.json

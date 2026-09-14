#!/bin/bash
#SBATCH -J native_agg_v1_base
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/native_aggregation_vlm.py \
    --arm base --eval-only --epochs 0 --seed 0 --data-seed 20260910 \
    --dataset-root /mnt/data/gabriele/gnn_transformer \
    --checkpoint-root /mnt/ckpts/gabriele/gnn_transformer \
    --manifest /mnt/data/gabriele/gnn_transformer/native_aggregation_vision_pilot/main_manifest.json \
    --eval-ns 8+16+32+64 --limit-eval 100 --resize 392 --max-seq-tokens 16000 \
    --max-new-tokens 4 --output outputs/native_aggregation_vlm/v1/base

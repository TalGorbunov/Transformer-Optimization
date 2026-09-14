#!/bin/bash
#SBATCH -J native_reason_vlm_check
#SBATCH -t 00:05:00
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
.venv/bin/python scripts/native_aggregation_vlm_v3.py \
    --model /mnt/ckpts/gabriele/gnn_transformer/cosmos_reason1_compat_440953 --arm lift_pre --seed 0 \
    --dataset-root /mnt/data/gabriele/gnn_transformer/v2_clean \
    --checkpoint-root /mnt/ckpts/gabriele/gnn_transformer \
    --manifest /mnt/data/gabriele/gnn_transformer/v2_clean/profile_manifest.json \
    --count-manifest /mnt/data/gabriele/gnn_transformer/v2_clean/profile_count_manifest.json \
    --layer-index 14 --rank 64 --lift-rank 8 --lora-layers 4 --lora-rank 8 --lora-alpha 16 \
    --lr 0.001 --lr-lora 0.0001 --accumulation 4 \
    --data-seed 20260910 --resize 392 --max-seq-tokens 16000 --max-new-tokens 4 --gold-max 16 \
    --train-ns 16 --dev-ns 16 --eval-ns 16+64 \
    --limit-train 2 --limit-dev 2 --limit-eval 2 --limit-count 2 \
    --epochs 1 --max-steps 1 --output outputs/native_aggregation_vlm/v3/reasoning_compatibility

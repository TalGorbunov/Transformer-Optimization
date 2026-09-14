#!/bin/bash
#SBATCH -J native_agg_v3
#SBATCH --array=0-3%4
#SBATCH -t 00:50:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON=(--dataset-root /mnt/data/gabriele/gnn_transformer/v2_clean
    --checkpoint-root /mnt/ckpts/gabriele/gnn_transformer
    --layer-index 14 --rank 64 --lift-rank 8
    --lora-layers 4 --lora-rank 8 --lora-alpha 16
    --lr 0.001 --lr-lora 0.0001 --accumulation 4
    --data-seed 20260910 --resize 392 --max-seq-tokens 16000
    --max-new-tokens 4 --gold-max 16)
ARMS=(lift_pre lift_post lift_pre lift_post)
SEEDS=(0 0 1 1)
ARM="${ARMS[${SLURM_ARRAY_TASK_ID:?}]}"
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:?}]}"
.venv/bin/python scripts/native_aggregation_vlm_v3.py "${COMMON[@]}" \
    --seed "$SEED" --arm "$ARM" \
    --manifest /mnt/data/gabriele/gnn_transformer/v2_clean/main_manifest.json \
    --count-manifest /mnt/data/gabriele/gnn_transformer/v2_clean/count_manifest.json \
    --train-ns 8+16 --dev-ns 8+16 --eval-ns 16+32+64 \
    --limit-train 90 --limit-dev 36 --limit-eval 108 --limit-count 64 \
    --epochs 9 --output "outputs/native_aggregation_vlm/v3/main/$ARM/seed$SEED"

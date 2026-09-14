#!/bin/bash
#SBATCH -J native_agg_diag
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON=(--model Qwen/Qwen2.5-3B-Instruct
    --dataset-root /mnt/data/gabriele/gnn_transformer
    --checkpoint-root /mnt/ckpts/gabriele/gnn_transformer
    --layer-index 18 --rank 64 --query-chunk-size 32
    --lora-layers 4 --lora-rank 8 --lora-alpha 16
    --lr 0.001 --lr-lora 0.0001 --accumulation 4
    --seed 0 --data-seed 1234 --max-new-tokens 4
    --train-ns 8+16+32 --dev-ns 8+32 --eval-ns 8+32+64+128
    --limit-train 90 --limit-dev 36 --limit-eval 72 --epochs 9 --test-epochs 3+9)
.venv/bin/python scripts/native_aggregation.py "${COMMON[@]}" --arm lora --block-size 64     --output outputs/native_aggregation/p1/lora
for block in 64 16; do
    .venv/bin/python scripts/native_aggregation.py "${COMMON[@]}" --arm sum --block-size "$block"         --output "outputs/native_aggregation/p1/raw$block"
    .venv/bin/python scripts/native_aggregation.py "${COMMON[@]}" --arm sum --block-size "$block"         --center-messages --output "outputs/native_aggregation/p1/centered$block"
done

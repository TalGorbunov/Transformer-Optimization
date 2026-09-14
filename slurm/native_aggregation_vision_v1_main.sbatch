#!/bin/bash
#SBATCH -J native_agg_v1
#SBATCH --array=0-6%4
#SBATCH -t 01:00:00
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
COMMON=(--dataset-root /mnt/data/gabriele/gnn_transformer
    --checkpoint-root /mnt/ckpts/gabriele/gnn_transformer
    --layer-index 14 --block-size 64 --rank 64 --routing-rank 8 --query-chunk-size 32
    --lora-layers 4 --lr 0.001 --lr-lora 0.0001 --accumulation 4
    --seed 0 --data-seed 20260910 --resize 392 --max-seq-tokens 16000
    --eval-ns 8+16+32+64 --max-new-tokens 4)
ARMS=(sum post_sum hierarchical lora mean post_mean global)
ARM="${ARMS[${SLURM_ARRAY_TASK_ID:?}]}"
LORA_RANK=8
LORA_ALPHA=16
if [[ "$ARM" == lora ]]; then
    LORA_RANK=16
    LORA_ALPHA=32
fi
.venv/bin/python scripts/native_aggregation_vlm.py "${COMMON[@]}" \
    --arm "$ARM" --manifest /mnt/data/gabriele/gnn_transformer/native_aggregation_vision_pilot/main_manifest.json \
    --train-ns 8+16 --dev-ns 8+16 --limit-train 90 --limit-dev 36 --limit-eval 100 \
    --epochs 9 --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
    --output "outputs/native_aggregation_vlm/v1/main/$ARM"

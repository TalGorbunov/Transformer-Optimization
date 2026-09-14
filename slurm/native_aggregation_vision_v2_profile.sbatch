#!/bin/bash
#SBATCH -J native_agg_v2_profile
#SBATCH --array=0-3%4
#SBATCH -t 00:05:00
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
    --layer-index 14 --block-size 64 --rank 64 --query-chunk-size 32
    --lora-layers 4 --lr 0.001 --lr-lora 0.0001 --accumulation 4
    --seed 0 --data-seed 20260910 --resize 392 --max-seq-tokens 16000
    --max-new-tokens 4 --gold-max 16)
ARMS=(global hidden middle_lora lora)
ARM="${ARMS[${SLURM_ARRAY_TASK_ID:?}]}"
LORA_RANK=8
LORA_ALPHA=16
if [[ "$ARM" == lora ]]; then
    LORA_RANK=16
    LORA_ALPHA=32
fi
 .venv/bin/python scripts/native_aggregation_vlm_v2.py "${COMMON[@]}" \
    --arm "$ARM" --manifest /mnt/data/gabriele/gnn_transformer/v2_clean/profile_manifest.json \
    --count-manifest /mnt/data/gabriele/gnn_transformer/v2_clean/profile_count_manifest.json \
    --train-ns 16 --dev-ns 16 --eval-ns 16+64 \
    --limit-train 2 --limit-dev 2 --limit-eval 2 --limit-count 2 \
    --epochs 1 --max-steps 1 --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
    --output "outputs/native_aggregation_vlm/v2/profile/$ARM"

#!/bin/bash
#SBATCH -J native_agg_v6_profile
#SBATCH --array=0-1%2
#SBATCH -t 00:03:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V6_SOURCES="${1:?Pass the frozen CPU source_hashes.json}"
COMMON=(--source-ledger "$V6_SOURCES" --dataset-root /mnt/data/gabriele/gnn_transformer/v4_diversity
    --checkpoint-root /mnt/ckpts/gabriele/gnn_transformer
    --teacher-index /mnt/data/gabriele/gnn_transformer/v6_local_teacher/teacher_cache.json
    --arm independent --layer-index 14 --rank 96
    --lora-layers 4 --lora-rank 8 --lora-alpha 16
    --lr 0.001 --lr-lora 0.0001 --max-grad-norm 1 --accumulation 4
    --data-seed 20260914 --resize 392 --max-seq-tokens 16000
    --max-new-tokens 4 --gold-max 16 --eval-modes all)
CONDITIONS=(control aligned)
CONDITION="${CONDITIONS[${SLURM_ARRAY_TASK_ID:?}]}"
.venv/bin/python scripts/native_aggregation_vlm_v6.py "${COMMON[@]}" \
    --condition "$CONDITION" --seed 6 --profile \
    --manifest /mnt/data/gabriele/gnn_transformer/v4_diversity/profile_manifest.json \
    --count-manifest /mnt/data/gabriele/gnn_transformer/v4_diversity/profile_count_manifest.json \
    --schedule /mnt/data/gabriele/gnn_transformer/v4_diversity/profile_schedule.json \
    --test-manifest /mnt/data/gabriele/gnn_transformer/v4_diversity/profile_manifest.json \
    --test-count-manifest /mnt/data/gabriele/gnn_transformer/v4_diversity/profile_count_manifest.json \
    --train-ns 8+16 --dev-ns 8+16 --eval-ns 16+32+64 \
    --limit-train 4 --limit-dev 2 --limit-eval 2 --limit-count 2 \
    --epochs 2 --output "outputs/native_aggregation_vlm/v6/profile/$CONDITION"

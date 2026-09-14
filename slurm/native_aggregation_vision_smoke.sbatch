#!/bin/bash
#SBATCH -J native_agg_vsmoke
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
.venv/bin/python scripts/native_aggregation_vlm.py --arm sum     --dataset-root /mnt/data/gabriele/gnn_transformer     --checkpoint-root /mnt/ckpts/gabriele/gnn_transformer     --train-ns 8 --dev-ns 8 --eval-ns 8 --limit-train 2 --limit-dev 2 --limit-eval 2     --epochs 1 --max-steps 1 --accumulation 2 --seed 0 --data-seed 1234     --block-size 64 --rank 64 --query-chunk-size 32 --max-new-tokens 4     --output outputs/native_aggregation_vlm/smoke

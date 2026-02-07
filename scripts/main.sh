#!/bin/bash
#SBATCH --job-name=transofrmer-optimization
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tlgpu

python main.py
EOF
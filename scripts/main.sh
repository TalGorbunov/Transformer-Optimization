#!/bin/bash
#SBATCH --job-name=transofrmer-optimization
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

source .venv/bin/activate

python main.py --data_root data/mmred_images/seq_len_8/train --limit 1
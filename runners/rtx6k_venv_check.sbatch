#!/bin/bash
#SBATCH --job-name=rtx_venvcheck
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
echo "node=$(hostname)"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; else echo "NO_VENV"; fi
.venv/bin/python -c "import torch; print('TORCH_OK', torch.__version__, 'cuda_avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')" \
  && echo "VENV_WORKS_ON_RTX" || echo "VENV_BROKEN_ON_RTX"

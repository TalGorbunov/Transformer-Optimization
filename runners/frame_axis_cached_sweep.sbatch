#!/bin/bash
#SBATCH --job-name=frame_agg_sweep
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT=experiments/glstm/frame_axis_aggregator_cached.py
COMMON="--train-seq-lens 1,2,3,4,5,6,7,8 --ood-seq-lens 99 --cap-per-seq 400 --epochs ${EPOCHS:-120}"
OUT=outputs/frame_axis_sweep

run () {  # $1=name  $2...=flags
  name="$1"; shift
  echo "================ SWEEP: ${name} ================"
  python -u "${SCRIPT}" ${COMMON} "$@" --output "${OUT}/${name}"
}

run deepsets          --aggregator deepsets
run deepsets_codebook --aggregator deepsets --phi codebook
run pna               --aggregator pna
run pna_codebook      --aggregator pna --phi codebook
run pna_balanced      --aggregator pna --balanced-loss
run pna_cb_balanced   --aggregator pna --phi codebook --balanced-loss
echo "==== SWEEP DONE ===="

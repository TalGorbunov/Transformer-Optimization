#!/bin/bash
#SBATCH -J native_agg_check
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
DATA_ROOT=/mnt/data/gabriele/gnn_transformer
mkdir -p "$DATA_ROOT/mmred_filtered_train" "$DATA_ROOT/mmred_filtered"
for n in 8 16 32; do
    rsync -a --ignore-existing "data/mmred_filtered_train/seq_len_$n" "$DATA_ROOT/mmred_filtered_train/"
done
for n in 8 32 64 128; do
    rsync -a --ignore-existing "data/mmred_filtered/seq_len_$n" "$DATA_ROOT/mmred_filtered/"
done
.venv/bin/python tests/test_native_aggregation.py
for test in tests/test_fencing.py tests/test_carrier_masks.py tests/test_scratchpad.py tests/test_data.py; do
    .venv/bin/python "$test"
done

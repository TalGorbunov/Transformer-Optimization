#!/bin/bash
# SPARSE wave 3 — S8 eval cascade, fired when the vn adapter lands.
# Usage: bash scripts/sparse/wave3_evals.sh <s8_adapter_dir>
set -euo pipefail
VN="${1:?usage: wave3_evals.sh <adapter dir>}"
test -d "$VN" || { echo "no adapter at $VN"; exit 1; }
EX=outputs/loramech/examdirs
sbatch -p l40s-shared --qos=4d_1g --time=6:00:00 --job-name=sp8_ladder \
  --export=ALL,LIMIT=24,ADAPTER=$VN,GATE=oracle,DIRS_FILES="$EX/exam_ff_N64.txt $EX/exam_ff_N128.txt",LONGN_LIMIT=150,OUTPUT=outputs/sparse/s8_ladder_64_128 \
  slurm/sparse_train.sbatch
sbatch -p l40s-shared --qos=4d_1g --time=8:00:00 --job-name=sp8_s4 \
  --export=ALL,LIMIT=24,ADAPTER=$VN,GATE=oracle,DIRS_FILES="outputs/sparse/s4/dirs_N64.txt outputs/sparse/s4/dirs_N128.txt",LONGN_LIMIT=300,OUTPUT=outputs/sparse/s8_s4 \
  slurm/sparse_train.sbatch
RUN_DIR=$(dirname "$VN")
sbatch -p l40s-shared --qos=4d_1g --time=3:00:00 --job-name=sp8_cap_tr \
  --export=ALL,DIRS_FILES="$RUN_DIR/train_dirs.txt",LAYER=20,ADAPTER=$VN,OUTPUT=outputs/sparse/s8_cap/cap_train \
  slurm/sparse_capture.sbatch
for N in 8 16 32 64 128; do
  sbatch -p rtx6k-shared --qos=4d_1g --time=1:50:00 --job-name=sp8_cap_N$N \
    --export=ALL,DIRS_FILES="$EX/exam_ff_N$N.txt",LAYER=20,ADAPTER=$VN,OUTPUT=outputs/sparse/s8_cap/cap_N$N \
    slurm/sparse_capture.sbatch
done
squeue -u $USER -h -o "%.9i %.12j %.13P %.8T %.8q"

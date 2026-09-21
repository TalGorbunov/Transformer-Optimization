#!/bin/bash
# SPARSE wave 2 — fired when the P1g adapter lands (S1 job 138975).
# Usage: bash scripts/sparse/wave2_submit.sh <p1g_adapter_dir>
set -euo pipefail
P1G="${1:?usage: wave2_submit.sh <adapter dir>}"
test -d "$P1G" || { echo "no adapter at $P1G"; exit 1; }
EX64=outputs/loramech/examdirs/exam_ff_N64.txt
EX128=outputs/loramech/examdirs/exam_ff_N128.txt
# S1 ladder tail (oracle gate, 150/cell)
sbatch -p l40s-shared --qos=4d_1g --time=6:00:00 --job-name=sp_s1_ladder \
  --export=ALL,LIMIT=24,ADAPTER=$P1G,GATE=oracle,DIRS_FILES="$EX64 $EX128",LONGN_LIMIT=150,OUTPUT=outputs/sparse/s1_ladder_64_128 \
  slurm/sparse_train.sbatch
# S4 capacity in k (oracle gate, topped-up strata)
sbatch -p l40s-shared --qos=4d_1g --time=8:00:00 --job-name=sp_s4 \
  --export=ALL,LIMIT=24,ADAPTER=$P1G,GATE=oracle,DIRS_FILES="outputs/sparse/s4/dirs_N64.txt outputs/sparse/s4/dirs_N128.txt",LONGN_LIMIT=300,OUTPUT=outputs/sparse/s4/eval \
  slurm/sparse_train.sbatch
# S3 vs-N chain (gated probe)
sbatch -p l40s-shared --qos=4d_1g --time=5:00:00 --job-name=sp_s3_n \
  --export=ALL,CHAIN=n,ADAPTER=$P1G,GATE=oracle,OUTPUT=outputs/sparse/s3 \
  slurm/sparse_s3_chain.sbatch
# S3 vs-k chain at N=64 (gated probe, redux pool)
sbatch -p l40s-shared --qos=4d_1g --time=5:00:00 --job-name=sp_s3_k \
  --export=ALL,CHAIN=k,ADAPTER=$P1G,GATE=oracle,OUTPUT=outputs/sparse/s3 \
  slurm/sparse_s3_chain.sbatch
# S2 pass-1 captures: S1 train split (gate training data)
RUN_DIR=$(dirname "$P1G")
sbatch -p l40s-shared --qos=4d_1g --time=3:00:00 --job-name=sp_cap_train \
  --export=ALL,DIRS_FILES="$RUN_DIR/train_dirs.txt",LAYER=20,ADAPTER=$P1G,OUTPUT=outputs/sparse/s2/cap_train \
  slurm/sparse_capture.sbatch
# S2 exam captures (one per N, chained in one job via LIMIT-free files)
for N in 8 16 32 64 128; do
  sbatch -p rtx6k-shared --qos=4d_1g --time=3:00:00 --job-name=sp_cap_N$N \
    --export=ALL,DIRS_FILES="outputs/loramech/examdirs/exam_ff_N$N.txt",LAYER=20,ADAPTER=$P1G,OUTPUT=outputs/sparse/s2/cap_N$N \
    slurm/sparse_capture.sbatch
done
squeue -u $USER -h -o "%.9i %.12j %.13P %.8T %.8q"

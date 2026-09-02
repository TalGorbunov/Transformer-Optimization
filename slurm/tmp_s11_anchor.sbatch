#!/bin/bash
#SBATCH --job-name=sp11_anchor
#SBATCH -p l40s-shared
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
run_logged outputs/_scratch/sparse_w3/s11_anchor python -u scripts/sparse/probe_hahn_gated.py \
  --data_root data/mmred_images_park/seq_len_8/all_uniform --limit 3 --controls 1 \
  --seed 0 --arms p1fence --peft-adapter checkpoints/sft_fenced_le16_ep10_adapter \
  --output outputs/_scratch/sparse_w3/s11_anchor
diff -q outputs/_scratch/sparse_w3/s11_anchor/pairs.csv outputs/_scratch/sparse_smoke/probe_anchor/pairs.csv \
  && echo "S11 ANCHOR BYTE-IDENTICAL" || { echo "S11 ANCHOR DIFFERS"; exit 1; }

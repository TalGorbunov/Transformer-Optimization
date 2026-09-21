#!/bin/bash
#SBATCH --job-name=probe_supply
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# A3 supply probe. Knobs: DATA_ROOT LIMIT LAYERS TASK ARMS OUTPUT (ARMS default = deployed A3)
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/carrier/probe}"
ARMS="${ARMS:---question-first --fence-frames --fence-blocks --reset-positions}"
run_logged "$OUTPUT" python -u scripts/probe_supply.py \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-300}" --layers "${LAYERS:-14,16}" --task "${TASK:-steps}" \
  --shuffle-dirs "${SHUFFLE_SEED:-0}" $ARMS --output "$OUTPUT"

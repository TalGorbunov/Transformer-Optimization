#!/bin/bash
#SBATCH --job-name=train_carrier_token
#SBATCH -p l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -o logs/%x-%j.out
# Stage-1 e_c distill. Knobs: DATA_ROOT LIMIT OBJECTIVE EPOCHS OUTPUT EXTRA
# (EXTRA e.g. "--eval-only --carrier-ckpt <pt>" for transfer cells)
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/carrier/token}"
run_logged "$OUTPUT" python -u scripts/train_carrier_token.py \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --limit "${LIMIT:-900}" --train-n "${TRAIN_N:-450}" \
  --objective "${OBJECTIVE:-distill}" --epochs "${EPOCHS:-15}" \
  --shuffle-dirs "${SHUFFLE_SEED:-0}" ${EXTRA:-} --output "$OUTPUT"

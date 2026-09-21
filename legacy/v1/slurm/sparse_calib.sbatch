#!/bin/bash
#SBATCH --job-name=sp_calib
#SBATCH -p a100-public
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# Minimal-fix ladder rung A0: frozen + fence + oracle gate, digit-readout calibration
# (no LoRA). Knobs: ROOTS LIMIT DIRS_FILES EXCLUDES LONGN_LIMIT OUTPUT EXTRA.
# Always submit with explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
OUTPUT="${OUTPUT:-outputs/sparse/a0_calib}"
DF_ARGS=""; for f in ${DIRS_FILES:-}; do DF_ARGS="$DF_ARGS --eval-dirs-file $f"; done
EX_ARGS=""; for f in ${EXCLUDES:-}; do EX_ARGS="$EX_ARGS --exclude-dirs-file $f"; done
run_logged "$OUTPUT" python -u scripts/sparse/calib_digits.py \
  ${ROOTS:+--data_root "$ROOTS"} --limit "${LIMIT:-400}" $DF_ARGS $EX_ARGS ${EXTRA:-} \
  --eval-longn-limit "${LONGN_LIMIT:-150}" --output "$OUTPUT"

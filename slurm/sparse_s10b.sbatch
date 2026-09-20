#!/bin/bash
#SBATCH --job-name=sp_s10b
#SBATCH -p rtx6k-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH -o logs/%x-%j.out
# S10b: attention-photograph fill-in cells (2026-09-19). ARM=p1b|frozen|gated, NLIST=dash-separated
# (e.g. 8-16-64; never commas in --export). Optional TAULIST=dash-separated sharpen factors (P1b only)
# -> outputs/sparse/s10/<ARM>_N<N>_tau<TAU>/. Explicit --time at submit.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
ARM="${ARM:?}"; NLIST="${NLIST:?}"; TAULIST="${TAULIST:-}"; LIMIT="${LIMIT:-30}"; KLIST="${KLIST:-2,4,8}"
for N in ${NLIST//-/ }; do
  if [ -z "$TAULIST" ]; then
    run_logged "outputs/sparse/s10/${ARM}_N${N}" python -u scripts/sparse/probe_attn_photo.py \
      --arm "$ARM" --n "$N" --k-list "$KLIST" --limit "$LIMIT" --output "outputs/sparse/s10/${ARM}_N${N}"
  else
    for TAU in ${TAULIST//-/ }; do
      run_logged "outputs/sparse/s10/${ARM}_N${N}_tau${TAU}" python -u scripts/sparse/probe_attn_photo.py \
        --arm "$ARM" --n "$N" --k-list "$KLIST" --limit "$LIMIT" --attn-sharpen "$TAU" --sharpen-from-layer 12 \
        --output "outputs/sparse/s10/${ARM}_N${N}_tau${TAU}"
    done
  fi
done
echo "S10b ($ARM N=$NLIST TAU=$TAULIST) COMPLETE"

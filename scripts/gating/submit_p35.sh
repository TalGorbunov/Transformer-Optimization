#!/bin/bash
# Submit the P3.5 discriminator grids for every finished P3 arm.
#
# Three grids per arm:
#   grid      the brief's N=2..8 sweep  -- kept ON THE RECORD to document that the
#             teacher-forced COUNT metric saturates at 1.000 there and cannot discriminate
#   highN     N=8,16,32,64,128 -- the regime where the method still loses. seq_len_128 is
#             HELD OUT of the training mixture; 16/32/64 are in it (matched-data only)
#   capacity  N=8, evidence count 0..8 -- the pure capacity axis
#
# tf mode ONLY: eval_gated builds attention masks only when mode != tf, and at N=128 the
# sequence is ~25k tokens, so a dense seq^2 mask is ~1.25 GB per record and ~2.5 GB fp32
# on the GPU. tf mode caches just h at L* (~180 MB/sample).
#
# Usage: bash scripts/gating/submit_p35.sh [grid|highN|capacity ...]
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

WHICH=("$@")
[ ${#WHICH[@]} -eq 0 ] && WHICH=(highN capacity grid)

declare -A ROOTS=(
  [grid]=slurm/lib/roots_gating_grid.txt
  [highN]=slurm/lib/roots_gating_highN.txt
  [capacity]=slurm/lib/roots_gating_capacity.txt
)
declare -A PERGOLD=([grid]=40 [highN]=15 [capacity]=100)
declare -A LIMIT=([grid]=900 [highN]=400 [capacity]=100)

n_sub=0
for arm_dir in outputs/gating/p3_arms/*/; do
    arm=$(basename "$arm_dir")
    ck=$(ls -t "$arm_dir"/*/*/carrier_layer_best.pt 2>/dev/null | head -1)
    if [ -z "$ck" ]; then
        echo "[skip] $arm: no carrier_layer_best.pt yet"
        continue
    fi
    for g in "${WHICH[@]}"; do
        out="outputs/gating/p35_${g}/${arm}"
        echo "[submit] $arm / $g  <- $ck"
        sbatch -J "gate_p35_${g}_${arm}" -p l40s-shared --qos=12h_4g \
            --time=08:00:00 --mem=200G --gres=gpu:1 --cpus-per-task=8 \
            --export=ALL,CKPT="$ck",ROOTS_FILE="${ROOTS[$g]}",PER_GOLD="${PERGOLD[$g]}",LIMIT="${LIMIT[$g]}",MODE=tf,OUTPUT="$out" \
            slurm/gating_eval.sbatch
        n_sub=$((n_sub + 1))
    done
done
echo "submitted $n_sub job(s)"

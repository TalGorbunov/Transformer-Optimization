#!/usr/bin/env bash
# DIAG Tier-0 submission (docs/DIAGNOSTICS_2026-09-22.md §2; outputs/diag/CAMPAIGN_BRIEF.md). Idempotent per wave:
#   bash sbatch/diag_submit_tier0.sh wave1   # D4 train captures (8) + D1 N=8/16/32 + D3 N=32 + D2 N=8/16/32 + faithful grid N=16/32
#   bash sbatch/diag_submit_tier0.sh wave2   # D4 test captures (4) + D1/D2 N=64 + faithful grid N=64/128
#   bash sbatch/diag_submit_tier0.sh h200    # the N=128 fenced/gated cells on h200-shared (D1 N=128, D2 N=128, D3 N=128, D4 N=128 x2)
# Every submission is appended to outputs/diag/jobs.tsv (jobid, wave, cell, cmd). Never a comma inside an --export value.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DRY_RUN
WAVE="${1:?wave1|wave2|h200}"
LOG=outputs/diag/jobs.tsv; touch "$LOG"
sub() {  # sub <cell-name> <sbatch args...>
  local cell="$1"; shift
  local id; id=$(sbatch "$@" | awk '{print $NF}')
  printf '%s\t%s\t%s\t%s\n' "$id" "$WAVE" "$cell" "$*" >> "$LOG"
  echo "submitted $id  $cell"
}
D1Q="steps_in_room char_at_frame n_char_at_frame"
D2Q="char_at_frame steps_in_room n_char_at_frame where_spend"
case "$WAVE" in
wave1)
  for SPLIT in seq_len_8_train seq_len_16_train; do
    for ARM in plain qfirst fenced_qlast fenced_qfirst; do
      sub "D4 $SPLIT $ARM" -p a100-public --qos=4d_1g --time=03:00:00 --export=ALL,SPLIT=$SPLIT,ARM=$ARM sbatch/gate_capture.sbatch
    done
  done
  for N in 8 16 32; do
    sub "D1 N$N" -p a100-public --qos=24h_1g --time=05:00:00 --export=ALL,SPLIT=seq_len_${N}_test,QTYPES="$D1Q",ARMS="plain qfirst fenced gated" sbatch/probe_hahn.sbatch
  done
  sub "D3 N32" -p a100-public --qos=12h_4g --time=08:00:00 --export=ALL,SPLIT=seq_len_32_test,NOBLOCK=1 sbatch/diag_d3.sbatch
  for N in 8 16 32; do
    sub "D2 N$N" -p l40s-shared --qos=12h_4g --time=06:00:00 --export=ALL,SPLIT=seq_len_${N}_test,QTYPES="$D2Q",ARMS="plain qfirst fenced gated",NOBLOCK=1 sbatch/diag_d2.sbatch
  done
  for N in 16 32; do
    sub "grid N$N faithful" -p l40s-shared --qos=24h_1g --time=06:00:00 --export=ALL,SPLIT=seq_len_${N}_test,LAYOUT=paper,OUTPUT=outputs/diag/eval/grid/N$N sbatch/evaluate.sbatch
  done
  ;;
wave2)
  for SPLIT in seq_len_32_test seq_len_64_test; do
    for ARM in fenced_qlast fenced_qfirst; do
      sub "D4 $SPLIT $ARM" -p a100-public --qos=4d_1g --time=04:00:00 --export=ALL,SPLIT=$SPLIT,ARM=$ARM,QTYPES_FILE=sbatch/lib/splits/qtypes_d4_long.txt sbatch/gate_capture.sbatch
    done
  done
  sub "D1 N64" -p a100-public --qos=24h_1g --time=10:00:00 --export=ALL,SPLIT=seq_len_64_test,QTYPES="$D1Q",ARMS="plain qfirst fenced gated" sbatch/probe_hahn.sbatch
  sub "D2 N64" -p l40s-shared --qos=24h_1g --time=10:00:00 --export=ALL,SPLIT=seq_len_64_test,QTYPES="$D2Q",ARMS="plain qfirst fenced gated",NOBLOCK=1 sbatch/diag_d2.sbatch
  for N in 64 128; do
    sub "grid N$N faithful" -p a100-public --qos=4d_1g --time=12:00:00 --export=ALL,SPLIT=seq_len_${N}_test,LAYOUT=paper,OUTPUT=outputs/diag/eval/grid/N$N sbatch/evaluate.sbatch
  done
  ;;
h200)
  sub "D1 N128" -p h200-shared --qos=24h_1g --time=14:00:00 --mem=96G --export=ALL,SPLIT=seq_len_128_test,QTYPES="$D1Q",ARMS="plain qfirst fenced gated" sbatch/probe_hahn.sbatch
  sub "D2 N128" -p h200-shared --qos=24h_1g --time=14:00:00 --mem=96G --export=ALL,SPLIT=seq_len_128_test,QTYPES="$D2Q",ARMS="plain qfirst fenced gated",NOBLOCK=1 sbatch/diag_d2.sbatch
  sub "D3 N128" -p h200-shared --qos=24h_1g --time=16:00:00 --mem=96G --export=ALL,SPLIT=seq_len_128_test,NOBLOCK=1 sbatch/diag_d3.sbatch
  for ARM in fenced_qlast fenced_qfirst; do
    sub "D4 seq_len_128_test $ARM" -p h200-shared --qos=4d_1g --time=08:00:00 --mem=96G --export=ALL,SPLIT=seq_len_128_test,ARM=$ARM,QTYPES_FILE=sbatch/lib/splits/qtypes_d4_long.txt sbatch/gate_capture.sbatch
  done
  ;;
*) echo "unknown wave $WAVE"; exit 2 ;;
esac

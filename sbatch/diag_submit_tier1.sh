#!/usr/bin/env bash
# Tier 1 + comparison table (docs/DIAGNOSTICS_2026-09-22.md §2 Tier 1; Tal 2026-09-23 "run tier 1 ... do the comparison table").
#   bash sbatch/diag_submit_tier1.sh train          # 5 trainers (3 qtypes, seq 8+16 train, 5 epochs)
#   bash sbatch/diag_submit_tier1.sh eval           # after the adapters exist: eval chains N=1..128 per arm (+ tau/logN eval-only arms)
#   bash sbatch/diag_submit_tier1.sh probes         # D1b/D2b (flip + photo with adapters C and D), D3b (tau on C at N=32), D8 headscan
set -euo pipefail; cd "$(dirname "$0")/.."; unset DRY_RUN
LOG=outputs/diag/jobs.tsv; WAVE="tier1-${1:?train|eval|probes}"
sub() { local cell="$1"; shift; local id; id=$(sbatch "$@" | awk '{print $NF}'); printf '%s\t%s\t%s\t%s\n' "$id" "$WAVE" "$cell" "$*" >> "$LOG"; echo "submitted $id  $cell"; }
QT=sbatch/lib/splits/qtypes_tier1.txt
A=outputs/diag/train
case "$1" in
train)
  sub "A plain LoRA, paper layout, no fence"      -p a100-public,l40s-shared --qos=24h_1g --time=22:00:00 --export=ALL,QTYPES_FILE=$QT,NOFENCE=1,LAYOUT=paper,OUTPUT=$A/A_plain_paper sbatch/train.sbatch
  sub "B plain LoRA, question-first, no fence"    -p a100-public,l40s-shared --qos=24h_1g --time=22:00:00 --export=ALL,QTYPES_FILE=$QT,NOFENCE=1,LAYOUT=question-first,OUTPUT=$A/B_plain_qfirst sbatch/train.sbatch
  sub "C fenced-SFT, question-first (THE read)"   -p a100-public,l40s-shared --qos=24h_1g --time=22:00:00 --export=ALL,QTYPES_FILE=$QT,LAYOUT=question-first,OUTPUT=$A/C_fenced sbatch/train.sbatch
  sub "D gated fenced-SFT (oracle gate)"          -p a100-public,l40s-shared --qos=24h_1g --time=22:00:00 --export=ALL,QTYPES_FILE=$QT,LAYOUT=question-first,GATE=oracle,OUTPUT=$A/D_gated sbatch/train.sbatch
  sub "E fenced-SFT + log-N training prior"       -p a100-public --qos=24h_4g --time=22:00:00 --export=ALL,QTYPES_FILE=$QT,LAYOUT=question-first,LOGN=3000,OUTPUT=$A/E_fenced_logn sbatch/train.sbatch
  ;;
eval)
  ad() { ls -d $A/$1/*/adapter | tail -1; }
  sub "eval frozen deployed"        -p a100-public --qos=4d_1g --time=12:00:00 --export=ALL,ARM=frozen_plain,LAYOUT=paper sbatch/diag_tier1_eval.sbatch
  sub "eval frozen question-first"  -p a100-public --qos=4d_1g --time=12:00:00 --export=ALL,ARM=frozen_qfirst,LAYOUT=question-first sbatch/diag_tier1_eval.sbatch
  sub "eval A plain paper"          -p a100-public --qos=4d_1g --time=12:00:00 --export=ALL,ARM=A_plain_paper,ADAPTER=$(ad A_plain_paper) sbatch/diag_tier1_eval.sbatch
  sub "eval B plain qfirst"         -p a100-public --qos=4d_1g --time=12:00:00 --export=ALL,ARM=B_plain_qfirst,ADAPTER=$(ad B_plain_qfirst) sbatch/diag_tier1_eval.sbatch
  sub "eval C fenced (N<=64)"       -p a100-public,l40s-shared --qos=24h_1g --time=12:00:00 --export=ALL,ARM=C_fenced,ADAPTER=$(ad C_fenced),NLIST="1 2 4 8 16 32 64" sbatch/diag_tier1_eval.sbatch
  sub "eval C fenced N=128 (h200)"  -p h200-shared --qos=24h_4g --time=12:00:00 --mem=96G --export=ALL,ARM=C_fenced,ADAPTER=$(ad C_fenced),NLIST="128" sbatch/diag_tier1_eval.sbatch
  sub "eval D gated oracle (N<=64)" -p a100-public,l40s-shared --qos=24h_1g --time=12:00:00 --export=ALL,ARM=D_gated_oracle,ADAPTER=$(ad D_gated),GATE=oracle,NLIST="1 2 4 8 16 32 64" sbatch/diag_tier1_eval.sbatch
  sub "eval D gated oracle N=128"   -p h200-shared --qos=24h_4g --time=12:00:00 --mem=96G --export=ALL,ARM=D_gated_oracle,ADAPTER=$(ad D_gated),GATE=oracle,NLIST="128" sbatch/diag_tier1_eval.sbatch
  sub "eval E fenced+logN prior (N<=64)" -p a100-public,l40s-shared --qos=24h_1g --time=12:00:00 --export=ALL,ARM=E_fenced_logn,ADAPTER=$(ad E_fenced_logn),NLIST="1 2 4 8 16 32 64" sbatch/diag_tier1_eval.sbatch
  sub "eval C + eval-time logN (N<=64)" -p a100-public,l40s-shared --qos=4d_1g --time=12:00:00 --export=ALL,ARM=C_fenced,ADAPTER=$(ad C_fenced),LOGN=3000,NLIST="8 16 32 64" sbatch/diag_tier1_eval.sbatch
  sub "eval C + eval-time tau2 (N<=64)" -p a100-public,l40s-shared --qos=4d_1g --time=12:00:00 --export=ALL,ARM=C_fenced,ADAPTER=$(ad C_fenced),TAU=2,NLIST="8 16 32 64" sbatch/diag_tier1_eval.sbatch
  ;;
probes)
  ad() { ls -d $A/$1/*/adapter | tail -1; }
  D1Q="steps_in_room char_at_frame n_char_at_frame"; D2Q="char_at_frame steps_in_room n_char_at_frame where_spend"
  for N in 8 16 32 64; do
    sub "D1b flips, adapter C, N$N" -p a100-public --qos=4d_1g --time=08:00:00 --export=ALL,SPLIT=seq_len_${N}_test,QTYPES="$D1Q",ARMS="qfirst fenced gated",ADAPTER=$(ad C_fenced),OUTPUT=outputs/diag/hahn/adapterC/N$N sbatch/probe_hahn.sbatch
  done
  sub "D1b flips, adapter C, N128 (h200)" -p h200-shared --qos=24h_4g --time=14:00:00 --mem=96G --export=ALL,SPLIT=seq_len_128_test,QTYPES="$D1Q",ARMS="qfirst fenced gated",ADAPTER=$(ad C_fenced),OUTPUT=outputs/diag/hahn/adapterC/N128 sbatch/probe_hahn.sbatch
  for N in 8 16 32 64; do
    sub "D2b photo, adapter C, N$N" -p a100-public --qos=4d_1g --time=06:00:00 --export=ALL,SPLIT=seq_len_${N}_test,QTYPES="$D2Q",ARMS="qfirst fenced gated",ADAPTER=$(ad C_fenced),OUTPUT=outputs/diag/photo/adapterC sbatch/diag_d2.sbatch
  done
  sub "D2b photo, adapter D (gated-trained), N32" -p a100-public --qos=4d_1g --time=06:00:00 --export=ALL,SPLIT=seq_len_32_test,QTYPES="$D2Q",ARMS="fenced gated",ADAPTER=$(ad D_gated),OUTPUT=outputs/diag/photo/adapterD sbatch/diag_d2.sbatch
  sub "D3b tau/logN on adapter C, N32" -p a100-public --qos=12h_4g --time=08:00:00 --export=ALL,SPLIT=seq_len_32_test,ARM=fenced,ADAPTER=$(ad C_fenced) sbatch/diag_d3.sbatch
  ;;
esac

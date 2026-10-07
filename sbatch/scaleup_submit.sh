#!/usr/bin/env bash
# SCALEUP wave submitter: one dataset x one backbone x stage(s). Records every job in outputs/scaleup/jobs.tsv.
#   [PROTOCOL=planted|isolated] [PER_QTYPE=50] [QTYPES="AGAR AGBI"] [SKIP_UNIFORM=1] bash sbatch/scaleup_submit.sh <dataset> <backbone> <s1|s2|s3|all>
# Job shape (Qwen-7B basis): mask-free chains (N 8..64 in one job, N=128 alone) on 12h_4g / 4d_1g;
# masked N=128 cells (fenced captures, gated eval) on h200-shared. Count FREE slots (preflight) before a wave.
set -euo pipefail
DS="$1"; BB="$2"; ST="$3"; PROTOCOL="${PROTOCOL:-planted}"; PER_QTYPE="${PER_QTYPE:-50}"
LEDGER=outputs/scaleup/jobs.tsv; mkdir -p outputs/scaleup logs
DRY="${DRY_RUN:-0}"
submit() {  # submit <stage> <arm> <nlist> <partition> <qos> <time> <note> [mem]
  local stage="$1" arm="$2" nlist="$3" part="$4" qos="$5" time="$6" note="$7" mem="${8:-48G}" jid
  local cmd=(sbatch -p "$part" --qos="$qos" --time="$time" --mem="$mem" --export="ALL,DATASET=$DS,BACKBONE=$BB,STAGE=$stage,ARM=$arm,NLIST=$nlist,PROTOCOL=$PROTOCOL,PER_QTYPE=$PER_QTYPE${QTYPES:+,QTYPES=$QTYPES}" sbatch/scaleup_cells.sbatch)
  if [ "$DRY" = "1" ]; then echo "[dry] ${cmd[*]}"; return; fi
  jid=$("${cmd[@]}" | grep -o "[0-9]\{6\}$")
  printf "%s\t%s\t%s_%s_N%s\t%s\t%s\t%s\t%s\t%s\trunning\t%s\n" "$jid" "$stage" "$stage" "$arm" "${nlist// /-}" "$DS" "$BB" "$(date '+%Y-%m-%d %H:%M')" "$part" "$qos" "$note" >> "$LEDGER"
  echo "submitted $jid: $stage $arm N=[$nlist] on $part/$qos"
}
MF="a100-public,l40s-shared,l40s-public"     # mask-free cells
H2="h200-shared"                             # masked N=128
if [ "$ST" = "s1" ] || [ "$ST" = "all" ]; then
  [ "${SKIP_UNIFORM:-0}" = "1" ] || PROTOCOL=uniform PER_QTYPE=0 submit s1 plain "16" "$MF" 12h_4g 06:00:00 "official comparability row (uniform N=16, frames-first, ALL rows)"
  for arm in plain qfirst; do
    submit s1 "$arm" "8 16 32 64" "$MF" 12h_4g 08:00:00 "frozen ladder"
    submit s1 "$arm" "128" "$MF" 4d_1g 08:00:00 "frozen ladder N=128 (mask-free)"
  done
fi
if [ "$ST" = "s2" ] || [ "$ST" = "all" ]; then
  for arm in fenced_qfirst qfirst fenced_qlast; do
    submit s2 "$arm" "8 16 32 64" "$MF" 4d_1g 08:00:00 "per-block evidence captures"
    if [ "$arm" = "qfirst" ]; then submit s2 "$arm" "128" "$MF" 4d_1g 06:00:00 "captures N=128 (mask-free)"
    else submit s2 "$arm" "128" "$H2" 24h_4g 08:00:00 "captures N=128 (masked -> h200)" 96G; fi
  done
fi
if [ "$ST" = "s3" ] || [ "$ST" = "all" ]; then
  submit s3 gated "8 16 32 64" "$MF" 4d_1g 10:00:00 "oracle-gate upper bound"
  submit s3 gated "128" "$H2" 24h_4g 08:00:00 "oracle-gate N=128 (masked -> h200)" 96G
fi

#!/usr/bin/env bash
# UNIT baseline wave (plan 2026-10-06), one dataset x one backbone. Records every job in outputs/scaleup/jobs.tsv.
#   [DRY_RUN=1] [QOS_POOL="12h_4g:3 24h_1g:4 24h_4g:3 4d_1g:8"] bash sbatch/unit_submit.sh <herbench|minerva|mmred> <backbone> <smoke|T|E|D|all>
# Unit sets: the W1 sweep (default) or, with UNIT_SET=w23, the reduced set W1 kept for MINERVA / the other backbones:
#   E frame@lo + {frame clip3_d1 clip9_d2}@hi, D frame@lo + {frame clip3_d1}@hi (E_HI_UNITS / D_HI_UNITS override; InternVL's 12-tile
#   hi mode makes clip9_d2 ~140 K tokens -> E_HI_UNITS="frame clip3_d1" there). RES_HI (default hi; "" = the processor defaults).
# Jobs are spread over the QOS pool in order, one slot per job (the per-user caps; run preflight first and pass the
# FREE counts as QOS_POOL when the queue is not empty). Every cell is single-GPU and mask-free.
# HERBench cells (pinned 582 rows, sbatch/lib/splits/qids_herbench_native582.txt):
#   smoke  one job on 2h_2g: T, E (frame + clip9_d2 @hi), D (frame + clip9_d2 @hi) on 1 row per task
#   T      uniform N=8 rows, no frames                                                    (< 0.5 GPU-h)
#   E      frame@lo | frame clip3_d1 clip5_d1 @hi | clip5_d2 clip9_d2 @hi (3 jobs)        (~1-2 GPU-h each)
#   D      frame@lo | {frame clip3_d1} {clip5_d1 clip5_d2} {clip9_d2} @hi (4 jobs)        (~2-3 GPU-h per unit)
# MMReD anchor (seq_len_8 and seq_len_16 test, 50 rows per qtype): smoke = one job (T E D, 1 row/qtype, seq 8); T E D per split.
# The long evidence-only prompts (clip9_d2 @hi: up to 45 frames ~ 55 K tokens) go to the 48 GB L40S / H200 partitions.
set -euo pipefail
DS="$1"; BB="$2"; ST="$3"
LEDGER=outputs/scaleup/jobs.tsv; mkdir -p outputs/scaleup logs
DRY="${DRY_RUN:-0}"
MF="a100-public,l40s-shared,l40s-public"; BIG="l40s-shared,l40s-public,h200-shared"
QIDS=sbatch/lib/splits/qids_herbench_native582.txt; QIDS_MINERVA=sbatch/lib/splits/qids_minerva_50.txt
UNIT_SET="${UNIT_SET:-w1}"; RES_HI="${RES_HI-hi}"
if [ "$UNIT_SET" = "w23" ]; then E_HI_UNITS="${E_HI_UNITS:-frame clip3_d1 clip9_d2}"; D_HI_UNITS="${D_HI_UNITS:-frame clip3_d1}"; fi
read -r -a POOL <<< "${QOS_POOL:-12h_4g:3 24h_1g:4 24h_4g:3 4d_1g:8}"
next_qos() {  # pop one slot from the pool, in order -> $NEXT_QOS (a global: a $(subshell) would not persist the pop — the 2026-10-06 incident)
  local i spec name n
  for i in "${!POOL[@]}"; do
    spec="${POOL[$i]}"; name="${spec%%:*}"; n="${spec##*:}"
    if [ "$n" -gt 0 ]; then POOL[$i]="$name:$((n - 1))"; NEXT_QOS="$name"; return 0; fi
  done
  echo "QOS pool exhausted" >&2; return 1
}
submit() {  # submit <regimes> <units> <res> <extra export> <partition> <qos|auto> <time> <mem> <note>
  local regimes="$1" units="$2" res="$3" extra="$4" part="$5" qos="$6" time="$7" mem="$8" note="$9" jid ex
  if [ "$qos" = "auto" ]; then next_qos || exit 1; qos="$NEXT_QOS"; fi
  ex="ALL,STAGE_DATA=0,DATASET=$DS,BACKBONE=$BB,REGIME=$regimes,UNITS=$units"
  [ -n "$res" ] && ex="$ex,RES=$res"
  [ -n "$extra" ] && ex="$ex,$extra"
  local cmd=(sbatch -p "$part" --qos="$qos" --time="$time" --mem="$mem" --export="$ex" sbatch/unit_baseline.sbatch)
  if [ "$DRY" = "1" ]; then echo "[dry] ${cmd[*]}"; return; fi
  jid=$(DRY_RUN= "${cmd[@]}" | grep -o "[0-9]\{6\}$")
  printf "%s\ts1u\t%s_%s%s\t%s\t%s\t%s\t%s\t%s\trunning\t%s\n" "$jid" "${regimes// /-}" "${units// /-}" "${res:+_$res}" "$DS" "$BB" \
    "$(date '+%Y-%m-%d %H:%M')" "$part" "$qos" "$note" >> "$LEDGER"
  echo "submitted $jid: regimes=[$regimes] units=[$units] res=${res:-default} on $part/$qos ($time)"
}
if [ "$DS" = "herbench" ] || [ "$DS" = "minerva" ]; then
  ROWS="QIDS_FILE=$QIDS"; [ "$DS" = "minerva" ] && ROWS="QIDS_FILE=$QIDS_MINERVA"
  if [ "$ST" = "smoke" ]; then
    submit "T E D" "frame clip9_d2" hi "PER_QTYPE=1,OUT_BASE=outputs/_scratch/unit_smoke/herbench" "$BIG" 2h_2g 01:30:00 48G "UNIT smoke: T, E, D on 1 row per task (frame + clip9_d2 @hi)"
  fi
  if [ "$ST" = "T" ] || [ "$ST" = "all" ]; then
    submit T frame "" "$ROWS" "$MF" auto 01:00:00 32G "regime T: question alone, 582 rows"
  fi
  if [ "$ST" = "E" ] || [ "$ST" = "all" ]; then
    submit E "frame" lo "$ROWS" "$MF" auto 02:00:00 48G "regime E: evidence-only, one frame per unit, lo resolution"
    if [ "$UNIT_SET" = "w23" ]; then
      submit E "$E_HI_UNITS" "$RES_HI" "$ROWS" "$BIG" auto 08:00:00 64G "regime E: evidence-only, reduced unit set, ${RES_HI:-default} resolution"
    else
      submit E "frame clip3_d1 clip5_d1" hi "$ROWS" "$BIG" auto 05:00:00 64G "regime E: evidence-only, hi resolution"
      submit E "clip5_d2 clip9_d2" hi "$ROWS" "$BIG" auto 05:00:00 64G "regime E: evidence-only, hi resolution, 2 s clips (up to 45 frames)"
    fi
  fi
  if [ "$ST" = "D" ] || [ "$ST" = "all" ]; then
    submit D "frame" lo "$ROWS" "$MF" auto 04:00:00 48G "regime D: per-unit yes/no, one frame, lo"
    if [ "$UNIT_SET" = "w23" ]; then
      submit D "$D_HI_UNITS" "$RES_HI" "$ROWS" "$MF" auto 10:00:00 48G "regime D: per-unit yes/no, reduced unit set, ${RES_HI:-default}"
    else
      submit D "frame clip3_d1" hi "$ROWS" "$MF" auto 08:00:00 48G "regime D: per-unit yes/no, hi"
      submit D "clip5_d1 clip5_d2" hi "$ROWS" "$MF" auto 10:00:00 48G "regime D: per-unit yes/no, hi"
      submit D "clip9_d2" hi "$ROWS" "$BIG" auto 08:00:00 48G "regime D: per-unit yes/no, 9-frame clips, hi"
    fi
  fi
elif [ "$DS" = "mmred" ]; then
  if [ "$ST" = "smoke" ]; then
    submit "T E D" frame "" "SPLIT=seq_len_8_test,PER_QTYPE=1,OUT_BASE=outputs/_scratch/unit_smoke/mmred" "$MF" 2h_2g 01:00:00 48G "MMReD UNIT smoke: T, E(frame), D(frame) on 1 row per qtype"
  fi
  for split in seq_len_8_test seq_len_16_test; do
    EX="SPLIT=$split,PER_QTYPE=50"
    if [ "$ST" = "chain" ]; then submit "T E D" frame "" "$EX" "$MF" auto 07:00:00 48G "MMReD anchor T + E(frame) + D(frame) chained, $split"; continue; fi
    if [ "$ST" = "T" ] || [ "$ST" = "all" ]; then submit T frame "" "$EX" "$MF" auto 01:00:00 32G "MMReD anchor T, $split"; fi
    if [ "$ST" = "E" ] || [ "$ST" = "all" ]; then submit E frame "" "$EX" "$MF" auto 02:00:00 48G "MMReD anchor E(frame), $split"; fi
    if [ "$ST" = "D" ] || [ "$ST" = "all" ]; then submit D frame "" "$EX" "$MF" auto 04:00:00 48G "MMReD anchor D(frame), $split"; fi
  done
else
  echo "unknown dataset $DS"; exit 2
fi

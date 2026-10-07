#!/usr/bin/env bash
# SCALEUP wave, corrected protocol (docs/SCALEUP_2026-09-23.md, addendum 2026-09-29): UNIFORM sampling,
# frames at the video's own resolution, the model-side resolution set by the processor's max_pixels,
# fenced arms through the batched path. One dataset x one backbone. Records every job in outputs/scaleup/jobs.tsv.
#   [PER_QTYPE=50] [DRY_RUN=1] bash sbatch/scaleup_submit_native.sh <dataset> <backbone> <s1|s2|s3|all>
# Cells (Qwen token counts for a 1080p frame: uncapped 2,691 · 1 MP cap 1,222 · 151,200 px cap 180):
#   s1  unfenced plain + qfirst: uncapped N=8,16 (N=16 plain on ALL rows = the paper's row); 1 MP N=8,16,32;
#       151,200 px N=8..256. One context must hold N x tokens, so each cap stops where ~45 K tokens are reached.
#   s2  per-block captures, batched: fenced_qfirst + fenced_qlast, uncapped N=16,64; 1 MP N=16,64,256.
#   s3  fenced + oracle gate, batched: uncapped and 1 MP, N=8..256 (the read sees prefix + kept frames only).
# Count FREE slots (preflight) before a wave. Frames are read from /rg (STAGE_DATA=0).
set -euo pipefail
DS="$1"; BB="$2"; ST="$3"; PER_QTYPE="${PER_QTYPE:-50}"
LEDGER=outputs/scaleup/jobs.tsv; mkdir -p outputs/scaleup logs
DRY="${DRY_RUN:-0}"
MF="a100-public,l40s-shared,l40s-public"; H2="h200-shared"; MP1=1003520; MP0=151200
submit() {  # submit <stage> <arm> <nlist> <max_pixels|0> <fast 0|1> <per_qtype> <partition> <qos> <time> <mem> <note>
  local stage="$1" arm="$2" nlist="$3" mp="$4" fast="$5" pq="$6" part="$7" qos="$8" time="$9" mem="${10}" note="${11}" jid ex
  ex="ALL,STAGE_DATA=0,DATASET=$DS,BACKBONE=$BB,STAGE=$stage,ARM=$arm,NLIST=$nlist,PROTOCOL=uniform,PER_QTYPE=$pq,FRAME_SET=native,FAST=$fast,MAX_SEQ=60000"
  [ "$mp" != "0" ] && ex="$ex,MAX_PIXELS=$mp"
  local cmd=(sbatch -p "$part" --qos="$qos" --time="$time" --mem="$mem" --export="$ex" sbatch/scaleup_cells.sbatch)
  if [ "$DRY" = "1" ]; then echo "[dry] ${cmd[*]}"; return; fi
  jid=$(DRY_RUN= "${cmd[@]}" | grep -o "[0-9]\{6\}$")
  printf "%s\t%s\t%s_%s_uniform_native%s_N%s%s\t%s\t%s\t%s\t%s\t%s\trunning\t%s\n" "$jid" "$stage" "$stage" "$arm" \
    "$([ "$mp" != "0" ] && echo "_mp$mp")" "${nlist// /-}" "$([ "$fast" = "1" ] && echo "_fast")" "$DS" "$BB" \
    "$(date '+%Y-%m-%d %H:%M')" "$part" "$qos" "$note" >> "$LEDGER"
  echo "submitted $jid: $stage $arm N=[$nlist] max_pixels=$mp fast=$fast per_qtype=$pq on $part/$qos"
}
if [ "$ST" = "s1" ] || [ "$ST" = "all" ]; then
  submit s1 plain "16" 0 0 0 "$H2" 24h_4g 12:00:00 96G "THE PAPER'S ROW: uniform N=16, frames then question, native resolution, all rows"
  submit s1 plain "8" 0 0 "$PER_QTYPE" "$MF" 12h_4g 06:00:00 64G "unfenced, uncapped"
  submit s1 qfirst "8 16" 0 0 "$PER_QTYPE" "$H2" 24h_4g 12:00:00 96G "unfenced question-first, uncapped"
  for arm in plain qfirst; do
    submit s1 "$arm" "8 16 32" "$MP1" 0 "$PER_QTYPE" "$MF" 24h_1g 12:00:00 64G "unfenced, 1 MP cap"
    submit s1 "$arm" "8 16 32 64 128 256" "$MP0" 0 "$PER_QTYPE" "$MF" 4d_1g 16:00:00 64G "unfenced, 151,200 px cap"
  done
fi
if [ "$ST" = "s2" ] || [ "$ST" = "all" ]; then
  for arm in fenced_qfirst fenced_qlast; do
    submit s2 "$arm" "16 64" 0 1 "$PER_QTYPE" "$MF" 4d_1g 24:00:00 64G "per-block captures, batched, uncapped"
    submit s2 "$arm" "16 64 256" "$MP1" 1 "$PER_QTYPE" "$MF" 4d_1g 36:00:00 64G "per-block captures, batched, 1 MP cap"
  done
fi
if [ "$ST" = "s3" ] || [ "$ST" = "all" ]; then
  submit s3 gated "8 16 32 64 128 256" 0 1 "$PER_QTYPE" "$MF" 12h_4g 10:00:00 64G "oracle gate, batched, uncapped"
  submit s3 gated "8 16 32 64 128 256" "$MP1" 1 "$PER_QTYPE" "$MF" 12h_4g 10:00:00 64G "oracle gate, batched, 1 MP cap"
fi

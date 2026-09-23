#!/usr/bin/env bash
# preflight.sh: the look-before-you-submit snapshot. Free GPUs per node on the partitions we use,
# your queue, and how many jobs you hold per QOS. Run it right before every sbatch.
#   bash .claude/skills/sbatch-submit/scripts/preflight.sh            # default partition list
#   PARTS=a100-public,h200-shared bash .../preflight.sh                # narrower
set -uo pipefail
PARTS="${PARTS:-l40s-shared,l40s-public,a100-public,h200-shared,rtx6k-shared}"
EXCLUDE="${EXCLUDE:-n317}"   # nodes we never use (silent stalls); shown but flagged

printf '## free GPUs by node   %s\n' "$(date '+%F %T')"
printf '%-14s %-12s %-10s %5s %5s %5s  %s\n' PARTITION NODE STATE TOTAL USED FREE NOTE
sinfo -h -N -p "$PARTS" -O "Partition:20,NodeHost:16,StateLong:14,Gres:60,GresUsed:60" \
| while read -r part node state rest; do
    counts=$(grep -oE 'gpu:[A-Za-z0-9_]+:[0-9]+' <<<"$rest" | grep -oE '[0-9]+$' | paste -sd' ')
    tot=${counts%% *}; used=${counts##* }
    [ -z "$counts" ] && { tot=0; used=0; }
    free=$((tot - used))
    note=""
    case " ${EXCLUDE//,/ } " in *" $node "*) note="EXCLUDE (stalls)";; esac
    case "$state" in drain*|down*|fail*) note="${note:+$note; }unusable ($state)"; free=0;; esac
    printf '%-14s %-12s %-10s %5s %5s %5s  %s\n' "${part%\*}" "$node" "$state" "$tot" "$used" "$free" "$note"
  done

echo
echo "## your queue"
squeue -u "$USER" -o "%.9i %20j %13P %7q %9T %10M %10l %R"

echo
echo "## your jobs per QOS vs the per-user job caps (free = cap - running - pending)"
# caps from sacctmgr (2026-09-21); single-GPU jobs are accepted by 24h_4g and 72h_8g too (verified 2026-09-22)
printf '%-8s %4s %4s %4s %5s  %s\n' QOS CAP RUN PEND FREE NOTE
for spec in "2h_2g 3 smokes <2h; 2 GPUs total per user" "12h_4g 3 standard <12h" "24h_1g 4 long 1-GPU <24h" \
            "24h_4g 3 1-GPU jobs accepted, <24h" "4d_1g 8 many parallel 1-GPU slots" "72h_8g 1 1-GPU jobs accepted, <72h" \
            "4h_0g 10 CPU only, mem<=16G, cpu<=8"; do
    set -- $spec; q=$1; cap=$2; shift 2; note="$*"
    run=$(squeue -u "$USER" -h -q "$q" -t R 2>/dev/null | wc -l); pend=$(squeue -u "$USER" -h -q "$q" -t PD 2>/dev/null | wc -l)
    free=$((cap - run - pend)); [ "$free" -lt 0 ] && free=0
    printf '%-8s %4s %4s %4s %5s  %s\n' "$q" "$cap" "$run" "$pend" "$free" "$note"
done
echo "  Rule: a batch never queues more jobs on a QOS than its FREE column; spread 1-GPU jobs over 12h_4g -> 24h_1g -> 24h_4g -> 4d_1g -> 72h_8g,"
echo "  or chain several cells inside one job. A pending job can be moved: scontrol update JobId=<id> QOS=<qos> [TimeLimit=HH:MM:SS] [Partition=<p>]."
echo "  DefaultTime 2h everywhere: pass --time. Pending reason QOSMaxJobsPerUserLimit = cap (move it); Resources/Priority = waiting for hardware."

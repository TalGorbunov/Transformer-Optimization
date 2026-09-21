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
echo "## your jobs per QOS (running + pending) vs caps"
usage=$(squeue -u "$USER" -h -o "%q" | sort | uniq -c | awk '{printf "  %-8s %s\n",$2,$1}')
echo "${usage:-  (none)}"
cat <<'CAPS'
  caps (sacctmgr, 2026-09-21): 4h_0g 10 queued, 0 GPU, mem<=16G, cpu<=8 | 2h_2g 3 jobs, 2 GPU total/user
        12h_4g 3 | 24h_1g 4 | 4d_1g 8 | 24h_4g 3 | 72h_8g 1.   DefaultTime 2h everywhere: pass --time.
CAPS

#!/usr/bin/env bash
# verify_submit.sh <jobid> [expected-substring ...]
# Right after sbatch: wait for the job's log, print the command run_logged actually executed, and
# check every expected substring appears in it. Exists because sbatch --export silently truncates
# values at the first comma (three campaigns ran on partial inputs before anyone looked) and
# because an exported DRY_RUN=1 turns a real submit into a no-op.
#   bash .claude/skills/sbatch-submit/scripts/verify_submit.sh 154999 "--limit 50" "--epochs 5"
# Exit 0 = all checks passed, 1 = a check failed (scancel the job), 2 = still pending, re-run later.
set -uo pipefail
jid="${1:?usage: verify_submit.sh <jobid> [expected-substring ...]}"; shift
TIMEOUT="${TIMEOUT:-300}"     # seconds to wait for the cmd line before giving up (pending jobs)
cd "${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

name=""; state=""; node=""; reason=""; log=""
job_info() {
    local s
    if s=$(scontrol show job "$jid" 2>/dev/null); then
        name=$(grep -oP 'JobName=\K\S+' <<<"$s" | head -1)
        state=$(grep -oP 'JobState=\K\S+' <<<"$s" | head -1)
        reason=$(grep -oP 'Reason=\K\S+' <<<"$s" | head -1)
        node=$(grep -oP ' NodeList=\K\S+' <<<"$s" | head -1)
        log=$(grep -oP 'StdOut=\K\S+' <<<"$s" | head -1)
    else   # finished a while ago: scontrol forgets, sacct remembers
        IFS='|' read -r name state node < <(sacct -j "$jid" -X -n -P -S "$(date -d '-60 days' +%F)" -o JobName,State,NodeList | head -1)
        reason=""; log="logs/${name}-${jid}.out"
    fi
}

job_info
[ -z "$name" ] && { echo "[verify] job $jid unknown to scontrol and sacct"; exit 1; }
echo "[verify] job $jid  name=$name  state=$state${reason:+ ($reason)}  node=${node:-?}  log=$log"

deadline=$((SECONDS + TIMEOUT))
while ! { [ -f "$log" ] && grep -q '\[run_logged\] cmd:' "$log"; }; do
    if [[ "$state" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY) ]]; then break; fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "[verify] no cmd line after ${TIMEOUT}s (state=$state${reason:+, $reason}); re-run once it starts"; exit 2
    fi
    sleep 15; job_info
done

[ -f "$log" ] || { echo "[verify] FAIL: log $log does not exist (state=$state)"; exit 1; }
rc=0
grep -m1 '\[run_logged\] output_root=' "$log" || echo "[verify] note: no output_root line (wrapper without run_logged)"
cmd=$(grep -m1 '\[run_logged\] cmd:' "$log" | sed 's/.*\[run_logged\] cmd: //')
if [ -z "$cmd" ]; then
    echo "[verify] FAIL: no '[run_logged] cmd:' line in $log; first lines:"; head -15 "$log"; exit 1
fi
echo "[verify] cmd: $cmd"
grep -q '\[DRY_RUN\]' "$log" && { echo "[verify] FAIL: DRY_RUN=1 reached the job (exported in your shell?); nothing will run"; rc=1; }
for want in "$@"; do
    if grep -qF -- "$want" <<<"$cmd"; then echo "[verify] ok      $want"
    else echo "[verify] MISSING $want    (comma in --export? see skill)"; rc=1; fi
done
dups=$(grep -oE -- '(^| )--[A-Za-z0-9_-]+' <<<"$cmd" | tr -d ' ' | sort | uniq -d | paste -sd' ')
[ -n "$dups" ] && echo "[verify] warn: repeated flags: $dups (EXTRA collision? fine if the flag is repeatable)"
grep -m1 '\[stage_split\]' "$log" || true
[ "$rc" -eq 0 ] && echo "[verify] PASS" || echo "[verify] FAILED: scancel $jid and fix the submit line"
exit $rc

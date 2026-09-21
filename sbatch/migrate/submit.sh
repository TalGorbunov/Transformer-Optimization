#!/usr/bin/env bash
# Submit copy_tree jobs from a manifest, at most MAXQ (default 10 = the 4h_0g cap) of ours in the queue.
#   bash sbatch/migrate/submit.sh sbatch/migrate/manifest_lab_data.txt        # submits what fits
# Re-run it later to submit the remaining lines; finished lines are skipped via a done-marker file.
set -uo pipefail
manifest="${1:?manifest}"; MAXQ="${MAXQ:-10}"
done_file="${manifest%.txt}.submitted"
touch "$done_file"
cd "$(dirname "$0")/../.."
grep -vE '^\s*#|^\s*$' "$manifest" | while read -r src dst ex; do
    grep -qxF "$src" "$done_file" && continue
    inq=$(squeue -u "$USER" -h -n copy_tree | wc -l)
    if [ "$inq" -ge "$MAXQ" ]; then echo "queue full ($inq) — stop; re-run later"; break; fi
    [ -e "$src" ] || { echo "SKIP missing $src"; continue; }
    jid=$(sbatch --parsable --export=ALL,SRC="$src",DST="$dst",EXCLUDES="$ex" sbatch/migrate/copy_tree.sbatch) || { echo "sbatch failed for $src"; break; }
    echo "$jid  $src -> $dst  [${ex:-}]"
    echo "$src" >> "$done_file"
done

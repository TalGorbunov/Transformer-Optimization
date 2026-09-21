#!/usr/bin/env bash
#SBATCH --job-name=copy_tree
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
# One-off data migration (2026-09-21): copy a tree from /home to /rg with rsync.
# Non-destructive (nothing is removed from the source). Resumable: re-submit on TIMEOUT.
#   sbatch --export=ALL,SRC=data/mmred_hf/json,DST=/rg/shocher_prj/lab_data/mmred/json sbatch/migrate/copy_tree.sbatch
# Optional EXCLUDES="dirs images" (space-separated rsync --exclude patterns, relative to SRC).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${SRC:?SRC required}" "${DST:?DST required}"
ex=()
for p in ${EXCLUDES:-}; do ex+=(--exclude "$p"); done
mkdir -p "$DST"
echo "[copy_tree] $(date)  $SRC -> $DST  excludes: ${EXCLUDES:-none}"
rsync -a --no-inc-recursive --info=stats2 --partial "${ex[@]}" "$SRC/" "$DST/"
rc=$?
echo "[copy_tree] rsync exit $rc  $(date)"
n_dst=$(find "$DST" -type f | wc -l)
if [ -z "${EXCLUDES:-}" ]; then
    n_src=$(find "$SRC" -type f | wc -l)
    echo "[copy_tree] files src=$n_src dst=$n_dst $([ "$n_src" -eq "$n_dst" ] && echo MATCH || echo MISMATCH)"
else
    echo "[copy_tree] files dst=$n_dst (src count skipped: excludes in effect)"
fi
[ "$rc" -eq 0 ] && echo "[copy_tree] OK (rsync clean)" || echo "[copy_tree] INCOMPLETE — resubmit the same SRC/DST"

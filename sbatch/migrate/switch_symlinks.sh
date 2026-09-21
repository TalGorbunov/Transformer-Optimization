#!/usr/bin/env bash
# After copy_tree jobs report OK: verify each SRC/DST pair (file count + a sampled checksum), then
# replace SRC by a symlink to DST. The original is renamed to <SRC>.moved-YYYYMMDD first (NOT deleted);
# delete those yourself once you are happy (they are listed at the end).
#   bash sbatch/migrate/switch_symlinks.sh sbatch/migrate/manifest_lab_data.txt [--dry-run]
set -uo pipefail
manifest="${1:?manifest}"; dry="${2:-}"
cd "$(dirname "$0")/../.."
stamp=$(date +%Y%m%d); moved=()
grep -vE '^\s*#|^\s*$' "$manifest" | while read -r src dst ex; do
    [ -n "$ex" ] && { echo "SKIP $src (has excludes; handled by its sub-trees)"; continue; }
    [ -L "$src" ] && { echo "already a symlink: $src"; continue; }
    [ -d "$src" ] && [ -d "$dst" ] || { echo "MISSING side: $src / $dst"; continue; }
    # a parent of $src may already be a symlink to /rg (e.g. data/mmred_hf): then src IS dst — never switch
    [ "$(realpath "$src")" = "$(realpath "$dst")" ] && { echo "same tree (parent already switched): $src"; continue; }
    n_src=$(find "$src" -type f | wc -l); n_dst=$(find "$dst" -type f | wc -l)
    if [ "$n_src" -ne "$n_dst" ]; then echo "COUNT MISMATCH $src ($n_src) vs $dst ($n_dst) — not switched"; continue; fi
    # sampled checksum: 20 files spread through the tree
    bad=0
    for f in $(find "$src" -type f | awk 'NR%997==1' | head -20); do
        rel="${f#$src/}"
        [ "$(md5sum < "$f")" = "$(md5sum < "$dst/$rel")" ] || { bad=1; echo "CHECKSUM DIFF $rel"; }
    done
    [ "$bad" -eq 0 ] || { echo "not switched: $src"; continue; }
    echo "OK $src ($n_src files) -> symlink to $dst"
    if [ "$dry" != "--dry-run" ]; then
        mv "$src" "$src.moved-$stamp" && ln -s "$dst" "$src" && echo "   moved original to $src.moved-$stamp"
    fi
done
echo "When satisfied, remove the *.moved-$stamp originals yourself:  ls -d data/*.moved-* outputs*/*.moved-*"

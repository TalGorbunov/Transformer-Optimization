#!/usr/bin/env bash
#SBATCH --job-name=tar_split
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
# Write the per-split tarball that sbatch/lib/common.sh:stage_split copies to the node-local NVMe:
#   <root>/images/<split>.tar  (uncompressed: PNGs do not compress; tar exists to move 1 file, not 40 K)
#   sbatch --export=ALL,SPLIT=seq_len_8_train sbatch/migrate/tar_split.sbatch
# Idempotent: skips if the tarball exists and its member count equals the tree's file count.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${SPLIT:?SPLIT required}"
ROOT="${ROOT:-/rg/shocher_prj/lab_data/mmred}"
tar_path="$ROOT/images/$SPLIT.tar"
n_tree=$(find "$ROOT/images/$SPLIT" -type f | wc -l)
if [ -f "$tar_path" ] && [ "$(tar tf "$tar_path" | grep -vc '/$')" -eq "$n_tree" ]; then
    echo "[tar_split] $tar_path already complete ($n_tree files)"; exit 0
fi
echo "[tar_split] $(date) $SPLIT: $n_tree files -> $tar_path"
tar cf "$tar_path.partial" -C "$ROOT/images" "$SPLIT" && mv "$tar_path.partial" "$tar_path"
n_tar=$(tar tf "$tar_path" | grep -vc '/$')
echo "[tar_split] members=$n_tar tree=$n_tree $([ "$n_tar" -eq "$n_tree" ] && echo OK || echo MISMATCH) size=$(du -h "$tar_path" | cut -f1) $(date)"

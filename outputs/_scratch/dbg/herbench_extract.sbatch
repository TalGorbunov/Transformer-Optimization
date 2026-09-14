#!/bin/bash
#SBATCH -J hb_extract
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
echo "[$(date +%T)] extracting HD_EPIC members from the concatenated parts with --ignore-zeros (handles standalone-tar parts)"
cat data/herbench_raw/videos/videos.tar.part.?? | tar -x -i -C data/herbench_raw --wildcards '*HD_EPIC*' 2> logs/hb_tar_err-$SLURM_JOB_ID.txt; echo "tar exit $? ; stderr lines $(wc -l < logs/hb_tar_err-$SLURM_JOB_ID.txt)"; head -3 logs/hb_tar_err-$SLURM_JOB_ID.txt
echo "mp4 count $(find data/herbench_raw/HD_EPIC -name '*.mp4' | wc -l)"
[ -e data/herbench_raw/videos/HD_EPIC ] || ln -s "$(realpath data/herbench_raw/HD_EPIC)" data/herbench_raw/videos/HD_EPIC
missing=0; while read v; do [ -f "data/herbench_raw/$v" ] || missing=$((missing+1)); done < data/herbench_ac/ac_videos.txt; echo "AC videos missing: $missing / $(wc -l < data/herbench_ac/ac_videos.txt)"
if [ "$missing" = 0 ]; then echo "[$(date +%T)] removing tar parts"; rm -f data/herbench_raw/videos/videos.tar.part.*; fi
du -sh data/herbench_raw; echo "[$(date +%T)] DONE"

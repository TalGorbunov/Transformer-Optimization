#!/bin/bash
#SBATCH -J hb_download
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
unset HF_HUB_OFFLINE; export PYTHONNOUSERSITE=1
mkdir -p data/herbench_raw/videos
echo "[$(date +%T)] downloading tar parts"; .venv/bin/huggingface-cli download DanBenAmi/HERBench --repo-type dataset --include "videos/videos.tar.part.*" --local-dir data/herbench_raw --max-workers 8 || { echo "download failed"; exit 1; }
ls -la data/herbench_raw/videos/ | head -30; du -sh data/herbench_raw/videos
echo "[$(date +%T)] listing first members"; cat data/herbench_raw/videos/videos.tar.part.00 | tar -t 2>/dev/null | head -5
echo "[$(date +%T)] extracting HD_EPIC members"; cat data/herbench_raw/videos/videos.tar.part.?? | tar -x -C data/herbench_raw --wildcards '*HD_EPIC*' 2> logs/hb_tar_err-$SLURM_JOB_ID.txt; echo "tar exit $? ; stderr lines $(wc -l < logs/hb_tar_err-$SLURM_JOB_ID.txt)"
find data/herbench_raw -name "*.mp4" | head -3; echo "mp4 count $(find data/herbench_raw -name '*.mp4' | wc -l)"
# normalise location to data/herbench_raw/videos/HD_EPIC/<P>/<file>.mp4 (what video_path expects)
if [ ! -d data/herbench_raw/videos/HD_EPIC ]; then d=$(find data/herbench_raw -type d -name HD_EPIC | head -1); [ -n "$d" ] && ln -s "$(realpath $d)" data/herbench_raw/videos/HD_EPIC && echo "linked $d"; fi
missing=0; while read v; do [ -f "data/herbench_raw/$v" ] || { missing=$((missing+1)); }; done < data/herbench_ac/ac_videos.txt; echo "AC videos missing: $missing / $(wc -l < data/herbench_ac/ac_videos.txt)"
if [ "$missing" = 0 ]; then echo "[$(date +%T)] removing tar parts"; rm -f data/herbench_raw/videos/videos.tar.part.*; fi
du -sh data/herbench_raw; echo "[$(date +%T)] DONE"

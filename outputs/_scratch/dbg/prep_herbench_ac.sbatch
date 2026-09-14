#!/bin/bash
#SBATCH -J hb_prep
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH --array=0-33
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1
.venv/bin/python outputs/_scratch/dbg/prep_herbench_ac.py --video-index "$SLURM_ARRAY_TASK_ID" --n-videos 34 --per-video "${PER_VIDEO:-6}" --ns "${NS:-16+32}"

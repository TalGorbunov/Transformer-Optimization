#!/bin/bash
#SBATCH --job-name=rv_probe
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=n314

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1

RUN_DIRS="${RUN_DIRS:?set RUN_DIRS as space-separated run dirs}"
LABELS="${LABELS:-}"
OUTPUT="${OUTPUT:-outputs/probe_rooms_visited_messages}"

if [[ -n "${LABELS}" ]]; then
  python -u evaluations/scripts/probe_rooms_visited_messages.py --run-dirs ${RUN_DIRS} --labels ${LABELS} --output "${OUTPUT}"
else
  python -u evaluations/scripts/probe_rooms_visited_messages.py --run-dirs ${RUN_DIRS} --output "${OUTPUT}"
fi

#!/bin/bash
#SBATCH --job-name=gate_tally
#SBATCH -p l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -o logs/%x-%j.out
# CPU gate->tally on a messages/carrier-states cache. Knobs: CACHE LAYER OUTPUT EXTRA
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
run_logged "${OUTPUT:?set OUTPUT}" python -u scripts/gate_tally.py \
  --cache "${CACHE:?set CACHE}" --layer "${LAYER:-16}" ${EXTRA:-} --output "$OUTPUT"

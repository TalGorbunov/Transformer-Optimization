#!/bin/bash
#SBATCH --job-name=posreset_tally
#SBATCH --partition=a100-public
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# gate->tally on every posreset-necessity sweep cache: does the d' gap cost accuracy?
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
ROOT=outputs/ladder/image_longN
for name in ${NAMES:-noreset_N2 posreset_qf_N2 noreset_N4 posreset_qf_N4 noreset_N16 posreset_qf_N16}; do
  d=$(ls -d $ROOT/$name/*/ 2>/dev/null | tail -1)
  c="${d}messages_cache.pt"
  if [[ ! -f "$c" ]]; then echo "### SKIP $name (no cache)"; continue; fi
  echo "### === $name  ($d)"
  python -u experiments/glstm/replica_gate_tally.py \
    --cache "$c" --layer 16 --output "${d}gate_tally" 2>&1 | sed "s/^/[$name] /"
done
echo "### DONE"

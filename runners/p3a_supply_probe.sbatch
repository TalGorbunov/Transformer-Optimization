#!/bin/bash
#SBATCH --job-name=nat_supply
#SBATCH --partition=a100-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# P3a L1 (p0p2, PREREG plans/p3a_natural_PREREG.md): one-forward supply on natural v2 —
# replicas + blockfence + posreset + Q-first, n=300 (v1 ref was n=50). Writes
# messages_cache.pt for the L2 CPU gate->tally. 24h_1g because the cache writes at END
# (walltime kill loses everything; ~2.5h expected, 6h wall for margin).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/glstm/replica_carrier_probe.py \
  --natural --fence-frames --fence-blocks --reset-positions --question-first \
  --shuffle-dirs 0 \
  --data_root data/mmred_natural_v2/dist_far \
  --limit 300 \
  --output outputs/ladder/natural_mm/replica_supply_dist_far

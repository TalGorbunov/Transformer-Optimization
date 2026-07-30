#!/bin/bash
#SBATCH --job-name=gate_tally
#SBATCH --partition=a100-public
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# CPU analysis of the E1a full-prior probe cache: gate->tally + matched-n d' check.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
CACHE=outputs/ladder/image_longN/replica_blockfence_qfirst_full900/20260718_130546/messages_cache.pt

python -u experiments/glstm/replica_gate_tally.py \
  --cache "$CACHE" --layer 16 \
  --output outputs/ladder/image_longN/replica_blockfence_qfirst_full900/20260718_130546/gate_tally

python - "$CACHE" << 'EOF'
import sys, numpy as np, torch
sys.path.insert(0, ".")
from experiments.glstm.dprime_vs_n import dprime_pair
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
X = np.asarray(c["rep"][16], dtype=np.float32); Y = np.asarray(c["labels"], int)
for seed in range(3):
    idx = np.random.default_rng(seed).permutation(X.shape[0])[:300]
    d, s, a = dprime_pair(X[idx], Y[idx])
    print(f"[matched-n] subsample n=300 seed {seed}: d' {d:.2f}+-{s:.2f} (auc {a:.2f})")
EOF

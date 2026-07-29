#!/bin/bash
#SBATCH --job-name=car_tally
#SBATCH --partition=a100-public
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
python - << 'EOF'
import numpy as np
from sklearn.linear_model import LogisticRegression
z = np.load("outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/messages_best.npz")
X, Y = z["X"].astype(np.float32), z["Y"].astype(int)
G = Y.sum(1)
n, NF, H = X.shape
hist = {int(g): int((G == g).sum()) for g in np.unique(G)}
print("[gold-hist]", sorted(hist.items()))
accs, ferrs = [], []
for seed in range(5):
    idx = np.random.default_rng(seed).permutation(n)
    tr, ev = idx[:n // 2], idx[n // 2:]
    clf = LogisticRegression(max_iter=2000).fit(X[tr].reshape(-1, H), Y[tr].reshape(-1))
    pr = clf.predict(X[ev].reshape(-1, H)).reshape(len(ev), NF)
    ferrs.append((pr != Y[ev]).mean()); accs.append((pr.sum(1) == G[ev]).mean())
print(f"CARRIER STACK full-prior gate->tally: per-frame err {np.mean(ferrs):.4f}, "
      f"exact {np.mean(accs):.3f}+-{np.std(accs):.3f}")
EOF

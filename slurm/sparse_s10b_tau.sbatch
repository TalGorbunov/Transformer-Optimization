#!/bin/bash
#SBATCH --job-name=sp_s10b_tau
#SBATCH -p h200-shared
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH -o logs/%x-%j.out
# S10b temperature sweep with a SAME-HARDWARE anchor: tau-off vs tau=1.0 smokes must agree to 1e-4
# on every mass.csv row before the full cells run. NLIST/TAULIST dash-separated. Explicit --time.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
NLIST="${NLIST:-32-128}"; TAULIST="${TAULIST:-1.5-2-3-4}"; LIMIT="${LIMIT:-30}"
SM=outputs/_scratch/sparse_w3
run_logged $SM/s10b_anchor_off python -u scripts/sparse/probe_attn_photo.py --arm p1b --n 8 --k-list 2,4 --limit 3 --output $SM/s10b_anchor_off
run_logged $SM/s10b_anchor_tau1 python -u scripts/sparse/probe_attn_photo.py --arm p1b --n 8 --k-list 2,4 --limit 3 --attn-sharpen 1.0 --sharpen-from-layer 12 --output $SM/s10b_anchor_tau1
python - <<'PY'
import csv,sys
def load(p): return {(r['sample'],r['layer'],r['head']):(float(r['evid_mass']),float(r['nonevid_mass'])) for r in csv.DictReader(open(p))}
a=load("outputs/_scratch/sparse_w3/s10b_anchor_off/mass.csv"); b=load("outputs/_scratch/sparse_w3/s10b_anchor_tau1/mass.csv")
common=[k for k in a if k in b]; d=max(abs(a[k][0]-b[k][0])+abs(a[k][1]-b[k][1]) for k in common)
print(f"[anchor] same-GPU tau-off vs tau=1.0: {len(common)} rows, max|diff|={d:.2e}")
sys.exit(0 if (len(common)>=800 and d<1e-4) else 1)
PY
echo "[anchor] PASS"
for N in ${NLIST//-/ }; do for TAU in ${TAULIST//-/ }; do
  run_logged "outputs/sparse/s10/p1b_N${N}_tau${TAU}" python -u scripts/sparse/probe_attn_photo.py \
    --arm p1b --n "$N" --k-list 2,4,8 --limit "$LIMIT" --attn-sharpen "$TAU" --sharpen-from-layer 12 \
    --output "outputs/sparse/s10/p1b_N${N}_tau${TAU}"
done; done
echo "S10b TAU SWEEP COMPLETE"

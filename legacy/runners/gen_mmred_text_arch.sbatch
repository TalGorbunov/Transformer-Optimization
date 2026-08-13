#!/bin/bash
#SBATCH --job-name=gen_text_arch
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
echo "=== node storage check ==="; hostname; df -h /rg /scratch /tmp 2>&1; ls -ld /rg/shocher_prj/tal.gorbunov 2>&1; echo "=========================="
# long-N convention: low band 0-8 + spread points <= N; per-count sized for ~150 samples/N
python -u datasets/mmred/generate_mmred_balanced.py --task steps_in_room \
  --seq-len 64 --per-count 10 --n-chars 5 --no-render --seed 7 \
  --counts "0,1,2,3,4,5,6,7,8,16,24,32,40,48,56,64" \
  --out-root data/mmred_text_arch
python -u datasets/mmred/generate_mmred_balanced.py --task steps_in_room \
  --seq-len 128 --per-count 9 --n-chars 5 --no-render --seed 7 \
  --counts "0,1,2,3,4,5,6,7,8,16,32,48,64,80,96,112,128" \
  --out-root data/mmred_text_arch
echo DONE

#!/bin/bash
#SBATCH --job-name=deploy_fix
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# Deploy Garcia-style digit-row repair on the real model: constrained + free-generation accuracy.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LIMIT="${LIMIT:-200}"; GEN="${GEN:-30}"
RUN="${OUT_ROOT:-$PWD/outputs/frame_axis/probes/deploy_readout_fix}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"; echo "OUT=$RUN LIMIT=$LIMIT GEN=$GEN DATA=${DATA:-steps_5char}"
DATA="${DATA:-$PWD/data/mmred_steps_balanced/seq_len_8/all_uniform}"
python -u evaluations/scripts/patch_importence/deploy_readout_fix.py \
  --data_root "$DATA" --tag "${TAG:-steps_5char}" --limit "$LIMIT" --gen-check "$GEN" --output "$RUN" 2>&1 | tee "$RUN/deploy.log"
cp "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$RUN/" 2>/dev/null || true
echo "DONE -> $RUN"

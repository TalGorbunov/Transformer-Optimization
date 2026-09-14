#!/bin/bash
# Fully training-free select-and-repack on MMRED vision: one job = one MODEL, loops NS (frames).
# LAYER=auto (default): --scan on seq_len_${SCAN_N:-32} first (recall@${SCAN_K:-8}, the gold bound), pick the shallowest layer with the
# best frame recall@k, then run the arms at every N. Lists use '+' (sbatch --export splits commas).
#   sbatch --export=ALL,MODEL=,NS=8+16+32+64 slurm/vlm_frozen.sbatch                 # 7B (repo default)
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-VL-3B-Instruct,NS=8+16+32+64 slurm/vlm_frozen.sbatch
#SBATCH -J vlm_frozen
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
MARG=(); [ -n "${MODEL:-}" ] && MARG=(--model "$MODEL")
OUT="${OUTPUT:-outputs/mmred_vfrozen}"
LAYER="${LAYER:-auto}"
if [ "$LAYER" = auto ]; then
  $PY scripts/condmask/vlm_frozen.py ${MARG[@]+"${MARG[@]}"} --scan --root "data/mmred_vfiltered/seq_len_${SCAN_N:-32}/test" \
    --limit "${SCAN_LIMIT:-60}" --k "${SCAN_K:-8}" --output "$OUT" | tee logs/vlm_frozen_scan-$SLURM_JOB_ID.txt
  SCAN=$(grep -- '^-> ' logs/vlm_frozen_scan-$SLURM_JOB_ID.txt | tail -1 | cut -d' ' -f2)
  LAYER=$($PY -c "import json,sys; r=json.load(open(sys.argv[1]))['recall_at_k']; print(r.index(max(r)))" "$SCAN/scan.json")
  echo "scan -> $SCAN ; picked LAYER=$LAYER"
fi
for N in ${NS//+/ }; do
  $PY scripts/condmask/vlm_frozen.py ${MARG[@]+"${MARG[@]}"} --root "data/mmred_vfiltered/seq_len_$N/test" \
    --limit "${LIMIT:-200}" --k "${K:-16}" --sel-layer "$LAYER" \
    --arms "${ARMS:-full+oracle+judge+attn+attn_norp+rand}" --output "$OUT" ${EXTRA:-}
done

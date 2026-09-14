#!/bin/bash
# BABILong training-free select-and-repack eval: one job = one MODEL, loops over TASKS x
# LENGTHS. Lists use '+' separators (sbatch --export splits commas).
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-3B-Instruct,TASKS=qa1+qa2,LENGTHS=0k+4k+32k,LAYER=20 slurm/babilong.sbatch
# EXTRA passes raw flags (e.g. "--sel-chunk 32768 --sel-tournament 64 --probe").
#SBATCH -J babilong
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${MMRED_NS:-}" ]; then   # MMRED text, same frozen protocol: MMRED_NS=8+16+...+1024 (frames), K default 16
for N in ${MMRED_NS//+/ }; do
  .venv/bin/python scripts/condmask/eval_babilong.py \
    --model "${MODEL:?}" --mmred "data/mmred_filtered/seq_len_$N/${SPLIT:-test}" --limit "${LIMIT:-200}" \
    --k "${K:-16}" --sel-layer "${LAYER:?}" --arms "${ARMS:-full+oracle+attn+attn_norp+lex+emb+rand}" \
    --max-new 4 --output "${OUTPUT:-outputs/mmred_frozen}" ${EXTRA:-}
done
exit 0
fi
TASKS="${TASKS:-${TASK:?}}"
for TASK in ${TASKS//+/ }; do
for LEN in ${LENGTHS//+/ }; do
  .venv/bin/python scripts/condmask/eval_babilong.py \
    --model "${MODEL:?}" --task "$TASK" --length "$LEN" --limit "${LIMIT:-200}" \
    --k "${K:-32}" --sel-layer "${LAYER:?}" --arms "${ARMS:-full+oracle+attn+attn_norp+lex+emb+rand}" \
    --output "${OUTPUT:-outputs/babilong}" ${EXTRA:-}
done
done

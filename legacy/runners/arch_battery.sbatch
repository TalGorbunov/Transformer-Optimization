#!/bin/bash
#SBATCH --job-name=arch_batt
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
# NEW venv (py3.11 on /rg) -- NEVER the shared .venv
source /rg/shocher_prj/tal.gorbunov/venv_arch/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/rg/shocher_prj/tal.gorbunov/hf_arch
: "${MODEL:?set MODEL}"; : "${ARCH:?set ARCH}"; : "${OUT:?set OUT}"
mkdir -p "$OUT"
EXTRA=()
[[ "${FOURBIT:-0}" == "1" ]] && EXTRA+=(--load-in-4bit)
[[ -n "${LIMIT_N:-}" ]] && EXTRA+=(--limit-n "${LIMIT_N}")
[[ -n "${SYSTEM:-}" ]] && EXTRA+=(--system "${SYSTEM}")
echo "node: $(hostname)  model: $MODEL  arch: $ARCH  4bit: ${FOURBIT:-0}"
python -u experiments/arch_battery/run_text_battery.py \
  --model-name "$MODEL" --arch-class "$ARCH" --output "$OUT" "${EXTRA[@]}"

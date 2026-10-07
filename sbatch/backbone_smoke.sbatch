#!/usr/bin/env bash
#SBATCH --job-name=backbone_smoke
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
# experiments/backbone_smoke.py: load variants of one backbone on a short MMReD prompt (finite logits? what is generated?).
# Knobs (env; space-separated): BACKBONE  VARIANTS ("spec bf16 skipvis full")  ROWS (3)  OUTPUT
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
BACKBONE="${BACKBONE:-gemma-3-12b}"
read -r -a V <<< "${VARIANTS:-spec bf16 skipvis full}"
OUTPUT="${OUTPUT:-outputs/scaleup/_backbones/smoke}"
run_logged "$OUTPUT" python -u experiments/backbone_smoke.py --backbone "$BACKBONE" --variants "${V[@]}" --rows "${ROWS:-3}" --output "$OUTPUT"

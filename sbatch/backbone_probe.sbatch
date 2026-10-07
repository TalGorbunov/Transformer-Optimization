#!/usr/bin/env bash
#SBATCH --job-name=backbone_probe
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
# experiments/backbone_probe.py: token layout of candidate backbones (config + processor only, offline).
# Knobs (env; space-separated): MODEL_IDS  SET_IP ("k=v ..." image-processor attrs, applied as a SECOND probe per model)  OUT
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
read -r -a MODELS <<< "${MODEL_IDS:-OpenGVLab/InternVL3_5-8B-HF google/gemma-3-12b-it}"
OUT="${OUT:-outputs/scaleup/_backbones}"
for M in "${MODELS[@]}"; do
  run_logged "$OUT" python -u experiments/backbone_probe.py --model-id "$M" --out "$OUT"
  if [ -n "${SET_IP:-}" ]; then read -r -a IP <<< "$SET_IP"; run_logged "$OUT" python -u experiments/backbone_probe.py --model-id "$M" --set-image-processor "${IP[@]}" --out "$OUT"; fi
done

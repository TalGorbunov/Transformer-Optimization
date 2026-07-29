#!/bin/bash
#SBATCH --job-name=p12_ceiling
#SBATCH --partition=a100-public
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# P1.2 (p0p2 campaign): MEASURED best-linear-readout ceiling per N on the existing joint
# image caches (deployed carrier locus L16/off9) — replaces the law-predicted curve in
# outputs/_scratch/figs/pre_stage1_squashed_readout.png. Reuses probe_dprime_parity.py
# (ridge/logit-on-sum, 60/40 sample-disjoint split, seeds 0,1,2).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1

J=outputs/ladder/image_longN/joint
python -u experiments/glstm/probe_dprime_parity.py \
  --carrier-caches \
    joint_N8:16:9:$J/N8/20260710_215405/count/messages_cache.pt \
    joint_N16:16:9:$J/N16/20260710_215406/count/messages_cache.pt \
    joint_N32:16:9:$J/N32/20260710_215405/count/messages_cache.pt \
    joint_N64:16:9:$J/N64/20260710_215405/count/messages_cache.pt \
    joint_N128:16:9:$J/N128/20260710_215411/count/messages_cache.pt \
  --seeds 0,1,2 \
  --output outputs/ladder/image_longN/measured_ceiling

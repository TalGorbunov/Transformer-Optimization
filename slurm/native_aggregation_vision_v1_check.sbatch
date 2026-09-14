#!/bin/bash
#SBATCH -J native_agg_v1_check
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
.venv/bin/python tests/test_native_aggregation.py
.venv/bin/python tests/test_fencing.py
.venv/bin/python tests/test_carrier_masks.py
.venv/bin/python tests/test_scratchpad.py
.venv/bin/python tests/test_data.py
.venv/bin/python -m py_compile gnnformer/native_aggregation.py scripts/native_aggregation.py scripts/native_aggregation_vlm.py scripts/stage_native_vision_pilot.py scripts/report_native_vision_v1.py

#!/bin/bash
#SBATCH -J v8_loss_check
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:03:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
V8_CHECK_OUT="outputs/native_aggregation_vlm/v8/loss_check_${SLURM_JOB_ID}"
mkdir -p "$V8_CHECK_OUT/code"
cp gnnformer/aggregation_consistency.py "$V8_CHECK_OUT/code/"
cp tests/test_aggregation_consistency.py "$V8_CHECK_OUT/code/"
cp slurm/native_vision_v8_loss_check.sbatch "$V8_CHECK_OUT/code/"
sha256sum gnnformer/aggregation_consistency.py tests/test_aggregation_consistency.py slurm/native_vision_v8_loss_check.sbatch > "$V8_CHECK_OUT/source_hashes.txt"
.venv/bin/python tests/test_aggregation_consistency.py > "$V8_CHECK_OUT/tests.txt" 2>&1
printf 'passed\n' > "$V8_CHECK_OUT/PASSED"
printf '# V8 loss checks\n\n[Test output](tests.txt) · [Sources](source_hashes.txt)\n' > "$V8_CHECK_OUT/INDEX.md"
cat "$V8_CHECK_OUT/tests.txt"

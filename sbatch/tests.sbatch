#!/usr/bin/env bash
#SBATCH --job-name=tests
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
# Every CPU test file (tests/test_*.py), one after another; the last line says how many passed.
# The login node kills python processes that load several processors (exit 137, 2026-09-24): run the suite here.
# Knobs: TESTS (space-separated file stems, default all)
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
if [ -n "${TESTS:-}" ]; then read -r -a T <<< "$TESTS"; else T=(); for f in tests/test_*.py; do T+=("$(basename "$f" .py)"); done; fi
ok=0; bad=()
for t in "${T[@]}"; do
  echo "=== $t"
  python -u "tests/$t.py" 2>&1 | grep -v "NVML\|warnings.warn" > "${TMPDIR:-/tmp}/test_$t.log"
  tail -40 "${TMPDIR:-/tmp}/test_$t.log"
  if grep -q "^ALL OK\|^ok$" "${TMPDIR:-/tmp}/test_$t.log"; then ok=$((ok+1)); else bad+=("$t"); fi
done
echo "TESTS PASSED $ok/${#T[@]}  FAILED: ${bad[*]:-none}"

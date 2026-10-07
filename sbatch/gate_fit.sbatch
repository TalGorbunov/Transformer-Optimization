#!/usr/bin/env bash
#SBATCH --job-name=gate_fit
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
# experiments/gate_fit.py on a CPU queue (the fit is minutes of sklearn on ~20k x 3.5k features; not for the login node).
# Knobs (env; lists space-separated, globs allowed; no commas): RUNS  or  TRAIN_RUNS + TEST_RUNS ; GROUP_BY (group|qid)
#   FOLDS (group K-fold)  TRAIN_N ("8 16": train on these N only, test on all)  PER_QTYPE_FIT (1 = also one classifier per task)  VAL_FRAC  OUT (json path, required)  EXTRA
# Submit:
#   sbatch --export=ALL,RUNS="outputs/scaleup/herbench/qwen2.5-vl-7b/s2_gate/planted_N*/fenced_qfirst/2*",OUT=outputs/scaleup/herbench/qwen2.5-vl-7b/s2_fit/heldout.json sbatch/gate_fit.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
: "${OUT:?OUT=<json path> is required}"
ARGS=()
if [ -n "${RUNS:-}" ]; then read -r -a R <<< "$RUNS"; ARGS+=(--runs "${R[@]}"); fi
if [ -n "${TRAIN_RUNS:-}" ]; then read -r -a TR <<< "$TRAIN_RUNS"; ARGS+=(--train-runs "${TR[@]}"); fi
if [ -n "${TEST_RUNS:-}" ]; then read -r -a TE <<< "$TEST_RUNS"; ARGS+=(--test-runs "${TE[@]}"); fi
[ "${PER_QTYPE_FIT:-0}" = "1" ] && ARGS+=(--per-qtype)
[ -n "${FOLDS:-}" ] && ARGS+=(--folds "$FOLDS")
if [ -n "${TRAIN_N:-}" ]; then read -r -a TN <<< "$TRAIN_N"; ARGS+=(--train-n "${TN[@]}"); fi
run_logged "$(dirname "$OUT")" python -u experiments/gate_fit.py "${ARGS[@]}" --group-by "${GROUP_BY:-group}" \
    ${VAL_FRAC:+--val-frac "$VAL_FRAC"} ${EXTRA:-} --out "$OUT"

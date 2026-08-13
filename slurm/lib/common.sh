# Shared preamble for every slurm wrapper. Source it right after the #SBATCH headers:
#   source "$(dirname "$0")/lib/common.sh"     (sbatch copies the script; use $SLURM_SUBMIT_DIR)
# Provides:
#   - repo-root cd + .venv activation + standard env exports
#   - run_logged <output_root> <cmd...>: DRY_RUN=1 support, tee into
#     <output_root>/runner-<jobid>.log (restores the log-copy convention the of_*
#     runner generation dropped — logs/ must never be the only copy of stdout)
#   - $ROOTS_INLENGTH: the 16-root in-length training mixture (from lib/roots_inlength.txt;
#     files instead of --export values because sbatch --export silently splits on commas)

set -uo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

_lib_dir="$REPO_ROOT/slurm/lib"
if [ -f "$_lib_dir/roots_inlength.txt" ]; then
    ROOTS_INLENGTH="$(grep -v '^\s*#' "$_lib_dir/roots_inlength.txt" | grep -v '^\s*$' | paste -sd, -)"
    export ROOTS_INLENGTH
fi

run_logged() {
    local out="$1"
    shift
    mkdir -p "$out"
    echo "[run_logged] output_root=$out"
    echo "[run_logged] cmd: $*"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY_RUN] not executing"
        return 0
    fi
    "$@" 2>&1 | tee "$out/runner-${SLURM_JOB_ID:-local}.log"
}

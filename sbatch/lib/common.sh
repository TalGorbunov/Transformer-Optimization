# Shared preamble for every sbatch wrapper. Source it right after the #SBATCH headers:
#   source "$SLURM_SUBMIT_DIR/sbatch/lib/common.sh"
# Provides:
#   - repo-root cd + .venv activation + standard env exports
#   - run_logged <output_root> <cmd...>: DRY_RUN=1 support, tee into <output_root>/runner-<jobid>.log
#     (logs/ must never be the only copy of stdout)
#   - stage_split <split>: copy one benchmark split's tarball to the node-local NVMe ($TMPDIR) and
#     export DATA_ROOT so scripts read frames locally (per-split tarballs are written by
#     experiments/prepare_data.py next to the extracted tree). No-op if already staged.
# Twin: legacy/v1/slurm/lib/common.sh (minus the park-era ROOTS_INLENGTH list).

set -uo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # the model is in the cache; never block on the hub

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

# stage_split seq_len_8_train   -> $TMPDIR/mmred/images/seq_len_8_train (+ json/ next to it)
stage_split() {
    local split="$1"
    local src="${DATA_SRC:-data/mmred_hf}"
    local dst="${TMPDIR:-/scratch/tmp/$UID}/mmred"
    if [ "${STAGE:-1}" != "1" ] || [ -z "${TMPDIR:-}" ]; then
        export DATA_ROOT="$src"; echo "[stage_split] not staging; DATA_ROOT=$DATA_ROOT"; return 0
    fi
    mkdir -p "$dst/images" "$dst/json"
    cp -u "$src"/json/*.json "$dst/json/" 2>/dev/null || true
    if [ ! -f "$dst/images/$split/.staged" ]; then
        local tar="$src/images/$split.tar"
        if [ -f "$tar" ]; then
            echo "[stage_split] $tar -> $dst/images/"
            tar xf "$tar" -C "$dst/images/" && touch "$dst/images/$split/.staged"
        else
            echo "[stage_split] no tarball for $split; reading from $src"; export DATA_ROOT="$src"; return 0
        fi
    fi
    export DATA_ROOT="$dst"
    echo "[stage_split] DATA_ROOT=$DATA_ROOT ($(find "$dst/images/$split" -type f | wc -l) files)"
}

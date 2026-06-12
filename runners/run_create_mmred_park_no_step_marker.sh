#!/bin/bash
#SBATCH --job-name=gen-no-step
#SBATCH --mem=16G
#SBATCH -p l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/tal.gorbunov/projects/Transformer-Optimization
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -d "${PWD}/evaluations" && -d "${PWD}/models" ]]; then
    REPO_ROOT="${PWD}"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/evaluations" && -d "${SLURM_SUBMIT_DIR}/models" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
  fi
fi

cd "${REPO_ROOT}"
mkdir -p logs

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1

SOURCE_ROOT="${SOURCE_ROOT:-/home/tal.gorbunov/projects/Transformer-Optimization/data/mmred_images_park}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/tal.gorbunov/projects/Transformer-Optimization/data/mmred_images_park_no_step_marker}"

SLURM_LOG_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME:-gen-no-step}-${SLURM_JOB_ID:-local}.out"
RUN_LOG_DIR="${REPO_ROOT}/outputs/mmred_images_park_no_step_marker_creation"
mkdir -p "${RUN_LOG_DIR}"
copy_slurm_log() {
  if [[ -f "${SLURM_LOG_PATH}" && -d "${OUTPUT_ROOT}" ]]; then
    cp "${SLURM_LOG_PATH}" "${OUTPUT_ROOT}/"
  fi
}
trap copy_slurm_log EXIT
exec > >(tee -a "${RUN_LOG_DIR}/run-${SLURM_JOB_ID:-local}.log") 2>&1

CMD=(
  python -u scripts/create_mmred_park_no_step_marker.py
  --source-root "${SOURCE_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --overwrite
)

printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" "$@"

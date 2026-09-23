#!/usr/bin/env bash
#SBATCH --job-name=diag_cpu
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
# CPU collection for DIAG: fits (+ figures unless FITS_ONLY=1) over whatever run dirs exist. Re-run any time.
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$$"
export OMP_NUM_THREADS=8
bash experiments/diag_analyze.sh ${FITS_ONLY:+fits} 2>&1 | tee "outputs/diag/fits/analyze-${SLURM_JOB_ID}.log"

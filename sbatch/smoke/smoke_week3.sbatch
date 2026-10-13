#!/usr/bin/env bash
#SBATCH --job-name=smoke_week3
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --time=01:55:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# Week-3 smoke (2026-10-12): the two-pass read off the planted protocol, one row per task, chained in one job.
#   1  uniform N=16, fitted gate (fold A rows, gate fitted on fold B)   = the try-0 uniform curve's path
#   2  vnbench4 uniform N=64, fitted gate, the four-try fold file        = the circular score's path
#   3  uniform N=256, fitted gate                                        = memory and time of a 256-frame scan
#   4  planted N=16, fitted gate, --two-pass-positions original          = variant 2 of the read
# Rows: --qids-file pins the rows (PER_QTYPE does not limit them), so the smoke reads 9-row fold-A files written next to
# its outputs (one row per task; x4 option orders for vnbench4). Knobs: STEPS ("1 2 3 4"; the steps to run).
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
export STAGE_DATA=0 OUT_BASE=outputs/_scratch/week3_smoke MAX_PIXELS=151200 DATASET=vnbench STAGE=s4 PER_QTYPE=1
export GATE_NPZ=outputs/scaleup/vnbench/qwen2.5-vl-7b/s2_gate/fits/gate_L20_fitB.npz GATE_TAG=_foldA
ROWS="--qids-file $OUT_BASE/qids_foldA_9.txt"
ROWS4="--qids-file $OUT_BASE/qids4_foldA_36.txt"
STEPS="${STEPS:-1 2 3 4}"
step() {   # step <n> <description>: prints the header; false when <n> is not in STEPS, so `step N ... && cmd` skips cmd
  local number="$1"
  shift
  echo
  echo "=================== [$(date '+%H:%M:%S')] $number $*"
  case " $STEPS " in
    *" $number "*) return 0 ;;
    *) echo "skipped (STEPS=$STEPS)"; return 1 ;;
  esac
}
step 1 uniform N16 fitted &&          PROTOCOL=uniform NLIST="16" EXTRA="$ROWS" bash sbatch/scaleup_cells.sbatch
step 2 vnbench4 uniform N64 fitted && DATASET=vnbench4 PROTOCOL=uniform NLIST="64" EXTRA="$ROWS4" bash sbatch/scaleup_cells.sbatch
step 3 uniform N256 fitted &&         PROTOCOL=uniform NLIST="256" EXTRA="$ROWS" bash sbatch/scaleup_cells.sbatch
step 4 planted N16 original positions && NLIST="16" POSITIONS=original EXTRA="$ROWS" bash sbatch/scaleup_cells.sbatch
echo; echo "=================== [$(date '+%H:%M:%S')] done"

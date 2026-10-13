#!/usr/bin/env bash
#SBATCH --job-name=smoke_week2
#SBATCH --partition=a100-public
#SBATCH --qos=2h_2g
#SBATCH --time=01:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# Week-2 smoke (2026-10-11): the three new pieces on a handful of rows, chained in one job.
#   1-3  evaluate.py --two-pass (fitted gate / letters / oracle) on Qwen, planted N=16, 1 row per task
#   4    evaluate.py --readout letters on the dense plain path, Qwen N=8
#   5-6  InternVL3.5-8B plain + oracle-gated with --readout letters, N=8, --res lo
#   7-8  probe_attention.py on VNBench planted N=8, plain + gated, 1 row of two tasks
#   9    probe_attention.py MMReD seq_len_8 char_at_frame plain, 2 rows = the 2026-09-22 anchor rows (0003834, 0003841)
# Knobs: STEPS ("1 2 ... 9"; the steps to run, so a rerun after a node failure skips the finished ones).
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
export STAGE_DATA=0 OUT_BASE=outputs/_scratch/week2_smoke
G=outputs/scaleup/vnbench/qwen2.5-vl-7b/s2_gate/fits/gate_L20_fitB.npz
STEPS="${STEPS:-1 2 3 4 5 6 7 8 9}"
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
step 1 two-pass fitted gate &&   DATASET=vnbench STAGE=s4 NLIST="16" PER_QTYPE=1 MAX_PIXELS=151200 GATE_NPZ=$G GATE_TAG=_foldA bash sbatch/scaleup_cells.sbatch
step 2 two-pass fitted letters && DATASET=vnbench STAGE=s4 NLIST="16" PER_QTYPE=1 MAX_PIXELS=151200 GATE_NPZ=$G GATE_TAG=_foldA READOUT=letters bash sbatch/scaleup_cells.sbatch
step 3 two-pass oracle &&        DATASET=vnbench STAGE=s4 NLIST="16" PER_QTYPE=1 MAX_PIXELS=151200 bash sbatch/scaleup_cells.sbatch
step 4 qwen dense letters &&     DATASET=vnbench STAGE=s1 ARM=plain NLIST="8" PER_QTYPE=1 MAX_PIXELS=151200 READOUT=letters bash sbatch/scaleup_cells.sbatch
step 5 internvl plain letters && DATASET=vnbench BACKBONE=internvl3.5-8b STAGE=s1 ARM=plain NLIST="8" PER_QTYPE=1 READOUT=letters EXTRA="--res lo" bash sbatch/scaleup_cells.sbatch
step 6 internvl gated letters && DATASET=vnbench BACKBONE=internvl3.5-8b STAGE=s3 NLIST="8" PER_QTYPE=1 READOUT=letters EXTRA="--res lo" bash sbatch/scaleup_cells.sbatch
step 7 photo vnbench plain &&    DATASET=vnbench ARM=plain NLIST="8" MAX_PIXELS=151200 LIMIT=1 QTYPES="cnt_edit1 ret_edit1" bash sbatch/probe_attention.sbatch
step 8 photo vnbench gated &&    DATASET=vnbench ARM=gated NLIST="8" MAX_PIXELS=151200 LIMIT=1 QTYPES="cnt_edit1 ret_edit1" bash sbatch/probe_attention.sbatch
step 9 photo mmred anchor &&     STAGE=0 SPLIT=seq_len_8_test ARM=plain QTYPES="char_at_frame" LIMIT=2 OUTPUT=outputs/_scratch/week2_smoke/photo_mmred_N8_plain bash sbatch/probe_attention.sbatch
echo; echo "=================== [$(date '+%H:%M:%S')] done"

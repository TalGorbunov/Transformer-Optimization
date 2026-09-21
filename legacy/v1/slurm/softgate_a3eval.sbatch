#!/bin/bash
#SBATCH --job-name=sg_a3eval
#SBATCH -p h200-shared
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x-%j.out
# SOFTGATE A3 evals: MODE=ladder (each arm at its own B, N=64/128) | cross (N=32: arms under the other gates)
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
MODE="${MODE:?}"; A8="${A8:?}"; A128="${A128:?}"; AH="${AH:?}"
E="outputs/loramech/examdirs"
ev() { # name adapter bonus dirs...
  local name=$1 ad=$2 bonus=$3; shift 3; local DF=""; for f in "$@"; do DF="$DF --eval-dirs-file $f"; done
  local BF=""; [ "$bonus" != "hard" ] && BF="--gate-bonus $bonus"
  run_logged "outputs/softgate/a3eval_${name}" python -u scripts/sparse/train_sft_gated.py --limit 24 --eval-only-adapter "$ad" --gate oracle $BF --nfree-prompt $DF --eval-longn-limit 150 --output "outputs/softgate/a3eval_${name}"
}
if [ "$MODE" = ladder ]; then
  ev B8_at_B8_N64_128 "$A8" 8 $E/exam_ff_N64.txt $E/exam_ff_N128.txt
  ev B128_at_B128_N64_128 "$A128" 128 $E/exam_ff_N64.txt $E/exam_ff_N128.txt
  ev hard_at_hard_N64_128 "$AH" hard $E/exam_ff_N64.txt $E/exam_ff_N128.txt
elif [ "$MODE" = cross ]; then
  ev B8_at_hard_N32 "$A8" hard $E/exam_ff_N32.txt
  ev B128_at_hard_N32 "$A128" hard $E/exam_ff_N32.txt
  ev hard_at_B8_N32 "$AH" 8 $E/exam_ff_N32.txt
  ev hard_at_B128_N32 "$AH" 128 $E/exam_ff_N32.txt
  ev B8_at_B128_N32 "$A8" 128 $E/exam_ff_N32.txt
fi
echo "SOFTGATE A3EVAL ($MODE) COMPLETE"

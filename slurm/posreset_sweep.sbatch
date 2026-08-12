#!/bin/bash
#SBATCH --job-name=posreset_sweep
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
# Posreset dose-response, gate-accuracy edition: probe_supply with/without --reset-positions.
# Knobs: GROUP=A (N=4,8,16) | B (N=32,64). Caches land under outputs/presentation/posreset_sweep/.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
GROUP="${GROUP:?set GROUP=A|B}"

cell() {  # cell <N> <root> <limit> <arm reset|noreset>
  local N=$1 ROOT=$2 LIMIT=$3 ARM=$4
  local ARMS="--question-first --fence-frames --fence-blocks"
  [ "$ARM" = "reset" ] && ARMS="$ARMS --reset-positions"
  run_logged "outputs/presentation/posreset_sweep/N${N}_${ARM}" \
    python -u scripts/probe_supply.py --data_root "$ROOT" --limit "$LIMIT" \
    --layers 16 --task steps --shuffle-dirs 0 $ARMS \
    --output "outputs/presentation/posreset_sweep/N${N}_${ARM}"
}

if [ "$GROUP" = "A" ]; then
  for ARM in reset noreset; do
    cell 4  data/mmred_images_park/seq_len_4/all_uniform 150 $ARM
    cell 8  data/mmred_images_park/seq_len_8/all_uniform 150 $ARM
    cell 16 data/mmred_longN_park/seq_len_16/all_uniform 120 $ARM
  done
else
  for ARM in reset noreset; do
    cell 32 data/mmred_longN_park/seq_len_32/all_uniform 100 $ARM
    cell 64 data/mmred_longN_park/seq_len_64/all_uniform 40 $ARM
  done
fi

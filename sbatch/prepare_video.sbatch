#!/usr/bin/env bash
#SBATCH --job-name=prepare_video
#SBATCH --partition=l40s-shared
#SBATCH --qos=4h_0g
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
# experiments/prepare_video.py on a CPU queue: the stages that need no network (videos for herbench = tar
# extraction; frames; tar; verify). `rows` and MINERVA `videos` need the network -> run them on the login node.
# Knobs (env; no commas): DATASET (herbench|minerva) ROOT STAGES (space-separated, default "frames tar verify")
#   FRAME_SET (native|512)  POOL LIMIT (frames smoke) KEEP_PARTS (1) MAX_FAILED (undecodable videos tolerated) EXTRA OUTPUT
# Submit:
#   SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/prepare_video.sbatch
#   sbatch -p l40s-shared --qos=4h_0g --time=04:00:00 --export=ALL,DATASET=herbench,STAGES="videos frames tar verify" sbatch/prepare_video.sbatch
source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"
DATASET="${DATASET:-herbench}"
read -r -a STAGE_ARR <<< "${STAGES:-frames tar verify}"
OUTPUT="${OUTPUT:-outputs/prepare/${DATASET}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
export MPLBACKEND=Agg
run_logged "$OUTPUT" python -u experiments/prepare_video.py --dataset "$DATASET" ${ROOT:+--root "$ROOT"} \
    --stage "${STAGE_ARR[@]}" ${POOL:+--pool "$POOL"} ${LIMIT:+--limit "$LIMIT"} \
    ${FRAME_SET:+--frame-set "$FRAME_SET"} ${KEEP_PARTS:+--keep-parts} ${MAX_FAILED:+--max-failed "$MAX_FAILED"} --workers "${SLURM_CPUS_PER_TASK:-4}" ${EXTRA:-}

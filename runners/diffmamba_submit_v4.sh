#!/bin/bash
# v4: patched code (defensive mkdir before checkpoint save) + UNIQUE OUTPUT_ROOT per job
# (avoids any cross-job contention on a shared output dir). count/distractor/order only;
# co_occ jobs (97045/46/47) kept running separately.
set -euo pipefail
cd "$(dirname "$0")/.."
R=runners/rooms_visited_adapter_seq8.sbatch
BASE="EPOCHS=6,LAYERS=14 15 16 17,SEQ_LEN=4 6 8"
DATA="--train-per-count 35 --val-per-count 12 --length-ood-per-count 12 --candidate-max 10"
go() { sbatch --parsable -p "$1" --qos="$2" --time="$3" -J "$4" --export="$5" "$R"; }

# ----- count/neutral -----
go h200-shared 4d_1g 24:00:00 dm4_c_diffon \
"ALL,TASK=count,$BASE,VARIANTS=carrier_diff,RUN_PREFIX=v,OUTPUT_ROOT=outputs/dm4_count_diffon,EXTRA_ARGS=--filler-kind neutral --diff-output-norm $DATA"
go h200-shared 4d_1g 24:00:00 dm4_c_diffoff \
"ALL,TASK=count,$BASE,VARIANTS=carrier_diff,RUN_PREFIX=v,OUTPUT_ROOT=outputs/dm4_count_diffoff,EXTRA_ARGS=--filler-kind neutral --no-diff-output-norm $DATA"
go h200-shared 4d_1g 24:00:00 dm4_c_mamba \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=v,OUTPUT_ROOT=outputs/dm4_count_mamba,EXTRA_ARGS=--filler-kind neutral $DATA"
go a100-public 24h_1g 18:00:00 dm4_c_base \
"ALL,TASK=count,$BASE,VARIANTS=carrier_direct_sum carrier_pna,RUN_PREFIX=v,OUTPUT_ROOT=outputs/dm4_count_base,EXTRA_ARGS=--filler-kind neutral $DATA"

# ----- count/distractor -----
go a100-public 4d_1g 24:00:00 dm4_d_diffon \
"ALL,TASK=count,$BASE,VARIANTS=carrier_diff,RUN_PREFIX=v,OUTPUT_ROOT=outputs/dm4_distract_diffon,EXTRA_ARGS=--filler-kind distractor --diff-output-norm $DATA"
go a100-public 4d_1g 24:00:00 dm4_d_mamba \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=v,OUTPUT_ROOT=outputs/dm4_distract_mamba,EXTRA_ARGS=--filler-kind distractor $DATA"
go a100-public 24h_1g 18:00:00 dm4_d_base \
"ALL,TASK=count,$BASE,VARIANTS=carrier_direct_sum carrier_pna,RUN_PREFIX=v,OUTPUT_ROOT=outputs/dm4_distract_base,EXTRA_ARGS=--filler-kind distractor $DATA"

# ----- mamba order-sensitivity -----
go h200-shared 4d_1g 24:00:00 dm4_j4b \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=v,OUTPUT_ROOT=outputs/dm4_order_permEval,EXTRA_ARGS=--filler-kind neutral --mamba-eval-permute $DATA"
go h200-shared 4d_1g 24:00:00 dm4_j4c \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=v,OUTPUT_ROOT=outputs/dm4_order_aug,EXTRA_ARGS=--filler-kind neutral --mamba-order-aug --mamba-eval-permute $DATA"
echo "v4 submitted"

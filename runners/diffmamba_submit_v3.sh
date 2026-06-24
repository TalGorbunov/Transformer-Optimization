#!/bin/bash
# v3: resubmit COUNT-task jobs with --candidate-max 10 (fixes KeyError on length-OOD len-10 where
# count reaches 9-10; also makes len-10 a genuine count-extrapolation test). co_occ jobs unaffected.
set -euo pipefail
cd "$(dirname "$0")/.."
R=runners/rooms_visited_adapter_seq8.sbatch
BASE="EPOCHS=6,LAYERS=14 15 16 17,SEQ_LEN=4 6 8"
DATA="--train-per-count 35 --val-per-count 12 --length-ood-per-count 12 --candidate-max 10"
go() { sbatch --parsable -p "$1" --qos="$2" --time="$3" -J "$4" --export="$5" "$R"; }

# slow operators on 4d_1g
go h200-shared 4d_1g 24:00:00 dm3_c_diffon \
"ALL,TASK=count,$BASE,VARIANTS=carrier_diff,RUN_PREFIX=count_diffon,OUTPUT_ROOT=outputs/diffmamba2_count,EXTRA_ARGS=--filler-kind neutral --diff-output-norm $DATA"
go h200-shared 4d_1g 24:00:00 dm3_c_diffoff \
"ALL,TASK=count,$BASE,VARIANTS=carrier_diff,RUN_PREFIX=count_diffoff,OUTPUT_ROOT=outputs/diffmamba2_count,EXTRA_ARGS=--filler-kind neutral --no-diff-output-norm $DATA"
go h200-shared 4d_1g 24:00:00 dm3_c_mamba \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=count_mamba,OUTPUT_ROOT=outputs/diffmamba2_count,EXTRA_ARGS=--filler-kind neutral $DATA"
go a100-public 4d_1g 24:00:00 dm3_d_diffon \
"ALL,TASK=count,$BASE,VARIANTS=carrier_diff,RUN_PREFIX=distract_diffon,OUTPUT_ROOT=outputs/diffmamba2_distractor,EXTRA_ARGS=--filler-kind distractor --diff-output-norm $DATA"
go a100-public 4d_1g 24:00:00 dm3_d_mamba \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=distract_mamba,OUTPUT_ROOT=outputs/diffmamba2_distractor,EXTRA_ARGS=--filler-kind distractor $DATA"
go a100-public 4d_1g 24:00:00 dm3_j4b \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=order_permEval,OUTPUT_ROOT=outputs/diffmamba2_order,EXTRA_ARGS=--filler-kind neutral --mamba-eval-permute $DATA"
go h200-shared 4d_1g 24:00:00 dm3_j4c \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=order_aug,OUTPUT_ROOT=outputs/diffmamba2_order,EXTRA_ARGS=--filler-kind neutral --mamba-order-aug --mamba-eval-permute $DATA"

# fast baselines on 24h_1g
go a100-public 24h_1g 18:00:00 dm3_c_base \
"ALL,TASK=count,$BASE,VARIANTS=carrier_direct_sum carrier_pna,RUN_PREFIX=count_base,OUTPUT_ROOT=outputs/diffmamba2_count,EXTRA_ARGS=--filler-kind neutral $DATA"
go l40s-public 24h_1g 18:00:00 dm3_d_base \
"ALL,TASK=count,$BASE,VARIANTS=carrier_direct_sum carrier_pna,RUN_PREFIX=distract_base,OUTPUT_ROOT=outputs/diffmamba2_distractor,EXTRA_ARGS=--filler-kind distractor $DATA"
echo "v3 count jobs submitted"

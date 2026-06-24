#!/bin/bash
# Right-sized Diff/Mamba matrix v2: 6 epochs, smaller data, slow mamba isolated on 4d_1g,
# fast baselines on 24h_1g. Spread across h200/a100-public/l40s-public.
set -euo pipefail
cd "$(dirname "$0")/.."
R=runners/rooms_visited_adapter_seq8.sbatch
BASE="EPOCHS=6,LAYERS=14 15 16 17,SEQ_LEN=4 6 8"
DATA="--train-per-count 35 --val-per-count 12 --length-ood-per-count 12"

go() { # partition qos time jobname export
  sbatch --parsable -p "$1" --qos="$2" --time="$3" -J "$4" --export="$5" "$R"
}

# ---------- slow operators (diff/mamba) on 4d_1g, time 24h ----------
go h200-shared 4d_1g 24:00:00 dm2_c_diffon \
"ALL,TASK=count,$BASE,VARIANTS=carrier_diff,RUN_PREFIX=count_diffon,OUTPUT_ROOT=outputs/diffmamba2_count,EXTRA_ARGS=--filler-kind neutral --diff-output-norm $DATA"
go h200-shared 4d_1g 24:00:00 dm2_c_diffoff \
"ALL,TASK=count,$BASE,VARIANTS=carrier_diff,RUN_PREFIX=count_diffoff,OUTPUT_ROOT=outputs/diffmamba2_count,EXTRA_ARGS=--filler-kind neutral --no-diff-output-norm $DATA"
go h200-shared 4d_1g 24:00:00 dm2_c_mamba \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=count_mamba,OUTPUT_ROOT=outputs/diffmamba2_count,EXTRA_ARGS=--filler-kind neutral $DATA"
go a100-public 4d_1g 24:00:00 dm2_d_diffon \
"ALL,TASK=count,$BASE,VARIANTS=carrier_diff,RUN_PREFIX=distract_diffon,OUTPUT_ROOT=outputs/diffmamba2_distractor,EXTRA_ARGS=--filler-kind distractor --diff-output-norm $DATA"
go a100-public 4d_1g 24:00:00 dm2_d_mamba \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=distract_mamba,OUTPUT_ROOT=outputs/diffmamba2_distractor,EXTRA_ARGS=--filler-kind distractor $DATA"
go l40s-public 4d_1g 24:00:00 dm2_o_diffon \
"ALL,TASK=co_occupancy,$BASE,VARIANTS=carrier_diff,RUN_PREFIX=coocc_diffon,OUTPUT_ROOT=outputs/diffmamba2_coocc,EXTRA_ARGS=--co-occ-evidence-only --diff-output-norm $DATA"
go l40s-public 4d_1g 24:00:00 dm2_o_mamba \
"ALL,TASK=co_occupancy,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=coocc_mamba,OUTPUT_ROOT=outputs/diffmamba2_coocc,EXTRA_ARGS=--co-occ-evidence-only $DATA"
# order-sensitivity test B: mamba (natural train) evaluated under permuted frames
go h200-shared 4d_1g 24:00:00 dm2_j4b \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=order_permEval,OUTPUT_ROOT=outputs/diffmamba2_order,EXTRA_ARGS=--filler-kind neutral --mamba-eval-permute $DATA"

# ---------- fast baselines on 24h_1g, time 18h ----------
go a100-public 24h_1g 18:00:00 dm2_c_base \
"ALL,TASK=count,$BASE,VARIANTS=carrier_direct_sum carrier_pna,RUN_PREFIX=count_base,OUTPUT_ROOT=outputs/diffmamba2_count,EXTRA_ARGS=--filler-kind neutral $DATA"
go a100-public 24h_1g 18:00:00 dm2_d_base \
"ALL,TASK=count,$BASE,VARIANTS=carrier_direct_sum carrier_pna,RUN_PREFIX=distract_base,OUTPUT_ROOT=outputs/diffmamba2_distractor,EXTRA_ARGS=--filler-kind distractor $DATA"
go l40s-public 24h_1g 18:00:00 dm2_o_base \
"ALL,TASK=co_occupancy,$BASE,VARIANTS=carrier_direct_sum carrier_glstm_layerwise,RUN_PREFIX=coocc_base,OUTPUT_ROOT=outputs/diffmamba2_coocc,EXTRA_ARGS=--co-occ-evidence-only $DATA"
# order-sensitivity test C: mamba order-augmented training, eval permuted
go h200-shared 24h_1g 18:00:00 dm2_j4c \
"ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=order_aug,OUTPUT_ROOT=outputs/diffmamba2_order,EXTRA_ARGS=--filler-kind neutral --mamba-order-aug --mamba-eval-permute $DATA"
echo "v2 submitted"

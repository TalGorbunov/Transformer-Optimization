#!/bin/bash
# Submit the Diff/Mamba experiment matrix. Run from repo root AFTER the smoke job validates.
# SLURM --export is comma-separated KEY=VALUE; multi-token values (VARIANTS, EXTRA_ARGS, LAYERS,
# SEQ_LEN) use SPACES (never commas) and the whole --export is ONE quoted arg.
set -euo pipefail
cd "$(dirname "$0")/.."
R=runners/rooms_visited_adapter_seq8.sbatch
BASE="EPOCHS=8,LAYERS=14 15 16 17,SEQ_LEN=4 6 8"

# Job 1a: counting mechanism (clean fillers), diff output-norm ON
sbatch --parsable -p h200-shared --qos=24h_1g --time=12:00:00 -J dm_j1a \
  --export="ALL,TASK=count,$BASE,VARIANTS=carrier_direct_sum carrier_glstm_layerwise_softmax carrier_pna carrier_diff carrier_mamba,RUN_PREFIX=j1_count,OUTPUT_ROOT=outputs/diffmamba_job1_count,EXTRA_ARGS=--filler-kind neutral --diff-output-norm" "$R"

# Job 1b: diff output-norm OFF (decisive normalization ablation)
sbatch --parsable -p h200-shared --qos=24h_1g --time=08:00:00 -J dm_j1b \
  --export="ALL,TASK=count,$BASE,VARIANTS=carrier_diff,RUN_PREFIX=j1_diffnonorm,OUTPUT_ROOT=outputs/diffmamba_job1_count,EXTRA_ARGS=--filler-kind neutral --no-diff-output-norm" "$R"

# Job 2: selection (distractor fillers)
sbatch --parsable -p a100-public --qos=24h_1g --time=12:00:00 -J dm_j2 \
  --export="ALL,TASK=count,$BASE,VARIANTS=carrier_direct_sum carrier_pna carrier_diff carrier_mamba,RUN_PREFIX=j2_distract,OUTPUT_ROOT=outputs/diffmamba_job2_distractor,EXTRA_ARGS=--filler-kind distractor --diff-output-norm" "$R"

# Job 3: relational clean ceiling (co-occupancy evidence-only, oracle 0.98)
sbatch --parsable -p h200-shared --qos=4d_1g --time=12:00:00 -J dm_j3 \
  --export="ALL,TASK=co_occupancy,$BASE,VARIANTS=carrier_direct_sum carrier_glstm_layerwise carrier_pna carrier_diff carrier_mamba,RUN_PREFIX=j3_coocc,OUTPUT_ROOT=outputs/diffmamba_job3_coocc,EXTRA_ARGS=--co-occ-evidence-only --diff-output-norm" "$R"

# Job 4: Mamba order-sensitivity (natural vs order-aug; both eval-permuted)
sbatch --parsable -p a100-public --qos=2h_2g --time=02:00:00 -J dm_j4a \
  --export="ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=j4_natural,OUTPUT_ROOT=outputs/diffmamba_job4_order,EXTRA_ARGS=--filler-kind neutral --mamba-eval-permute" "$R"
sbatch --parsable -p a100-public --qos=2h_2g --time=02:00:00 -J dm_j4b \
  --export="ALL,TASK=count,$BASE,VARIANTS=carrier_mamba,RUN_PREFIX=j4_orderaug,OUTPUT_ROOT=outputs/diffmamba_job4_order,EXTRA_ARGS=--filler-kind neutral --mamba-order-aug --mamba-eval-permute" "$R"
echo "all submitted"

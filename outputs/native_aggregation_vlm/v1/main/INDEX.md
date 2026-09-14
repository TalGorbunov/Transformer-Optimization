# V1 main comparison

Slurm array 440504: all seven tasks completed. Same data, nine epochs, development-selected checkpoint. All registered criteria failed. [Full results](../REPORT.md).

| Arm | Canonical run | Selected epoch | N32/64 exact |
|---|---|---:|---:|
| sum | [sum_seed0_20260910_144258_440505_2770151](sum/sum_seed0_20260910_144258_440505_2770151/summary.json) | 9 | 6.5% |
| mean | [mean_seed0_20260910_150701_440567_1332288](mean/mean_seed0_20260910_150701_440567_1332288/summary.json) | 9 | 30.5% |
| post_sum | [post_sum_seed0_20260910_144259_440506_1322809](post_sum/post_sum_seed0_20260910_144259_440506_1322809/summary.json) | 9 | 6.5% |
| post_mean | [post_mean_seed0_20260910_151127_440570_1334912](post_mean/post_mean_seed0_20260910_151127_440570_1334912/summary.json) | 5 | 40.0% |
| global | [global_seed0_20260910_151215_440504_1335620](global/global_seed0_20260910_151215_440504_1335620/summary.json) | 9 | 40.5% |
| hierarchical | [hierarchical_seed0_20260910_144258_440507_1322920](hierarchical/hierarchical_seed0_20260910_144258_440507_1322920/summary.json) | 7 | 36.0% |
| lora | [lora_seed0_20260910_144258_440508_1323108](lora/lora_seed0_20260910_144258_440508_1323108/summary.json) | 7 | 21.5% |

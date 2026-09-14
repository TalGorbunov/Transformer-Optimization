# Failed orientation head-audit diagnostic

The original registered report remains FAILED. This CPU diagnostic does not relax its argmax rule or release native/fresh inference.

| Arm | GPU saved correct | CPU replay correct | Argmax disagreements | Optimistic disagreement bound |
|---|---:|---:|---:|---:|
| product | 79/216 | 77/216 | 2 | 79/216 |
| additive | 72/216 | 72/216 | 0 | 72/216 |

All432 rows and full-vocabulary CPU outputs are retained. Sensitivity bounds are deterministic and are not statistical confidence intervals. The one extra fixed head route uses the saved GPU-normalized product batch10; it does not replace the registered normalization-plus-head replay. Training-window summaries describe logged quantities only.

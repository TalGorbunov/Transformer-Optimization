# V9 selected-checkpoint software audit

All four models completed. Native replay failures: 0/40; descriptive cache/full failures: 5/656.

| Condition | Seed | Selected step | Replay passed | Cache/full passed |
|---|---:|---:|---|---|
| path | 12 | 4860 | True | False |
| path | 13 | 4860 | True | False |
| residual | 12 | 3888 | True | False |
| residual | 13 | 3888 | True | False |

The original mixed/cache failure remains failed. No software result changes checkpoint selection.

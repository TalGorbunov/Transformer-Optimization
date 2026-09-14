# V7 selected-checkpoint software audit

All four models completed. Native replay failures: 0/40; descriptive cache/full failures: 6/336.

| Arm | Seed | Selected step | Replay passed | Cache/full passed |
|---|---:|---:|---|---|
| joint | 8 | 4860 | True | False |
| joint | 9 | 4860 | True | False |
| parallel | 8 | 4860 | True | False |
| parallel | 9 | 3888 | True | False |

The original mixed/cache failure remains failed. No software result changes checkpoint selection.

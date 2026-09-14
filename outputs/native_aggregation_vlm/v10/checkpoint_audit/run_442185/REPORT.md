# V10 selected-checkpoint software audit

All four models completed. Native replay failures: 0/40; descriptive cache/full failures: 4/656.

| Condition | Seed | Selected step | Replay passed | Cache/full passed |
|---|---:|---:|---|---|
| ce | 14 | 4590 | True | False |
| ce | 15 | 4590 | True | False |
| consistency | 14 | 4590 | True | False |
| consistency | 15 | 4590 | True | False |

The original mixed/cache failure remains failed. No software result changes checkpoint selection.

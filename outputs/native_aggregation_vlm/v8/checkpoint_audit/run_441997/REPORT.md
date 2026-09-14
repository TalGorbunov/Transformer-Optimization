# V8 selected-checkpoint software audit

All four models completed. Native replay failures: 0/40; descriptive cache/full failures: 5/656.

| Condition | Seed | Selected step | Replay passed | Cache/full passed |
|---|---:|---:|---|---|
| ce | 10 | 2916 | True | False |
| ce | 11 | 4860 | True | False |
| consistency | 10 | 4860 | True | False |
| consistency | 11 | 4860 | True | False |

The original mixed/cache failure remains failed. No software result changes checkpoint selection.

# V7 independent native evaluation

Audit passed. Both-seed primary: **True**. Both-seed practical threshold: **False**.

Exact match requires the correct integer and EOS completion within four generated tokens. Every one of the 452 test examples per run remains in the denominator.

| Arm | Seed | N16 familiar | Familiar OOD | Nonzero OOD | Unseen counts | Selected step |
|---|---:|---:|---:|---:|---:|---:|
| joint | 8 | 42/108 (38.9%) | 48/216 (22.2%) | 39/192 (20.3%) | 0/128 (0.0%) | 4860 |
| joint | 9 | 37/108 (34.3%) | 58/216 (26.9%) | 45/192 (23.4%) | 0/128 (0.0%) | 4860 |
| parallel | 8 | 108/108 (100.0%) | 122/216 (56.5%) | 110/192 (57.3%) | 0/128 (0.0%) | 4860 |
| parallel | 9 | 108/108 (100.0%) | 90/216 (41.7%) | 90/192 (46.9%) | 0/128 (0.0%) | 3888 |

| Seed | OOD additional correct | OOD difference | Family bootstrap 95% CI | N16 difference | Primary | Practical |
|---:|---:|---:|---:|---:|---|---|
| 8 | 74/216 | 34.26 pp | [28.24, 39.81] pp | 61.11 pp | True | False |
| 9 | 32/216 | 14.81 pp | [8.33, 21.30] pp | 65.74 pp | True | False |

Pooled fixed-two-seed OOD difference: 24.54 percentage points (descriptive).

Primary integer gates, required separately in both seeds: at least 11 more correct OOD answers out of 216; no more than 5 fewer N16 answers out of 108. Practical requires primary plus at least 152/216 OOD and 116/192 nonzero OOD correct.

This contrast tests the complete parallel local computation and fusion treatment. It does not isolate SUM, equalize inference compute, or establish reasoning composition. Training data were newly balanced; local image atoms can recur. Family intervals condition on these two fitted seeds.

Raw full-vocabulary logits, all five development selections, all checkpoint digests, all 4,860 target batches, native call counts, and the 19-manifest data exclusions were independently checked. First-token NLL is distinct from whole-answer likelihood for multi-digit counts.

[Complete per-N/K metrics, provenance, numerical/runtime exceptions, and bootstrap](analysis.json).

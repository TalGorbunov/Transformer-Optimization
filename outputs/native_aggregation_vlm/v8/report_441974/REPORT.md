# V8 independent native evaluation

Audit passed. Both-seed primary: **True**. Both-seed practical: **False**.

The two conditions use the same native parallel architecture and paired scenes. Only the training objective coefficient differs.

| Condition | Seed | N16 | N32 | N64 | Familiar OOD | Nonzero OOD | Unseen counts | Selected step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ce | 10 | 108/108 (100.0%) | 79/108 (73.1%) | 0/108 (0.0%) | 79/216 (36.6%) | 79/192 (41.1%) | 0/128 (0.0%) | 2916 |
| ce | 11 | 108/108 (100.0%) | 55/108 (50.9%) | 12/108 (11.1%) | 67/216 (31.0%) | 55/192 (28.6%) | 0/128 (0.0%) | 4860 |
| consistency | 10 | 108/108 (100.0%) | 108/108 (100.0%) | 72/108 (66.7%) | 180/216 (83.3%) | 156/192 (81.2%) | 0/128 (0.0%) | 4860 |
| consistency | 11 | 108/108 (100.0%) | 108/108 (100.0%) | 53/108 (49.1%) | 161/216 (74.5%) | 140/192 (72.9%) | 0/128 (0.0%) | 4860 |

| Seed | OOD gain | Family bootstrap95%CI | N16 change | Primary | Practical |
|---:|---:|---:|---:|---|---|
| 10 | 46.76pp | [43.98,50.00]pp | 0.00pp | True | False |
| 11 | 43.52pp | [39.35,48.15]pp | 0.00pp | True | False |

Exact requires the correct integer and native EOS within four tokens. All452 examples per run remain in the denominator.

Primary requires at least11/216 additional familiar-OOD answers and no more than5/108 fewer N16 answers in each seed. Practical additionally requires at least152/216 OOD,116/192 nonzero OOD, and76/108 N64 correct.

Independent checks cover21 prior-manifest exclusions, unchanged V7 training/dev data and cache,972same-question/count pairs,77760scene presentations,4860logged target batches, objective component arithmetic, all five development checkpoints, and raw native argmax/EOS/NLL.

Final acceptance also requires the separate selected-checkpoint audit. A strong CE-only result does not establish a consistency-objective benefit. This experiment makes no reasoning-composition or new-attention claim.

[Complete per-N/K metrics, losses, provenance and fixed-seed family intervals](analysis.json).

# V9 independent native evaluation

Audit passed. Both-seed primary: **False**. Both-seed practical: **False**.

The two conditions use the same native parallel architecture and paired scenes. Only the selected training regularizer differs; both coefficients equal one.

| Condition | Seed | N16 | N32 | N64 | Familiar OOD | Nonzero OOD | Unseen counts | Selected step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| residual | 12 | 108/108 (100.0%) | 108/108 (100.0%) | 47/108 (43.5%) | 155/216 (71.8%) | 131/192 (68.2%) | 0/128 (0.0%) | 3888 |
| residual | 13 | 108/108 (100.0%) | 108/108 (100.0%) | 50/108 (46.3%) | 158/216 (73.1%) | 146/192 (76.0%) | 0/128 (0.0%) | 3888 |
| path | 12 | 108/108 (100.0%) | 108/108 (100.0%) | 18/108 (16.7%) | 126/216 (58.3%) | 102/192 (53.1%) | 0/128 (0.0%) | 4860 |
| path | 13 | 108/108 (100.0%) | 108/108 (100.0%) | 18/108 (16.7%) | 126/216 (58.3%) | 102/192 (53.1%) | 0/128 (0.0%) | 4860 |

| Seed | OOD gain | Family bootstrap95%CI | N16 change | Primary | Practical |
|---:|---:|---:|---:|---|---|
| 12 | -13.43pp | [-16.20,-10.65]pp | 0.00pp | False | False |
| 13 | -14.81pp | [-18.06,-11.11]pp | 0.00pp | False | False |

Exact requires the correct integer and native EOS within four tokens. All452 examples per run remain in the denominator.

Primary requires at least11/216 additional familiar-OOD answers and no more than5/108 fewer N16 answers in each seed. Practical additionally requires at least152/216 OOD,116/192 nonzero OOD, and76/108 N64 correct.

Independent checks cover23 prior-manifest exclusions, unchanged V7 training/dev data and cache,972same-question/count pairs,77760scene presentations,4860logged target batches, objective component arithmetic, all five development checkpoints, and raw native argmax/EOS/NLL.

Both coefficients equal one, which does not match effective regularization strength or clipping. The path penalty bounds the FP32 branch along observed training directions; it does not certify new-scene or native-answer accuracy.

Final acceptance also requires the separate selected-checkpoint audit. A strong residual-consistency result does not establish an additional path-bound benefit. This experiment makes no reasoning-composition or new-attention claim.

[Complete per-N/K metrics, losses, provenance and fixed-seed family intervals](analysis.json).

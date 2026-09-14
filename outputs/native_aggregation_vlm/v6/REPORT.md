# Native aggregation with training-only local distillation (V6)

Fresh sequence evaluation; identical native SUM inference, training scenes and decoding. The auxiliary head is absent at inference.

| Training | Seed | N16 | N32 | N64 | Familiar OOD | Nonzero OOD | Unseen K9–16 | Selected block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6 | 25/108 (23.1%) | 14/108 (13.0%) | 1/108 (0.9%) | 15/216 (6.9%) | 15/192 (7.8%) | 0/128 (0.0%) | 7 |
| control | 7 | 18/108 (16.7%) | 14/108 (13.0%) | 0/108 (0.0%) | 14/216 (6.5%) | 14/192 (7.3%) | 0/128 (0.0%) | 7 |
| aligned | 6 | 25/108 (23.1%) | 12/108 (11.1%) | 12/108 (11.1%) | 24/216 (11.1%) | 0/192 (0.0%) | 0/128 (0.0%) | 9 |
| aligned | 7 | 24/108 (22.2%) | 12/108 (11.1%) | 0/108 (0.0%) | 12/216 (5.6%) | 12/192 (6.2%) | 0/128 (0.0%) | 9 |

Both-seed relative efficacy criterion: **False**.
Both-seed practical milestone: **False**.

| Aligned minus control | Familiar OOD difference | Paired-anchor 95% interval | N16 difference |
|---|---:|---:|---:|
| seed6 | +4.17pp | [-2.78,+11.57]pp | +0.00pp |
| seed7 | -0.93pp | [-2.78,+0.93]pp | +5.56pp |

Aligned-control familiar OOD >=5pp and N16 loss <=5pp, EACH seed.
Primary plus aligned familiar OOD >=70% and nonzero-K OOD >=60%, EACH seed.

Every invalid output remains incorrect. MAE and bias condition on parsed answers; full denominators, per-K results and length transitions are in analysis.json.
This tests the auxiliary training objective. Without a permuted-correspondence control, it does not isolate alignment-specific learning from other effects of that objective. It does not establish reasoning composition or a new attention family.
Teacher judgments and local supervision are additional training resources. Local rendered atoms recur in this finite compositional benchmark; fresh complete sequences do not establish unseen perception or natural-video generalization.
The unchanged native numerical thresholds are descriptive post-training checks. Earlier V5 engineering failures are preserved; no full/cache or long-reasoning equivalence claim follows.


# Native single-versus-mixed cache localization

Original441729numerical gate remains **FAIL**. Every original artifact and threshold is unchanged.

New132cached/full comparisons: FAIL under the same.02TV/top1 rule.

| Case | Forced prefix step | Focus-row TV | Top1 equal | Gate |
|---|---:|---:|---|---|
| single | 1 | 0.000237567692 | True | PASS |
| single | 2 | 0.0174366228 | True | PASS |
| mixed | 1 | 8.6454299e-05 | True | PASS |
| mixed | 2 | 0.0202964636 | True | FAIL |

Repeated mixed raw-logit rows exact: 325/325.

The original441729failed gate remains failed, regardless of this localization.
One row was chosen because it failed. This is numerical localization, not representative model accuracy.
Same native NF4/bfloat16-compute backend, actual FP16 native states/head; no kernel or precision intervention.
Single-versus-mixed and original-repeat differences are descriptive, not replacement eligibility criteria.
No training, aggregate answer, new prompt search or claimed reasoning competence.

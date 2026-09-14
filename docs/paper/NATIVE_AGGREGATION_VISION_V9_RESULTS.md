# V9: the path bound worsens extrapolation

The stronger sensitivity penalty failed its matched test. All models remained perfect at 16 and 32 frames, but the path objective reduced 64-frame accuracy in both seeds. The proposed bound penalty is not retained as the method improvement.

| Objective | Seed | N16 | N32 | N64 | Familiar OOD | Unseen K9–16 |
|---|---:|---:|---:|---:|---:|---:|
| Residual consistency | 12 | 108/108 | 108/108 | 47/108 | 155/216 | 0/128 |
| Residual consistency | 13 | 108/108 | 108/108 | 50/108 | 158/216 | 0/128 |
| Path bound | 12 | 108/108 | 108/108 | 18/108 | 126/216 | 0/128 |
| Path bound | 13 | 108/108 | 108/108 | 18/108 | 126/216 | 0/128 |

Familiar OOD differences were −13.43 and −14.81 percentage points, with paired family-bootstrap 95% intervals [−16.20, −10.65] and [−18.06, −11.11]. Both registered primary and practical criteria failed. These intervals condition on the two fitted seeds; pooling cannot rescue either decision.

The conditions shared the paired examples, initialization within seed, 4,860 updates, native parallel SUM/SiLU core and inference. Only the selected regularizer changed; both coefficients were one. Both residual controls selected step 3,888, and both path models selected step 4,860, using the fixed N16 development rule.

The path loss was successfully reduced during training. Mean path penalties in the final 972-update block were approximately 0.000532 and 0.000629 in the path conditions, versus 10.58 and 0.918 in the residual conditions. Recorded residual penalties were also smaller in the path conditions. None of the 311,040 logged pair-position comparisons violated the descriptive bound tolerance. These observations do not identify why the stronger constraint hurts: it can be conservative, effective regularization strengths differ, and observed directions do not cover every longer-scene change. They show that optimizing the bound did not yield the intended extrapolation benefit.

Independent report 442051 verified every test/development output and the data, loss and checkpoint bindings. Selected-checkpoint audit 442081 passed all 40 native-head replays; five of 656 cache/full numerical comparisons retain their descriptive failure status, all with equal top-1. Four are the known local N64 comparison (TV 0.02030); the fifth is the residual seed-13 global comparison (TV 0.05785). Finalizer 442084 preserved the failed efficacy decisions and bound all selected checkpoints.

Total V9 cost was 4,940 allocated GPU-seconds (1.3722 hours): 67 for profiles, 4,788 for mains and 85 for the selected-checkpoint audit. CPU work and all artifacts are recorded separately. The fixed-state follow-up is diagnostic only; it cannot alter this result. V10 preparation keeps the simpler residual consistency objective as the candidate and expands native answer supervision without extending maximum training length beyond 16.

[Independent report](../../outputs/native_aggregation_vlm/v9/report_442051/REPORT.md) · [Figure and final verification](../../outputs/native_aggregation_vlm/v9/finalization/final_442084/REPORT.md) · [Method and limitations](NATIVE_AGGREGATION_PATH_BOUND.md)

The completed fixed-state follow-up442099 evaluated all432registered training-prototype points without a VLM call or fit. Allfourmodels were54/54atN16. AtN64, path12/13were32/54and24/54, versusresidual12/13at28/54and29/54. There is no consistent synthetic advantage for the path penalty. These repeated-state sets are outside the scene generator law and cannot substitute for the negative held-out result. [Fixed-state follow-up](../../outputs/native_aggregation_vlm/v9/response_surface/run_442099/REPORT.md).

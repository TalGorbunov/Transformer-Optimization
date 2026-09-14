# Training-estimated background drift explains much of the first-token failure

**A fixed diagnostic, not full-answer success.** All four selected V10 models failed length extrapolation despite perfect N16 development. Their SUM representation can retain a small average contribution from irrelevant images; that contribution grows with input length.

For each model and question, we estimated the mean local message from all 24 image occurrences in its N8/N16 zero-count training scenes. We then subtracted `(N−16) Wagg mean_message` from the aggregate decoder preactivation at test time. This used no test labels or test-state fitting. A fixed orthogonal direction with the same norm served as a control. The nonlinear native decoder stayed unchanged.

One preselected family for each answer 0–16 gave 17 cases at each length. All four models and all three interventions were retained.

| Model | N32 original / centered | N64 original / centered | N64 arbitrary-direction control |
|---|---:|---:|---:|
| CE 14 | 2 / 17 | 1 / 15 | 1 |
| CE 15 | 3 / 17 | 0 / 17 | 3 |
| Consistency 14 | 6 / 17 | 4 / 17 | 3 |
| Consistency 15 | 9 / 17 | 4 / 17 | 2 |

Every number is first-token correct out of 17. At N64, centered CE14 still predicted0 for K1 and9 for K10. The other centered models got every first token correct. All baseline argmaxes matched the original V10 outputs; all 34 captured-state head checks and 136 branch decompositions passed. The run used 88 GPU-seconds, with no parameter fitting or generated continuations.

This direction-specific rescue supports a correctable background contribution. It does not yet establish exact counting: answers10–16 all begin with the same token1. The reference estimate is question-specific, uses known zero-count training scenes, and is anchored at training length16. It is a diagnostic intervention, not yet an elegant general deployment rule.

The next check will compute reference messages at the actual generated prefix and test complete native integer-plus-EOS answers. A small set of reference streams can share the same model invocation as the actual images, so arbitrary prefixes need no lookup table. That increases computation and still requires valid null references. If this works, a learned neutral representation with a nonlinear decoder becomes a better-supported research direction. The separately verified V11 memory write remains useful for testing persistence, but is not the current explanation for length failure.

[Canonical report](../../outputs/native_aggregation_vlm/v10/background_diagnostic/run_442260/REPORT.md) · [All results](../../outputs/native_aggregation_vlm/v10/background_diagnostic/run_442260/analysis.json)

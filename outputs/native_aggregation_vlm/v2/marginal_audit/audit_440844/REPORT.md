# V2 oracle-marginal sampling-law audit

These are fixed oracle references using known generator probabilities, not VLM results or a proposed method.

| Data cell | n | Joint-marginal MAE | Joint bias | Joint rounded exact | Character-only MAE | Room-only MAE |
|---|---:|---:|---:|---:|---:|---:|
| train_N8 | 90 | 0.575 | +0.025 | 42/90 (46.7%) | 1.139 | 1.089 |
| train_N16 | 90 | 0.922 | -0.117 | 25/90 (27.8%) | 1.944 | 2.000 |
| dev_N8 | 36 | 0.444 | +0.083 | 22/36 (61.1%) | 1.306 | 1.222 |
| dev_N16 | 36 | 1.056 | -0.083 | 11/36 (30.6%) | 1.847 | 2.125 |
| test_N16 | 108 | 1.030 | -0.062 | 32/108 (29.6%) | 2.069 | 2.074 |
| test_N32 | 108 | 1.697 | +0.035 | 24/108 (22.2%) | 2.926 | 3.310 |
| test_N64 | 108 | 2.532 | -0.042 | 13/108 (12.0%) | 3.806 | 4.926 |
| familiar_count_N32_N64 | 216 | 2.115 | -0.003 | 37/216 (17.1%) | 3.366 | 4.118 |
| count_N32 | 64 | 1.176 | +0.223 | 18/64 (28.1%) | 2.531 | 2.711 |
| count_N64 | 64 | 2.078 | +0.219 | 10/64 (15.6%) | 4.430 | 3.836 |
| unseen_count_K9 | 16 | 1.938 | +1.031 | 1/16 (6.2%) | 4.594 | 3.969 |
| unseen_count_K10_16 | 112 | 1.583 | +0.105 | 27/112 (24.1%) | 3.321 | 3.174 |

For each negative frame, the character/room marginal indicator is (1,0), (0,1) or (0,0), each with probability 1/3. Consequently C+R=2K+B with B~Binomial(N-K,2/3). The joint estimator is unbiased with variance (N-K)/8 under the nominal law; each individual estimator has variance (N-K)/2.

At N=4, category counts (match, character-only, room-only, neither)=(2,0,0,2) and (1,1,1,1) both yield C=R=2 but have different K. This illustrates both the predictive shortcut and its inability to identify individual conjunction counts exactly.

All 704 QA hashes, gold counts and manifest semantic counts were independently checked. Per-K metrics, all three estimators' bias/exact scores, empirical error variances and nominal variances are in [analysis.json](analysis.json); sample-level estimates are in [rows.json](rows.json).

## Interpretation limits

- Oracle marginals and known generator probabilities; no images, model features/predictions, model fitting, or proposed inference method.
- No clipping or tuning: continuous estimates are rounded only by floor(x+0.5) for secondary exact accuracy.
- Under the nominal conditional law E[C]=E[R]=N/3+2K/3, so marginal frequencies predict K without observing which frames realize the conjunction.
- Marginals do not determine K exactly; equal-marginal scenes can have different conjunction counts. The audit quantifies a statistical shortcut, not a complete substitute for binding.
- Unbiasedness and variance formulas hold under the nominal IID-negative law before content-exclusion/rejection conditioning. Empirical bias/variance need not equal nominal moments in this finite selected sample.
- Paired lengths share anchors and these descriptive rows are not independent inferential replicates. No confidence or causal claim is made.
- Oracle-reference accuracies must remain separate from competitive model tables. This audit cannot show that any model uses the shortcut.

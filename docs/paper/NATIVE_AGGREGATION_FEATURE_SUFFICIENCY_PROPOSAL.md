# Frozen local-feature sufficiency assay — held proposal

This training-only diagnostic asks whether the cached local state makes the depicted person and room linearly recoverable. It is motivated by the unresolved distinction between representation, aggregation/readout and optimization failure; it does not depend on favorable or unfavorable outcomes from the current four-arm trainability pilot. No implementation, fit or job is released here. The readout will never supply runtime gates, labels, counts or answers.

## Bound inputs and split

Use only the completed `identity_join_learned/features/merge_443031/feature_cache.json` and its `stage_443005/plan.json`, bound to the passed merge proof, original training manifest, renderer atom table, native identity, source ledgers and consumed feature-file hashes. Reconstruct the union of each training scene's first local feature ID from all 6,048 N8/N16 training contexts. Require exact equality with the plan's `local_empty` group: **9,007 unique IDs**, each `kind=local`, `prefix_ids=[]`, with an exact image-SHA/question identity and a finite native FP16 vector of width 3,584. No global state is a probe input; no nonempty-prefix, dev or test feature enters fitting or prediction.

Resolve each image SHA to its unique canonical `(person, room, physical Step)` renderer atom; verify all repeated occurrences agree. Labels come from this bound image metadata, not another model or a scene's answer. Record unknown person/room categories, missing atoms and ambiguous occupants explicitly; any nonzero count fails input validation before fitting, with no dropped rows. All question-conditioned copies of one rendered image stay together. Odd Steps 1,3,...,15 define fitting; even Steps 2,4,...,16 define evaluation. Existing metadata fixes:

| Partition | Unique feature IDs | Unique image SHAs | Questions | Question–person–room cells |
|---|---:|---:|---:|---:|
| Odd / fit | 4,472 | 432 | 12 | 648 |
| Even / evaluate | 4,535 | 432 | 12 | 648 |

The 648 cells are all 12 questions × 9 people × 6 rooms, with 3–8 distinct Step images per cell on each side. Require disjoint image SHAs and all 864 training images/Steps 1–16 covered. Publish exact per-question, person, room, joint person/room and three-way counts before fitting. This holds out rendered images and Step values, not people, rooms, artwork or training-corpus semantics; it is not a fresh task test.

## One fixed readout

For each stored state h, compute **the core's existing FP32 input transform** `x = h.float() / sqrt(mean(h.float()²) + 1e-6)`, then promote x to FP64. Add no question, filename, Step or semantic metadata to x. No column standardization, dimensionality reduction or feature selection.

Concatenate person and room one-hot targets into 15 columns. With m=4,472 equally weighted unique fit rows, solve exactly one ridge objective:

`min(W,b) (1/m) * ||XW + 1bᵀ − Y||²_F + 1.0 * ||W||²_F`.

The loss is averaged over rows, **not over the 15 output columns**. The intercept b is unpenalized. Center X and Y using fit-only means, solve `(XcᵀXc/m + I)W = XcᵀYc/m` in FP64 by Cholesky, and recover `b = mean(Y) − mean(X)W`. There is one fixed lambda, no iterative schedule, restart or hyperparameter search. Predict each marginal by argmax over its 9 or 6 columns; resolve exact ties in the manifest's frozen person/room order. Report person accuracy, room accuracy and their jointly correct conjunction; ridge scores are not probabilities. Predictions are forced choices among the fixed classes, so an unknown prediction is unavailable by construction, not evidence that the model detects unfamiliar categories. Record argmax ties.

Controls are the fit-set majority person/room predictions globally and separately per question, with the same tie rule. These frequency-only controls see no evaluation labels. Primary scores weight each unique feature equally. Also publish original empty-prefix image-occurrence-weighted scores separately for fit and evaluation: 36,288 occurrences each, totaling all 72,576 training image occurrences across 6,048 contexts. Never pool fit and evaluation accuracy or count duplicated occurrences as independent evidence. Keep all mistakes and all strata.

## Exact collisions and interpretation

Before fitting, group raw state hashes within each question and verify any candidate collision by exact tensor equality. Conflicting person/room labels prove that this stored local state cannot distinguish those particular atoms; an irrelevant local distinction need not matter to the join. Separately group training scenes by question, exact global empty-state identity and the **multiset, with multiplicity**, of raw local state identities. Verify candidate tensor equality. Conflicting canonical answers within such a group establish ambiguity for a deterministic permutation-invariant reader of this cached interface. Report group sizes and label counts. Hash uniqueness alone proves no useful accessibility; these checks concern stored FP16 states, not every possible native execution or model layer.

Predeclare the component diagnostic thresholds as **at least 4,490/4,535 correct for person and separately for room**. Strong linear-accessibility success additionally requires **at least 4,490/4,535 jointly correct held-image feature rows**, at least 98% joint accuracy within every question, at least 95% within every person/room combination, and an overall joint advantage of at least 50 percentage points over the question-frequency control. Report all 648 three-way cells without filtering. Every threshold is diagnostic only; none releases native training or certifies method efficacy. Passing supports recoverable atomic bindings on this panel and redirects attention toward composition/readout/optimization; it does not establish capacity or trainability of the rank-96 native adapter. Exact conflicting collisions identify specific unavoidable ambiguities. Failure of the ridge thresholds without such collisions is inconclusive about missing information and licenses no automatic probe-capacity sweep.

## One bounded CPU execution

Proposed budget: **one CPU job, four cores, 16 GB RAM, 600 allocated seconds (10 minutes); zero GPU/model/vision/head calls**. Freeze source and resolved input/label/split inventories before the solve. Save fit-only means, coefficients, all predictions, coverage/collision records, objective and linear-system residual, source/file identities and stage timings. Nonfinite results, failed Cholesky or exhaustion preserve partial evidence and stop; no alternate solver, penalty, split or automatic retry. Numerical work and any eventual release require the parent's separate authorization.

# A finite native identity-join experiment

This is the proposed efficacy experiment after the [fixed inclusion audit](NATIVE_AGGREGATION_IDENTITY_JOIN_PROPOSAL.md). **Fitting remains held** until the complete gate audit, data checks, native runtime check, feature inventory and measured resource profiles pass. It tests whether local contents support answers that an exact relevance count cannot determine. It does not establish a new pooling operator, a minimum dimensionality, or reasoning composition.

The two conditions use the same frozen Qwen backbone and the same 1,041,600 trainable parameters, with matched seeds 22 and 23. The vector branch sums binary-gated local payloads. The scalar branch computes the binary count exactly, then multiplies one payload derived only from the current global state. Both execute the same dense projection shapes. Store the condition explicitly with each checkpoint; never reconstruct the scalar aggregate by summing its descriptive per-item messages.

| Split | Room pairs | Trios per pair | Families | Lengths | Contexts |
|---|---:|---:|---:|---|---:|
| Train | 12 | All 84 | 1,008 | 8, 16 | 6,048 |
| Development | Same 12 | 3 | 36 | 16 | 108 |
| Seen-pair test | 3 | 12 | 36 | 32, 64 | 216 |
| Held-pair test | 3 | 12 | 36 | 32, 64 | 216 |

Each family contains all three answer-changing variants. K is always six. The exact pairs, balanced trios, canonical rendering and insertion law are specified in the proposal and append-only registration. Data seed is 20261122. The test has 108 contexts per length and regime, including 12 answers for each name. No test state enters a training cache.

Train for 12 complete epochs with batch 16 containing eight same-variant N8/N16 pairs: 4,536 updates and 72,576 scene presentations per condition. Use the native name-plus-EOS target at every strict causal prefix, scene-mean cross entropy and coefficient-one residual consistency between the two lengths. Consistency divides the squared residual difference by the frozen global-state squared norm plus 1e-6. Do not impose consistency across answer-changing variants. The independent scorer must recompute the exact target-position inventory after tokenizer inspection.

Keep the existing AdamW recipe: learning rate .001, 50-step warmup, cosine decay to 1e-5, no weight decay, gradient clipping at one. Save only the fixed final endpoint for inference, verify its actual deployed tensors, then run the 108-context development set once descriptively and all 432 test contexts. Development performance cannot select, extend or replace a fit. The scalar control has deliberately insufficient information and does not need a high development score to be a valid control.

Generate the complete canonical name and native EOS greedily within four tokens, using the full unmasked vocabulary. Permit surrounding whitespace only. Invalid local measurements, malformed names, extra text, nonterminal special tokens and truncation remain failures in the denominator. Preserve all raw outputs. A training measurement failure stops fitting; a test measurement failure is retained and invalidates any strict scalar-ceiling claim for its family.

Proposed primary screen: in **each seed**, the vector condition exceeds the matched scalar condition by at least 18/108 whole answers at N64 in **both** seen and held regimes; at N32 it must exceed the scalar condition by the same amount. Proposed practical target: in each seed, vector accuracy reaches at least 98/108 in the seen regime and 87/108 in the held regime at **both** lengths. These thresholds must be registered before any fit, without later adjustment. Report complete-triple accuracy, per-name results, actual gate correctness and all control-input identity violations regardless of the screen.

A correct binary count and identical global state imply identical scalar inputs within each contrast triple, hence at most one correct answer per triple. The global question also omits the selected trio: if global states are identical across all families sharing a room pair and length, balanced names tighten the scalar ceiling to one ninth. Verify actual inputs before claiming either ceiling. The larger comparison remains a content-versus-count attribution test, not a sufficient competitive baseline.

Use 10,000 bootstrap resamples of complete families, stratified by room pair, with the same draws across lengths, conditions and the two fixed seeds. Report conditional example uncertainty; two seeds do not estimate the training-seed population reliably. Preserve all three variants and both lengths in every resampled family.

The proposed campaign ceiling is 18,000 GPU-seconds including every failed or zero allocation, with at most four concurrent project GPUs. Reserve 900 seconds for measurement/software prerequisites, 300 for a feature profile, four 1,200-second cache shards, two 300-second training profiles, and four 2,700-second mains; this leaves 600 seconds contingency. These are ceilings rather than measured estimates. The already completed one-second CPU core suite is not GPU cost.

Feature profiling must measure the complete deduplicated strict-prefix inventory before cache release. Training profiles use 32 updates per arm and nine training-derived timing trajectories: three variants at each of N16, N32 and N64, with no fresh test outcome. Pool worst native trajectory timings across conditions and variants. Each main must satisfy:

`setup + 1.25 * (first4_steps + 4532 * max_steady_step + 108*T16 + 216*T32 + 216*T64) + 120 <= 2700 seconds`.

All prior allocations plus full reservations must also fit the campaign ceiling. Native origin probes, ordinary heads, same-state audit replays and every generated token must be included in the measured costs. If a prerequisite or cost gate fails, stop this registered route before fitting and retain the failure.

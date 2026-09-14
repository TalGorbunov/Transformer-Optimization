# Learned selection for the native identity join

**Separate held proposal; no fitting or GPU release.** The frozen binary-prompt route remains stopped: its complete audit produced 1,122/4,860 valid correct judgments, 3,738 invalid outputs and no literal-one judgment on the 1,620 included cases. Native numerical verification passed, so this was a measurement-interface failure. Do not reinterpret the invalid names as gates or search another binary prompt on those outcomes.

This proposal removes the measurement interface. Every local stream receives its image and the **unaltered global join question**, with no binary wrapper. A learned score selects contextual vector values using only native whole-answer supervision and the existing length-pair objective. The question is whether an exact-zero region improves useful association aggregation over a smooth gate and standard normalized attention. This is a comparison of known primitives, not a new attention operator. Conditional elementwise transformation followed by summation already has the Deep Sets form. [Deep Sets](https://arxiv.org/abs/1703.06114), [join task and limits](NATIVE_AGGREGATION_IDENTITY_JOIN_CLAIM_LIMITS.md).

## Three fixed operators

Retain the existing question-conditioned 96-dimensional tanh payload `v_i`, SUM aggregate, SiLU residual readout, native FP16 cast-before-add and frozen native norm/head. Add one shared score `s_i = w · v_i + b`: 96 weights and one bias, bringing each parameter table to **1,041,697 FP32 parameters**. Initialize `w=0`, `b=0.5`, and the residual output projection U to zero. At every actual causal query, recompute payloads and scores from that query's local/global states; there is no frozen original gate, extra vocabulary-head probe, reference bank, predictor or local-label input.

| Condition | Item weight | Aggregate |
|---|---|---|
| `learnedclip` | `clamp(s_i, 0, 1)` | `SUM_i weight_i * v_i` |
| `matchedsigmoid` | `sigmoid(4 * (s_i - 0.5))` | `SUM_i weight_i * v_i` |
| `standardsoftmax` | `softmax_i(4 * (s_i - 0.5))` | `SUM_i weight_i * v_i` |

Clip and sigmoid start at the same weight 0.5 and derivative one with respect to the score. This matching is **local to initialization**: their interior curvature, saturation and gradient behavior later differ. Softmax starts at 1/N and is deliberately a normalized baseline, not an initialization-matched SUM. All native logits initially match through zero U; first output-projection gradients need not match softmax's different preactivation.

The shared score bias cancels analytically in softmax. Keep it in the declared parameter table, but disclose that one redundant degree of freedom and exempt its analytically zero gradient from nonzero-gradient checks. Score weights remain trainable. Clip has open regions with exactly zero or one weight. Sigmoid is strictly between them in real arithmetic; actual FP32 underflow/rounding can produce zeros or ones and must be reported. Neither gate construction identifies irrelevant images by itself.

## Data, supervision and fixed endpoint

Use exactly the registered [join matrix](NATIVE_AGGREGATION_IDENTITY_JOIN_EXPERIMENT.md): 6,048 training contexts at N8/N16, 108 N16 development contexts, and 216 seen-pair plus 216 held-pair test contexts at N32/N64. Preserve all three answer-changing variants, six relevant images, person/room marginals, occurrence slots, background insertions, balanced names/trios and complete-context disjointness. Data seed remains 20261122. The symbolic data law can be reused; publication and new Qwen feature preparation require an explicit new release without the retired binary-gate dependency. Preserve the failed audit rather than weakening its verifier.

Use seeds **22 and 23**, matching initial parameter bytes, paired presentation order and optimization across all three arms. Train 12 complete epochs with eight same-variant length pairs per batch: **4,536 updates, 72,576 scene presentations**, and 161,280 native target positions under the inspected name tokenizer. Apply full name-plus-EOS CE, averaged within each scene and then across scenes, plus coefficient-one paired residual consistency normalized by the identical frozen global-state squared norm plus 1e-6. Consistency applies only to a variant's N8/N16 pair, never across its different-answer variants. There is no local gate supervision, sparsity penalty, path loss or ancillary classifier objective.

Keep AdamW at .001, 50-step warmup, cosine decay to 1e-5, weight decay zero and clip norm one. Save the fixed final checkpoint only. Evaluate development once descriptively, then all tests, with no selection, extension or replacement. Generate an unmasked canonical name plus native EOS within four tokens; surrounding whitespace is permitted, aliases/partial names/extra text/nonterminal specials/truncations are failures. Every case remains in its denominator.

## Prospective decisions and evidence

The primary screen requires, **in each seed and each seen/held regime**, at least **6/108 additional N64 whole answers** for clip over **each** control. At N32, clip may lose no more than **3/108** against either control. The practical criterion independently requires clip to reach **98/108 seen and 87/108 held**, at both lengths in both seeds. Passing only one control or regime does not pass the primary screen. If all arms work similarly, that supports the shared pipeline without a distinctive clipping advantage.

Report all errors, per-name results, complete-triple correctness, Step strata and control comparisons. Use 10,000 complete-family bootstrap resamples, seed 20261124, stratified by room pair; share each draw across arms/seeds and preserve all three variants and both lengths. Intervals describe these examples conditional on two fixed seeds, not a well-estimated population of training seeds.

Retain per-query scores, weights, payloads, messages and native outputs. Only offline, use the known room membership to partition positive P and negative Z contributions. Report closed positives/negatives, saturation, their norms and projected contributions for every case. These records can establish an exactly zero **current** message and absence of an ungated value path. They cannot establish that clipping caused an accuracy gain: curvature, optimization and saturation also differ. Earlier emitted tokens may already contain evidence from an image that is closed now. Without a separately registered causal intervention, describe the gate geometry as an association, not the demonstrated mechanism of improvement. No exact-count scalar ceiling applies to these learned weights.

## Risks and bounded release

Answer CE and endpoint consistency may still learn cancellations or short-length shortcuts instead of irrelevant-item suppression. Closed clip units have zero score gradient; premature closure can discard needed evidence. Sigmoid may retain small coherent distractor contributions. Softmax bounds aggregate magnitude for bounded values but can dilute useful evidence as distractors increase. With six relevant items fixed, this task tests association preservation and distractor robustness, not increasing information bandwidth or reasoning composition.

The proposed ceiling is **24,000 GPU-seconds** including failures: six mains at 2,700 seconds, feature preparation at 5,100, three training profiles totaling 900, software totaling 900, and 900 contingency. Maximum four project GPUs. New prompt-specific caches and runtime checks are required; old binary artifacts are ineligible. Before main release, pool conservative measured timings across all arms and reserve every remaining allocation. Each main must satisfy `setup + 1.25*(first4_steps + 4532*max_steady_step + 108*T16 + 216*T32 + 216*T64) + 120 <= 2700`. Include actual native calls, captures, I/O and failed allocations. A failed prerequisite or cost gate stops this route; it does not authorize another prompt, coefficient or training sweep.

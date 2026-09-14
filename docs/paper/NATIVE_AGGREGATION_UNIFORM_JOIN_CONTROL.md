# Held: fixed-uniform join control

**Status: conditional design only; no implementation or execution released.** Proceed only if the completed independent report for all four seed24 optimization-pilot arms is computationally valid and every arm fails the already registered training-fit criterion. An invalid or incomplete run does not satisfy that trigger. Preserve every pilot result, including a near-threshold failure. This proposal was written while those fits were running, without inspecting their outcomes.

The current join experiment has fitted clip, sigmoid and softmax selection. Its bare/zero-output software checks are not trained ungated baselines. V18 tested all-open aggregation on counting with different prompts and data. The missing join control is a question-conditioned DeepSets map with a fixed uniform item weight.

## One change

For every actual item and every current assistant-query position, use

\[
p_i=\tanh(W_l\operatorname{RMS}(h_i)+q+b_l),\quad
z=\tfrac12\sum_i p_i,\quad
\delta=U\operatorname{SiLU}(W_{agg}z+q+b_{agg}).
\]

Padding contributes exactly zero. Keep the existing question-conditioned payload, SUM, nonlinear readout, frozen native FP16 head/norm, cast-before-add, and unaltered local/global question. The factor one half preserves the existing clip/sigmoid initialization exactly; it is an ordinary ungated SUM up to a constant absorbable into the aggregation projection. With the same tensors and first batch, the aggregate, preactivation and initial gradient of `U` must match the pilot's CE-only arm. No scalar selector is optimized.

The smallest implementation can reuse the immutable sigmoid core and runtime: retain `selection_weight=0` and `selection_bias=.5` as frozen constants throughout optimization and inference. Its gate is then exactly `.5` on valid items, even as the payload changes. Record the experimental condition as `uniform_half_ce` and the underlying runtime mode as `sigmoid`; audit the actual frozen selector bytes and every captured gate. Do not relabel it as adaptive selection. The state table may retain **1,041,697 coordinates**, but only **1,041,600 are trainable**; the 97 selector coordinates are unused as degrees of freedom. This is not a matched-capacity claim.

## Frozen recipe and observations

Reuse the pilot's exact hash-bound 108 training contexts, 54 intact N8/N16 pairs, six questions, three person trios, and 54 development contexts. Keep seed24, identical initial core tensors, the same concatenated shuffled-pair order, eight pairs per update, and **600 fixed updates**. AdamW remains lr `.001`, 50-step warmup, cosine decay to `1e-5`, weight decay zero, and global trainable-gradient clipping at one. Optimize the unchanged mean-scene mean-token native name-plus-EOS CE. Residual consistency remains a diagnostic with coefficient zero. No local labels, teacher gates, output masks, new images, or parameter search enter training.

Retain first-query/continuation/EOS losses, all gate statistics, and the fixed diagnostic steps 1, 2, 32, 128, 300 and 600. Gradient audits must distinguish frozen selector constants from trainable tensors, rather than inventing trainable selector gradients. Verify the fixed final checkpoint by reset and restricted reload. Then generate once on all 108 training and 54 development contexts, unmasked, with the same maximum four tokens and strict whole-name/EOS scorer. Retain all raw outputs and failures.

Keep both criteria unchanged: training requires first-token and whole-answer correctness each at least **103/108**, plus **16/18 complete six-context families**; development requires each at least **49/54**, plus **16/18 complete triples**. Computational acceptance is separate from these diagnostic outcomes. Development is reused, not fresh confirmation; no test/extrapolation evaluation is authorized.

## Release and source scope

No new architecture-wide native software campaign is needed for the unchanged sigmoid runtime. Retain **one 150-second profile**: 32 updates using this actual frozen-selector route, then the existing nine training-derived N16/N32/N64 timing cases, without accuracy scoring. It checks the changed optimizer/gradient contract and measures actual work; it does not select a checkpoint. Require an independent CPU release using the existing formula

`setup + 1.25*(first4 + 596*steady_max + 162*T16) + 120 <= 840`.

Allow one **840-second main**, a separate **990 allocated GPU-second campaign cap including every failed/zero allocation**, and at most one GPU for this control. No automatic retry or budget extension. CPU preparation/reporting use four cores, 16 GB and five minutes each. Reuse native/source identities and subset artifacts; the N64 timing images remain training-derived extensions. No new benchmark trajectory is introduced by profiling.

Before any job, create new control orchestration, independent reporter, small frozen-selector/algebra/gradient tests and CPU/profile/main/report wrappers. Version only the condition, selector trainability, optimizer/gradient audit handling, source/output identities and resources. Reuse immutable core/runtime/loss/evaluation helpers where their contracts apply; never edit pilot files or bypass a guard. Bind the completed four-arm report and its hashes as the conditional release input.

Success would show that this fixed recipe can learn the selected association task without learning selection, and would support gate/payload coadaptation as a contributor to the failed learned-gate fits. It would not establish a general benefit, new capacity, or improved reasoning. Failure would leave payload representation, native readout and optimization limitations unresolved. Repeated question components and cardinality-dependent background accumulation remain in this control; neither outcome identifies or removes that separate nuisance.

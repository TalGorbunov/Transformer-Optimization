# Learned conditional null means: contingent proposal

This is a design proposal, not a training release. The V10 background diagnostic supports a correctable accumulated background component at the first answer token. The verified V12 dynamic-reference continuation test now shows a real but partial complete-answer rescue: N64 background gives10/17,12/17,12/17,12/17 across the four cores, versus1/17,0/17,3/17,4/17 for their matched bases. Across those fixed models, K0–9 contributes39/40 correct answers but K10–16 only7/28. These are exploratory outcomes on17 reused families, not independent fresh confirmation. Neither result establishes a general method, an unseen-query benefit, or reasoning composition. The current evidence and reference-bank assumptions are documented in [the null-reference design](NATIVE_AGGREGATION_NULL_REFERENCE_DESIGN.md).

## Minimal operator

Keep the existing conditional message map, SUM, nonlinear readout, native vocabulary, and one parallel native forward per generated token. Let `q = Wq RMS(g)` be the current 96-dimensional global query, and predict the mean *message* of irrelevant images directly:

\[
 c_\eta(q)=C_2\operatorname{SiLU}(C_1q+d_1)+d_2,\qquad
 z_c=\sum_{i=1}^{N}\phi(h_i,g)-N c_\eta(q).
\]

The original decoder consumes `z_c`: `delta = U SiLU(Wagg z_c + b_rho + q)`, followed by the unchanged native addition, final RMS and vocabulary head. Training begins in these centered coordinates. There is no anchor16, reference stream, count-specific decoder, output mask, or table of answer prefixes at inference. Current `g` supplies the actual generated prefix.

Two 96-by-96 matrices and two biases add18,624 parameters, giving1,060,224 total branch parameters. A linear map from the full3584-dimensional global state would add344,160 parameters and change the capacity substantially; it should not be silently substituted if the smaller predictor fails. Using the existing query is the minimal first hypothesis. It also ties calibration to a trainable query representation, which must be acknowledged.

Do not replace the current zero-hidden baseline with a learned pseudo-null image state and assume that this estimates the mean. In general, `E[SiLU(a+q)]` differs from `SiLU(E[a]+q)`. The predictor targets the nonlinear message mean itself.

The current message subtracts `SiLU(q+b_local)` as its zero-hidden origin. Exact centering cancels that arbitrary baseline: `phi - E0[phi] = SiLU(a+q+b_local) - E0[SiLU(a+q+b_local)]`. The proposed predictor therefore needs a signed output, supplied by its final linear layer. The mechanism is a learned change of origin for the additive statistic.

For fixed query/prefix, under the generator's mixture law and accurate calibration,

\[
 \mathbb E[z_c\mid K,N]=K(\mu_1-\mu_0).
\]

This removes the expected negative contribution's dependence on N. It does not remove negative variance or guarantee exact counting: independent negative fluctuations still accumulate, while a conditional-mean error `e` introduces bias `-N e`. This is calibration of an additive statistic, not a new attention operator or a neutrality theorem.

## Necessary training-only prefix coverage

Ordinary K0 teacher forcing supplies null examples only at the empty prefix and prefix `0`. It cannot establish calibration at positive-count continuations. The V10 answer support has18 unique strict prefixes: empty plus numerals0–16. Across54 questions this is972 global states.

For the auxiliary objective only, evaluate K0 training images under these independently assigned prefixes. Their known irrelevance comes from the existing zero-count bag label; no new frame labels are needed. Never apply zero-answer CE to an image/prefix combination inconsistent with its ordinary answer sequence. The ordinary count/EOS CE stream remains unchanged.

The existing24 K0 image occurrences per query require at most54×24×18 =23,328 occurrence-feature positions before deduplication and reuse. These are training-only frozen native features. They replace runtime reference computation; they are not a lookup table used by the inference operator. Balanced prefix sampling should be fixed before outcomes, so a prefix does not receive an auxiliary weight merely because it appears in more answer sequences.

This supplies finite answer-prefix coverage, not general reasoning-prefix coverage. Native global representations may generalize beyond it, but that remains a separate hypothesis. The same54 questions also permit memorization despite a continuous input. Fresh nuisance ensembles and held-out question combinations are necessary before claiming query generality.

## Two bounded comparisons, with different purposes

The smallest diagnostic can first freeze each existing V10 core and fit only `c_eta` to its training null means. Compare its anchored correction `(N-16)c_eta` against the actual dynamic reference mean and the matched augmented baseline, using identical frozen cores. Fixed messages prevent branch collapse and isolate whether a small query-conditioned predictor can approximate the useful reference statistic. This remains an anchored frozen-model diagnostic; it neither establishes anchor-free training nor licenses a larger predictor after seeing test results.

The subsequent method comparison, if justified, trains exactly two matched arms from the same initialization:

| Arm | Aggregate entering the existing decoder |
|---|---|
| Query-offset control | `sum(phi) - c_eta(q)` |
| Centered SUM | `sum(phi) - N*c_eta(q)` |

Both have the same active predictor, ordinary CE, null examples, auxiliary objective, data order, optimizer budget and selection rule. The control makes the additional predictor useful as an ordinary query-conditioned offset. Only its multiplication by set cardinality differs. An unmodified smaller branch alone would confound centering with added parameters and auxiliary data.

A concrete calibration target is the live bag mean `mu = mean_j phi(h_j^0,g)`. Regress `c_eta(q)` toward `stop_gradient(mu)`, with a fixed prospective loss weight and a documented normalization. A relative mean error can divide by the detached null-message second moment plus a fixed numerical floor. Report raw errors as well. Common rescaling is approximately canceled away from that floor; neither this normalization nor target detachment prevents all coadaptation or collapse.

For the first trained method comparison, keep both ordinary CE and residual consistency fully differentiable through `c_eta(q)` and the shared query. Restrict only the auxiliary loss to the predictor: evaluate it as `c_eta(stop_gradient(q))` against `stop_gradient(mu)`. Thus the main losses optimize the actual deployed forward, while null supervision calibrates the predictor without directly forcing the original messages or query toward zero. Use the same coefficient1 residual-consistency objective in both arms. The predictor may still trade calibration for predictive utility; report that rather than calling it an exact null estimator.

A stricter calibration-only alternative also blocks the main losses through the predictor. That is not the default recommendation for a moving core. An update that adds a query-only component to every message can produce an N-scaled forward change until the private predictor catches up, although ideal instantaneous centering would cancel it. Such follower lag may create the very length failure being studied. Frozen-core mean fitting can assess approximation without this confound; sending the auxiliary loss through live null messages instead directly reshapes their representation. These alternatives should not become a grid. Fix the chosen routing before execution. If predictor and original-core clipping/optimizer groups are separated, apply exactly the same rule in both arms and report the inevitable difference between N-scaled and unit-scaled main gradients.

Track message/predictor norms, their projected error `Wagg(c_eta-mu)`, native carry scale, count CE and per-prefix calibration. A small auxiliary loss alone is not success: branch/readout rescaling can conceal errors, and CE count variation constrains rather than proves noncollapse.

## Decisive falsifier and limits

The central prediction is that learned centering lowers the held-out negative mean in the decoder's actual coordinates and improves *complete integer-plus-EOS* extrapolation over the equally sized query-offset control, across fixed seeds. Measure both on fresh scenes; retain every K and every prefix. The N64 test must keep maximum training length16.

If the predictor matches independent null means but complete answers do not improve over the matched control, null-mean removal is insufficient as the proposed repair. If errors instead grow with N because the predictor misses those means, the small conditional-calibration hypothesis fails; this does not establish absence of usable image information. Improvement restricted to the54 trained questions or registered numeric prefixes cannot support general aggregation or reasoning-composition claims.

Zero-count bags supply a valid all-irrelevant reference here. In a task where a zero aggregate can arise through cancellation, that implication does not hold. A broader method must specify how null examples are obtained and validate changed nuisance distributions. No training, cache harvesting, new test selection or additional model size is released by this proposal.

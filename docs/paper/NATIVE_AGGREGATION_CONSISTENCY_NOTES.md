> Status update2026-09-11: V8 verified complete, both relative criteria pass but practical N64 criterion fails both seeds. See [verified results](NATIVE_AGGREGATION_VISION_V8_RESULTS.md). The prospective reasoning below is preserved.

# Candidate training change after V7

Status: registered V8 comparison running (2026-09-11), no independently verified V8 efficacy result yet. See [transfer criteria](NATIVE_AGGREGATION_TRANSFER_CRITERIA.md) and [execution ledger](../../outputs/native_aggregation_vlm/v8/execution.json).

V7 learns correct native outputs for every16-frame test and most32-frame tests, but fails at64. Its centered message map makes raw zero padding neutral; it does not make an irrelevant image neutral. A fixed frozen-state diagnostic will test whether unfamiliar visual inputs are necessary for a synthetic version of this failure.

If that diagnostic supports merger drift, the next bounded comparison will keep the entire deployed operator unchanged and test answer-equivalent evidence consistency. Pair training scenes with the same question and count, one at8frames and one at16. Keep count and EOS-prefix positions separate. With native residual outputs delta8 and delta16 and identical frozen global state g, add

```
L_cons = mean_pairs,positions ||delta8 - delta16||² / (||g||² + 1e-6)
L_total = native_answer_CE + L_cons
```

The control receives exactly the same paired examples/order and uses CE alone. No local classification labels, oracle selection or additional inference work are introduced. The numerator lives after all learned projections, avoiding a trivial latent shrinkage/compensating-readout escape. The frozen denominator fixes its scale. CE remains necessary to prevent an uninformative constant residual.

Under a stationary two-class toy model, expected SUM is K*mu_positive+(N-K)*mu_negative. Equal-count differences across set sizes reveal the background mean. Actual Step-labeled inputs need not satisfy stationarity. Matching residuals after a nonlinear decoder also does not prove the local negative mean is zero, or guarantee transfer outside the trained lengths. Unseen count values remain a separate failure; this objective is not assumed to repair them.

This is a known consistency-training family, not a new attention operator. [SizeShiftReg](https://arxiv.org/abs/2207.07888) regularizes representations under simulated graph-size changes. [Representation Learning by Learning to Count](https://arxiv.org/abs/1708.06734) learns representations through counting-related transformation constraints. [From Local Structures to Size Generalization in GNNs](https://proceedings.mlr.press/v139/yehudai21a.html) studies why a size-capable architecture need not learn the size-generalizing solution. These are positioning anchors, not evidence that the proposed native residual objective works.

The distinction between sampled consistency and an aggregation law matters. In a toy representation z=(K,N), the readout could implement delta(K,N)=d(K)+v*f(N), where d correctly decodes familiar counts and

```
t = (N - 12) / 4
f(N) = SiLU(t) + SiLU(-t) - SiLU(1) - SiLU(-1)
```

This nuisance term is expressible by the existing SiLU readout. It is exactly zero at N8 and N16 but approximately12.54 at N64. Equal training residuals therefore do not guarantee extrapolation, even without a complex memorization construction. This is a mathematical counterexample, not a description of a fitted checkpoint.

A stronger sufficient property is that every irrelevant item's message is zero, and the relevant-item messages stay unchanged when distractors are inserted. The sum, residual, native normalization and vocabulary output then remain unchanged for arbitrarily many such distractors at the same query. For approximate neutrality, adding M messages of norm at most epsilon perturbs the sum by at most M*epsilon. A bound on downstream sensitivity and an answer margin are additionally needed to turn this into output robustness. V7's raw-zero centering supplies none of these guarantees for real irrelevant images; changing Step labels can also change relevant-item states.

Even an exact neutral-element law preserves only the readout's existing answer mapping. Native decoding of unseen count values and operation under later reasoning prefixes require separate evidence. These distinctions motivate the fixed-state comparison and the proposed answer-value holdout; neither diagnostic changes the registered V8 acceptance criteria.

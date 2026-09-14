# Readout-only query: the trainability screen still fails

Independent [report 443418](../../outputs/native_aggregation_vlm/identity_join_readout_query/report_443418/REPORT.md) passed its computation and provenance audits. **The single query-placement ablation failed the registered training screen:** 80/108 cached first tokens and 5/18 complete families, compared with the frozen parent's 79/108 and 5/18. The required thresholds were 103/108 and 16/18. This architecture branch ends; no native evaluation, continuation or further variant is released.

All 108 training contexts remain in the denominator: 18 families, each containing three answer-changing variants at N8 and N16. Only first queries are scored; these are neither full-name completions nor held-out results.

| Product model / factor pairing | Correct first tokens /108 | Complete families /18 | Mean first-token NLL | Paired screen |
|---|---:|---:|---:|---|
| Frozen parent, original pairing |79|5|0.868483189501582|FAIL|
| Readout-only query, original pairing |80|5|0.8611090023408852|FAIL|
| Frozen parent, fixed cyclic permutation |38|0|1.402758027895851|Descriptive only|
| Readout-only query, fixed cyclic permutation |11|0|3.273114197614856|Descriptive only|

The new run reused the exact **unfitted** seed 24 parameter packet from factor check 443373, with 1,385,760 trainable parameters and 97 frozen selector coordinates. Both models started with the same native output because U was zero. Internal factors, aggregates and the first U gradient can differ. The only architectural change removed the explicit global query from both local tanh factors while retaining it after the SUM in the SiLU readout. Local hidden states still depend on the question and prefix.

Training otherwise matched the parent: fixed global conditioning statistics, the original 4,800 pair presentations, 600 mean-scene/mean-token full-name-plus-EOS CE updates, and residual consistency coefficient 0. Each fit consumed 21,334 target positions. The fixed step 600 checkpoint was reset and restricted-reloaded before evaluation; no checkpoint was selected by accuracy. The new final minibatch CE was 0.437431, compared with the parent's 0.414147; these minibatch losses are distinct from the all-context first-token NLL above.

Every transition is retained below. “Correct” refers to the first token, and each row totals 108 contexts.

| Comparison, before → after | Wrong → wrong | Wrong → correct | Correct → wrong | Correct → correct |
|---|---:|---:|---:|---:|
| Parent paired → new paired |21|8|7|72|
| Parent cyclic → new cyclic |68|2|29|9|
| Parent paired → parent cyclic |22|7|48|31|
| New paired → new cyclic |25|3|72|8|

The new model therefore corrected 8 parent errors but introduced 7 others, with no complete-family gain. Its predetermined cyclic factor-B permutation lost 72 correct predictions and recovered 3. Exact original-factor/query equality across paired and permuted execution was verified. This establishes sensitivity to within-item factor pairing in the fitted model; it does not assign person/room semantics to the factors, demonstrate disentanglement, or rescue the failed screen.

The preceding [geometry audit](../../outputs/native_aggregation_vlm/identity_join_factor_binding/query_geometry_443401/summary.json) found lower saturation and larger local payload partial derivatives when the explicit query was removed from the frozen parent's final factors, across all 216 scene/factor rows. That was a pointwise counterfactual on saved parent states. Retraining the altered architecture produced only one additional correct first token and the same five complete families. The geometry observation therefore did not yield successful fitting under this fixed recipe. It does not establish saturation as the cause of the remaining errors or show that saturation is irrelevant; optimization changes the learned states and readout together.

All 20 captured training/final batches passed the prescribed functional checks. Independent CPU native-head replay covered 14 final batches and 216 rows, with **216/216 exact argmax matches** and maximum full-vocabulary TV **0.003819364123046398**, below the unchanged 0.02 gate. FP32 functional checks retained atol/rtol 1e-4; FP64-reference NLL checks retained 2e-6. The frozen parent was joined through its passed report, initial/statistics/order identities, saved predictions and raw file hashes, without loading its tensors or repeating its head calls.

The one GPU attempt, job 443416, completed in **28 allocated GPU-seconds**, within the 90-second cap and maximum one GPU. It used 614 core calls, 614 norm calls and 614 head calls over 21,550 rows, including the fixed permutation diagnostic; VLM and vision calls were zero. There were no failed GPU attempts in this ablation. Independent reporting was CPU-only. The result establishes neither complete native-generation equivalence, whole-answer accuracy, generalization, increased communication capacity, a new attention primitive, nor reasoning composition.

Provenance: [proposal](NATIVE_AGGREGATION_READOUT_QUERY_PROPOSAL.md), [CPU plan](../../outputs/native_aggregation_vlm/identity_join_readout_query/check_443413/plan.json), [summary](../../outputs/native_aggregation_vlm/identity_join_readout_query/report_443418/summary.json), [analysis](../../outputs/native_aggregation_vlm/identity_join_readout_query/report_443418/analysis.json), [parent comparison](../../outputs/native_aggregation_vlm/identity_join_readout_query/report_443418/parent_query_placement_comparison.json), [reporter](../../scripts/report_native_identity_join_readout_query.py), [core](../../gnnformer/parallel_local_readout_query_binding.py). The report binds 108 source files, the fixed endpoint, all inputs and raw captures. The [factor-binding results](NATIVE_AGGREGATION_FACTOR_BINDING_RESULTS.md) document the parent comparison and bilinear prior art.

Summary SHA256: `fea7ecb1a2f71a943aa126e1f395163a91e2f7ca8bc189c8a2d4ebb1374733a2`.
Analysis SHA256: `06f544d0e046be13dda1dd13aa2c41f2cafd4e91f707858bac2f687dd9cc2235`.

# Saved local-code readout geometry

CPU diagnostic443772 completed in4 allocated CPU-seconds (3.976 measured seconds), with zero core, norm, head, backbone, backward or optimizer calls. It examined44 saved captures from the216-context original-order code fit443716 and paired-order fit443735: eight pre-update batches and fourteen final6000 batches per run. The988 query occurrences comprise432 final first queries and, per run,128 training first queries,128 EOS queries and22 name-continuation queries.

The table describes all216 final first queries per run. Norms are means over queries. Each derivative fraction has20,736 coordinates (216×96) as its denominator; derivatives are evaluated at the actual saved preactivation.

| Code fit | Mean query norm | Mean projected aggregate norm | Mean preactivation norm | Small absolute slope | Small absolute curvature |
|---|---:|---:|---:|---:|---:|
| Original order443716 |96.054|12.210|96.909|11,456/20,736 (55.25%)|12,360/20,736 (59.61%)|
| Paired order443735 |138.870|10.932|139.401|16,706/20,736 (80.57%)|18,048/20,736 (87.04%)|

The fixed bins are `abs(SiLU') <= .01` and `abs(SiLU'') <= .01`. Near-unit slope, separately defined by `abs(SiLU'−1) <= .01`, occurs in656/20,736 coordinates for the original order and1,058/20,736 for the paired order. Small curvature alone does not imply inactivity. Mean bias norms are only0.058 and0.061. The maximum discrepancy between saved preactivation and the FP64 decomposition from saved query, aggregate matrix and bias is9.93e-7 and1.87e-6, respectively.

This operating regime develops during training: in the saved pre-update1 first-query batches, mean query norms are5.521 and5.490, and neither run has any coordinates in the small-curvature bin. The two initial batches contain different scene occurrences. Later snapshots are also different minibatches, so these are descriptive training observations rather than controlled trajectories on an identical cohort.

The findings support a specific concern: the learned query term places many readout coordinates in regions with little local slope or curvature, while it is substantially larger than the projected aggregate. They do not show that this causes the failed join fit. Some coordinates retain nonlinear response; output weights and feature directions matter, and a large norm alone is not a capacity or gradient-flow certificate. No query-removed input, altered architecture or intervention score was evaluated. The pending first-query-only supervision control remains a separate test. These results alone do not establish that moving the query outside SiLU would restore learning.

The original report443718 remains numerically failed; this diagnostic uses its complete saved core/NLL-audited captures without accepting its failed native-head comparisons. Report443742 passed its computational audit, while its training screen failed. Neither result is changed, and no native or fresh evaluation is released.

Independent review checked all3 new and187 inherited source hashes,47 output JSON hashes,26 small input bindings, all988 query labels and103 strata. Counts and extrema matched exactly; recomputed descriptive means agreed within floating summation roundoff (maximum absolute difference2.71e-13). Tensor artifacts and large order files were not reloaded or rehashed in this lightweight review; their bound execution evidence is preserved in the diagnostic.

Evidence: [protocol](NATIVE_AGGREGATION_CODE_READOUT_GEOMETRY_PROPOSAL.md), [summary](../../outputs/native_aggregation_vlm/identity_join_code_readout_geometry/geometry_443772/summary.json), [analysis](../../outputs/native_aggregation_vlm/identity_join_code_readout_geometry/geometry_443772/analysis.json), [all query rows](../../outputs/native_aggregation_vlm/identity_join_code_readout_geometry/geometry_443772/rows.json), [strata](../../outputs/native_aggregation_vlm/identity_join_code_readout_geometry/geometry_443772/strata.json). Summary SHA256: `27851411d7cf6977cb53d4f94149a0c94e84ced35569e236f451e03c9614a76a`; analysis SHA256: `de4dfc1bab3943bec76d22cce1c0bb028f4365b4aefe752f037ea0b80d5ebaa4`.
